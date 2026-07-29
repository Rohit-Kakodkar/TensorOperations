// ===========================================================================
// bench_permuted_combine.cpp
//
// Measures what it costs a COMBINE node to consume a contraction operand whose
// axis order differs from the combine's output order. Every variant computes
// the same logical result
//
//     P[i,l] = c[i,l] * D[i,l] + 100*i + l ,
//     c[i,l] = sum_{j,k} A[i,j,k] * B[j,k,l]
//
// but presents the fused operand `c` in a different axis order, so each takes a
// different staging path inside the combine evaluator:
//
//   canonical    c[i,l], no hook  -> zero-copy passthrough  (baseline)
//   permOperand  c[l,i], no hook  -> zero-copy RELABEL      (reorder_view)
//   canonHook    c[i,l], hooked   -> passthrough + hook     (hook baseline)
//   permHook     c[l,i], hooked   -> in-place REORDER + hook
//
// The question this exists to answer: a relabeled operand is read at a stride
// from shared memory, where the reorder it replaces leaves the data contiguous
// but pays a full transposition pass plus a team barrier first. Counting
// touches favours the relabel (1 strided read vs. 2 contiguous touches + a
// barrier + 1 read, since a combine reads each operand exactly once), but on
// GPU the strided read can hit shared-memory bank conflicts, and that cost is
// not predictable from first principles.
//
// The hook is the instrument, not the subject. A hooked operand is deliberately
// excluded from the relabel path, so hooking is the only way to force the
// in-place reorder for the same shape -- hence the two ratios to compare:
//
//   permOperand / canonical   isolates the RELABEL   (strided reads)
//   permHook    / canonHook   isolates the REORDER   (transpose + barrier)
//
// A relabel that wins reads < 1.00 on the first ratio relative to the second.
// The hook itself is a no-op pass over the tile and cancels in its own ratio.
//
// Extents are square (I == L) only because the in-place reorder requires equal
// extents on every transposed axis pair -- the relabel does not, which is the
// capability it adds and which no benchmark can measure. J,K are small on
// purpose: this measures OPERAND STAGING, so the GEMM must not dominate.
//
// Sizes auto-select CPU- vs GPU-appropriate presets. Override with argv:
//     bench [N [reps [warmup]]].
// ===========================================================================

#include <Kokkos_Core.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/Graph.hpp>
#include <TensorOperations/Tiling.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace TensorOperations;

// ---------------------------------------------------------------------------
// Compile-time configuration. J,K (contracted) and all tile extents are
// compile-time so StaticTile can use them; I = L = N are runtime.
// ---------------------------------------------------------------------------
namespace cfg {
inline constexpr bool kIsGPU =
    !Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace,
                                Kokkos::HostSpace>::accessible;

inline constexpr int J  = 2;   // contracted mode j (small: staging is the
inline constexpr int K  = 4;   // contracted mode k  subject, not the GEMM)
inline constexpr int TI = 32;  // output tile along i
inline constexpr int TL = 32;  // output tile along l

inline constexpr int KTOT = J * K;  // total contracted extent
}  // namespace cfg

using R  = Kokkos::LayoutRight;
using V3 = Kokkos::View<float***, R>;
using V2 = Kokkos::View<float**, R>;

// Per-operand tile bundles. The contraction operand's `.c` slot is in that
// operand's own order, so the permuted variants transpose it along with the
// A/B slots; the combine output tile stays [i,l] throughout.
using OutTile   = StaticTile<cfg::TI, cfg::TL>;
using CanonBund = Tile<StaticTile<cfg::TI, cfg::J, cfg::K>,
                       StaticTile<cfg::J, cfg::K, cfg::TL>, OutTile>;
using PermBund =
    Tile<StaticTile<cfg::TL, cfg::J, cfg::K>,
         StaticTile<cfg::J, cfg::K, cfg::TI>, StaticTile<cfg::TL, cfg::TI>>;

// Logical tensor generators -- deterministic, bounded, not all-ones (so an
// indexing/transpose bug surfaces in the correctness check).
KOKKOS_INLINE_FUNCTION float fA(int i, int j, int k) {
  return static_cast<float>((i + j + k) % 3 + 1) * 0.5f;
}
KOKKOS_INLINE_FUNCTION float fB(int j, int k, int l) {
  return static_cast<float>((2 * j + k + l) % 3 + 1) * 0.25f;
}
KOKKOS_INLINE_FUNCTION float fD(int i, int l) {
  return static_cast<float>((i + 3 * l) % 7 + 1) * 0.125f;
}

struct MulPlusCoord {
  KOKKOS_FUNCTION float operator()(int i, int l, float c, float d) const {
    return c * d + static_cast<float>(i) * 100.0f + static_cast<float>(l);
  }
};

// A hook that touches every element and changes none, so its cost is the pass
// itself. Present only to push an operand off the relabel path (which requires
// NoHook) and onto the in-place reorder, for the A/B comparison above.
struct TouchHook {
  KOKKOS_FUNCTION void operator()(int, int, float& v) const { v += 0.0f; }
};

// ---------------------------------------------------------------------------
// The four staging-path variants of the identical logical combine, as one
// template: they differ only in the operand's two output modes, its tile
// bundle, and whether it carries a hook.
//
// The permuted variants feed the SAME numbers through a transposed operand:
// A2[l,j,k] = B[j,k,l] and B2[j,k,i] = A[i,j,k], so the sub-contraction
// computes c'[l,i] == c[i,l] and every variant must produce identical P.
//
// This BUILDS the graph and returns a callable that only launches it. Graph
// construction is pure host work -- node/view copies, tile-count and scratch
// sizing, a fresh TeamPolicy -- and folding it into the timed region would add
// the same constant to every variant, compressing exactly the ratios this
// benchmark exists to produce. It matters more here than in the sibling
// permuted-contraction benchmark, which uses a 32x larger contracted extent and
// so drowns the setup in device work.
// ---------------------------------------------------------------------------
template <char M0, char M1, typename Bundle, typename Hook>
auto make_variant(const V3& OpA, const V3& OpB, const V2& D, const V2& P) {
  auto nc = make_contraction_node<float, Kokkos::DefaultExecutionSpace, M0, M1>(
      make_input_node(make_handle<M0, 'j', 'k'>(OpA)),
      make_input_node(make_handle<'j', 'k', M1>(OpB)), Hook{});
  auto nd       = make_input_node(make_handle<'i', 'l'>(D));
  auto g        = make_graph();
  auto [g1, o1] = g.ops(make_combine_node<'i', 'l'>(nc, nd, MulPlusCoord{}));
  return [g1, P] {
    g1.execute(TeamPolicyTag<>{},
               make_combine_tile(OutTile{}, Bundle{}, OutTile{}), P);
  };
}

// ---------------------------------------------------------------------------
// Fills. Written out per operand rather than templated on a generator: fA/fB
// are device functions, and passing them through a functor parameter would mean
// taking their address on the host.
// ---------------------------------------------------------------------------
void fillA(const V3& A, int I) {  // stored [I,J,K]
  Kokkos::parallel_for(
      "fillA",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {I, cfg::J, cfg::K}),
      KOKKOS_LAMBDA(int i, int j, int k) { A(i, j, k) = fA(i, j, k); });
}
void fillB(const V3& B, int L) {  // stored [J,K,L]
  Kokkos::parallel_for(
      "fillB",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {cfg::J, cfg::K, L}),
      KOKKOS_LAMBDA(int j, int k, int l) { B(j, k, l) = fB(j, k, l); });
}
void fillA2(const V3& A2, int L) {  // stored [L,J,K], holds B's numbers
  Kokkos::parallel_for(
      "fillA2",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {L, cfg::J, cfg::K}),
      KOKKOS_LAMBDA(int l, int j, int k) { A2(l, j, k) = fB(j, k, l); });
}
void fillB2(const V3& B2, int I) {  // stored [J,K,I], holds A's numbers
  Kokkos::parallel_for(
      "fillB2",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {cfg::J, cfg::K, I}),
      KOKKOS_LAMBDA(int j, int k, int i) { B2(j, k, i) = fA(i, j, k); });
}
void fillD(const V2& D, int I, int L) {
  Kokkos::parallel_for(
      "fillD", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {I, L}),
      KOKKOS_LAMBDA(int i, int l) { D(i, l) = fD(i, l); });
}

std::vector<double> to_host(V2 P, int I, int L) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, P);
  std::vector<double> out(static_cast<std::size_t>(I) * L);
  for (int i = 0; i < I; ++i)
    for (int l = 0; l < L; ++l)
      out[static_cast<std::size_t>(i) * L + l] = h(i, l);
  return out;
}
double max_rel_diff(const std::vector<double>& x,
                    const std::vector<double>& y) {
  double m = 0.0;
  for (std::size_t n = 0; n < x.size(); ++n)
    m = std::max(m, std::abs(x[n] - y[n]) / (std::abs(y[n]) + 1e-6));
  return m;
}

// Time a callable: warmup iters, then min wall-clock over reps -> GFLOP/s.
template <class Fn>
double gflops_of(Fn&& fn, int warmup, int reps, double flops) {
  for (int w = 0; w < warmup; ++w) fn();
  Kokkos::fence();
  double best = 1e300;
  for (int r = 0; r < reps; ++r) {
    Kokkos::Timer timer;
    fn();
    Kokkos::fence();
    best = std::min(best, timer.seconds());
  }
  return flops / best / 1e9;
}

struct VRow {
  const char* name;
  const char* path;
  double      gflops;
  double      ratio;    // vs this variant's own baseline
  double      reldiff;  // vs the canonical result
};

std::vector<VRow> run_case(int N, int warmup, int reps) {
  const int I = N, L = N;
  // Contraction FLOPs plus the combine's own multiply-add per output element.
  const double flops =
      2.0 * double(I) * double(L) * double(cfg::KTOT) + 2.0 * double(I) * L;

  V3 A("A", I, cfg::J, cfg::K);
  V3 B("B", cfg::J, cfg::K, L);
  V3 A2("A2", L, cfg::J, cfg::K);
  V3 B2("B2", cfg::J, cfg::K, I);
  V2 D("D", I, L);
  fillA(A, I);
  fillB(B, L);
  fillA2(A2, L);
  fillB2(B2, I);
  fillD(D, I, L);

  V2 P("P", I, L);

  // Built once, outside every timed region (see make_variant).
  auto canonical = make_variant<'i', 'l', CanonBund, NoHook>(A, B, D, P);
  auto perm      = make_variant<'l', 'i', PermBund, NoHook>(A2, B2, D, P);
  auto canonHook = make_variant<'i', 'l', CanonBund, TouchHook>(A, B, D, P);
  auto permHook  = make_variant<'l', 'i', PermBund, TouchHook>(A2, B2, D, P);

  const double g_canon = gflops_of(canonical, warmup, reps, flops);
  const auto   ref     = to_host(P, I, L);

  auto measure = [&](const char* name, const char* path, auto&& fn,
                     double baseline) {
    const double g = gflops_of(fn, warmup, reps, flops);
    const double d = max_rel_diff(to_host(P, I, L), ref);
    return VRow{name, path, g, g / baseline, d};
  };

  std::vector<VRow> rows;
  rows.push_back(VRow{"canonical", "passthrough", g_canon, 1.0, 0.0});
  rows.push_back(measure("permOperand", "relabel", perm, g_canon));

  // Hooked pair: its own baseline, so the hook's cost cancels in the ratio.
  const double g_canon_hook = gflops_of(canonHook, warmup, reps, flops);
  rows.push_back(VRow{"canonHook", "passthrough+hook", g_canon_hook, 1.0,
                      max_rel_diff(to_host(P, I, L), ref)});
  rows.push_back(
      measure("permHook", "in-place reorder+hook", permHook, g_canon_hook));
  return rows;
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int reps = 10, warmup = 3, override_N = 0;
    if (argc > 1) override_N = std::atoi(argv[1]);
    if (argc > 2) reps = std::atoi(argv[2]);
    if (argc > 3) warmup = std::atoi(argv[3]);

    std::vector<int> Ns;
    if (override_N > 0)
      Ns = {override_N};
    else if (cfg::kIsGPU)
      Ns = {1024, 2048, 4096};
    else
      Ns = {256, 512, 1024};

    std::printf(
        "\n=== Permuted-combine (operand staging) benchmark ===========\n");
    std::printf("execution space : %s%s\n",
                Kokkos::DefaultExecutionSpace::name(),
                cfg::kIsGPU ? "  (GPU preset)" : "  (CPU preset)");
    std::printf(
        "logical         : P[i,l] = c[i,l]*D[i,l] + 100i + l,\n"
        "                  c[i,l] = sum_{j,k} A[i,j,k]*B[j,k,l]\n");
    std::printf("J=%d K=%d (contracted=%d)   tile: TI=%d TL=%d\n", cfg::J,
                cfg::K, cfg::KTOT, cfg::TI, cfg::TL);
    std::printf(
        "reps=%d warmup=%d  (GFLOP/s from min wall-clock; ratio vs that "
        "variant's baseline)\n",
        reps, warmup);
    std::printf(
        "Compare permOperand's ratio (relabel) against permHook's (in-place "
        "reorder):\n"
        "the higher ratio is the cheaper way to consume a permuted operand.\n"
        "\n");

    bool all_ok = true;
    for (int N : Ns) {
      if (N % cfg::TI != 0 || N % cfg::TL != 0) {
        std::printf("  (skipping N=%d: not divisible by tile %d/%d)\n", N,
                    cfg::TI, cfg::TL);
        continue;
      }
      std::printf("N=%d\n", N);
      std::printf("  %-12s %-22s %11s %8s %10s\n", "variant", "staging path",
                  "lib G/s", "ratio", "check");
      std::printf(
          "  ------------ ---------------------- ----------- -------- "
          "----------\n");
      for (const VRow& r : run_case(N, warmup, reps)) {
        const bool result_ok = r.reldiff < 1e-2;
        all_ok               = all_ok && result_ok;
        std::printf("  %-12s %-22s %11.3f %7.2fx %s(%.1e)\n", r.name, r.path,
                    r.gflops, r.ratio, result_ok ? "PASS" : "FAIL", r.reldiff);
      }
      std::printf("\n");
    }

    std::printf("Overall: %s\n\n",
                all_ok ? "PASS (every staging path agrees on the result)"
                       : "FAIL (a variant produced a different result)");
  }
  Kokkos::finalize();
  return 0;
}
