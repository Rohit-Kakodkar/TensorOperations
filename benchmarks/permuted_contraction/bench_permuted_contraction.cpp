// ===========================================================================
// bench_permuted_contraction.cpp
//
// Verifies that the library's team-policy contraction throughput is invariant
// to the *index (axis-label) order* of the operands and output. Every variant
// computes the SAME logical contraction
//
//     C[i,l] = sum_{j,k} A[i,j,k] * B[j,k,l]            (a GEMM I x (J*K) x L)
//
// but presents A / B / C axes in a different label order, so each exercises a
// different internal permutation (permA / permB / permC):
//
//   canonical    A[i,j,k]  B[j,k,l]  -> C[i,l]   (all identity)
//   permInputA   A[j,i,k]  B[j,k,l]  -> C[i,l]   (A input permuted)
//   permInputB   A[i,j,k]  B[k,j,l]  -> C[i,l]   (B input permuted)
//   permOutput   A[i,j,k]  B[j,k,l]  -> C[l,i]   (output permuted)
//   permAll      A[j,i,k]  B[k,j,l]  -> C[l,i]   (all permuted)
//
// A permuted operand is presented to the GEMM as a strided relabel; the library
// takes the tile subview in the operand's *native* axis order (the compile-time
// ordered, register-resident subview path) and reorders it, so the permuted
// variants should match the canonical throughput (ratio ~ 1.0). Before that
// change they fell onto a runtime strided subview (local-memory spill on GPU)
// and were measurably slower.
//
// Each stored view is filled from the same logical tensor (transposed into the
// variant's layout), so the results are bit-comparable to the canonical run and
// double as a correctness check. Inputs are LayoutRight throughout: the axis
// (index) order is the variable under test, held apart from the memory-layout
// robustness covered by bench_team_contraction.
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
#include <string>
#include <vector>

using namespace TensorOperations;

// ---------------------------------------------------------------------------
// Compile-time configuration (CPU vs GPU). J,K (contracted) and all tile
// extents are compile-time so StaticTile can use them; I = L = N are runtime.
// ---------------------------------------------------------------------------
namespace cfg {
inline constexpr bool kIsGPU =
    !Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace,
                                Kokkos::HostSpace>::accessible;

inline constexpr int J  = kIsGPU ? 16 : 8;  // contracted mode j
inline constexpr int K  = kIsGPU ? 16 : 8;  // contracted mode k
inline constexpr int TI = 32;               // output tile along i
inline constexpr int TL = 32;               // output tile along l
inline constexpr int TJ = 8;                // contracted tile along j
inline constexpr int TK = 8;                // contracted tile along k

inline constexpr int KTOT = J * K;  // total contracted extent
}  // namespace cfg

using R  = Kokkos::LayoutRight;
using V3 = Kokkos::View<float***, R>;
using V2 = Kokkos::View<float**, R>;

// Logical tensor generators — deterministic, bounded, not all-ones (so an
// indexing/transpose bug surfaces in the correctness check).
KOKKOS_INLINE_FUNCTION float fA(int i, int j, int k) {
  return static_cast<float>((i + j + k) % 3 + 1) * 0.5f;
}
KOKKOS_INLINE_FUNCTION float fB(int j, int k, int l) {
  return static_cast<float>((2 * j + k + l) % 3 + 1) * 0.25f;
}

// ---------------------------------------------------------------------------
// The five index-order variants of the identical logical contraction. Each
// operand/output tile is given in that operand's own label order (matching the
// handle labels), so the library sees a distinct permA / permB / permC.
// ---------------------------------------------------------------------------
void con_canonical(V3 A, V3 B, V2 C) {  // A[i,j,k] B[j,k,l] -> C[i,l]
  auto hA       = make_input_node(make_handle<'i', 'j', 'k'>(A));
  auto hB       = make_input_node(make_handle<'j', 'k', 'l'>(B));
  auto g        = make_graph();
  auto [g1, o1] = g.ops(make_contraction_node<'i', 'l'>(hA, hB));
  g1.execute(TeamPolicyTag<>{},
             Tile<StaticTile<cfg::TI, cfg::TJ, cfg::TK>,
                  StaticTile<cfg::TJ, cfg::TK, cfg::TL>,
                  StaticTile<cfg::TI, cfg::TL>>{},
             C);
}
void con_permA(V3 A, V3 B, V2 C) {  // A[j,i,k] B[j,k,l] -> C[i,l]
  auto hA       = make_input_node(make_handle<'j', 'i', 'k'>(A));
  auto hB       = make_input_node(make_handle<'j', 'k', 'l'>(B));
  auto g        = make_graph();
  auto [g1, o1] = g.ops(make_contraction_node<'i', 'l'>(hA, hB));
  g1.execute(TeamPolicyTag<>{},
             Tile<StaticTile<cfg::TJ, cfg::TI, cfg::TK>,
                  StaticTile<cfg::TJ, cfg::TK, cfg::TL>,
                  StaticTile<cfg::TI, cfg::TL>>{},
             C);
}
void con_permB(V3 A, V3 B, V2 C) {  // A[i,j,k] B[k,j,l] -> C[i,l]
  auto hA       = make_input_node(make_handle<'i', 'j', 'k'>(A));
  auto hB       = make_input_node(make_handle<'k', 'j', 'l'>(B));
  auto g        = make_graph();
  auto [g1, o1] = g.ops(make_contraction_node<'i', 'l'>(hA, hB));
  g1.execute(TeamPolicyTag<>{},
             Tile<StaticTile<cfg::TI, cfg::TJ, cfg::TK>,
                  StaticTile<cfg::TK, cfg::TJ, cfg::TL>,
                  StaticTile<cfg::TI, cfg::TL>>{},
             C);
}
void con_permOut(V3 A, V3 B, V2 C) {  // A[i,j,k] B[j,k,l] -> C[l,i]
  auto hA       = make_input_node(make_handle<'i', 'j', 'k'>(A));
  auto hB       = make_input_node(make_handle<'j', 'k', 'l'>(B));
  auto g        = make_graph();
  auto [g1, o1] = g.ops(make_contraction_node<'l', 'i'>(hA, hB));
  g1.execute(TeamPolicyTag<>{},
             Tile<StaticTile<cfg::TI, cfg::TJ, cfg::TK>,
                  StaticTile<cfg::TJ, cfg::TK, cfg::TL>,
                  StaticTile<cfg::TL, cfg::TI>>{},
             C);
}
void con_permAll(V3 A, V3 B, V2 C) {  // A[j,i,k] B[k,j,l] -> C[l,i]
  auto hA       = make_input_node(make_handle<'j', 'i', 'k'>(A));
  auto hB       = make_input_node(make_handle<'k', 'j', 'l'>(B));
  auto g        = make_graph();
  auto [g1, o1] = g.ops(make_contraction_node<'l', 'i'>(hA, hB));
  g1.execute(TeamPolicyTag<>{},
             Tile<StaticTile<cfg::TJ, cfg::TI, cfg::TK>,
                  StaticTile<cfg::TK, cfg::TJ, cfg::TL>,
                  StaticTile<cfg::TL, cfg::TI>>{},
             C);
}

// ---------------------------------------------------------------------------
// Fills: store the same logical A/B into the variant's axis order.
// ---------------------------------------------------------------------------
void fillA_ijk(V3 A, int I) {  // stored [I,J,K]
  Kokkos::parallel_for(
      "fillA_ijk",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {I, cfg::J, cfg::K}),
      KOKKOS_LAMBDA(int i, int j, int k) { A(i, j, k) = fA(i, j, k); });
}
void fillA_jik(V3 A, int I) {  // stored [J,I,K]
  Kokkos::parallel_for(
      "fillA_jik",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {cfg::J, I, cfg::K}),
      KOKKOS_LAMBDA(int j, int i, int k) { A(j, i, k) = fA(i, j, k); });
}
void fillB_jkl(V3 B, int L) {  // stored [J,K,L]
  Kokkos::parallel_for(
      "fillB_jkl",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {cfg::J, cfg::K, L}),
      KOKKOS_LAMBDA(int j, int k, int l) { B(j, k, l) = fB(j, k, l); });
}
void fillB_kjl(V3 B, int L) {  // stored [K,J,L]
  Kokkos::parallel_for(
      "fillB_kjl",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {cfg::K, cfg::J, L}),
      KOKKOS_LAMBDA(int k, int j, int l) { B(k, j, l) = fB(j, k, l); });
}

// Copy a result view into a canonical [I,L] host buffer (row-major i*L+l).
// `transposed` handles the C[l,i] output layout.
std::vector<double> to_host_IL(V2 C, bool transposed, int I, int L) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  std::vector<double> out(static_cast<std::size_t>(I) * L);
  for (int i = 0; i < I; ++i)
    for (int l = 0; l < L; ++l)
      out[static_cast<std::size_t>(i) * L + l] = transposed ? h(l, i) : h(i, l);
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
  const char* shape;
  double      gflops;
  double      ratio;    // vs canonical
  double      reldiff;  // vs canonical result
};

std::vector<VRow> run_case(int N, int warmup, int reps) {
  const int    I = N, L = N;
  const double flops = 2.0 * double(I) * double(L) * double(cfg::KTOT);

  // Shared inputs in each needed layout (filled once, reused across reps).
  V3 A_ijk("A_ijk", I, cfg::J, cfg::K);
  V3 A_jik("A_jik", cfg::J, I, cfg::K);
  V3 B_jkl("B_jkl", cfg::J, cfg::K, L);
  V3 B_kjl("B_kjl", cfg::K, cfg::J, L);
  fillA_ijk(A_ijk, I);
  fillA_jik(A_jik, I);
  fillB_jkl(B_jkl, L);
  fillB_kjl(B_kjl, L);

  V2 C_il("C_il", I, L);
  V2 C_li("C_li", L, I);

  // Canonical reference (throughput baseline + correctness reference).
  const double g0  = gflops_of([&] { con_canonical(A_ijk, B_jkl, C_il); },
                               warmup, reps, flops);
  const auto   ref = to_host_IL(C_il, false, I, L);

  auto measure = [&](const char* name, const char* shape, auto&& fn, V2 out,
                     bool transposed) {
    const double g = gflops_of(fn, warmup, reps, flops);
    const double d = max_rel_diff(to_host_IL(out, transposed, I, L), ref);
    return VRow{name, shape, g, g / g0, d};
  };

  std::vector<VRow> rows;
  rows.push_back(VRow{"canonical", "A[i,j,k] B[j,k,l] C[i,l]", g0, 1.0, 0.0});
  rows.push_back(measure(
      "permInputA", "A[j,i,k]", [&] { con_permA(A_jik, B_jkl, C_il); }, C_il,
      false));
  rows.push_back(measure(
      "permInputB", "B[k,j,l]", [&] { con_permB(A_ijk, B_kjl, C_il); }, C_il,
      false));
  rows.push_back(measure(
      "permOutput", "C[l,i]", [&] { con_permOut(A_ijk, B_jkl, C_li); }, C_li,
      true));
  rows.push_back(measure(
      "permAll", "A[j,i,k] B[k,j,l] C[l,i]",
      [&] { con_permAll(A_jik, B_kjl, C_li); }, C_li, true));
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
        "\n=== Permuted-contraction (index-order) benchmark ===========\n");
    std::printf("execution space : %s%s\n",
                Kokkos::DefaultExecutionSpace::name(),
                cfg::kIsGPU ? "  (GPU preset)" : "  (CPU preset)");
    std::printf("logical         : C[i,l] = sum_{j,k} A[i,j,k] * B[j,k,l]\n");
    std::printf("J=%d K=%d (contracted=%d)   tile: TI=%d TL=%d TJ=%d TK=%d\n",
                cfg::J, cfg::K, cfg::KTOT, cfg::TI, cfg::TL, cfg::TJ, cfg::TK);
    std::printf(
        "reps=%d warmup=%d  (GFLOP/s from min wall-clock; ratio vs "
        "canonical)\n",
        reps, warmup);
    std::printf(
        "PASS = ratio within 15%% of canonical AND result matches "
        "canonical.\n\n");

    bool all_ok = true;
    for (int N : Ns) {
      if (N % cfg::TI != 0 || N % cfg::TL != 0) {
        std::printf("  (skipping N=%d: not divisible by tile %d/%d)\n", N,
                    cfg::TI, cfg::TL);
        continue;
      }
      std::printf("N=%d\n", N);
      std::printf("  %-11s %-26s %11s %8s %10s %6s\n", "variant", "shape",
                  "lib G/s", "ratio", "check", "perf");
      std::printf(
          "  ----------- -------------------------- ----------- -------- "
          "---------- ------\n");
      for (const VRow& r : run_case(N, warmup, reps)) {
        const bool result_ok = r.reldiff < 1e-2;
        const bool perf_ok   = r.ratio > 0.85;  // within 15% of canonical
        all_ok               = all_ok && result_ok && perf_ok;
        std::printf("  %-11s %-26s %11.3f %7.2fx %s(%.1e) %5s\n", r.name,
                    r.shape, r.gflops, r.ratio, result_ok ? "PASS" : "FAIL",
                    r.reldiff, perf_ok ? "ok" : "SLOW");
      }
      std::printf("\n");
    }

    std::printf("Overall: %s\n\n",
                all_ok ? "PASS (index order does not affect performance)"
                       : "FAIL (a permuted variant regressed or mismatched)");
  }
  Kokkos::finalize();
  return 0;
}
