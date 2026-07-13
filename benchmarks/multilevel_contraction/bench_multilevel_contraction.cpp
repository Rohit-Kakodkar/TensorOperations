// ===========================================================================
// bench_multilevel_contraction.cpp
//
// Benchmarks the library's FUSED multi-level (chained) contraction path against
// two hand-written Kokkos baselines, for the representative chained contraction
//
//     M[i,j] = sum_k A[i,k] B[k,j]        (a GEMM  I x K x J)
//     E[i,l] = sum_j M[i,j] D[j,l]        (a GEMM  I x J x L)
//     =>  E = (A·B)·D                      (matrix-chain / two-level GEMM)
//
// This is the matricized form of the operator/transfer chains ubiquitous in
// scientific computing (triple products X^T H X / B^T D B, change-of-basis, the
// reduced form of sum-factorization). Three implementations of the SAME chain
// are compared:
//
//   1. library — Graph::execute(TeamPolicyTag, flat tile list, E). Generic and
//                FUSED: the intermediate M never touches global memory; it is
//                produced tile-by-tile in team scratch (recomputed per output
//                tile). Driven by einsum-style mode labels + StaticTile
//                extents.
//   2. naive   — two passes that MATERIALIZE M to global memory: kernel 1
//                M = A·B, kernel 2 E = M·D, each a plain team kernel reading
//                its operands straight from global memory (no scratch). The
//                obvious first approach; memory-bound on the global M write +
//                repeated M reads.
//   3. tiled   — a straightforward hand-written FUSED team+scratch kernel (the
//                fused kernel a practitioner realistically writes, NOT
//                register- blocked), hardcoded for this chain: one team per E
//                output tile [TI x TL], looping over j-tiles — stage A/B into
//                scratch, compute the M[TI,TJ] tile in scratch, stage D,
//                accumulate into E; no global M. Recomputes M per output tile
//                (same fusion strategy as library). Getting it as fast as the
//                library needs the register-blocking/tiling the library
//                supplies for free.
//
// library and tiled both FUSE (trade global-M traffic for scratch recompute);
// naive materializes. On the memory-bound GPU the fused library path should
// beat naive, and the library-vs-tiled gap shows what the library's
// register-blocking adds on top of the same fusion strategy. Each impl is run
// with A,B,D in BOTH LayoutRight and LayoutLeft (library coalesces either way;
// the baselines degrade on LayoutLeft), and in TWO intermediate regimes:
// single-tile (J = TJ) and multi-tiled (J = 4*TJ, exercises recompute).
//
// Sizes auto-select CPU- vs GPU-appropriate presets from the default execution
// space. Override with argv: bench [N [reps [warmup]]].
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
#include <tuple>
#include <vector>

using namespace TensorOperations;

// ---------------------------------------------------------------------------
// Compile-time configuration (CPU vs GPU). K (inner contracted extent) and all
// tile extents are compile-time so scratch sizes are constexpr and StaticTile
// can use them; I = L = N and J are runtime (J selects the regime).
// ---------------------------------------------------------------------------
namespace cfg {
inline constexpr bool kIsGPU =
    !Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace,
                                Kokkos::HostSpace>::accessible;

inline constexpr int K  = kIsGPU ? 16 : 8;  // inner contracted mode k
inline constexpr int TI = 32;               // output tile along i
inline constexpr int TL = 32;               // output tile along l
inline constexpr int TJ = 32;               // intermediate tile along j
inline constexpr int TK = K;                // inner contracted tile (all of K)

inline constexpr int J_single = TJ;      // single-tile intermediate regime
inline constexpr int J_multi  = 4 * TJ;  // multi-tiled intermediate regime
}  // namespace cfg

// Convenience view aliases. Inputs are templated on layout; intermediate and
// output are LayoutRight.
template <class L>
using V2  = Kokkos::View<float**, L>;
using V2R = Kokkos::View<float**, Kokkos::LayoutRight>;

// ===========================================================================
// Implementation 1: the library. Rank- and pattern-generic, FUSED.
// ===========================================================================
inline constexpr int kLibSrcBegin = __LINE__;
template <class LA>
void library_chain(V2<LA> A, V2<LA> B, V2<LA> D, V2R E) {
  auto hA = make_input_node(make_handle<'i', 'k'>(A));
  auto hB = make_input_node(make_handle<'k', 'j'>(B));
  auto hD = make_input_node(make_handle<'j', 'l'>(D));
  auto M  = make_contraction_node<'i', 'j'>(hA, hB);  // M[i,j] = sum_k
  auto g  = make_graph();
  [[maybe_unused]] auto [g1, En] =
      g.ops(make_contraction_node<'i', 'l'>(M, hD));  // E = sum_j
  // Flat, pre-order tile list: top E=M·D, then inner M=A·B. The top's M-slot
  // tile must equal M's output tile (BundleM.c).
  using BundleTop =
      Tile<StaticTile<cfg::TI, cfg::TJ>, StaticTile<cfg::TJ, cfg::TL>,
           StaticTile<cfg::TI, cfg::TL>>;
  using BundleM =
      Tile<StaticTile<cfg::TI, cfg::TK>, StaticTile<cfg::TK, cfg::TJ>,
           StaticTile<cfg::TI, cfg::TJ>>;
  g1.execute(TeamPolicyTag<>{}, std::make_tuple(BundleTop{}, BundleM{}), E);
}
inline constexpr int kLibSrcEnd = __LINE__;

// ===========================================================================
// Implementation 2: naive — two passes materializing M to global memory. One
// team per output row; threads parallelize over output columns; operands read
// straight from global memory in the inner loop.
// ===========================================================================
inline constexpr int kNaiveSrcBegin = __LINE__;
template <class LA>
void naive_chain(V2<LA> A, V2<LA> B, V2<LA> D, V2R M, V2R E, int I, int J,
                 int L) {
  using member_t = Kokkos::TeamPolicy<>::member_type;
  // Pass 1: M[i,j] = sum_k A[i,k] B[k,j].
  Kokkos::parallel_for(
      "naive_M", Kokkos::TeamPolicy<>(I, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_t& team) {
        const int i = team.league_rank();
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team, J), [=](int j) {
          float acc = 0.0f;
          for (int k = 0; k < cfg::K; ++k) acc += A(i, k) * B(k, j);
          M(i, j) = acc;
        });
      });
  // Pass 2: E[i,l] = sum_j M[i,j] D[j,l].
  Kokkos::parallel_for(
      "naive_E", Kokkos::TeamPolicy<>(I, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_t& team) {
        const int i = team.league_rank();
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team, L), [=](int l) {
          float acc = 0.0f;
          for (int j = 0; j < J; ++j) acc += M(i, j) * D(j, l);
          E(i, l) = acc;
        });
      });
}
inline constexpr int kNaiveSrcEnd = __LINE__;

// ===========================================================================
// Implementation 3: representative hand-written FUSED team+scratch kernel,
// hardcoded for this chain. One team per E output tile [TI x TL]; loop over
// j-tiles: stage A/B into scratch, compute the M[TI,TJ] tile in scratch, stage
// D, accumulate into an E-tile in scratch; write E once. No global M (the
// intermediate is recomputed per output tile, same fusion strategy as the
// library). Scratch views carry runtime extents; contracted inner loops (K, TJ)
// are compile-time so they unroll.
// ===========================================================================
inline constexpr int kTiledSrcBegin = __LINE__;
template <class LA>
void tiled_chain(V2<LA> A, V2<LA> B, V2<LA> D, V2R E, int I, int J, int L) {
  using ExecSpace    = Kokkos::DefaultExecutionSpace;
  using ScratchSpace = ExecSpace::scratch_memory_space;
  using Scratch2     = Kokkos::View<float**, ScratchSpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using member_t     = Kokkos::TeamPolicy<ExecSpace>::member_type;

  constexpr int TI = cfg::TI, TL = cfg::TL, TJ = cfg::TJ, K = cfg::K;
  const int     ntl    = L / TL;
  const int     nteams = (I / TI) * ntl;
  const int     njt    = J / TJ;  // number of j-tiles

  const size_t                  bytes = Scratch2::shmem_size(TI, K) +   // As
                                        Scratch2::shmem_size(K, TJ) +   // Bs
                                        Scratch2::shmem_size(TI, TJ) +  // Ms
                                        Scratch2::shmem_size(TJ, TL) +  // Ds
                                        Scratch2::shmem_size(TI, TL);   // Es
  Kokkos::TeamPolicy<ExecSpace> policy(nteams, Kokkos::AUTO);
  policy.set_scratch_size(0, Kokkos::PerTeam(bytes));

  Kokkos::parallel_for(
      "tiled_chain", policy, KOKKOS_LAMBDA(const member_t& team) {
        const int t  = team.league_rank();
        const int i0 = (t / ntl) * TI;
        const int l0 = (t % ntl) * TL;

        Scratch2 As(team.team_scratch(0), TI, K);
        Scratch2 Bs(team.team_scratch(0), K, TJ);
        Scratch2 Ms(team.team_scratch(0), TI, TJ);
        Scratch2 Ds(team.team_scratch(0), TJ, TL);
        Scratch2 Es(team.team_scratch(0), TI, TL);

        // Zero the E-tile accumulator.
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team, TI * TL),
                             [=](int e) { Es(e / TL, e % TL) = 0.0f; });
        // A[i0:TI, :K] is independent of j — stage it once.
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team, TI * K), [=](int e) {
          const int di = e / K, dk = e % K;
          As(di, dk) = A(i0 + di, dk);
        });
        team.team_barrier();

        for (int jt = 0; jt < njt; ++jt) {
          const int j0 = jt * TJ;
          // Stage B[:K, j-tile] and compute M[TI,TJ] = A·B in scratch.
          Kokkos::parallel_for(Kokkos::TeamVectorRange(team, K * TJ),
                               [=](int e) {
                                 const int dk = e / TJ, dj = e % TJ;
                                 Bs(dk, dj) = B(dk, j0 + dj);
                               });
          team.team_barrier();
          Kokkos::parallel_for(Kokkos::TeamVectorRange(team, TI * TJ),
                               [=](int e) {
                                 const int di = e / TJ, dj = e % TJ;
                                 float     acc = 0.0f;
                                 for (int dk = 0; dk < K; ++dk)
                                   acc += As(di, dk) * Bs(dk, dj);
                                 Ms(di, dj) = acc;
                               });
          // Stage D[j-tile, l0:TL].
          Kokkos::parallel_for(Kokkos::TeamVectorRange(team, TJ * TL),
                               [=](int e) {
                                 const int dj = e / TL, dl = e % TL;
                                 Ds(dj, dl) = D(j0 + dj, l0 + dl);
                               });
          team.team_barrier();
          // Accumulate E[TI,TL] += M[TI,TJ] · D[TJ,TL].
          Kokkos::parallel_for(Kokkos::TeamVectorRange(team, TI * TL),
                               [=](int e) {
                                 const int di = e / TL, dl = e % TL;
                                 float     acc = 0.0f;
                                 for (int dj = 0; dj < TJ; ++dj)
                                   acc += Ms(di, dj) * Ds(dj, dl);
                                 Es(di, dl) += acc;
                               });
          team.team_barrier();
        }
        // Write the E-tile to global memory.
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team, TI * TL),
                             [=](int e) {
                               const int di = e / TL, dl = e % TL;
                               E(i0 + di, l0 + dl) = Es(di, dl);
                             });
      });
}
inline constexpr int kTiledSrcEnd = __LINE__;

// ---------------------------------------------------------------------------
// Input fill — deterministic, bounded pattern (not all-ones, so an indexing or
// summation bug surfaces in the correctness check).
// ---------------------------------------------------------------------------
template <class LA>
void fill_inputs(V2<LA> A, V2<LA> B, V2<LA> D, int I, int J, int L) {
  Kokkos::parallel_for(
      "fillA", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {I, cfg::K}),
      KOKKOS_LAMBDA(int i, int k) {
        A(i, k) = static_cast<float>((i + 2 * k) % 3 + 1) * 0.5f;
      });
  Kokkos::parallel_for(
      "fillB", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {cfg::K, J}),
      KOKKOS_LAMBDA(int k, int j) {
        B(k, j) = static_cast<float>((3 * k + j) % 4 + 1) * 0.25f;
      });
  Kokkos::parallel_for(
      "fillD", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {J, L}),
      KOKKOS_LAMBDA(int j, int l) {
        D(j, l) = static_cast<float>((j + 2 * l) % 3 + 1) * 0.5f;
      });
}

// Max relative difference between two output views (host-side comparison).
double max_rel_diff(V2R X, V2R Y, int I, int L) {
  auto   xh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, X);
  auto   yh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Y);
  double m  = 0.0;
  for (int i = 0; i < I; ++i)
    for (int l = 0; l < L; ++l) {
      const double a = xh(i, l), b = yh(i, l);
      m = std::max(m, std::abs(a - b) / (std::abs(b) + 1e-6));
    }
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

struct Row {
  int         N, J;
  const char* layout;
  double      lib, naive, tiled, reldiff;
};

template <class LA>
Row run_case(int N, int J, const char* layout_name, int warmup, int reps) {
  const int    I = N, L = N;
  const double flops =
      2.0 * double(I) * double(J) * double(cfg::K) +  // M = A·B
      2.0 * double(I) * double(L) * double(J);        // E = M·D

  V2<LA> A("A", I, cfg::K);
  V2<LA> B("B", cfg::K, J);
  V2<LA> D("D", J, L);
  fill_inputs<LA>(A, B, D, I, J, L);

  V2R Elib("Elib", I, L), Enai("Enai", I, L), Etil("Etil", I, L);
  V2R Mtmp("Mtmp", I, J);  // naive's materialized intermediate

  const double g_lib =
      gflops_of([&] { library_chain<LA>(A, B, D, Elib); }, warmup, reps, flops);
  const double g_nai =
      gflops_of([&] { naive_chain<LA>(A, B, D, Mtmp, Enai, I, J, L); }, warmup,
                reps, flops);
  const double g_til = gflops_of(
      [&] { tiled_chain<LA>(A, B, D, Etil, I, J, L); }, warmup, reps, flops);

  const double d =
      std::max(max_rel_diff(Elib, Enai, I, L), max_rel_diff(Elib, Etil, I, L));
  return Row{N, J, layout_name, g_lib, g_nai, g_til, d};
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

    const int Js[] = {cfg::J_single, cfg::J_multi};

    std::printf(
        "\n=== Multi-level (chained) contraction benchmark ===========\n");
    std::printf("execution space : %s%s\n",
                Kokkos::DefaultExecutionSpace::name(),
                cfg::kIsGPU ? "  (GPU preset)" : "  (CPU preset)");
    std::printf(
        "chain           : E = (A·B)·D   "
        "M[i,j]=sum_k A[i,k]B[k,j], E[i,l]=sum_j M[i,j]D[j,l]\n");
    std::printf(
        "K=%d   tile: TI=%d TL=%d TJ=%d TK=%d   "
        "J: %d (single-tile), %d (multi-tiled)\n",
        cfg::K, cfg::TI, cfg::TL, cfg::TJ, cfg::TK, cfg::J_single,
        cfg::J_multi);
    std::printf("reps=%d warmup=%d  (GFLOP/s from min wall-clock)\n\n", reps,
                warmup);

    std::printf("%6s %5s %7s %11s %11s %11s %10s %10s %14s\n", "N", "J",
                "layout", "lib G/s", "naive G/s", "tiled G/s", "sx/naive",
                "sx/tiled", "check");
    std::printf(
        "------ ----- ------- ----------- ----------- ----------- ---------- "
        "---------- --------------\n");

    std::vector<Row> rows;
    for (int N : Ns) {
      if (N % cfg::TI != 0 || N % cfg::TL != 0) {
        std::printf("  (skipping N=%d: not divisible by tile %d/%d)\n", N,
                    cfg::TI, cfg::TL);
        continue;
      }
      for (int J : Js) {
        rows.push_back(
            run_case<Kokkos::LayoutRight>(N, J, "Right", warmup, reps));
        rows.push_back(
            run_case<Kokkos::LayoutLeft>(N, J, "Left", warmup, reps));
      }
    }

    for (const Row& r : rows) {
      std::printf("%6d %5d %7s %11.3f %11.3f %11.3f %9.2fx %9.2fx   %s(%.1e)\n",
                  r.N, r.J, r.layout, r.lib, r.naive, r.tiled, r.lib / r.naive,
                  r.lib / r.tiled, r.reldiff < 1e-2 ? "PASS" : "FAIL",
                  r.reldiff);
    }

    // Layout-robustness summary: average GFLOP/s per impl per layout, and the
    // Left/Right ratio (closer to 1.0 == more layout-robust).
    auto avg = [&](const char* layout, double Row::* field) {
      double s = 0;
      int    n = 0;
      for (const Row& r : rows)
        if (std::string(r.layout) == layout) {
          s += r.*field;
          ++n;
        }
      return n ? s / n : 0.0;
    };
    struct Imp {
      const char* name;
      double Row::* field;
    };
    const Imp imps[] = {
        {"library", &Row::lib}, {"naive", &Row::naive}, {"tiled", &Row::tiled}};
    std::printf("\nLayout robustness (avg GFLOP/s; ratio = Left/Right):\n");
    std::printf("%10s %12s %12s %10s\n", "impl", "Right", "Left", "L/R");
    for (const Imp& im : imps) {
      const double R = avg("Right", im.field), Lt = avg("Left", im.field);
      std::printf("%10s %12.3f %12.3f %9.2f\n", im.name, R, Lt,
                  R > 0 ? Lt / R : 0.0);
    }

    // Lines-of-code / generality. The library call is a few generic lines that
    // handle arbitrary rank and chain from mode labels; the two baselines are
    // hardcoded to this one chained contraction.
    std::printf("\nImplementation size (source lines) & generality:\n");
    std::printf("%10s %8s   %s\n", "impl", "lines", "generality");
    std::printf("%10s %8d   %s\n", "library", kLibSrcEnd - kLibSrcBegin - 1,
                "generic: any rank / chain via einsum modes + StaticTile");
    std::printf("%10s %8d   %s\n", "naive", kNaiveSrcEnd - kNaiveSrcBegin - 1,
                "hardcoded, materializes M to global memory");
    std::printf("%10s %8d   %s\n", "tiled", kTiledSrcEnd - kTiledSrcBegin - 1,
                "hardcoded fused team+scratch for this chain");
    std::printf("\n");
  }
  Kokkos::finalize();
  return 0;
}
