// ===========================================================================
// bench_team_contraction.cpp
//
// Benchmarks the library's team-policy contraction path against two
// hand-written Kokkos baselines, for the representative contraction
//
//     C[i,l] = sum_{j,k} A[i,j,k] * B[j,k,l]            (a GEMM I x (J*K) x L)
//
// Three implementations of the SAME contraction are compared:
//
//   1. library  — Graph::execute(TeamPolicyTag, Tile<...>, C). Generic: driven
//   by
//                 einsum-style mode labels and StaticTile extents; stages tiles
//                 into team scratch with a *memory-order* (coalesced) access
//                 pattern regardless of the input view's layout.
//   2. naive    — a plain Kokkos team kernel that reads A/B straight from
//   global
//                 memory in the inner loop (no scratch, no reuse). The obvious
//                 first kernel a user writes.
//   3. tiled    — a competent hand-written team+scratch GEMM, hardcoded for
//   this
//                 one rank-3 x rank-3 contraction (stage tiles -> block GEMM ->
//                 store). Mirrors what the library does, but its staging uses a
//                 fixed row-major (LayoutRight-tuned) thread->index mapping.
//
// Each implementation is run with the input views A,B in BOTH
// Kokkos::LayoutRight and Kokkos::LayoutLeft. The library holds throughput
// across layouts (coalesced either way); the baselines degrade on LayoutLeft
// (strided/uncoalesced global reads). Output C is always LayoutRight.
//
// Sizes auto-select CPU- vs GPU-appropriate presets from the default execution
// space. Override the sweep with argv: bench [N [reps [warmup]]].
// ===========================================================================

#include <Kokkos_Core.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/Graph.hpp>
#include <TensorOperations/Tiling.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace TensorOperations;

// ---------------------------------------------------------------------------
// Compile-time configuration (CPU vs GPU), selected from the default exec
// space. J,K (contracted extents) and all tile extents are compile-time so
// scratch sizes are constexpr and StaticTile can use them; I = L = N are
// runtime.
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

inline constexpr int KTOT = J * K;    // total contracted extent
inline constexpr int KK   = TJ * TK;  // contracted tile (flattened)
}  // namespace cfg

// Convenience view aliases. Inputs are templated on layout; output is
// LayoutRight.
template <class L>
using V3 = Kokkos::View<float***, L>;
template <class L>
using V2  = Kokkos::View<float**, L>;
using V2R = Kokkos::View<float**, Kokkos::LayoutRight>;

// ===========================================================================
// Implementation 1: the library. Rank- and pattern-generic.
// ===========================================================================
inline constexpr int kLibSrcBegin = __LINE__;
template <class LA>
void library_contract(V3<LA> A, V3<LA> B, V2R C) {
  auto hA =
      make_input_node(make_handle(A, std::array<int32_t, 3>{'i', 'j', 'k'}));
  auto hB =
      make_input_node(make_handle(B, std::array<int32_t, 3>{'j', 'k', 'l'}));
  auto g = make_graph();
  auto [g1, o1] =
      g.ops(make_contraction_node(hA, hB, std::array<int32_t, 2>{'i', 'l'}));
  g1.execute(TeamPolicyTag<>{},
             Tile<StaticTile<cfg::TI, cfg::TJ, cfg::TK>,
                  StaticTile<cfg::TJ, cfg::TK, cfg::TL>,
                  StaticTile<cfg::TI, cfg::TL>>{},
             C);
}
inline constexpr int kLibSrcEnd = __LINE__;

// ===========================================================================
// Implementation 2: naive team kernel — global-memory reads, no scratch reuse.
// One team per output row i; threads parallelize over output columns l.
// ===========================================================================
inline constexpr int kNaiveSrcBegin = __LINE__;
template <class LA>
void naive_contract(V3<LA> A, V3<LA> B, V2R C, int I, int L) {
  using member_t = Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for(
      "bench_naive", Kokkos::TeamPolicy<>(I, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_t& team) {
        const int i = team.league_rank();
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team, L), [=](int l) {
          float acc = 0.0f;
          for (int j = 0; j < cfg::J; ++j)
            for (int k = 0; k < cfg::K; ++k) acc += A(i, j, k) * B(j, k, l);
          C(i, l) = acc;
        });
      });
}
inline constexpr int kNaiveSrcEnd = __LINE__;

// ===========================================================================
// Implementation 3: hand-tiled team+scratch GEMM, hardcoded for rank-3 x
// rank-3. One team per output tile [TI x TL]; stage A/B tiles into scratch,
// block GEMM, accumulate over contracted tiles, store. The staging fills use a
// fixed row-major (k-fastest / l-fastest) thread->index mapping tuned for
// LayoutRight.
// ===========================================================================
inline constexpr int kTiledSrcBegin = __LINE__;
template <class LA>
void tiled_contract(V3<LA> A, V3<LA> B, V2R C, int I, int L) {
  using ExecSpace    = Kokkos::DefaultExecutionSpace;
  using ScratchSpace = ExecSpace::scratch_memory_space;
  using Pad          = Kokkos::View<float*, ScratchSpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using member_t     = Kokkos::TeamPolicy<ExecSpace>::member_type;

  constexpr int TI = cfg::TI, TL = cfg::TL, TJ = cfg::TJ, TK = cfg::TK,
                KK     = cfg::KK;
  const int     ntl    = L / TL;
  const int     nteams = (I / TI) * ntl;
  const int     nkt_j  = cfg::J / TJ;
  const int     nkt_k  = cfg::K / TK;

  const size_t bytes = Pad::shmem_size(TI * KK) + Pad::shmem_size(KK * TL) +
                       Pad::shmem_size(TI * TL);
  Kokkos::TeamPolicy<ExecSpace> policy(nteams, Kokkos::AUTO);
  policy.set_scratch_size(0, Kokkos::PerTeam(bytes));

  Kokkos::parallel_for(
      "bench_tiled", policy, KOKKOS_LAMBDA(const member_t& team) {
        const int t  = team.league_rank();
        const int i0 = (t / ntl) * TI;
        const int l0 = (t % ntl) * TL;

        Pad As(team.team_scratch(0), TI * KK);
        Pad Bs(team.team_scratch(0), KK * TL);
        Pad Cs(team.team_scratch(0), TI * TL);

        Kokkos::parallel_for(Kokkos::TeamVectorRange(team, TI * TL),
                             [=](int e) { Cs(e) = 0.0f; });
        team.team_barrier();

        for (int ktj = 0; ktj < nkt_j; ++ktj) {
          for (int ktk = 0; ktk < nkt_k; ++ktk) {
            const int j0 = ktj * TJ;
            const int k0 = ktk * TK;
            // Stage A tile [TI x (TJ*TK)], k fastest (contiguous for
            // LayoutRight).
            Kokkos::parallel_for(Kokkos::TeamVectorRange(team, TI * KK),
                                 [=](int e) {
                                   const int dk = e % TK;
                                   const int dj = (e / TK) % TJ;
                                   const int di = e / (TK * TJ);
                                   As(e)        = A(i0 + di, j0 + dj, k0 + dk);
                                 });
            // Stage B tile [(TJ*TK) x TL], l fastest (contiguous for
            // LayoutRight).
            Kokkos::parallel_for(Kokkos::TeamVectorRange(team, KK * TL),
                                 [=](int e) {
                                   const int dl = e % TL;
                                   const int kk = e / TL;  // kk == dj*TK + dk
                                   const int dj = kk / TK;
                                   const int dk = kk % TK;
                                   Bs(e)        = B(j0 + dj, k0 + dk, l0 + dl);
                                 });
            team.team_barrier();
            // Block GEMM: Cs[TI x TL] += As[TI x KK] * Bs[KK x TL].
            Kokkos::parallel_for(Kokkos::TeamVectorRange(team, TI * TL),
                                 [=](int e) {
                                   const int dl  = e % TL;
                                   const int di  = e / TL;
                                   float     acc = Cs(e);
                                   for (int kk = 0; kk < KK; ++kk)
                                     acc += As(di * KK + kk) * Bs(kk * TL + dl);
                                   Cs(e) = acc;
                                 });
            team.team_barrier();
          }
        }
        // Store the accumulated output tile back to global memory.
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team, TI * TL),
            [=](int e) { C(i0 + e / TL, l0 + e % TL) = Cs(e); });
      });
}
inline constexpr int kTiledSrcEnd = __LINE__;

// ---------------------------------------------------------------------------
// Input fill — deterministic, bounded pattern (not all-ones, so an indexing or
// summation bug actually surfaces in the correctness check).
// ---------------------------------------------------------------------------
template <class LA>
void fill_inputs(V3<LA> A, V3<LA> B, int I, int L) {
  Kokkos::parallel_for(
      "fillA",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {I, cfg::J, cfg::K}),
      KOKKOS_LAMBDA(int i, int j, int k) {
        A(i, j, k) = static_cast<float>((i + j + k) % 3 + 1) * 0.5f;
      });
  Kokkos::parallel_for(
      "fillB",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {cfg::J, cfg::K, L}),
      KOKKOS_LAMBDA(int j, int k, int l) {
        B(j, k, l) = static_cast<float>((2 * j + k + l) % 3 + 1) * 0.25f;
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
  int         N;
  const char* layout;
  double      lib, naive, tiled, reldiff;
};

template <class LA>
Row run_case(int N, const char* layout_name, int warmup, int reps) {
  const int    I = N, L = N;
  const double flops = 2.0 * double(I) * double(L) * double(cfg::KTOT);

  V3<LA> A("A", I, cfg::J, cfg::K);
  V3<LA> B("B", cfg::J, cfg::K, L);
  fill_inputs<LA>(A, B, I, L);

  V2R Clib("Clib", I, L), Cnai("Cnai", I, L), Ctil("Ctil", I, L);

  const double g_lib =
      gflops_of([&] { library_contract<LA>(A, B, Clib); }, warmup, reps, flops);
  const double g_nai = gflops_of([&] { naive_contract<LA>(A, B, Cnai, I, L); },
                                 warmup, reps, flops);
  const double g_til = gflops_of([&] { tiled_contract<LA>(A, B, Ctil, I, L); },
                                 warmup, reps, flops);

  const double d =
      std::max(max_rel_diff(Clib, Cnai, I, L), max_rel_diff(Clib, Ctil, I, L));
  return Row{N, layout_name, g_lib, g_nai, g_til, d};
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
        "\n=== Team-policy contraction benchmark =====================\n");
    std::printf("execution space : %s%s\n",
                Kokkos::DefaultExecutionSpace::name(),
                cfg::kIsGPU ? "  (GPU preset)" : "  (CPU preset)");
    std::printf("contraction     : C[i,l] = sum_{j,k} A[i,j,k] * B[j,k,l]\n");
    std::printf("J=%d K=%d (contracted=%d)   tile: TI=%d TL=%d TJ=%d TK=%d\n",
                cfg::J, cfg::K, cfg::KTOT, cfg::TI, cfg::TL, cfg::TJ, cfg::TK);
    std::printf("reps=%d warmup=%d  (GFLOP/s from min wall-clock)\n\n", reps,
                warmup);

    std::printf("%6s %7s %11s %11s %11s %10s %10s %14s\n", "N", "layout",
                "lib G/s", "naive G/s", "tiled G/s", "sx/naive", "sx/tiled",
                "check");
    std::printf(
        "------ ------- ----------- ----------- ----------- ---------- "
        "---------- --------------\n");

    std::vector<Row> rows;
    for (int N : Ns) {
      if (N % cfg::TI != 0 || N % cfg::TL != 0) {
        std::printf("  (skipping N=%d: not divisible by tile %d/%d)\n", N,
                    cfg::TI, cfg::TL);
        continue;
      }
      rows.push_back(run_case<Kokkos::LayoutRight>(N, "Right", warmup, reps));
      rows.push_back(run_case<Kokkos::LayoutLeft>(N, "Left", warmup, reps));
    }

    for (const Row& r : rows) {
      std::printf("%6d %7s %11.3f %11.3f %11.3f %9.2fx %9.2fx   %s(%.1e)\n",
                  r.N, r.layout, r.lib, r.naive, r.tiled, r.lib / r.naive,
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
    // handle arbitrary rank and contraction pattern from mode labels; the two
    // baselines are hardcoded to this one rank-3 x rank-3 contraction.
    std::printf("\nImplementation size (source lines) & generality:\n");
    std::printf("%10s %8s   %s\n", "impl", "lines", "generality");
    std::printf("%10s %8d   %s\n", "library", kLibSrcEnd - kLibSrcBegin - 1,
                "generic: any rank / pattern via einsum modes + StaticTile");
    std::printf("%10s %8d   %s\n", "naive", kNaiveSrcEnd - kNaiveSrcBegin - 1,
                "hardcoded for this rank-3 x rank-3 contraction");
    std::printf("%10s %8d   %s\n", "tiled", kTiledSrcEnd - kTiledSrcBegin - 1,
                "hardcoded for this rank-3 x rank-3 contraction");
    std::printf("\n");
  }
  Kokkos::finalize();
  return 0;
}
