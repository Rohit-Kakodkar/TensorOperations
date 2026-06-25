// ===========================================================================
// bench_matrix_multiply.cpp
//
// Benchmarks the library's team-policy contraction path against a hand-written
// Kokkos baseline for plain matrix-matrix multiplication
//
//     C[i,j] = sum_{k} A[i,k] * B[k,j]                  (a dense GEMM I x K x
//     J)
//
// Three implementations of the SAME product are compared:
//
//   1. library  — Graph::execute(TeamPolicyTag, Tile<...>, C). Generic: driven
//                 by einsum-style mode labels and StaticTile extents; stages
//                 tiles into team scratch with a *memory-order* (coalesced)
//                 access pattern regardless of the input view's layout.
//   2. naive    — a plain Kokkos team kernel that reads A/B straight from
//                 global memory in the inner loop (no scratch, no reuse). The
//                 obvious first kernel a user writes.
//   3. kkblas   — KokkosKernels::gemm (BLAS3). Routes to cuBLAS on GPU and
//                 CPU BLAS (MKL/OpenBLAS) on CPU. Vendor-optimized reference.
//
// Each implementation is run with the input views A,B in BOTH
// Kokkos::LayoutRight and Kokkos::LayoutLeft. The library holds throughput
// across layouts (coalesced either way); the naive baseline degrades on
// LayoutLeft (strided/uncoalesced global reads). Output C is always
// LayoutRight.
//
// Sizes auto-select CPU- vs GPU-appropriate presets from the default execution
// space. Override the sweep with argv: bench [N [reps [warmup]]].
// ===========================================================================

#include <Kokkos_Core.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/Graph.hpp>
#include <TensorOperations/Tiling.hpp>
#include <KokkosBlas3_gemm.hpp>

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
// space. The tile extents are compile-time so scratch sizes are constexpr and
// StaticTile can use them; the three matrix extents I = J = K = N are runtime
// (a square N x N x N product). N only has to be divisible by the tile sizes.
// ---------------------------------------------------------------------------
namespace cfg {
inline constexpr bool kIsGPU =
    !Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace,
                                Kokkos::HostSpace>::accessible;

inline constexpr int TI = 64;  // output tile along i
inline constexpr int TJ = 64;  // output tile along j
inline constexpr int TK = 16;  // contracted tile along k
}  // namespace cfg

// Convenience view aliases. Inputs are templated on layout; output is
// LayoutRight.
template <class L>
using V2  = Kokkos::View<float**, L>;
using V2R = Kokkos::View<float**, Kokkos::LayoutRight>;

// ===========================================================================
// Implementation 1: the library. Rank- and pattern-generic.
// ===========================================================================
inline constexpr int kLibSrcBegin = __LINE__;
template <class LA>
void library_matmul(V2<LA> A, V2<LA> B, V2R C) {
  auto hA = make_input_node(make_handle(A, std::array<int32_t, 2>{'i', 'k'}));
  auto hB = make_input_node(make_handle(B, std::array<int32_t, 2>{'k', 'j'}));
  auto g  = make_graph();
  auto [g1, o1] =
      g.ops(make_contraction_node(hA, hB, std::array<int32_t, 2>{'i', 'j'}));
  g1.execute(TeamPolicyTag<>{},
             Tile<StaticTile<cfg::TI, cfg::TK>, StaticTile<cfg::TK, cfg::TJ>,
                  StaticTile<cfg::TI, cfg::TJ>>{},
             C);
}
inline constexpr int kLibSrcEnd = __LINE__;

// ===========================================================================
// Implementation 2: naive team kernel — global-memory reads, no scratch reuse.
// One team per output row i; threads parallelize over output columns j.
// ===========================================================================
inline constexpr int kNaiveSrcBegin = __LINE__;
template <class LA>
void naive_matmul(V2<LA> A, V2<LA> B, V2R C, int I, int J, int K) {
  using member_t = Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for(
      "bench_naive", Kokkos::TeamPolicy<>(I, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_t& team) {
        const int i = team.league_rank();
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team, J), [=](int j) {
          float acc = 0.0f;
          for (int k = 0; k < K; ++k) acc += A(i, k) * B(k, j);
          C(i, j) = acc;
        });
      });
}
inline constexpr int kNaiveSrcEnd = __LINE__;

// ===========================================================================
// Implementation 3: KokkosKernels BLAS3 GEMM — cuBLAS on GPU, CPU BLAS on CPU.
// ===========================================================================
inline constexpr int kKKSrcBegin = __LINE__;
template <class LA>
void kkblas_matmul(V2<LA> A, V2<LA> B, V2R C) {
  KokkosBlas::gemm("N", "N", 1.0f, A, B, 0.0f, C);
}
inline constexpr int kKKSrcEnd = __LINE__;

// ---------------------------------------------------------------------------
// Input fill — deterministic, bounded pattern (not all-ones, so an indexing or
// summation bug actually surfaces in the correctness check).
// ---------------------------------------------------------------------------
template <class LA>
void fill_inputs(V2<LA> A, V2<LA> B, int I, int J, int K) {
  Kokkos::parallel_for(
      "fillA", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {I, K}),
      KOKKOS_LAMBDA(int i, int k) {
        A(i, k) = static_cast<float>((i + k) % 3 + 1) * 0.5f;
      });
  Kokkos::parallel_for(
      "fillB", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {K, J}),
      KOKKOS_LAMBDA(int k, int j) {
        B(k, j) = static_cast<float>((2 * k + j) % 3 + 1) * 0.25f;
      });
}

// Max relative difference between two output views (host-side comparison).
double max_rel_diff(V2R X, V2R Y, int I, int J) {
  auto   xh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, X);
  auto   yh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Y);
  double m  = 0.0;
  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j) {
      const double a = xh(i, j), b = yh(i, j);
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
  double      lib, naive, kkblas, reldiff, reldiff_kk;
};

template <class LA>
Row run_case(int N, const char* layout_name, int warmup, int reps) {
  const int    I = N, J = N, K = N;
  const double flops = 2.0 * double(I) * double(J) * double(K);

  V2<LA> A("A", I, K);
  V2<LA> B("B", K, J);
  fill_inputs<LA>(A, B, I, J, K);

  V2R Clib("Clib", I, J), Cnai("Cnai", I, J), Ckk("Ckk", I, J);

  const double g_lib =
      gflops_of([&] { library_matmul<LA>(A, B, Clib); }, warmup, reps, flops);
  const double g_nai = gflops_of([&] { naive_matmul<LA>(A, B, Cnai, I, J, K); },
                                 warmup, reps, flops);
  const double g_kk =
      gflops_of([&] { kkblas_matmul<LA>(A, B, Ckk); }, warmup, reps, flops);

  const double d    = max_rel_diff(Clib, Cnai, I, J);
  const double d_kk = max_rel_diff(Clib, Ckk, I, J);
  return Row{N, layout_name, g_lib, g_nai, g_kk, d, d_kk};
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
      Ns = {256, 512, 1024};  // steady-state sizes; <=128 is overhead-bound

    std::printf(
        "\n=== Matrix-matrix multiply benchmark ======================\n");
    std::printf("execution space : %s%s\n",
                Kokkos::DefaultExecutionSpace::name(),
                cfg::kIsGPU ? "  (GPU preset)" : "  (CPU preset)");
    std::printf(
        "product         : C[i,j] = sum_{k} A[i,k] * B[k,j]  (N x N x N)\n");
    std::printf("tile: TI=%d TJ=%d TK=%d\n", cfg::TI, cfg::TJ, cfg::TK);
    std::printf("reps=%d warmup=%d  (GFLOP/s from min wall-clock)\n\n", reps,
                warmup);

    std::printf("%6s %7s %11s %11s %11s %10s %10s %14s\n", "N", "layout",
                "lib G/s", "naive G/s", "kk G/s", "sx/naive", "lib/kk",
                "check");
    std::printf(
        "------ ------- ----------- ----------- ----------- ---------- "
        "---------- --------------\n");

    std::vector<Row> rows;
    for (int N : Ns) {
      if (N % cfg::TI != 0 || N % cfg::TJ != 0 || N % cfg::TK != 0) {
        std::printf("  (skipping N=%d: not divisible by tile %d/%d/%d)\n", N,
                    cfg::TI, cfg::TJ, cfg::TK);
        continue;
      }
      rows.push_back(run_case<Kokkos::LayoutRight>(N, "Right", warmup, reps));
      rows.push_back(run_case<Kokkos::LayoutLeft>(N, "Left", warmup, reps));
    }

    for (const Row& r : rows) {
      char chk[24];
      std::snprintf(chk, sizeof(chk), "%s(%.0e)/%s(%.0e)",
                    r.reldiff < 1e-2 ? "P" : "F", r.reldiff,
                    r.reldiff_kk < 1e-2 ? "P" : "F", r.reldiff_kk);
      std::printf("%6d %7s %11.3f %11.3f %11.3f %9.2fx %9.2fx   %-14s\n", r.N,
                  r.layout, r.lib, r.naive, r.kkblas, r.lib / r.naive,
                  r.lib / r.kkblas, chk);
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
    const Imp imps[] = {{"library", &Row::lib},
                        {"naive", &Row::naive},
                        {"kkblas", &Row::kkblas}};
    std::printf("\nLayout robustness (avg GFLOP/s; ratio = Left/Right):\n");
    std::printf("%10s %12s %12s %10s\n", "impl", "Right", "Left", "L/R");
    for (const Imp& im : imps) {
      const double R = avg("Right", im.field), Lt = avg("Left", im.field);
      std::printf("%10s %12.3f %12.3f %9.2f\n", im.name, R, Lt,
                  R > 0 ? Lt / R : 0.0);
    }

    // Lines-of-code / generality. The library call is a few generic lines that
    // handle arbitrary rank and contraction pattern from mode labels; the naive
    // baseline is hardcoded to this one rank-2 x rank-2 product.
    std::printf("\nImplementation size (source lines) & generality:\n");
    std::printf("%10s %8s   %s\n", "impl", "lines", "generality");
    std::printf("%10s %8d   %s\n", "library", kLibSrcEnd - kLibSrcBegin - 1,
                "generic: any rank / pattern via einsum modes + StaticTile");
    std::printf("%10s %8d   %s\n", "naive", kNaiveSrcEnd - kNaiveSrcBegin - 1,
                "hardcoded for this rank-2 x rank-2 product");
    std::printf("%10s %8d   %s\n", "kkblas", kKKSrcEnd - kKKSrcBegin - 1,
                "generic: KokkosKernels BLAS3 (Internal backend)");
    std::printf("\n");
  }
  Kokkos::finalize();
  return 0;
}
