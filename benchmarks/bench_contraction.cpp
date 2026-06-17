#include <Kokkos_Core.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/Graph.hpp>
#include <TensorOperations/Tiling.hpp>
#include <TensorOperations/TimingInstrumentation.hpp>
#include <TensorOperations/TimingAnalysis.hpp>

#include <cmath>
#include <cstdio>

using namespace TensorOperations;

// C[I,L] = sum_{j,k} A[I,j,k] * B[j,k,L]
//
// J and K (the contracted modes) stay compile-time constants because the
// register tier fully stages a [.,J,K] / [J,K,.] block per work item — the
// StaticTile and RegisterArray sizes must be known at compile time. The free
// modes I and L are runtime: they only set the work-item count and the output
// shape, both of which the library reads from the View extents.
constexpr int J = 8;
constexpr int K = 8;

using View2  = Kokkos::View<float**, Kokkos::LayoutRight>;
using View3  = Kokkos::View<float***, Kokkos::LayoutRight>;
using View2L = Kokkos::View<float**, Kokkos::LayoutLeft>;
using View3L = Kokkos::View<float***, Kokkos::LayoutLeft>;

// ---------------------------------------------------------------------------
// Naive: one output element per work item, no data reuse between work items.
// Arithmetic intensity: J*K MACs / (J*K + J*K loads) = 0.5 MACs/load.
//
// For a fixed row i the inner sweep over l re-streams the entire B[J,K,L]
// tensor once. Over all I rows that is I full passes over B. When B fits in
// cache (small problem) those passes are nearly free; when B is larger than
// cache (large problem) every pass is paid in memory bandwidth — that is the
// regime tiling is built to win.
// ---------------------------------------------------------------------------
template <typename Layout>
struct NaiveFunctorT {
  Kokkos::View<float***, Layout> A, B;
  Kokkos::View<float**, Layout>  C;
  int                            L;

  KOKKOS_FUNCTION void operator()(int flat) const {
    const int i   = flat / L;
    const int l   = flat % L;
    float     acc = 0.0f;
    for (int j = 0; j < J; ++j)
      for (int k = 0; k < K; ++k) acc += A(i, j, k) * B(j, k, l);
    C(i, l) = acc;
  }
};

// With LayoutLeft inputs the naive inner loop (k innermost) strides across
// memory at step I*J for A and J for B — cache-hostile even in the small case.
template <typename Layout = Kokkos::LayoutRight>
double bench_naive(const Kokkos::View<float***, Layout>& A,
                   const Kokkos::View<float***, Layout>& B,
                   const Kokkos::View<float**, Layout>&  C,
                   int warmup, int iters) {
  const int I = static_cast<int>(C.extent(0));
  const int L = static_cast<int>(C.extent(1));
  for (int w = 0; w < warmup; ++w) {
    Kokkos::parallel_for("naive_warmup", Kokkos::RangePolicy<>(0, I * L),
                         NaiveFunctorT<Layout>{A, B, C, L});
    Kokkos::fence();
  }
  Kokkos::fence();
  Kokkos::Timer timer;
  for (int r = 0; r < iters; ++r)
    Kokkos::parallel_for("naive", Kokkos::RangePolicy<>(0, I * L),
                         NaiveFunctorT<Layout>{A, B, C, L});
  Kokkos::fence();
  return timer.seconds() * 1000.0 / iters;
}

// ---------------------------------------------------------------------------
// Library: TileI*TileL output elements per work item, staged into registers.
//
// A tile [TileI, J, K] is loaded once and reused TileL times (one per l in
// the tile); B tile [J, K, TileL] is loaded once and reused TileI times.
// Arithmetic intensity: TileI*TileL / (TileI + TileL) MACs/load.
//
// The traffic win: A is read L/TileL times total (vs L naive) and B is read
// I/TileI times total (vs I naive). That reduction only turns into wall-clock
// savings when those repeated reads would otherwise miss cache.
// ---------------------------------------------------------------------------
template <int TileI, int TileL, int TileJ = J, int TileK = K,
          typename VA, typename VB, typename VC>
double bench_library(const VA& A, const VB& B, const VC& C, int warmup,
                     int iters) {

  g_timing_stats.reset();
  auto hA =
      make_input_node(make_handle(A, std::array<int32_t, 3>{'i', 'j', 'k'}));
  auto hB =
      make_input_node(make_handle(B, std::array<int32_t, 3>{'j', 'k', 'l'}));

  auto g = make_graph();
  [[maybe_unused]] auto [g1, o1] =
      g.ops(make_contraction_node(hA, hB, std::array<int32_t, 2>{'i', 'l'}));

  constexpr auto tile = StaticTile<TileI, TileL, TileJ, TileK>{};

  for (int w = 0; w < warmup; ++w) {
    g1.execute(RangePolicyTag{}, tile, C);
    Kokkos::fence();
  }
  Kokkos::fence();
  // Reset counters AFTER warmup so the cycle counts cover exactly the timed
  // iterations — matching the wall_ms anchor passed to the analyzer below.
  g_timing_stats.reset();
  Kokkos::Timer timer;
  for (int r = 0; r < iters; ++r) g1.execute(RangePolicyTag{}, tile, C);
  Kokkos::fence();
  double ms      = timer.seconds() * 1000.0 / iters;
  double wall_ms = ms * iters;  // total wall time across the timed iterations

#if defined(TENSOR_OPS_ENABLE_TIMING)
  TensorOperations::TimingAnalyzer::print_detailed_analysis(g_timing_stats,
                                                            wall_ms);
#endif
  return ms;
}

static void print_row(const char* name, long work_items, double macs_per_load,
                      double ms, double flops) {
  std::printf("%-32s  %12ld  %9.1f  %9.3f  %9.3f\n", name, work_items,
              macs_per_load, ms, flops / (ms * 1e6));
}

static void print_header() {
  std::printf("%-32s  %12s  %9s  %9s  %9s\n", "Benchmark", "Work items",
              "MACs/load", "Avg (ms)", "GFLOP/s");
  std::printf("%-32s  %12s  %9s  %9s  %9s\n",
              "--------------------------------", "------------", "---------",
              "---------", "---------");
}

template <typename Layout = Kokkos::LayoutRight>
static bool correct(const Kokkos::View<float**, Layout>& C) {
  auto C_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  for (std::size_t i = 0; i < C_host.extent(0); ++i)
    for (std::size_t l = 0; l < C_host.extent(1); ++l)
      if (std::abs(C_host(i, l) - 1.0f) > 1e-4f) return false;
  return true;
}

// Allocate the operands, run the naive baseline plus a sweep of tile sizes, and
// print one table. `include_small_tiles` keeps the tiny (2x2/4x4) tiles off the
// large problem, where their millions of work items make the run needlessly
// long without adding signal.
static void run_suite(const char* title, int I, int L, int warmup, int iters,
                      bool include_small_tiles) {
  // LayoutRight (row-major): last dim contiguous for A, B, and C.
  View3 A("A", I, J, K);
  View3 B("B", J, K, L);
  View2 C("C", I, L);
  Kokkos::deep_copy(A, 1.0f / static_cast<float>(J * K));
  Kokkos::deep_copy(B, 1.0f);
  Kokkos::deep_copy(C, 0.0f);

  // LayoutLeft (column-major): first dim contiguous. For A[I,J,K] this is I
  // (stride 1), so the naive inner loop over k strides at I*J — cache-hostile.
  // The library's layout-aware gather picks dim 0 and streams I-runs instead.
  View3L AL("AL", I, J, K);
  View3L BL("BL", J, K, L);
  View2L CL("CL", I, L);
  Kokkos::deep_copy(AL, 1.0f / static_cast<float>(J * K));
  Kokkos::deep_copy(BL, 1.0f);
  Kokkos::deep_copy(CL, 0.0f);

  const double flops = 2.0 * I * J * K * L;
  const double a_mb  = double(I) * J * K * 4 / (1 << 20);
  const double b_mb  = double(J) * K * L * 4 / (1 << 20);
  const double c_mb  = double(I) * L * 4 / (1 << 20);

  std::printf("\n=== %s ===\n", title);
  std::printf("C[%d,%d] = sum_{j,k} A[%d,%d,%d] * B[%d,%d,%d]\n", I, L, I, J, K,
              J, K, L);
  std::printf("A=%.2f MB  B=%.2f MB  C=%.2f MB   (P-core L2 = 12 MB)\n", a_mb,
              b_mb, c_mb);
  std::printf("FLOPs/run: %.3e   Warmup: %d   Timed iters: %d\n\n", flops,
              warmup, iters);

  std::printf("  -- LayoutRight (row-major, last dim contiguous) --\n");
  print_header();
  print_row("Naive RangePolicy", long(I) * L, 0.5,
            bench_naive(A, B, C, warmup, iters), flops);
  if (include_small_tiles) {
    print_row("Library tile 2x2 (k=8x8)", long(I / 2) * (L / 2), 1.0,
              bench_library<2, 2>(A, B, C, warmup, iters), flops);
    print_row("Library tile 4x4 (k=8x8)", long(I / 4) * (L / 4), 2.0,
              bench_library<4, 4>(A, B, C, warmup, iters), flops);
  }
  print_row("Library tile 8x8 (k=8x8)", long(I / 8) * (L / 8), 4.0,
            bench_library<8, 8>(A, B, C, warmup, iters), flops);
  print_row("Library tile 16x16 (k=8x8)", long(I / 16) * (L / 16), 8.0,
            bench_library<16, 16>(A, B, C, warmup, iters), flops);
  print_row("Library tile 32x32 (k=8x8)", long(I / 32) * (L / 32), 16.0,
            bench_library<32, 32>(A, B, C, warmup, iters), flops);
  std::printf("Correctness (LayoutRight): %s\n",
              correct(C) ? "PASSED" : "FAILED");

  std::printf("\n  -- LayoutLeft (column-major, first dim contiguous) --\n");
  print_header();
  print_row("Naive RangePolicy", long(I) * L, 0.5,
            bench_naive(AL, BL, CL, warmup, iters), flops);
  if (include_small_tiles) {
    print_row("Library tile 2x2 (k=8x8)", long(I / 2) * (L / 2), 1.0,
              bench_library<2, 2>(AL, BL, CL, warmup, iters), flops);
    print_row("Library tile 4x4 (k=8x8)", long(I / 4) * (L / 4), 2.0,
              bench_library<4, 4>(AL, BL, CL, warmup, iters), flops);
  }
  print_row("Library tile 8x8 (k=8x8)", long(I / 8) * (L / 8), 4.0,
            bench_library<8, 8>(AL, BL, CL, warmup, iters), flops);
  print_row("Library tile 16x16 (k=8x8)", long(I / 16) * (L / 16), 8.0,
            bench_library<16, 16>(AL, BL, CL, warmup, iters), flops);
  print_row("Library tile 32x32 (k=8x8)", long(I / 32) * (L / 32), 16.0,
            bench_library<32, 32>(AL, BL, CL, warmup, iters), flops);
  std::printf("Correctness (LayoutLeft):  %s\n",
              correct(CL) ? "PASSED" : "FAILED");
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    // Cache-resident: A and B (64 KB each) live in L2, so the naive baseline's
    // repeated passes over B are free and tiling only adds staging overhead.
    run_suite("Cache-resident (small)", /*I=*/256, /*L=*/256, /*warmup=*/3,
              /*iters=*/10, /*include_small_tiles=*/true);

    // Cache-busting: B is 64 MB — far past L2/SLC. The naive baseline streams
    // all 64 MB of B from DRAM I=128 times; tiling cuts that to I/TileI passes,
    // so the reuse finally shows up as wall-clock savings.
    run_suite("Cache-busting (large)", /*I=*/128, /*L=*/262144, /*warmup=*/1,
              /*iters=*/3, /*include_small_tiles=*/false);
  }
  Kokkos::finalize();
  return 0;
}
