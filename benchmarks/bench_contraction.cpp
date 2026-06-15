#include <Kokkos_Core.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/Graph.hpp>
#include <TensorOperations/Tiling.hpp>

#include <cmath>
#include <cstdio>

using namespace TensorOperations;

// C[I,L] = sum_{j,k} A[I,j,k] * B[j,k,L]
constexpr int    I      = 256;
constexpr int    J      = 8;
constexpr int    K      = 8;
constexpr int    L      = 256;
constexpr int    WARMUP = 3;
constexpr int    ITERS  = 10;
constexpr double FLOPS  = 2.0 * I * J * K * L;

using View2 = Kokkos::View<float**, Kokkos::LayoutRight>;
using View3 = Kokkos::View<float***, Kokkos::LayoutRight>;

// ---------------------------------------------------------------------------
// Naive: one output element per work item, no data reuse between work items.
// Arithmetic intensity: J*K MACs / (J*K + J*K loads) = 0.5 MACs/load.
// ---------------------------------------------------------------------------
struct NaiveFunctor {
  View3 A, B;
  View2 C;

  KOKKOS_FUNCTION void operator()(int flat) const {
    const int i   = flat / L;
    const int l   = flat % L;
    float     acc = 0.0f;
    for (int j = 0; j < J; ++j)
      for (int k = 0; k < K; ++k) acc += A(i, j, k) * B(j, k, l);
    C(i, l) = acc;
  }
};

double bench_naive(const View3& A, const View3& B, const View2& C) {
  for (int w = 0; w < WARMUP; ++w) {
    Kokkos::parallel_for("naive_warmup", Kokkos::RangePolicy<>(0, I * L),
                         NaiveFunctor{A, B, C});
    Kokkos::fence();
  }
  Kokkos::fence();
  Kokkos::Timer timer;
  for (int r = 0; r < ITERS; ++r)
    Kokkos::parallel_for("naive", Kokkos::RangePolicy<>(0, I * L),
                         NaiveFunctor{A, B, C});
  Kokkos::fence();
  return timer.seconds() * 1000.0 / ITERS;
}

// ---------------------------------------------------------------------------
// Library: TileI*TileL output elements per work item, staged into registers.
//
// A tile [TileI, J, K] is loaded once and reused TileL times (one per l in
// the tile); B tile [J, K, TileL] is loaded once and reused TileI times.
// Arithmetic intensity: TileI*TileL / (TileI + TileL) MACs/load.
//
// StaticTile makes all loop bounds compile-time constants, letting the
// compiler fully unroll the inner accumulation loops and guarantee register
// residency via RegisterArray::at<I,J,...>().
//
// Note: each RegisterArray is a fold expression over tile elements; clang's
// default bracket depth is 256 so the CMakeLists target sets
// -fbracket-depth=2048 for this executable (max tile element count = 512).
// ---------------------------------------------------------------------------
template <int TileI, int TileL, int TileJ = J, int TileK = K>
double bench_library(const View3& A, const View3& B, const View2& C) {
  auto hA =
      make_input_node(make_handle(A, std::array<int32_t, 3>{'i', 'j', 'k'}));
  auto hB =
      make_input_node(make_handle(B, std::array<int32_t, 3>{'j', 'k', 'l'}));

  auto g = make_graph();
  [[maybe_unused]] auto [g1, o1] =
      g.ops(make_contraction_node(hA, hB, std::array<int32_t, 2>{'i', 'l'}));

  constexpr auto tile = StaticTile<TileI, TileL, TileJ, TileK>{};

  for (int w = 0; w < WARMUP; ++w) {
    g1.execute(RangePolicyTag{}, tile, C);
    Kokkos::fence();
  }
  Kokkos::fence();
  Kokkos::Timer timer;
  for (int r = 0; r < ITERS; ++r) g1.execute(RangePolicyTag{}, tile, C);
  Kokkos::fence();
  return timer.seconds() * 1000.0 / ITERS;
}

static void print_row(const char* name, int work_items, double macs_per_load,
                      double ms) {
  std::printf("%-32s  %10d  %9.1f  %9.3f  %9.3f\n", name, work_items,
              macs_per_load, ms, FLOPS / (ms * 1e6));
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    View3 A("A", I, J, K);
    View3 B("B", J, K, L);
    View2 C("C", I, L);

    // A[i,j,k] = 1/(J*K), B[j,k,l] = 1  →  C[i,l] = 1.0 after contraction
    Kokkos::deep_copy(A, 1.0f / static_cast<float>(J * K));
    Kokkos::deep_copy(B, 1.0f);
    Kokkos::deep_copy(C, 0.0f);

    std::printf("\n=== Tensor Contraction Benchmark ===\n");
    std::printf("C[%d,%d] = sum_{j,k} A[%d,%d,%d] * B[%d,%d,%d]\n", I, L, I, J,
                K, J, K, L);
    std::printf("FLOPs per run: %.0f   Warmup: %d   Timed iters: %d\n\n", FLOPS,
                WARMUP, ITERS);
    std::printf("%-32s  %10s  %9s  %9s  %9s\n", "Benchmark", "Work items",
                "MACs/load", "Avg (ms)", "GFLOP/s");
    std::printf("%-32s  %10s  %9s  %9s  %9s\n",
                "--------------------------------", "----------", "---------",
                "---------", "---------");

    print_row("Naive RangePolicy", I * L, 0.5, bench_naive(A, B, C));
    print_row("Library tile 2x2 (k=8x8)", (I / 2) * (L / 2), 1.0,
              bench_library<2, 2>(A, B, C));
    print_row("Library tile 4x4 (k=8x8)", (I / 4) * (L / 4), 2.0,
              bench_library<4, 4>(A, B, C));
    print_row("Library tile 8x8 (k=8x8)", (I / 8) * (L / 8), 4.0,
              bench_library<8, 8>(A, B, C));

    // Verify last library run: C[i,l] should equal 1.0 everywhere
    auto C_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
    bool ok     = true;
    for (int i = 0; i < I && ok; ++i)
      for (int l = 0; l < L && ok; ++l)
        if (std::abs(C_host(i, l) - 1.0f) > 1e-4f) ok = false;
    std::printf("\nCorrectness check: %s\n\n", ok ? "PASSED" : "FAILED");
  }
  Kokkos::finalize();
  return 0;
}
