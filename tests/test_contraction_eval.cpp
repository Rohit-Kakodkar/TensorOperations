#include <Kokkos_Core.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/TensorHandle.hpp>
#include <TensorOperations/Tiling.hpp>
#include <array>
#include <gtest/gtest.h>

using namespace TensorOperations;

// ---------------------------------------------------------------------------
// GEMM-shaped contraction  C_{ik} = sum_j A_{ij} B_{jk},  all 4x4.
// A is stored [i,j] (= [freeA, contracted]), B is stored [j,k] (= [contracted,
// freeB]), out_modes [i,k] (= [freeA, freeB]) — the Option A / GEMM convention.
// ---------------------------------------------------------------------------
struct Doubler {
  float operator()(float v) const { return 2.0f * v; }
};

static float reference(int i, int k) {
  float s = 0.f;
  for (int j = 0; j < 4; ++j)
    s += static_cast<float>(i * 4 + j) * static_cast<float>(j * 7 + k);
  return s;
}

// Build a contraction node from A, B views with the given output hook.
template <typename HookOp = NoHook>
static auto make_nc(HookOp hook = {}) {
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace> a("A", 4, 4);
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace> b("B", 4, 4);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) a(i, j) = static_cast<float>(i * 4 + j);
  for (int j = 0; j < 4; ++j)
    for (int k = 0; k < 4; ++k) b(j, k) = static_cast<float>(j * 7 + k);
  auto na = make_input_node(make_handle(a, std::array<int32_t, 2>{'i', 'j'}));
  auto nb = make_input_node(make_handle(b, std::array<int32_t, 2>{'j', 'k'}));
  return make_contraction_node<float>(na, nb, std::array<int32_t, 2>{'i', 'k'},
                                      hook);
}

// Drive the register-tier contraction Evaluator over the whole 4x4 output with
// the participating tile TileT = StaticTile<Ti, Tk, Tj>. Each evaluator call
// computes one complete output tile (summing over all of K internally); the
// store-evaluator writes it into the destination view, applying the hook at
// store time. The contracted Tj drives Spec 3's internal K loop.
template <typename TileT, typename NC>
static std::array<std::array<float, 4>, 4> compute(const NC& nc) {
  auto          cev = make_evaluator<RangePolicyTag>(nc, TileT{});
  constexpr int N   = 4;
  constexpr int Ti = TileT::extent(0), Tk = TileT::extent(1);

  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace> v("C", N, N);
  for (int ti = 0; ti < N / Ti; ++ti)
    for (int tk = 0; tk < N / Tk; ++tk) {
      auto tile_node = cev(Kokkos::Array<int, 2>{ti, tk});
      auto store_ev  = make_evaluator<RangePolicyTag>(tile_node, TileT{});
      store_ev({ti, tk}, v);
    }

  std::array<std::array<float, 4>, 4> out{};
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) out[i][j] = v(i, j);
  return out;
}

// Tile contracted extent 2 < K=4 -> two contracted blocks (accumulation across
// operator() calls is exercised).
TEST(ContractionEvalTest, GemmTwoContractedBlocks) {
  auto nc  = make_nc();
  auto out = compute<StaticTile<2, 2, 2>>(nc);
  for (int i = 0; i < 4; ++i)
    for (int k = 0; k < 4; ++k)
      EXPECT_FLOAT_EQ(out[i][k], reference(i, k)) << "i=" << i << " k=" << k;
}

// Tile contracted extent 4 == K -> a single contracted block per output tile.
TEST(ContractionEvalTest, GemmSingleContractedBlock) {
  auto nc  = make_nc();
  auto out = compute<StaticTile<2, 2, 4>>(nc);
  for (int i = 0; i < 4; ++i)
    for (int k = 0; k < 4; ++k)
      EXPECT_FLOAT_EQ(out[i][k], reference(i, k)) << "i=" << i << " k=" << k;
}

// The contraction node's output hook is applied by the caller at store time.
TEST(ContractionEvalTest, AppliesHookAtStore) {
  auto nc  = make_nc(Doubler{});
  auto out = compute<StaticTile<2, 2, 2>>(nc);
  for (int i = 0; i < 4; ++i)
    for (int k = 0; k < 4; ++k)
      EXPECT_FLOAT_EQ(out[i][k], 2.0f * reference(i, k));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
