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
struct AGrid {  // A(i,j)
  static constexpr int rank = 2;
  using value_type          = float;
  int   extent(int) const { return 4; }
  float operator()(int i, int j) const { return static_cast<float>(i * 4 + j); }
};
struct BGrid {  // B(j,k)
  static constexpr int rank = 2;
  using value_type          = float;
  int   extent(int) const { return 4; }
  float operator()(int j, int k) const { return static_cast<float>(j * 7 + k); }
};
struct Doubler {
  float operator()(float v) const { return 2.0f * v; }
};

static float reference(int i, int k) {
  float s = 0.f;
  for (int j = 0; j < 4; ++j) s += AGrid{}(i, j) * BGrid{}(j, k);
  return s;
}

// Build a contraction node from A, B grids with the given output hook.
template <typename HookOp = NoHook>
static auto make_nc(HookOp hook = {}) {
  auto na =
      make_input_node(make_handle(AGrid{}, std::array<int32_t, 2>{'i', 'j'}));
  auto nb =
      make_input_node(make_handle(BGrid{}, std::array<int32_t, 2>{'j', 'k'}));
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
  using CEval  = Evaluator<RangePolicyTag, NC, TileT>;
  using Interm = typename CEval::result_type;
  using SEval  = Evaluator<RangePolicyTag, Interm, TileT>;
  CEval         cev{nc};
  constexpr int N  = 4;
  constexpr int Ti = TileT::extent(0), Tk = TileT::extent(1);

  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace> v("C", N, N);
  for (int off_i = 0; off_i < N; off_i += Ti)
    for (int off_k = 0; off_k < N; off_k += Tk) {
      auto tile_node = cev(std::array<int, 2>{off_i, off_k});
      SEval{tile_node}(std::array<int, 2>{off_i, off_k}, v);
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
