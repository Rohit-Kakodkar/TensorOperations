#include <Kokkos_Core.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/RegisterArray.hpp>
#include <TensorOperations/TensorHandle.hpp>
#include <TensorOperations/Tiling.hpp>
#include <gtest/gtest.h>

using namespace TensorOperations;

// ---------------------------------------------------------------------------
// StaticTile — compile-time extents.
// ---------------------------------------------------------------------------
TEST(TilingTest, StaticTile) {
  using Tile = StaticTile<2, 3, 4>;
  static_assert(Tile::rank == 3);
  static_assert(Tile::is_static);
  static_assert(Tile::extent(0) == 2);
  static_assert(Tile::extent(1) == 3);
  static_assert(Tile::extent(2) == 4);

  // Degenerate unit tile is the un-staged baseline.
  static_assert(StaticTile<1, 1>::rank == 2);
  SUCCEED();
}

// ---------------------------------------------------------------------------
// DynamicTile — runtime extents, compile-time rank.
// ---------------------------------------------------------------------------
TEST(TilingTest, DynamicTile) {
  using Tile = DynamicTile<3>;
  static_assert(Tile::rank == 3);
  static_assert(!Tile::is_static);

  Tile t{{2, 4, 6}};
  EXPECT_EQ(t.extent(0), 2);
  EXPECT_EQ(t.extent(1), 4);
  EXPECT_EQ(t.extent(2), 6);
}

// Build a ContractionTag node type directly. Its NodeHandle specialization does
// not constrain its operands with TensorLike, so this sidesteps the
// pre-existing TensorLike/make_handle breakage and still exercises Evaluator
// specialization selection for the tiling types. FakeNode mimics the node
// interface the contraction reads (Rank, value_type). C_{ik} = sum_j A_{ij}
// B_{jk}: output rank 2, NumContracted 1, so 3 participating modes.
struct FakeNode {
  static constexpr int Rank = 2;
  using value_type          = float;
};
using FakeContraction = NodeHandle<ContractionTag, FakeNode, FakeNode,
                                   std::integral_constant<int, 2>, float,
                                   Kokkos::DefaultExecutionSpace, NoHook>;

// ---------------------------------------------------------------------------
// Evaluator specialization selection — purely type-level; no kernels run.
// ---------------------------------------------------------------------------
TEST(TilingTest, EvaluatorSelection) {
  static_assert(FakeContraction::Rank == 2);
  static_assert(FakeContraction::NumContracted == 1);

  // -- Range + Contraction + StaticTile -> register-blocked (register tier) --
  using REvalC =
      Evaluator<RangePolicyTag, FakeContraction, StaticTile<2, 2, 2>>;
  static_assert(std::is_same_v<REvalC::tiling_type, StaticTile<2, 2, 2>>);
  static_assert(REvalC::Rank == 2);  // output rank, not tile rank
  static_assert(
      std::is_same_v<REvalC::register_array_t, RegisterArray<float, 2, 2, 2>>);

  // -- Team + Contraction + DynamicTile -> scratch tier ----------------------
  using TEvalC = Evaluator<TeamPolicyTag, FakeContraction, DynamicTile<2>>;
  static_assert(std::is_same_v<TEvalC::tiling_type, DynamicTile<2>>);
  static_assert(TEvalC::Rank == 2);
  static_assert(TEvalC::scratch_view_t::rank == 2);

  // -- Team also accepts StaticTile on the scratch tier ----------------------
  using TEvalCS =
      Evaluator<TeamPolicyTag, FakeContraction, StaticTile<2, 2, 2>>;
  static_assert(std::is_same_v<TEvalCS::tiling_type, StaticTile<2, 2, 2>>);
  static_assert(TEvalCS::scratch_view_t::rank == 2);

  // NOTE: Evaluator<RangePolicyTag, FakeContraction, DynamicTile<2>> matches no
  // specialization (undefined primary template) — register tier is static-only.

  SUCCEED();
}

// A 4x4 readable TensorLike with element (i,j) = i*4 + j.
struct Grid4x4 {
  static constexpr int rank = 2;
  using value_type          = float;
  int   extent(int) const { return 4; }
  float operator()(int i, int j) const { return static_cast<float>(i * 4 + j); }
};

// A doubling hook to verify hooks are applied at load.
struct Doubler {
  float operator()(float v) const { return 2.0f * v; }
};

// ---------------------------------------------------------------------------
// Range + Input + StaticTile -> register-tier input stager. Functional test of
// operator(): fill a 2x2 RegisterArray from a tile origin.
// ---------------------------------------------------------------------------
TEST(TilingTest, InputStagerContiguous) {
  auto inp =
      make_input_node(make_handle(Grid4x4{}, std::array<int32_t, 2>{'i', 'j'}));
  using Eval = Evaluator<RangePolicyTag, decltype(inp), StaticTile<2, 2>>;
  Eval ev{inp};  // tiling + layout default-constructed (stride 1 = contiguous)

  Eval::register_array_t regs{};
  ev({2, 1}, regs);  // tile origin (2,1)

  // regs(a,b) == Grid4x4(2+a, 1+b) == (2+a)*4 + (1+b)
  EXPECT_FLOAT_EQ((regs(0, 0)), 9.f);   // (2,1)
  EXPECT_FLOAT_EQ((regs(0, 1)), 10.f);  // (2,2)
  EXPECT_FLOAT_EQ((regs(1, 0)), 13.f);  // (3,1)
  EXPECT_FLOAT_EQ((regs(1, 1)), 14.f);  // (3,2)
}

// Staggered read: stride 2 along the last mode (the coalescing pattern).
TEST(TilingTest, InputStagerStaggered) {
  auto inp =
      make_input_node(make_handle(Grid4x4{}, std::array<int32_t, 2>{'i', 'j'}));
  using Eval = Evaluator<RangePolicyTag, decltype(inp), StaticTile<2, 2>>;
  Eval ev{inp, {}, StridedLayout<2>{{1, 2}}};

  Eval::register_array_t regs{};
  ev({0, 0}, regs);

  // regs(a,b) == Grid4x4(0 + a*1, 0 + b*2) == a*4 + b*2
  EXPECT_FLOAT_EQ((regs(0, 0)), 0.f);  // (0,0)
  EXPECT_FLOAT_EQ((regs(0, 1)), 2.f);  // (0,2)
  EXPECT_FLOAT_EQ((regs(1, 0)), 4.f);  // (1,0)
  EXPECT_FLOAT_EQ((regs(1, 1)), 6.f);  // (1,2)
}

// Hook is applied at load time.
TEST(TilingTest, InputStagerAppliesHook) {
  auto inp = make_input_node(
      make_handle(Grid4x4{}, std::array<int32_t, 2>{'i', 'j'}), Doubler{});
  using Eval = Evaluator<RangePolicyTag, decltype(inp), StaticTile<2, 2>>;
  Eval ev{inp};

  Eval::register_array_t regs{};
  ev({0, 0}, regs);

  EXPECT_FLOAT_EQ((regs(1, 1)), 2.f * 5.f);  // 2 * Grid4x4(1,1)
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
