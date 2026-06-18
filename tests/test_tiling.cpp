#include <Kokkos_Core.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/RegisterArray.hpp>
#include <TensorOperations/TensorHandle.hpp>
#include <TensorOperations/Tiling.hpp>
#include <array>
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

using View2D = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace>;

// ---------------------------------------------------------------------------
// Evaluator specialization selection — purely type-level; no kernels run.
// C_{ik} = sum_j A_{ij} B_{jk}: output rank 2, NumContracted 1.
// ---------------------------------------------------------------------------
TEST(TilingTest, EvaluatorSelection) {
  using NC = decltype(make_contraction_node<float>(
      make_input_node(make_handle(std::declval<View2D>(), std::array<int32_t, 2>{'i', 'j'})),
      make_input_node(make_handle(std::declval<View2D>(), std::array<int32_t, 2>{'j', 'k'})),
      std::array<int32_t, 2>{'i', 'k'}));

  static_assert(NC::Rank == 2);
  static_assert(NC::NumContracted == 1);

  // -- Range + Contraction + StaticTile -> register-blocked (register tier) --
  using REvalC = Evaluator<RangePolicyTag, NC, StaticTile<2, 2, 2>>;
  static_assert(std::is_same_v<REvalC::tiling_type, StaticTile<2, 2, 2>>);
  static_assert(REvalC::Rank == 2);  // output rank, not tile rank
  static_assert(
      std::is_same_v<REvalC::register_array_t, RegisterArray<float, 2, 2, 2>>);

  // -- Team + Contraction + DynamicTile -> scratch tier ----------------------
  using TEvalC = Evaluator<TeamPolicyTag, NC, DynamicTile<2>>;
  static_assert(std::is_same_v<TEvalC::tiling_type, DynamicTile<2>>);
  static_assert(TEvalC::Rank == 2);
  static_assert(TEvalC::scratch_view_t::rank == 2);

  // -- Team also accepts StaticTile on the scratch tier ----------------------
  using TEvalCS = Evaluator<TeamPolicyTag, NC, StaticTile<2, 2, 2>>;
  static_assert(std::is_same_v<TEvalCS::tiling_type, StaticTile<2, 2, 2>>);
  static_assert(TEvalCS::scratch_view_t::rank == 2);

  // NOTE: Evaluator<RangePolicyTag, NC, DynamicTile<2>> matches no
  // specialization (undefined primary template) — register tier is static-only.

  SUCCEED();
}

// A doubling hook to verify hooks are applied at load.
struct Doubler {
  float operator()(float v) const { return 2.0f * v; }
};

// ---------------------------------------------------------------------------
// Range + Input + StaticTile -> register-tier input stager. Functional test of
// operator(): fill a 2x2 RegisterArray from tile index (1,0).
// ---------------------------------------------------------------------------
TEST(TilingTest, InputStagerContiguous) {
  View2D v("g", 4, 4);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      v(i, j) = static_cast<float>(i * 4 + j);

  auto inp = make_input_node(make_handle(v, std::array<int32_t, 2>{'i', 'j'}));
  auto ev  = make_evaluator<RangePolicyTag>(inp, StaticTile<2, 2>{});

  auto        node = ev({1, 0});  // tile index (1,0) -> element origin (2,0)
  const auto& regs = node.storage_;

  // regs(a,b) == v(2+a, 0+b) == (2+a)*4 + b
  EXPECT_FLOAT_EQ((regs(0, 0)), 8.f);   // (2,0)
  EXPECT_FLOAT_EQ((regs(0, 1)), 9.f);   // (2,1)
  EXPECT_FLOAT_EQ((regs(1, 0)), 12.f);  // (3,0)
  EXPECT_FLOAT_EQ((regs(1, 1)), 13.f);  // (3,1)
}

// Hook is applied at load time.
TEST(TilingTest, InputStagerAppliesHook) {
  View2D v("g", 4, 4);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      v(i, j) = static_cast<float>(i * 4 + j);

  auto inp = make_input_node(
      make_handle(v, std::array<int32_t, 2>{'i', 'j'}), Doubler{});
  auto ev = make_evaluator<RangePolicyTag>(inp, StaticTile<2, 2>{});

  auto        node = ev({0, 0});
  const auto& regs = node.storage_;

  EXPECT_FLOAT_EQ((regs(1, 1)), 2.f * 5.f);  // 2 * v(1,1)
}

// ---------------------------------------------------------------------------
// Regression: fill_slots must compile for a 512-element (16x32) RegisterArray.
// Clang's bracket-depth limit was raised to 2048 in CMakeLists for exactly
// this — instantiating the fold over 512 slots is the stress case.
// ---------------------------------------------------------------------------
TEST(TilingTest, LargeRegisterViewFillSlotsCompiles) {
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace> lv("lg", 512, 512);
  auto inp = make_input_node(make_handle(lv, std::array<int32_t, 2>{'i', 'j'}));
  auto ev  = make_evaluator<RangePolicyTag>(inp, StaticTile<16, 32>{});
  static_assert(decltype(ev)::register_array_t::size == 512);
  auto node = ev({0, 0});  // instantiates fill_slots over all 512 slots
  SUCCEED();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
