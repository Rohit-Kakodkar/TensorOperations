#include <gtest/gtest.h>
#include <TensorOperations/Layout.hpp>

using namespace TensorOperations;

// ---------------------------------------------------------------------------
// Impl::filled_array — constexpr fill used to default StridedLayout to
// stride 1.
// ---------------------------------------------------------------------------
TEST(LayoutTest, FilledArray) {
  constexpr auto a = Impl::filled_array<3>(1);
  static_assert(a.size() == 3);
  static_assert(a[0] == 1 && a[1] == 1 && a[2] == 1);

  constexpr auto z = Impl::filled_array<2>(0);
  static_assert(z[0] == 0 && z[1] == 0);
  SUCCEED();
}

// ---------------------------------------------------------------------------
// Default construction = contiguous: every stride is 1.
// ---------------------------------------------------------------------------
TEST(LayoutTest, DefaultIsContiguous) {
  StridedLayout<3> layout;
  EXPECT_EQ(layout.rank, 3);
  EXPECT_EQ(layout.stride[0], 1);
  EXPECT_EQ(layout.stride[1], 1);
  EXPECT_EQ(layout.stride[2], 1);

  // operator[] mirrors the stride array.
  EXPECT_EQ(layout[0], 1);
  EXPECT_EQ(layout[1], 1);
  EXPECT_EQ(layout[2], 1);
}

// ---------------------------------------------------------------------------
// rank is a compile-time property of the layout.
// ---------------------------------------------------------------------------
TEST(LayoutTest, RankIsStatic) {
  static_assert(StridedLayout<1>::rank == 1);
  static_assert(StridedLayout<4>::rank == 4);
  SUCCEED();
}

// ---------------------------------------------------------------------------
// Aggregate init with explicit strides (the staggered / coalesced pattern).
// ---------------------------------------------------------------------------
TEST(LayoutTest, ExplicitStrides) {
  StridedLayout<2> layout{{1, 8}};
  EXPECT_EQ(layout[0], 1);
  EXPECT_EQ(layout[1], 8);
  EXPECT_EQ(layout.stride[1], 8);
}

// ---------------------------------------------------------------------------
// Fully usable in a constant expression: default fill and operator[] are
// constexpr, so a StridedLayout can be evaluated at compile time.
// ---------------------------------------------------------------------------
TEST(LayoutTest, ConstexprUsable) {
  constexpr StridedLayout<2> def{};
  static_assert(def[0] == 1 && def[1] == 1);

  constexpr StridedLayout<3> staggered{{1, 4, 16}};
  static_assert(staggered[0] == 1);
  static_assert(staggered[1] == 4);
  static_assert(staggered[2] == 16);
  SUCCEED();
}
