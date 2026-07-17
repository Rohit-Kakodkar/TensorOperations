#include <Kokkos_Core.hpp>
#include <TensorOperations/TileLayout.hpp>
#include <TensorOperations/Tiling.hpp>
#include <TensorOperations/TiledLayout.hpp>
#include <gtest/gtest.h>

using namespace TensorOperations;

// ---------------------------------------------------------------------------
// Compile-time sanity — StaticTileLayoutStride<StaticTile<E...>, Ord...>
// ---------------------------------------------------------------------------

// Order = {1,0} (dim1 fastest) must match StaticTileLayoutRight<4,8>
namespace {
using StrideRight2D = StaticTileLayoutStride<StaticTile<4, 8>, 1, 0>;
static_assert(StrideRight2D::rank == 2);
static_assert(StrideRight2D::num_elements == 32);
static_assert(StrideRight2D::extent(0) == 4);
static_assert(StrideRight2D::extent(1) == 8);
static_assert(StrideRight2D::stride(1) == 1);  // fastest dim
static_assert(StrideRight2D::stride(0) == 8);  // e[1] = 8

// Template (consteval) forms
static_assert(StrideRight2D::extent<0>() == 4);
static_assert(StrideRight2D::extent<1>() == 8);
static_assert(StrideRight2D::stride<1>() == 1);
static_assert(StrideRight2D::stride<0>() == 8);
static_assert(StrideRight2D::base_offset() == 0);
static_assert(StrideRight2D::size() == 32);
}  // namespace

// Order = {0,1} (dim0 fastest) must match StaticTileLayoutLeft<4,8>
namespace {
using StrideLeft2D = StaticTileLayoutStride<StaticTile<4, 8>, 0, 1>;
static_assert(StrideLeft2D::stride(0) == 1);  // fastest dim
static_assert(StrideLeft2D::stride(1) == 4);  // e[0] = 4
static_assert(StrideLeft2D::stride<0>() == 1);
static_assert(StrideLeft2D::stride<1>() == 4);
}  // namespace

// 3-D arbitrary order: extents {4,8,3}, Order={2,0,1} (dim2→dim0→dim1)
// stride(2)=1, stride(0)=e[2]=3, stride(1)=e[2]*e[0]=3*4=12
namespace {
using Stride3D = StaticTileLayoutStride<StaticTile<4, 8, 3>, 2, 0, 1>;
static_assert(Stride3D::rank == 3);
static_assert(Stride3D::num_elements == 96);
static_assert(Stride3D::stride(2) == 1);
static_assert(Stride3D::stride(0) == 3);
static_assert(Stride3D::stride(1) == 12);
static_assert(Stride3D::stride<2>() == 1);
static_assert(Stride3D::stride<0>() == 3);
static_assert(Stride3D::stride<1>() == 12);
}  // namespace

// ---------------------------------------------------------------------------
// TEST: stride values equal StaticTileLayoutRight/Left at runtime
// ---------------------------------------------------------------------------
TEST(StaticTileLayoutStride, MatchesRight2D) {
  using R = StaticTileLayoutRight<4, 8>;
  using S = StaticTileLayoutStride<StaticTile<4, 8>, 1, 0>;
  for (int d = 0; d < 2; ++d) {
    EXPECT_EQ(S::extent(d), R::extent(d)) << "d=" << d;
    EXPECT_EQ(S::stride(d), R::stride(d)) << "d=" << d;
  }
  EXPECT_EQ(S::size(), R::size());
  EXPECT_EQ(S::base_offset(), R::base_offset());
}

TEST(StaticTileLayoutStride, MatchesLeft2D) {
  using L = StaticTileLayoutLeft<4, 8>;
  using S = StaticTileLayoutStride<StaticTile<4, 8>, 0, 1>;
  for (int d = 0; d < 2; ++d) {
    EXPECT_EQ(S::extent(d), L::extent(d)) << "d=" << d;
    EXPECT_EQ(S::stride(d), L::stride(d)) << "d=" << d;
  }
  EXPECT_EQ(S::size(), L::size());
}

// ---------------------------------------------------------------------------
// TEST: 3-D arbitrary order strides
// ---------------------------------------------------------------------------
TEST(StaticTileLayoutStride, ArbitraryOrder3D) {
  using S = StaticTileLayoutStride<StaticTile<4, 8, 3>, 2, 0, 1>;
  EXPECT_EQ(S::stride(2), 1);
  EXPECT_EQ(S::stride(0), 3);
  EXPECT_EQ(S::stride(1), 12);
  EXPECT_EQ(S::size(), 96);
}

// ---------------------------------------------------------------------------
// TEST: flat() → operator[] round-trip (2D, order {1,0} = Right)
// ---------------------------------------------------------------------------
TEST(StaticTileLayoutStride, RoundTrip2D) {
  using S = StaticTileLayoutStride<StaticTile<4, 8>, 1, 0>;
  S layout{};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j) {
      const auto f   = static_cast<int>(layout.flat(i, j));
      const auto idx = layout[f];
      EXPECT_EQ(idx[0], i) << "i=" << i << " j=" << j;
      EXPECT_EQ(idx[1], j) << "i=" << i << " j=" << j;
    }
}

// ---------------------------------------------------------------------------
// TEST: flat() → operator[] round-trip (3D, arbitrary order {2,0,1})
// ---------------------------------------------------------------------------
TEST(StaticTileLayoutStride, RoundTrip3D) {
  using S = StaticTileLayoutStride<StaticTile<4, 8, 3>, 2, 0, 1>;
  S layout{};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j)
      for (int k = 0; k < 3; ++k) {
        const auto f   = static_cast<int>(layout.flat(i, j, k));
        const auto idx = layout[f];
        EXPECT_EQ(idx[0], i) << "i=" << i << " j=" << j << " k=" << k;
        EXPECT_EQ(idx[1], j) << "i=" << i << " j=" << j << " k=" << k;
        EXPECT_EQ(idx[2], k) << "i=" << i << " j=" << j << " k=" << k;
      }
}

// ---------------------------------------------------------------------------
// TEST: flat() values are correct (spot-check explicit offsets)
// Extents {4,8,3}, Order {2,0,1}: stride(2)=1, stride(0)=3, stride(1)=12
// flat(i,j,k) = i*3 + j*12 + k*1
// ---------------------------------------------------------------------------
TEST(StaticTileLayoutStride, FlatValues3D) {
  using S = StaticTileLayoutStride<StaticTile<4, 8, 3>, 2, 0, 1>;
  S layout{};
  EXPECT_EQ(static_cast<int>(layout.flat(0, 0, 0)), 0);
  EXPECT_EQ(static_cast<int>(layout.flat(1, 0, 0)), 3);           // 1*3
  EXPECT_EQ(static_cast<int>(layout.flat(0, 1, 0)), 12);          // 1*12
  EXPECT_EQ(static_cast<int>(layout.flat(0, 0, 1)), 1);           // 1*1
  EXPECT_EQ(static_cast<int>(layout.flat(1, 2, 1)), 3 + 24 + 1);  // 28
}

// ---------------------------------------------------------------------------
// TEST: View write-through — write via View<Buf, StaticTileLayoutStride>,
// read back with explicit flat offsets.
// ---------------------------------------------------------------------------
TEST(StaticTileLayoutStride, ViewWriteThrough2D) {
  using Buf2D = Kokkos::View<float*, Kokkos::LayoutRight, Kokkos::HostSpace>;
  // Extents {4,8}, Order {1,0}: stride(0)=8, stride(1)=1 — same as row-major
  using Layout = StaticTileLayoutStride<StaticTile<4, 8>, 1, 0>;
  Buf2D buf("buf", 32);
  Kokkos::deep_copy(buf, 0.f);

  View<Buf2D, Layout> v{buf, Layout{}};
  // Write v(i,j) = i*8 + j
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j) v(i, j) = static_cast<float>(i * 8 + j);

  // Read back through raw buffer — flat offset for (i,j) is i*8+j
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j)
      EXPECT_FLOAT_EQ(buf(i * 8 + j), static_cast<float>(i * 8 + j))
          << "i=" << i << " j=" << j;
}

TEST(StaticTileLayoutStride, ViewWriteThrough3D) {
  using Buf1D = Kokkos::View<float*, Kokkos::LayoutRight, Kokkos::HostSpace>;
  // Extents {4,8,3}, Order {2,0,1}: stride(0)=3, stride(1)=12, stride(2)=1
  using Layout = StaticTileLayoutStride<StaticTile<4, 8, 3>, 2, 0, 1>;
  Buf1D buf("buf", 96);
  Kokkos::deep_copy(buf, 0.f);

  View<Buf1D, Layout> v{buf, Layout{}};
  // Write v(i,j,k) = i*3 + j*12 + k
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j)
      for (int k = 0; k < 3; ++k)
        v(i, j, k) = static_cast<float>(i * 3 + j * 12 + k);

  // Verify raw buffer entries match expected flat offsets
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j)
      for (int k = 0; k < 3; ++k) {
        const int flat = i * 3 + j * 12 + k;
        EXPECT_FLOAT_EQ(buf(flat), static_cast<float>(flat))
            << "i=" << i << " j=" << j << " k=" << k;
      }
}

// ---------------------------------------------------------------------------
// TEST: make_tile_layout factory produces the correct type
// ---------------------------------------------------------------------------
TEST(StaticTileLayoutStride, MakeTileLayoutFactory) {
  auto layout =
      make_tile_layout(StaticTile<4, 8>{}, std::integer_sequence<int, 1, 0>{});
  using Expected = StaticTileLayoutStride<StaticTile<4, 8>, 1, 0>;
  static_assert(std::is_same_v<decltype(layout), Expected>);
  EXPECT_EQ(layout.stride(0), 8);
  EXPECT_EQ(layout.stride(1), 1);
  SUCCEED();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
