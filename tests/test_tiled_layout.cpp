#include <Kokkos_Core.hpp>
#include <TensorOperations/Concept.hpp>
#include <TensorOperations/TiledLayout.hpp>
#include <TensorOperations/Tiling.hpp>
#include <gtest/gtest.h>

using namespace TensorOperations;

// ---------------------------------------------------------------------------
// TensorLike concept checks — compile-time.
// ---------------------------------------------------------------------------

TEST(TiledLayout, TiledViewSatisfiesConcepts) {
  using View2D = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace>;

  using TV = TiledView<View2D, StaticTile<4, 4>>;
  static_assert(TensorLike<TV>);

  using TVD = TiledView<View2D, DynamicTile<2>>;
  static_assert(TensorLike<TVD>);

  using SV = Subview<View2D, 2>;
  static_assert(TensorLike<SV>);

  SUCCEED();
}

// ---------------------------------------------------------------------------
// 1-D: exact divisibility  E=[8], T=[4]
// Expected tiled shape: outer_extent=2, inner_extent=4
//           strides:    outer=4, inner=1
// ---------------------------------------------------------------------------

TEST(TiledLayout, TileView1D_Exact) {
  Kokkos::View<float*, Kokkos::LayoutRight, Kokkos::HostSpace> v("v", 8);
  for (int i = 0; i < 8; ++i) v(i) = static_cast<float>(i);

  auto tv = tile_view(v, StaticTile<4>{});

  ASSERT_EQ(tv.rank, 2);
  EXPECT_EQ(tv.extent(0), 2);  // outer: 2 tiles
  EXPECT_EQ(tv.extent(1), 4);  // inner: tile size 4
  EXPECT_EQ(tv.stride(0), 4);  // outer stride = T*S = 4*1
  EXPECT_EQ(tv.stride(1), 1);  // inner stride = S = 1

  // Element access: tv(t, r) == v(t*4 + r)
  for (int t = 0; t < 2; ++t)
    for (int r = 0; r < 4; ++r)
      EXPECT_FLOAT_EQ(tv(t, r), v(t * 4 + r)) << "t=" << t << " r=" << r;
}

// ---------------------------------------------------------------------------
// 1-D: non-divisible  E=[10], T=[4]
// n = ceil(10/4) = 3; last tile is partial.
// ---------------------------------------------------------------------------

TEST(TiledLayout, TileView1D_Remainder) {
  Kokkos::View<float*, Kokkos::LayoutRight, Kokkos::HostSpace> v("v", 10);
  for (int i = 0; i < 10; ++i) v(i) = static_cast<float>(i);

  auto tv = tile_view(v, StaticTile<4>{});

  EXPECT_EQ(tv.extent(0), 3);  // outer: ceil(10/4) = 3
  EXPECT_EQ(tv.extent(1), 4);  // inner: tile size
  EXPECT_EQ(tv.stride(0), 4);
  EXPECT_EQ(tv.stride(1), 1);

  // Valid (t, r) combinations
  for (int t = 0; t < 2; ++t)
    for (int r = 0; r < 4; ++r) EXPECT_FLOAT_EQ(tv(t, r), v(t * 4 + r));
}

// ---------------------------------------------------------------------------
// 2-D row-major: spec worked example  E=[10,6], S=[6,1], T=[4,4]
// n = [3, 2]
// outer_strides = [24, 4], inner_strides = [6, 1]
// ---------------------------------------------------------------------------

TEST(TiledLayout, TileView2D_RowMajor) {
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace> v("v", 10, 6);
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 6; ++j) v(i, j) = static_cast<float>(i * 6 + j);

  auto tv = tile_view(v, StaticTile<4, 4>{});

  ASSERT_EQ(tv.rank, 4);
  // Outer extents
  EXPECT_EQ(tv.extent(0), 3);  // ceil(10/4) = 3
  EXPECT_EQ(tv.extent(1), 2);  // ceil(6/4)  = 2
  // Inner extents (tile sizes)
  EXPECT_EQ(tv.extent(2), 4);
  EXPECT_EQ(tv.extent(3), 4);
  // Outer strides = T[d]*S[d]
  EXPECT_EQ(tv.stride(0), 4 * 6);  // 24
  EXPECT_EQ(tv.stride(1), 4 * 1);  // 4
  // Inner strides = S[d]
  EXPECT_EQ(tv.stride(2), 6);
  EXPECT_EQ(tv.stride(3), 1);

  // Element access: tv(t0, t1, r0, r1) == v(t0*4+r0, t1*4+r1)
  // Clamp inner loops to the view's actual extents (last tile may be partial).
  for (int t0 = 0; t0 < 2; ++t0)
    for (int t1 = 0; t1 < 2; ++t1)
      for (int r0 = 0; r0 < std::min(4, 10 - t0 * 4); ++r0)
        for (int r1 = 0; r1 < std::min(4, 6 - t1 * 4); ++r1)
          EXPECT_FLOAT_EQ(tv(t0, t1, r0, r1), v(t0 * 4 + r0, t1 * 4 + r1));
}

// ---------------------------------------------------------------------------
// 2-D column-major (LayoutLeft): E=[6,10], S=[1,6], T=[4,4]
// inner_strides should reflect LayoutLeft strides [1, 6].
// ---------------------------------------------------------------------------

TEST(TiledLayout, TileView2D_LayoutLeft) {
  Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::HostSpace> v("v", 6, 10);
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 10; ++j) v(i, j) = static_cast<float>(i + j * 6);

  auto tv = tile_view(v, StaticTile<4, 4>{});

  // LayoutLeft strides: stride(0)=1, stride(1)=6
  EXPECT_EQ(tv.stride(2), 1);  // inner stride dim 0 = S[0] = 1
  EXPECT_EQ(tv.stride(3), 6);  // inner stride dim 1 = S[1] = 6

  EXPECT_EQ(tv.extent(0), 2);  // ceil(6/4) = 2
  EXPECT_EQ(tv.extent(1), 3);  // ceil(10/4) = 3

  for (int t0 = 0; t0 < 1; ++t0)
    for (int t1 = 0; t1 < 2; ++t1)
      for (int r0 = 0; r0 < 4; ++r0)
        for (int r1 = 0; r1 < 4; ++r1)
          EXPECT_FLOAT_EQ(tv(t0, t1, r0, r1), v(t0 * 4 + r0, t1 * 4 + r1));
}

// ---------------------------------------------------------------------------
// 2-D LayoutStride: verify inner strides pass through unchanged.
// ---------------------------------------------------------------------------

TEST(TiledLayout, TileView2D_LayoutStride) {
  // Build a LayoutStride view with strides [3, 1] over a (4,4) shape.
  Kokkos::LayoutStride                                           ls{4, 3, 4, 1};
  Kokkos::View<float**, Kokkos::LayoutStride, Kokkos::HostSpace> v("v", ls);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) v(i, j) = static_cast<float>(i * 10 + j);

  auto tv = tile_view(v, StaticTile<2, 2>{});

  EXPECT_EQ(tv.stride(2), 3);  // inner_stride[0] = S[0] = 3
  EXPECT_EQ(tv.stride(3), 1);  // inner_stride[1] = S[1] = 1

  for (int t0 = 0; t0 < 2; ++t0)
    for (int t1 = 0; t1 < 2; ++t1)
      for (int r0 = 0; r0 < 2; ++r0)
        for (int r1 = 0; r1 < 2; ++r1)
          EXPECT_FLOAT_EQ(tv(t0, t1, r0, r1), v(t0 * 2 + r0, t1 * 2 + r1));
}

// ---------------------------------------------------------------------------
// Tile larger than extent: T[d] > E[d] → clamped to single tile.
// ---------------------------------------------------------------------------

TEST(TiledLayout, TileViewClamp) {
  Kokkos::View<float*, Kokkos::LayoutRight, Kokkos::HostSpace> v("v", 3);
  for (int i = 0; i < 3; ++i) v(i) = static_cast<float>(i);

  auto tv = tile_view(v, StaticTile<8>{});  // T=8 > E=3, clamped to 3

  EXPECT_EQ(tv.extent(0), 1);  // outer: 1 tile
  EXPECT_EQ(tv.extent(1), 3);  // inner: clamped tile size = E = 3

  for (int r = 0; r < 3; ++r) EXPECT_FLOAT_EQ(tv(0, r), v(r));
}

// ---------------------------------------------------------------------------
// Write through TiledView: assign via tv, read back through original view.
// ---------------------------------------------------------------------------

TEST(TiledLayout, TiledViewWritable) {
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace> v("v", 8, 8);
  Kokkos::deep_copy(v, 0.f);

  auto tv        = tile_view(v, StaticTile<4, 4>{});
  tv(1, 0, 3, 2) = 99.f;  // (t0=1,t1=0,r0=3,r1=2) -> v(1*4+3, 0*4+2) = v(7,2)

  EXPECT_FLOAT_EQ(v(7, 2), 99.f);
}

// ---------------------------------------------------------------------------
// DynamicTile: same extents and strides as StaticTile for the same sizes.
// ---------------------------------------------------------------------------

TEST(TiledLayout, TiledViewDynamic) {
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace> v("v", 10, 6);

  auto tv_static  = tile_view(v, StaticTile<4, 4>{});
  auto tv_dynamic = tile_view(v, DynamicTile<2>{{4, 4}});

  for (int d = 0; d < 4; ++d) {
    EXPECT_EQ(tv_static.extent(d), tv_dynamic.extent(d)) << "d=" << d;
    EXPECT_EQ(tv_static.stride(d), tv_dynamic.stride(d)) << "d=" << d;
  }
}

// ---------------------------------------------------------------------------
// tile_view for View<ViewType, TileLayout> — same backing, 2N-dimensional
// *position-preserving* tiled layout. The (outer,inner) extent pair is
// interleaved per dimension (Right: outer,inner; Left: inner,outer) so the
// plain row/col-major strides reproduce the element's position in the un-tiled
// source: the interleaved coordinate maps to src(o_d*T_d + i_d). Uses
// StaticTileLayoutRight/Left as the source (stand-in for ScratchView).
// ---------------------------------------------------------------------------

using Buf1D = Kokkos::View<float*, Kokkos::LayoutRight, Kokkos::HostSpace>;

TEST(TiledLayout, TileViewScratchStaticRight) {
  Buf1D buf("buf", 96);  // 8*12 elements
  for (int k = 0; k < 96; ++k) buf(k) = static_cast<float>(k);

  View<Buf1D, StaticTileLayoutRight<8, 12>> src{buf, {}};
  auto tv = tile_view(src, StaticTile<2, 4>{});

  // Fully compile-time result type; interleaved <E0/T0, T0, E1/T1, T1>.
  static_assert(std::is_same_v<decltype(tv),
                               View<Buf1D, StaticTileLayoutRight<4, 2, 3, 4>>>);

  EXPECT_EQ(tv.data(), src.data());  // same backing
  EXPECT_EQ(tv.rank, 4);
  EXPECT_EQ(tv.extent(0), 4);  // outer dim 0: 8/2
  EXPECT_EQ(tv.extent(1), 2);  // inner dim 0: tile size 2
  EXPECT_EQ(tv.extent(2), 3);  // outer dim 1: 12/4
  EXPECT_EQ(tv.extent(3), 4);  // inner dim 1: tile size 4
  // Right: fastest dim last
  EXPECT_EQ(tv.stride(3), 1);
  EXPECT_EQ(tv.stride(2), 4);
  EXPECT_EQ(tv.stride(1), 12);
  EXPECT_EQ(tv.stride(0), 24);

  // position-preserving: tv(ti,ri,tj,rj) == src(ti*2+ri, tj*4+rj)
  for (int ti = 0; ti < 4; ++ti)
    for (int ri = 0; ri < 2; ++ri)
      for (int tj = 0; tj < 3; ++tj)
        for (int rj = 0; rj < 4; ++rj)
          EXPECT_FLOAT_EQ(tv(ti, ri, tj, rj),
                          static_cast<float>((ti * 2 + ri) * 12 + tj * 4 + rj));
}

TEST(TiledLayout, TileViewScratchStaticLeft) {
  Buf1D buf("buf", 96);
  for (int k = 0; k < 96; ++k) buf(k) = static_cast<float>(k);

  View<Buf1D, StaticTileLayoutLeft<8, 12>> src{buf, {}};
  auto tv = tile_view(src, StaticTile<2, 4>{});

  // Left interleaves inner-first: <T0, E0/T0, T1, E1/T1>.
  static_assert(std::is_same_v<decltype(tv),
                               View<Buf1D, StaticTileLayoutLeft<2, 4, 4, 3>>>);

  EXPECT_EQ(tv.data(), src.data());
  EXPECT_EQ(tv.extent(0), 2);  // inner dim 0: tile size 2
  EXPECT_EQ(tv.extent(1), 4);  // outer dim 0: 8/2
  EXPECT_EQ(tv.extent(2), 4);  // inner dim 1: tile size 4
  EXPECT_EQ(tv.extent(3), 3);  // outer dim 1: 12/4
  // Left: fastest dim first; col-major src has stride row=1, col=8
  EXPECT_EQ(tv.stride(0), 1);
  EXPECT_EQ(tv.stride(1), 2);
  EXPECT_EQ(tv.stride(2), 8);
  EXPECT_EQ(tv.stride(3), 32);

  // position-preserving: tv(ri,ti,rj,tj) == src(ti*2+ri, tj*4+rj)
  for (int ti = 0; ti < 4; ++ti)
    for (int ri = 0; ri < 2; ++ri)
      for (int tj = 0; tj < 3; ++tj)
        for (int rj = 0; rj < 4; ++rj)
          EXPECT_FLOAT_EQ(
              tv(ri, ti, rj, tj),
              static_cast<float>((ti * 2 + ri) + (tj * 4 + rj) * 8));
}

TEST(TiledLayout, TileViewScratchDynamicRight) {
  Buf1D buf("buf", 96);
  for (int k = 0; k < 96; ++k) buf(k) = static_cast<float>(k);

  View<Buf1D, DynamicTileLayoutRight<2>> src{
      buf, DynamicTileLayoutRight<2>{Kokkos::Array<int, 2>{8, 12}}};
  auto tv = tile_view(src, DynamicTile<2>{{2, 4}});

  static_assert(
      std::is_same_v<decltype(tv), View<Buf1D, DynamicTileLayoutRight<4>>>);

  EXPECT_EQ(tv.data(), src.data());
  EXPECT_EQ(tv.extent(0), 4);  // outer dim 0
  EXPECT_EQ(tv.extent(1), 2);  // inner dim 0
  EXPECT_EQ(tv.extent(2), 3);  // outer dim 1
  EXPECT_EQ(tv.extent(3), 4);  // inner dim 1
  EXPECT_EQ(tv.stride(3), 1);
  EXPECT_EQ(tv.stride(0), 24);

  for (int ti = 0; ti < 4; ++ti)
    for (int ri = 0; ri < 2; ++ri)
      for (int tj = 0; tj < 3; ++tj)
        for (int rj = 0; rj < 4; ++rj)
          EXPECT_FLOAT_EQ(tv(ti, ri, tj, rj),
                          static_cast<float>((ti * 2 + ri) * 12 + tj * 4 + rj));
}

TEST(TiledLayout, TileViewScratchDynamicLeft) {
  Buf1D buf("buf", 96);
  for (int k = 0; k < 96; ++k) buf(k) = static_cast<float>(k);

  View<Buf1D, DynamicTileLayoutLeft<2>> src{
      buf, DynamicTileLayoutLeft<2>{Kokkos::Array<int, 2>{8, 12}}};
  auto tv = tile_view(src, DynamicTile<2>{{2, 4}});

  static_assert(
      std::is_same_v<decltype(tv), View<Buf1D, DynamicTileLayoutLeft<4>>>);

  EXPECT_EQ(tv.data(), src.data());
  EXPECT_EQ(tv.extent(0), 2);  // inner dim 0
  EXPECT_EQ(tv.extent(1), 4);  // outer dim 0
  EXPECT_EQ(tv.extent(2), 4);  // inner dim 1
  EXPECT_EQ(tv.extent(3), 3);  // outer dim 1
  EXPECT_EQ(tv.stride(0), 1);
  EXPECT_EQ(tv.stride(3), 32);

  for (int ti = 0; ti < 4; ++ti)
    for (int ri = 0; ri < 2; ++ri)
      for (int tj = 0; tj < 3; ++tj)
        for (int rj = 0; rj < 4; ++rj)
          EXPECT_FLOAT_EQ(
              tv(ri, ti, rj, tj),
              static_cast<float>((ti * 2 + ri) + (tj * 4 + rj) * 8));
}

// ---------------------------------------------------------------------------
// reorder_view — logical axis permutation (transpose) of a subview.
//
// Gather convention: Perm[i] is the source dim that becomes new dim i, so
// r.extent(i) == sv.extent(Perm[i]) and r(new coords) == sv(old coords).
// Non-square, non-symmetric extents/data so a transpose bug can't hide.
// ---------------------------------------------------------------------------

TEST(TiledLayout, ReorderSubviewLayout2D) {
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace> v("v", 4, 6);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 6; ++j) v(i, j) = static_cast<float>(i * 6 + j);

  // Runtime SubviewLayout over the full view (as the DynamicTile staging path
  // would produce), built directly from the view's extents/strides.
  const Kokkos::Array<int, 2> ext{4, 6}, str{6, 1};
  auto                        sv = Subview<decltype(v), 2>{
      v, SubviewLayout<2>{0, ext, str, Impl::argsort_by_stride<2>(str),
                          Impl::reciprocals<2>(ext)}};
  static_assert(std::is_same_v<decltype(sv)::layout_t, SubviewLayout<2>>);

  auto r = reorder_view(sv, std::integer_sequence<int, 1, 0>{});
  // Type-preserving: SubviewLayout stays a SubviewLayout.
  static_assert(std::is_same_v<decltype(r)::layout_t, SubviewLayout<2>>);

  constexpr int perm[] = {1, 0};
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(r.extent(i), sv.extent(perm[i]));
    EXPECT_EQ(r.stride(i), sv.stride(perm[i]));
  }
  EXPECT_EQ(r.layout().base_offset(), sv.layout().base_offset());

  for (int a = 0; a < r.extent(0); ++a)
    for (int b = 0; b < r.extent(1); ++b)
      EXPECT_FLOAT_EQ(r(a, b), sv(b, a)) << "a=" << a << " b=" << b;
}

TEST(TiledLayout, ReorderSubviewLayout3D) {
  Kokkos::View<float***, Kokkos::LayoutRight, Kokkos::HostSpace> v("v", 2, 3,
                                                                   4);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 4; ++k)
        v(i, j, k) = static_cast<float>((i * 3 + j) * 4 + k);

  const Kokkos::Array<int, 3> ext{2, 3, 4}, str{12, 4, 1};
  auto                        sv = Subview<decltype(v), 3>{
      v, SubviewLayout<3>{0, ext, str, Impl::argsort_by_stride<3>(str),
                          Impl::reciprocals<3>(ext)}};
  static_assert(std::is_same_v<decltype(sv)::layout_t, SubviewLayout<3>>);

  // Perm {2,0,1}: new dim0<-old2, new dim1<-old0, new dim2<-old1.
  auto r = reorder_view(sv, std::integer_sequence<int, 2, 0, 1>{});
  static_assert(std::is_same_v<decltype(r)::layout_t, SubviewLayout<3>>);

  constexpr int perm[] = {2, 0, 1};
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(r.extent(i), sv.extent(perm[i]));
    EXPECT_EQ(r.stride(i), sv.stride(perm[i]));
  }

  for (int a = 0; a < r.extent(0); ++a)
    for (int b = 0; b < r.extent(1); ++b)
      for (int c = 0; c < r.extent(2); ++c)
        EXPECT_FLOAT_EQ(r(a, b, c), sv(b, c, a))
            << "a=" << a << " b=" << b << " c=" << c;
}

// Compile-time check: reorder_layout on an OrderedSubviewLayout is constexpr
// and produces the correct type/extents/strides in a constant expression.
namespace {
constexpr OrderedSubviewLayout<2, 1, 0> kOrdered2D{
    /*base=*/0, /*ext=*/{4, 6}, /*str=*/{6, 1}, /*inv=*/{0.25f, 1.0f / 6.0f}};
constexpr auto kReordered2D =
    reorder_layout(kOrdered2D, std::integer_sequence<int, 1, 0>{});
static_assert(
    std::is_same_v<std::decay_t<decltype(kReordered2D)>,
                   OrderedSubviewLayout<2, 0, 1>>,
    "reorder_layout must preserve the OrderedSubviewLayout family at compile "
    "time");
static_assert(kReordered2D.extent(0) == 6 && kReordered2D.extent(1) == 4);
static_assert(kReordered2D.stride(0) == 1 && kReordered2D.stride(1) == 6);
static_assert(kReordered2D.base_offset() == 0);
}  // namespace

TEST(TiledLayout, ReorderOrderedSubviewLayout2D) {
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace> v("v", 4, 6);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 6; ++j) v(i, j) = static_cast<float>(i * 6 + j);

  auto tv = tile_view(v, StaticTile<4, 6>{});  // shape (1,1,4,6)
  // subview_tile on a LayoutRight source yields an OrderedSubviewLayout with
  // compile-time order {N-1,...,0} = {1,0}.
  auto sv = subview_tile(tv, Kokkos::Array<int, 2>{0, 0});
  static_assert(
      std::is_same_v<decltype(sv)::layout_t, OrderedSubviewLayout<2, 1, 0>>);

  auto r = reorder_view(sv, std::integer_sequence<int, 1, 0>{});
  // Order {1,0} transposed by Perm {1,0} -> new memory order {0,1}.
  static_assert(
      std::is_same_v<decltype(r)::layout_t, OrderedSubviewLayout<2, 0, 1>>);

  constexpr int perm[] = {1, 0};
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(r.extent(i), sv.extent(perm[i]));
    EXPECT_EQ(r.stride(i), sv.stride(perm[i]));
  }
  EXPECT_EQ(r.layout().base_offset(), sv.layout().base_offset());

  for (int a = 0; a < r.extent(0); ++a)
    for (int b = 0; b < r.extent(1); ++b)
      EXPECT_FLOAT_EQ(r(a, b), sv(b, a)) << "a=" << a << " b=" << b;
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
