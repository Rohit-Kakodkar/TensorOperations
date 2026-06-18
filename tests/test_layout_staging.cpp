#include <Kokkos_Core.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/Layout.hpp>
#include <TensorOperations/RegisterArray.hpp>
#include <TensorOperations/TensorHandle.hpp>
#include <TensorOperations/Tiling.hpp>
#include <gtest/gtest.h>

#include <array>

using namespace TensorOperations;

namespace {

using HostRight  = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace>;
using HostLeft   = Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::HostSpace>;
using HostStride = Kokkos::View<float**, Kokkos::LayoutStride, Kokkos::HostSpace>;

// Logical data, independent of how it is laid out in memory.
float D(int i, int j) { return static_cast<float>(i * 10 + j); }

// A plain TensorLike grid: no array_layout -> layout_of defaults to LayoutRight.
struct PlainGrid {
  static constexpr int rank = 2;
  using value_type          = float;
};

template <typename ViewT>
void fill_logical(const ViewT& v) {
  for (int i = 0; i < static_cast<int>(v.extent(0)); ++i)
    for (int j = 0; j < static_cast<int>(v.extent(1)); ++j) v(i, j) = D(i, j);
}

// Stage the 2x2 tile at tile index (ti,tj) from `v` through the register-tier
// input evaluator and return the staged RegisterArray.
template <typename ViewT>
RegisterArray<float, 2, 2> stage(const ViewT& v, int ti, int tj) {
  auto inp = make_input_node(make_handle(v, std::array<int32_t, 2>{'i', 'j'}));
  auto ev  = make_evaluator<RangePolicyTag>(inp, StaticTile<2, 2>{});
  return ev({ti, tj}).storage_;
}

}  // namespace

// ---------------------------------------------------------------------------
// Layout traits resolve as expected; non-view grids default to LayoutRight.
// ---------------------------------------------------------------------------
TEST(LayoutStaging, Traits) {
  static_assert(!Impl::is_layout_left_v<HostRight>);
  static_assert(!Impl::is_layout_stride_v<HostRight>);
  static_assert(Impl::is_layout_left_v<HostLeft>);
  static_assert(!Impl::is_layout_stride_v<HostLeft>);
  static_assert(Impl::is_layout_stride_v<HostStride>);
  static_assert(!Impl::is_layout_left_v<HostStride>);

  // A plain TensorLike grid has no array_layout -> defaults to LayoutRight.
  static_assert(
      std::is_same_v<Impl::layout_of_t<PlainGrid>, Kokkos::LayoutRight>);
  static_assert(!Impl::is_layout_left_v<PlainGrid>);
  static_assert(!Impl::is_layout_stride_v<PlainGrid>);
  SUCCEED();
}

// ---------------------------------------------------------------------------
// Gather parity: the same logical tile staged from Right / Left / Stride views
// must yield identical RegisterArray contents (= the row-major slot mapping is
// independent of the visit order chosen per layout).
// ---------------------------------------------------------------------------
TEST(LayoutStaging, GatherParity) {
  constexpr int n0 = 4, n1 = 4;
  HostRight right("right", n0, n1);
  HostLeft  left("left", n0, n1);
  // Strided, contiguous along dim 0 (column-major-like): forces the f=0 branch.
  HostStride stride_col("stride_col",
                        Kokkos::LayoutStride(n0, 1, n1, n0));
  // Strided, contiguous along the last dim: forces the f=Rank-1 branch.
  HostStride stride_row("stride_row",
                        Kokkos::LayoutStride(n0, n1, n1, 1));

  fill_logical(right);
  fill_logical(left);
  fill_logical(stride_col);
  fill_logical(stride_row);

  // Tile indices: 2 tiles per dimension (n=4, tile=2).
  for (int ti = 0; ti < n0 / 2; ++ti)
    for (int tj = 0; tj < n1 / 2; ++tj) {
      auto r  = stage(right, ti, tj);
      auto l  = stage(left, ti, tj);
      auto sc = stage(stride_col, ti, tj);
      auto sr = stage(stride_row, ti, tj);
      for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b) {
          const float expect = D(ti * 2 + a, tj * 2 + b);
          EXPECT_FLOAT_EQ((r(a, b)), expect) << "right ti=" << ti << ",tj=" << tj;
          EXPECT_FLOAT_EQ((l(a, b)), expect) << "left ti=" << ti << ",tj=" << tj;
          EXPECT_FLOAT_EQ((sc(a, b)), expect) << "stride_col ti=" << ti << ",tj=" << tj;
          EXPECT_FLOAT_EQ((sr(a, b)), expect) << "stride_row ti=" << ti << ",tj=" << tj;
        }
    }
}

// ---------------------------------------------------------------------------
// Store parity: drain a known register tile into Right / Left / Stride
// destinations; every layout must land identical values.
// ---------------------------------------------------------------------------
TEST(LayoutStaging, StoreParity) {
  constexpr int n0 = 4, n1 = 4;
  // Source the register tile from a LayoutRight view (gather is already
  // covered above); the staged node is the input to the store evaluator.
  HostRight src("src", n0, n1);
  fill_logical(src);

  auto stage_node = [&](int ti, int tj) {
    auto inp =
        make_input_node(make_handle(src, std::array<int32_t, 2>{'i', 'j'}));
    auto ev = make_evaluator<RangePolicyTag>(inp, StaticTile<2, 2>{});
    return ev({ti, tj});
  };

  HostRight  right("dr", n0, n1);
  HostLeft   left("dl", n0, n1);
  HostStride scol("dsc", Kokkos::LayoutStride(n0, 1, n1, n0));
  HostStride srow("dsr", Kokkos::LayoutStride(n0, n1, n1, 1));

  // Tile indices: 2 tiles per dimension.
  for (int ti = 0; ti < n0 / 2; ++ti)
    for (int tj = 0; tj < n1 / 2; ++tj) {
      auto node = stage_node(ti, tj);
      make_evaluator<RangePolicyTag>(node, StaticTile<2, 2>{})({ti, tj}, right);
      make_evaluator<RangePolicyTag>(node, StaticTile<2, 2>{})({ti, tj}, left);
      make_evaluator<RangePolicyTag>(node, StaticTile<2, 2>{})({ti, tj}, scol);
      make_evaluator<RangePolicyTag>(node, StaticTile<2, 2>{})({ti, tj}, srow);
    }

  for (int i = 0; i < n0; ++i)
    for (int j = 0; j < n1; ++j) {
      EXPECT_FLOAT_EQ((right(i, j)), D(i, j));
      EXPECT_FLOAT_EQ((left(i, j)), D(i, j));
      EXPECT_FLOAT_EQ((scol(i, j)), D(i, j));
      EXPECT_FLOAT_EQ((srow(i, j)), D(i, j));
    }
}

// ---------------------------------------------------------------------------
// Boundary tile: a tile that hangs off the view edge must take the guarded
// fallback and write only the in-bounds slots (no out-of-range writes).
// ---------------------------------------------------------------------------
TEST(LayoutStaging, StoreBoundaryGuard) {
  constexpr int n = 4;
  HostRight src("src", n, n);
  fill_logical(src);
  auto inp = make_input_node(make_handle(src, std::array<int32_t, 2>{'i', 'j'}));
  auto sev = make_evaluator<RangePolicyTag>(inp, StaticTile<2, 2>{});
  auto node = sev({1, 1});  // tile index (1,1) -> element origin (2,2)

  // Destination is 3x3, sentinel-filled. Tile origin (2,2) with a 2x2 tile:
  // only (2,2) is in bounds; (2,3),(3,2),(3,3) are out of range.
  HostRight  dst_r("dr", 3, 3);
  HostStride dst_s("ds", Kokkos::LayoutStride(3, 1, 3, 3));  // contiguous dim 0
  Kokkos::deep_copy(dst_r, -1.f);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) dst_s(i, j) = -1.f;

  make_evaluator<RangePolicyTag>(node, StaticTile<2, 2>{})({1, 1}, dst_r);
  make_evaluator<RangePolicyTag>(node, StaticTile<2, 2>{})({1, 1}, dst_s);

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      const float expect = (i == 2 && j == 2) ? D(2, 2) : -1.f;
      EXPECT_FLOAT_EQ((dst_r(i, j)), expect) << "right " << i << "," << j;
      EXPECT_FLOAT_EQ((dst_s(i, j)), expect) << "stride " << i << "," << j;
    }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
