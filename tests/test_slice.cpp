// Tests for slice() -- Kokkos-subview-style mode selection on a StaticLayout
// (or a View over one). slice takes EXACTLY rank arguments; each is either an
// integer (fix that mode to the value) or TensorOperations::ALL (keep it). The
// surviving modes form a layout of rank == (number of ALL args) whose runtime
// base offset is the flat offset of the fixed coordinate.
//
// THE INVARIANT: for a source of rank R with a per-mode choice of fix f_d or
// keep,
//
//     slice(src, a_0,...,a_{R-1}).flat_offset(g)  ==  src.flat_offset(full)
//
// where full[d] = f_d on a fixed mode and full[d] = g[kept-rank of d] on a kept
// mode. The reduced layout carries the SAME per-mode stride math as the
// source's surviving modes -- only its base moves -- which is what lets a
// level's members share one coordinate decode and each reconstitute merely
// (member_base + shared_k_accessor).
//
// The fixed modes may be ANYWHERE, not only leading: the tests fix middle and
// trailing modes explicitly. Covered on both a FLAT StaticLayout and a NESTED
// (multi-leaf) one, since the SEM eta/gamma level-3 members produce multi-leaf
// surviving modes.

#include <TensorOperations/DeviceTuple.hpp>
#include <TensorOperations/TileLayout.hpp>
#include <TensorOperations/TiledLayout.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <utility>
#include <vector>

using namespace TensorOperations;

namespace {

struct Backing {
  using value_type = float;
  float*                 p_;
  KOKKOS_FUNCTION float* data() const noexcept { return p_; }
};

template <typename Layout>
using V = View<Backing, Layout>;

template <int... E>
using Tile = StaticTile<E...>;
template <int... S>
using Str = std::integer_sequence<int, S...>;
template <int... O>
using Ord = std::integer_sequence<int, O...>;

// --- source layout aliases --------------------------------------------------

using FlatSrc   = StaticLayout<Tile<4, 8>, Str<8, 1>, Ord<1, 0>>;
using PaddedSrc = StaticLayout<Tile<4, 8>, Str<9, 1>, Ord<1, 0>>;

// mode 0 single-leaf, mode 1 two-leaf (global leaves {1,2}).
using NestSrc = StaticLayout<DeviceTuple<Str<5>, Str<5, 5>>,
                             DeviceTuple<Str<1>, Str<5, 25>>,
                             DeviceTuple<Ord<0>, Ord<1, 2>>>;

using R3Right = StaticTileLayoutRight<3, 5, 7>;
using R3Left  = StaticTileLayoutLeft<3, 5, 7>;

// The nested SEM-shaped sources: a permuted (TE,N,N,N) tile regrouped at split
// 1 into (free, contracted). The col mode is multi-leaf for the middle- and
// trailing-axis contractions.
using U4    = StaticTileLayoutRight<4, 5, 5, 5>;
using Perm1 = std::integer_sequence<int, 1, 0, 2, 3>;
using Perm2 = std::integer_sequence<int, 2, 0, 1, 3>;
using Perm3 = std::integer_sequence<int, 3, 0, 1, 2>;

template <typename Perm>
using Reordered = decltype(reorder_tile(U4{}, Perm{}));
template <typename Perm>
using Regrouped =
    Impl::regroup_layout_t<Reordered<Perm>, Impl::split_groups_t<1, 4>>;

using Regroup1 = Regrouped<Perm1>;
using Regroup2 = Regrouped<Perm2>;
using Regroup3 = Regrouped<Perm3>;

// The contraction A-operand shape: regrouped at Split<3,4> so the ROW mode is
// multi-leaf (3 leaves, product SA) and the contracted COL k is single-leaf.
// This is exactly what the contraction evaluator slices at the free row index
// i.
template <typename Perm>
using ARegrouped =
    Impl::regroup_layout_t<Reordered<Perm>, Impl::split_groups_t<3, 4>>;

using ARegroup1 = ARegrouped<Perm1>;
using ARegroup2 = ARegrouped<Perm2>;
using ARegroup3 = ARegrouped<Perm3>;

// --- compile-time type surgery ---------------------------------------------
//
// Keeping the TRAILING mode of a flat 4x8 (row-major) layout renumbers it to
// dimension 0 and rebinds to DynamicOffset -- the fixed leading mode is
// dropped.
using SlicedFlatKeepCol =
    decltype(slice(std::declval<const FlatSrc&>(), 0, ALL));
static_assert(
    std::is_same_v<SlicedFlatKeepCol,
                   StaticLayout<Tile<8>, Str<1>, Ord<0>, Impl::DynamicOffset>>,
    "keeping the trailing mode drops the leading extent/stride and renumbers "
    "the survivor to 0");

// Keeping the LEADING mode (fixing a NON-leading index) keeps its extent/stride
// and renumbers its order entry to 0.
using SlicedFlatKeepRow =
    decltype(slice(std::declval<const FlatSrc&>(), ALL, 0));
static_assert(
    std::is_same_v<SlicedFlatKeepRow,
                   StaticLayout<Tile<4>, Str<8>, Ord<0>, Impl::DynamicOffset>>,
    "keeping the leading mode while fixing the trailing one preserves the "
    "leading extent/stride and renumbers its order to 0");

// Fixing mode 1 of the nested source keeps mode 0; the surviving global leaf
// index {0} stays dense.
using SlicedNestKeepFirst =
    decltype(slice(std::declval<const NestSrc&>(), ALL, 0));
static_assert(
    std::is_same_v<SlicedNestKeepFirst,
                   StaticLayout<DeviceTuple<Str<5>>, DeviceTuple<Str<1>>,
                                DeviceTuple<Ord<0>>, Impl::DynamicOffset>>,
    "fixing the trailing nested mode keeps the leading single-leaf mode");

// Fixing mode 0 of the nested source keeps the two-leaf mode 1; its global leaf
// indices {1,2} renormalize to a dense {0,1}.
using SlicedNestKeepSecond =
    decltype(slice(std::declval<const NestSrc&>(), 0, ALL));
static_assert(
    std::is_same_v<SlicedNestKeepSecond,
                   StaticLayout<DeviceTuple<Str<5, 5>>, DeviceTuple<Str<5, 25>>,
                                DeviceTuple<Ord<0, 1>>, Impl::DynamicOffset>>,
    "fixing the leading nested mode keeps the multi-leaf mode and renormalizes "
    "its global leaf indices to a dense range");

// --- runtime invariant ------------------------------------------------------
//
// Enumerate every coordinate of the kept modes; the reduced layout's
// flat_offset must equal the source's over the reconstituted full coordinate.
// `fixedVals` carries the fixed values (0 on kept modes); `keptDims` lists the
// source dims that were ALL, ascending -- exactly the modes the reduced layout
// renumbers to 0..K-1.
template <typename Src, typename Red, int R, int K>
void verify(Src src, Red red, Kokkos::Array<int, R> fixedVals,
            Kokkos::Array<int, K> keptDims, const char* what) {
  static_assert(Red::rank == K, "reduced rank must equal the number of ALLs");
  int ext[K];
  int total = 1;
  for (int a = 0; a < K; ++a) {
    ext[a] = src.extent(keptDims[a]);
    total *= ext[a];
  }
  for (int lin = 0; lin < total; ++lin) {
    Impl::Index<R> full{};
    for (int d = 0; d < R; ++d) full[d] = fixedVals[d];
    Impl::Index<K> g{};
    int            rem = lin;
    for (int a = K - 1; a >= 0; --a) {
      const int c = rem % ext[a];
      rem /= ext[a];
      g[a]              = c;
      full[keptDims[a]] = c;
    }
    ASSERT_EQ(red.flat_offset(g), src.flat_offset(full))
        << what << " lin=" << lin;
  }
}

}  // namespace

// --- flat sources, fixing a single mode at each position --------------------

TEST(Slice, FlatFixLeading) {
  R3Right src{};
  for (int i = 0; i < src.extent(0); ++i)
    verify<R3Right, decltype(slice(src, i, ALL, ALL)), 3, 2>(
        src, slice(src, i, ALL, ALL), {i, 0, 0}, {1, 2}, "right fix dim0");
  R3Left srcL{};
  for (int i = 0; i < srcL.extent(0); ++i)
    verify<R3Left, decltype(slice(srcL, i, ALL, ALL)), 3, 2>(
        srcL, slice(srcL, i, ALL, ALL), {i, 0, 0}, {1, 2}, "left fix dim0");
}

TEST(Slice, FlatFixMiddle) {
  R3Right src{};
  for (int j = 0; j < src.extent(1); ++j)
    verify<R3Right, decltype(slice(src, ALL, j, ALL)), 3, 2>(
        src, slice(src, ALL, j, ALL), {0, j, 0}, {0, 2}, "right fix dim1");
  R3Left srcL{};
  for (int j = 0; j < srcL.extent(1); ++j)
    verify<R3Left, decltype(slice(srcL, ALL, j, ALL)), 3, 2>(
        srcL, slice(srcL, ALL, j, ALL), {0, j, 0}, {0, 2}, "left fix dim1");
}

TEST(Slice, FlatFixTrailing) {
  R3Right src{};
  for (int k = 0; k < src.extent(2); ++k)
    verify<R3Right, decltype(slice(src, ALL, ALL, k)), 3, 2>(
        src, slice(src, ALL, ALL, k), {0, 0, k}, {0, 1}, "right fix dim2");
  R3Left srcL{};
  for (int k = 0; k < srcL.extent(2); ++k)
    verify<R3Left, decltype(slice(srcL, ALL, ALL, k)), 3, 2>(
        srcL, slice(srcL, ALL, ALL, k), {0, 0, k}, {0, 1}, "left fix dim2");
}

TEST(Slice, FlatFixTwoNonAdjacent) {
  R3Right src{};
  for (int i = 0; i < src.extent(0); ++i)
    for (int k = 0; k < src.extent(2); ++k)
      verify<R3Right, decltype(slice(src, i, ALL, k)), 3, 1>(
          src, slice(src, i, ALL, k), {i, 0, k}, {1}, "right fix dim0+dim2");
  R3Left srcL{};
  for (int i = 0; i < srcL.extent(0); ++i)
    for (int k = 0; k < srcL.extent(2); ++k)
      verify<R3Left, decltype(slice(srcL, i, ALL, k)), 3, 1>(
          srcL, slice(srcL, i, ALL, k), {i, 0, k}, {1}, "left fix dim0+dim2");
}

TEST(Slice, FlatRank2BothPositions) {
  FlatSrc   src{};
  PaddedSrc pad{};
  for (int i = 0; i < 4; ++i) {
    verify<FlatSrc, decltype(slice(src, i, ALL)), 2, 1>(
        src, slice(src, i, ALL), {i, 0}, {1}, "row-major fix dim0");
    verify<PaddedSrc, decltype(slice(pad, i, ALL)), 2, 1>(
        pad, slice(pad, i, ALL), {i, 0}, {1}, "padded fix dim0");
  }
  for (int j = 0; j < 8; ++j) {
    verify<FlatSrc, decltype(slice(src, ALL, j)), 2, 1>(
        src, slice(src, ALL, j), {0, j}, {0}, "row-major fix dim1");
    verify<PaddedSrc, decltype(slice(pad, ALL, j)), 2, 1>(
        pad, slice(pad, ALL, j), {0, j}, {0}, "padded fix dim1");
  }
}

// --- nested (multi-leaf) sources, fixing either mode ------------------------

TEST(Slice, NestedRegroupedFixCol) {
  Regroup2 src{};  // multi-leaf col mode
  for (int k = 0; k < src.extent(1); ++k)
    verify<Regroup2, decltype(slice(src, ALL, k)), 2, 1>(
        src, slice(src, ALL, k), {0, k}, {0}, "regroup2 fix col (keep free)");
  Regroup3 src3{};
  for (int k = 0; k < src3.extent(1); ++k)
    verify<Regroup3, decltype(slice(src3, ALL, k)), 2, 1>(
        src3, slice(src3, ALL, k), {0, k}, {0}, "regroup3 fix col (keep free)");
}

TEST(Slice, NestedRegroupedFixRow) {
  Regroup1 src{};
  for (int r = 0; r < src.extent(0); ++r)
    verify<Regroup1, decltype(slice(src, r, ALL)), 2, 1>(
        src, slice(src, r, ALL), {r, 0}, {1}, "regroup1 fix row (keep col)");
  Regroup2 src2{};
  for (int r = 0; r < src2.extent(0); ++r)
    verify<Regroup2, decltype(slice(src2, r, ALL)), 2, 1>(
        src2, slice(src2, r, ALL), {r, 0}, {1},
        "regroup2 fix row (keep multi-leaf col)");
}

// --- the contraction A-operand: fix the multi-leaf row by a linear index ----
//
// This is the shape the contraction evaluator relies on: a2_ = regroup at
// Split<3,4>, whose ROW is multi-leaf. slice(a2_, i, ALL) must fold the full
// multi-leaf i-decode into the base and leave a single-leaf k accessor.

TEST(Slice, ContractionARowFixMultiLeaf) {
  ARegroup1 a{};
  for (int i = 0; i < a.extent(0); ++i)
    verify<ARegroup1, decltype(slice(a, i, ALL)), 2, 1>(
        a, slice(a, i, ALL), {i, 0}, {1}, "A perm1 Split<3,4> fix row");
  ARegroup2 a2{};
  for (int i = 0; i < a2.extent(0); ++i)
    verify<ARegroup2, decltype(slice(a2, i, ALL)), 2, 1>(
        a2, slice(a2, i, ALL), {i, 0}, {1}, "A perm2 Split<3,4> fix row");
  ARegroup3 a3{};
  for (int i = 0; i < a3.extent(0); ++i)
    verify<ARegroup3, decltype(slice(a3, i, ALL)), 2, 1>(
        a3, slice(a3, i, ALL), {i, 0}, {1}, "A perm3 Split<3,4> fix row");
}

// --- the base offset lands exactly at the fixed coordinate ------------------

TEST(Slice, BaseOffsetEqualsFixedPrefix) {
  StaticTileLayoutRight<4, 8> src{};
  for (int i = 0; i < 4; ++i)
    EXPECT_EQ(slice(src, i, ALL).base_offset(), i * 8) << "fix dim0 i=" << i;
  for (int j = 0; j < 8; ++j)
    EXPECT_EQ(slice(src, ALL, j).base_offset(), j) << "fix dim1 j=" << j;
}

// --- View level: zero-copy + write-through, flat, fixing a non-leading mode --

TEST(Slice, ViewFlatFixTrailingWriteThrough) {
  std::vector<float>             buf(32, -1.0f);
  V<StaticTileLayoutRight<4, 8>> v{Backing{buf.data()}, {}};

  // Fix the trailing column j=3; keep the leading row. Base is 0 + 3, stride 8.
  auto col = slice(v, ALL, 3);
  EXPECT_EQ(col.data(), v.data() + 3) << "slice must not copy";
  EXPECT_EQ(decltype(col)::rank, 1);

  for (int r = 0; r < 4; ++r) col(r) = static_cast<float>(100 + r);
  for (int r = 0; r < 4; ++r)
    EXPECT_FLOAT_EQ(buf[static_cast<std::size_t>(r * 8 + 3)],
                    static_cast<float>(100 + r))
        << "write through slice must land at base + r*stride; r=" << r;
}

TEST(Slice, ViewFlatFixLeadingWriteThrough) {
  std::vector<float>             buf(32, -1.0f);
  V<StaticTileLayoutRight<4, 8>> v{Backing{buf.data()}, {}};

  auto row = slice(v, 2, ALL);
  EXPECT_EQ(row.data(), v.data() + 2 * 8) << "slice must not copy";
  EXPECT_EQ(decltype(row)::rank, 1);

  for (int k = 0; k < 8; ++k) row(k) = static_cast<float>(100 + k);
  for (int k = 0; k < 8; ++k)
    EXPECT_FLOAT_EQ(buf[static_cast<std::size_t>(16 + k)],
                    static_cast<float>(100 + k))
        << "k=" << k;
}

// --- View level: nested source, write-through lands where the source says ---

TEST(Slice, ViewNestedWriteThrough) {
  std::vector<float> buf(500, 0.0f);

  V<Reordered<Perm2>> v{Backing{buf.data()}, {}};
  auto                mat = regroup_view(v, Split<1, 4>{});

  Regroup2  matL{};
  const int nrow = matL.extent(0), ncol = matL.extent(1);

  for (int r = 0; r < nrow; ++r) {
    auto col = slice(mat, r, ALL);
    EXPECT_EQ(decltype(col)::rank, 1);
    for (int k = 0; k < ncol; ++k) col(k) = static_cast<float>(r * 1000 + k);
  }

  for (int r = 0; r < nrow; ++r)
    for (int k = 0; k < ncol; ++k) {
      const int off = matL.flat_offset(Impl::Index<2>{r, k});
      EXPECT_FLOAT_EQ(buf[static_cast<std::size_t>(off)],
                      static_cast<float>(r * 1000 + k))
          << "r=" << r << " k=" << k;
    }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
