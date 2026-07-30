#include <Kokkos_Core.hpp>
#include <TensorOperations/TileLayout.hpp>
#include <TensorOperations/TiledLayout.hpp>
#include <TensorOperations/Tiling.hpp>
#include <gtest/gtest.h>

using namespace TensorOperations;

// ---------------------------------------------------------------------------
// StaticLayout<StaticTile<E...>, Strides, Order> — the generic static layout
// the three named layouts (Right / Left / Stride) are packed instances of.
//
// Two things need proving:
//   1. Equivalence — a StaticLayout carrying the packed strides for a memory
//      order behaves exactly like the named layout for that order (and *is*
//      its base class). This is what makes the three names safe to derive.
//   2. The generalisation — non-packed (padded) strides, which no named layout
//      can express: flat() must skip the pad while operator[] keeps decoding
//      densely over [0, size()).
// ---------------------------------------------------------------------------

namespace {

template <int... S>
using strides = std::integer_sequence<int, S...>;
template <int... O>
using order = std::integer_sequence<int, O...>;

// ---------------------------------------------------------------------------
// 1. Equivalence with the named layouts
// ---------------------------------------------------------------------------

// Row-major 4x8: dim 1 fastest, strides {8,1}
using Right2D  = StaticTileLayoutRight<4, 8>;
using Manual2D = StaticLayout<StaticTile<4, 8>, strides<8, 1>, order<1, 0>>;
static_assert(std::is_base_of_v<Manual2D, Right2D>,
              "StaticTileLayoutRight must derive from the packed StaticLayout "
              "for row-major order");

// Column-major 4x8: dim 0 fastest, strides {1,4}
using Left2D = StaticTileLayoutLeft<4, 8>;
static_assert(
    std::is_base_of_v<
        StaticLayout<StaticTile<4, 8>, strides<1, 4>, order<0, 1>>, Left2D>);

// Arbitrary order 4x8x3, Order={2,0,1}: stride(2)=1, stride(0)=3, stride(1)=12
using Stride3D = StaticTileLayoutStride<StaticTile<4, 8, 3>, 2, 0, 1>;
static_assert(std::is_base_of_v<StaticLayout<StaticTile<4, 8, 3>,
                                             strides<3, 12, 1>, order<2, 0, 1>>,
                                Stride3D>);

// Right<4,8> and Stride<{4,8}, 1,0> describe the same memory, so they share a
// base — the consolidation made that identity explicit rather than incidental.
static_assert(std::is_base_of_v<
              Manual2D, StaticTileLayoutStride<StaticTile<4, 8>, 1, 0>>);

// The full Layout interface is inherited and constant-evaluable.
static_assert(Manual2D::rank == 2);
static_assert(Manual2D::num_elements == 32);
static_assert(Manual2D::size() == 32);
static_assert(Manual2D::base_offset() == 0);
static_assert(Manual2D::extent(0) == 4 && Manual2D::extent(1) == 8);
static_assert(Manual2D::extent<0>() == 4 && Manual2D::extent<1>() == 8);
static_assert(Manual2D::stride(0) == 8 && Manual2D::stride(1) == 1);
static_assert(Manual2D::stride<0>() == 8 && Manual2D::stride<1>() == 1);
static_assert(Manual2D{}.flat(2, 3) == 19);
static_assert(Manual2D::flat_offset(Impl::Index<2>{2, 3}) == 19);

// ---------------------------------------------------------------------------
// 2. Padded strides — the capability no named layout has.
//
// A 4x8 tile with a row pitch of 9: extents {4,8}, strides {9,1}, dim 1
// fastest. Element (i,j) sits at 9*i + j, leaving one pad slot per row.
// ---------------------------------------------------------------------------
using Padded2D = StaticLayout<StaticTile<4, 8>, strides<9, 1>, order<1, 0>>;

static_assert(Padded2D::rank == 2);
static_assert(Padded2D::num_elements == 32);  // elements, not the 36 slots
static_assert(Padded2D::size() == 32);
static_assert(Padded2D::stride(0) == 9 && Padded2D::stride(1) == 1);
static_assert(Padded2D::stride<0>() == 9 && Padded2D::stride<1>() == 1);
static_assert(Padded2D{}.flat(0, 7) == 7);
static_assert(Padded2D{}.flat(1, 0) == 9);   // skips the pad slot at 8
static_assert(Padded2D{}.flat(3, 7) == 34);  // last element, buffer needs 35
static_assert(Padded2D::flat_offset(Impl::Index<2>{1, 0}) == 9);

// A padded 3-D layout with an arbitrary memory order: extents {4,8,3},
// Order={2,0,1} (dim 2 fastest), packed strides would be {3,12,1}; pad the
// middle axis to a pitch of 16 instead of 12.
using Padded3D =
    StaticLayout<StaticTile<4, 8, 3>, strides<3, 16, 1>, order<2, 0, 1>>;
static_assert(Padded3D{}.flat(1, 1, 2) == 3 + 16 + 2);

// ---------------------------------------------------------------------------
// 3. Nested (hierarchical) layouts — one logical mode backed by several memory
//    leaves. Indexing stays one scalar per mode; the split into leaves happens
//    inside flat()/operator[].
// ---------------------------------------------------------------------------

template <int... E>
using ext = std::integer_sequence<int, E...>;
template <typename... S>
using DT = DeviceTuple<S...>;

// --- 3a. Degenerate case: every mode arity 1 == the flat layout -------------
// ((4),(8)) : ((8),(1)) : ((1),(0))  must behave exactly like
// StaticTile<4,8> : {8,1} : {1,0}, i.e. Manual2D above.
using Nested1 =
    StaticLayout<DT<ext<4>, ext<8>>, DT<ext<8>, ext<1>>, DT<ext<1>, ext<0>>>;

static_assert(Nested1::rank == 2);
static_assert(Nested1::num_elements == 32);
static_assert(Nested1::size() == 32);
static_assert(Nested1::base_offset() == 0);
static_assert(Nested1::extent(0) == 4 && Nested1::extent(1) == 8);
static_assert(Nested1::extent<0>() == 4 && Nested1::extent<1>() == 8);
static_assert(Nested1{}.flat(2, 3) == 19);  // == Manual2D{}.flat(2, 3)
static_assert(Nested1::flat_offset(Impl::Index<2>{2, 3}) == 19);

// The mode_* family agrees with the flat layout's, which now spells it too.
static_assert(Nested1::mode_arity<0>() == 1 && Nested1::mode_arity<1>() == 1);
static_assert(Manual2D::mode_arity<0>() == 1);
static_assert(std::is_same_v<decltype(Nested1::mode_extents<0>()), ext<4>>);
static_assert(std::is_same_v<decltype(Manual2D::mode_extents<0>()), ext<4>>);
static_assert(std::is_same_v<decltype(Nested1::mode_strides<1>()), ext<1>>);
static_assert(std::is_same_v<decltype(Manual2D::mode_strides<1>()), ext<1>>);
static_assert(Nested1::mode_extent<0, 0>() == Manual2D::mode_extent<0, 0>());
static_assert(Nested1::mode_stride<0, 0>() == Manual2D::mode_stride<0, 0>());

// --- 3b. The motivating interleave -----------------------------------------
// A 25x25 matrix whose memory is a column-major (5,5,5,5) tensor indexed
// (i0,j0,i1,j1): mode 0 splits as i = i0 + 5*i1, mode 1 as j = j0 + 5*j1.
using Interleave =
    StaticLayout<DT<ext<5, 5>, ext<5, 5>>, DT<ext<1, 25>, ext<5, 125>>,
                 DT<ext<0, 2>, ext<1, 3>>>;

static_assert(Interleave::rank == 2);
static_assert(Interleave::extent(0) == 25 && Interleave::extent(1) == 25);
static_assert(Interleave::num_elements == 625);
static_assert(Interleave::mode_arity<0>() == 2);
static_assert(
    std::is_same_v<decltype(Interleave::mode_extents<0>()), ext<5, 5>>);
static_assert(
    std::is_same_v<decltype(Interleave::mode_strides<1>()), ext<5, 125>>);
static_assert(Interleave::mode_stride<1, 1>() == 125);

static_assert(Interleave{}.flat(0, 0) == 0);
static_assert(Interleave{}.flat(1, 0) == 1);    // i0 -> stride 1
static_assert(Interleave{}.flat(5, 0) == 25);   // i1 -> stride 25
static_assert(Interleave{}.flat(0, 1) == 5);    // j0 -> stride 5
static_assert(Interleave{}.flat(0, 5) == 125);  // j1 -> stride 125
static_assert(Interleave{}.flat(6, 6) == 1 + 25 + 5 + 125);
static_assert(Interleave{}.flat(24, 24) == 624);  // last element

// --- 3c. Ragged arity, pairwise-distinct extents ---------------------------
// extents ((2,3),(4,5,6)) strides ((1,8),(2,24,120)) order ((0,2),(1,3,4)).
// Memory order (fastest first) is leaves 0,2,1,3,4 with dense strides
// 1,2,8,24,120 — a mixed-radix bijection onto [0, 720). Unequal mode arities
// and no repeated extent, so a mode- or leaf-order transposition cannot pass.
using Ragged =
    StaticLayout<DT<ext<2, 3>, ext<4, 5, 6>>, DT<ext<1, 8>, ext<2, 24, 120>>,
                 DT<ext<0, 2>, ext<1, 3, 4>>>;

static_assert(Ragged::rank == 2);
static_assert(Ragged::extent(0) == 6 && Ragged::extent(1) == 120);
static_assert(Ragged::num_elements == 720);
static_assert(Ragged::mode_arity<0>() == 2 && Ragged::mode_arity<1>() == 3);
static_assert(Ragged::mode_extent<1, 2>() == 6);
static_assert(Ragged::mode_stride<1, 2>() == 120);

static_assert(Ragged{}.flat(1, 0) == 1);     // i0 -> stride 1
static_assert(Ragged{}.flat(2, 0) == 8);     // i1 -> stride 8
static_assert(Ragged{}.flat(0, 1) == 2);     // j0 -> stride 2
static_assert(Ragged{}.flat(0, 4) == 24);    // j1 -> stride 24
static_assert(Ragged{}.flat(0, 20) == 120);  // j2 -> stride 120
// i=5 -> (1,2); j=119 -> (3,4,5): 1 + 16 + 6 + 96 + 600
static_assert(Ragged{}.flat(5, 119) == 719);
static_assert(Ragged::flat_offset(Impl::Index<2>{5, 119}) == 719);

}  // namespace

// ---------------------------------------------------------------------------
// TEST: packed StaticLayout agrees with the named layout it is the base of,
// element for element — strides, flat(), and decode.
// ---------------------------------------------------------------------------
TEST(StaticLayout, MatchesNamedLayouts2D) {
  Manual2D manual{};
  Right2D  named{};

  for (int d = 0; d < 2; ++d) {
    EXPECT_EQ(manual.stride(d), named.stride(d)) << "d=" << d;
    EXPECT_EQ(manual.extent(d), named.extent(d)) << "d=" << d;
  }

  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j)
      EXPECT_EQ(manual.flat(i, j), named.flat(i, j)) << "i=" << i << " j=" << j;

  for (int f = 0; f < manual.size(); ++f) {
    const auto mc = manual[f];
    const auto nc = named[f];
    EXPECT_EQ(mc[0], nc[0]) << "f=" << f;
    EXPECT_EQ(mc[1], nc[1]) << "f=" << f;
  }
}

TEST(StaticLayout, MatchesNamedLayouts3DArbitraryOrder) {
  using Manual3D =
      StaticLayout<StaticTile<4, 8, 3>, strides<3, 12, 1>, order<2, 0, 1>>;
  Manual3D manual{};
  Stride3D named{};

  for (int d = 0; d < 3; ++d)
    EXPECT_EQ(manual.stride(d), named.stride(d)) << "d=" << d;

  for (int f = 0; f < manual.size(); ++f) {
    const auto mc = manual[f];
    const auto nc = named[f];
    for (int d = 0; d < 3; ++d)
      EXPECT_EQ(mc[d], nc[d]) << "f=" << f << " d=" << d;
  }
}

// ---------------------------------------------------------------------------
// TEST: decode is the inverse of encode, for both packed and padded strides.
// operator[] walks the dense range [0, size()); feeding its coordinate back
// through flat() must land on that element's (possibly padded) offset.
// ---------------------------------------------------------------------------
TEST(StaticLayout, RoundTripPacked3D) {
  StaticLayout<StaticTile<4, 8, 3>, strides<3, 12, 1>, order<2, 0, 1>> l{};

  bool hit[96] = {};  // packed: 96 elements in 96 slots
  for (int f = 0; f < l.size(); ++f) {
    const auto c   = l[f];
    const auto off = l.flat(c[0], c[1], c[2]);
    ASSERT_LT(off, 96u) << "f=" << f;
    EXPECT_FALSE(hit[off]) << "flat offset " << off << " reached twice";
    hit[off] = true;
  }
  // Packed strides: the decode covers every slot exactly once.
  for (int i = 0; i < 96; ++i) EXPECT_TRUE(hit[i]) << "slot " << i << " missed";
}

TEST(StaticLayout, RoundTripPadded3D) {
  Padded3D l{};  // extents {4,8,3}, strides {3,16,1}, dim 2 fastest

  // Highest element (3,7,2) → 3*3 + 7*16 + 2 = 123, so 124 slots.
  bool hit[124] = {};
  int  covered  = 0;
  for (int f = 0; f < l.size(); ++f) {
    const auto c   = l[f];
    const auto off = l.flat(c[0], c[1], c[2]);
    ASSERT_LT(off, 124u) << "f=" << f;
    EXPECT_FALSE(hit[off]) << "flat offset " << off << " reached twice";
    hit[off] = true;
    ++covered;
  }
  // 96 distinct elements spread over 124 slots: injective, and the 28 pad slots
  // are never addressed.
  EXPECT_EQ(covered, 96);
  EXPECT_EQ(l.size(), 96);
}

// ---------------------------------------------------------------------------
// TEST: View write-through on a padded layout — the pad slots must stay
// untouched. Values are distinct per element and the extents are unequal, so a
// transposed or pitch-ignoring offset cannot pass.
// ---------------------------------------------------------------------------
TEST(StaticLayout, ViewWriteThroughPadded2D) {
  using Buf1D = Kokkos::View<float*, Kokkos::LayoutRight, Kokkos::HostSpace>;
  Buf1D buf("buf", 36);  // 4 rows x pitch 9
  Kokkos::deep_copy(buf, -1.f);

  View<Buf1D, Padded2D> v{buf, Padded2D{}};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j) v(i, j) = static_cast<float>(100 * (i + 1) + j);

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 8; ++j)
      EXPECT_FLOAT_EQ(buf(9 * i + j), static_cast<float>(100 * (i + 1) + j))
          << "i=" << i << " j=" << j;
    // Slot 8 of each row is pad and must still hold the sentinel.
    EXPECT_FLOAT_EQ(buf(9 * i + 8), -1.f) << "pad of row " << i << " written";
  }

  // Reading back through the View agrees with the raw offsets.
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j)
      EXPECT_FLOAT_EQ(v(i, j), buf(9 * i + j)) << "i=" << i << " j=" << j;
}

// ---------------------------------------------------------------------------
// TEST: a padded layout still round-trips through View::operator[](Index),
// i.e. the decode path used by every element loop in the evaluators.
// ---------------------------------------------------------------------------
TEST(StaticLayout, ViewDecodeLoopPadded2D) {
  using Buf1D = Kokkos::View<float*, Kokkos::LayoutRight, Kokkos::HostSpace>;
  Buf1D buf("buf", 36);
  Kokkos::deep_copy(buf, -1.f);

  View<Buf1D, Padded2D> v{buf, Padded2D{}};
  Padded2D              layout{};

  // Flat loop over the dense element range, exactly as the tier kernels do.
  for (int f = 0; f < layout.size(); ++f) v[layout[f]] = static_cast<float>(f);

  // Element f decodes to (f/8, f%8) — row-major over the *logical* extents.
  for (int f = 0; f < 32; ++f)
    EXPECT_FLOAT_EQ(buf(9 * (f / 8) + (f % 8)), static_cast<float>(f))
        << "f=" << f;
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(buf(9 * i + 8), -1.f) << "pad of row " << i << " written";
}

// ---------------------------------------------------------------------------
// TEST: an arity-1 nested layout is indistinguishable from the flat layout it
// degenerates to — strides, flat(), and decode, element for element.
// ---------------------------------------------------------------------------
TEST(NestedStaticLayout, DegeneratesToFlatLayout) {
  Nested1  nested{};
  Manual2D flat{};

  for (int d = 0; d < 2; ++d) EXPECT_EQ(nested.extent(d), flat.extent(d));

  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j)
      EXPECT_EQ(nested.flat(i, j), flat.flat(i, j)) << "i=" << i << " j=" << j;

  for (int f = 0; f < flat.size(); ++f) {
    const auto nc = nested[f];
    const auto fc = flat[f];
    EXPECT_EQ(nc[0], fc[0]) << "f=" << f;
    EXPECT_EQ(nc[1], fc[1]) << "f=" << f;
  }
}

// ---------------------------------------------------------------------------
// TEST: flat() computes the interleave by hand-checked arithmetic, and decode
// inverts it. 625 elements over 625 slots, each reached exactly once.
// ---------------------------------------------------------------------------
TEST(NestedStaticLayout, InterleaveMatchesDigitSplit) {
  Interleave l{};

  for (int i = 0; i < 25; ++i)
    for (int j = 0; j < 25; ++j) {
      const std::size_t expect =
          (i % 5) * 1 + ((i / 5) % 5) * 25 + (j % 5) * 5 + ((j / 5) % 5) * 125;
      EXPECT_EQ(l.flat(i, j), expect) << "i=" << i << " j=" << j;
    }
}

TEST(NestedStaticLayout, InterleaveRoundTrip) {
  Interleave l{};

  bool hit[625] = {};
  for (int f = 0; f < l.size(); ++f) {
    const auto c = l[f];
    ASSERT_GE(c[0], 0);
    ASSERT_LT(c[0], 25) << "f=" << f;
    ASSERT_GE(c[1], 0);
    ASSERT_LT(c[1], 25) << "f=" << f;
    const auto off = l.flat(c[0], c[1]);
    ASSERT_LT(off, 625u) << "f=" << f;
    EXPECT_FALSE(hit[off]) << "flat offset " << off << " reached twice";
    hit[off] = true;
  }
  for (int i = 0; i < 625; ++i)
    EXPECT_TRUE(hit[i]) << "slot " << i << " missed";
}

// ---------------------------------------------------------------------------
// TEST: the ragged case — unequal mode arities, no repeated extent. Both a
// hand-checked encode and a full bijection over [0, 720).
// ---------------------------------------------------------------------------
TEST(NestedStaticLayout, RaggedMatchesLeafSplit) {
  Ragged l{};

  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 120; ++j) {
      const std::size_t expect = (i % 2) * 1 + (i / 2) * 8 + (j % 4) * 2 +
                                 ((j / 4) % 5) * 24 + (j / 20) * 120;
      EXPECT_EQ(l.flat(i, j), expect) << "i=" << i << " j=" << j;
    }
}

TEST(NestedStaticLayout, RaggedRoundTrip) {
  Ragged l{};

  bool hit[720] = {};
  for (int f = 0; f < l.size(); ++f) {
    const auto c = l[f];
    ASSERT_LT(c[0], 6) << "f=" << f;
    ASSERT_LT(c[1], 120) << "f=" << f;
    const auto off = l.flat(c[0], c[1]);
    ASSERT_LT(off, 720u) << "f=" << f;
    EXPECT_FALSE(hit[off]) << "flat offset " << off << " reached twice";
    hit[off] = true;
  }
  for (int i = 0; i < 720; ++i)
    EXPECT_TRUE(hit[i]) << "slot " << i << " missed";
}

// ---------------------------------------------------------------------------
// TEST: View write-through on a nested layout. Values are distinct per element
// and the interleave is non-trivial, so a collapsed or transposed offset cannot
// pass — each backing slot is checked against the hand-computed address.
// ---------------------------------------------------------------------------
TEST(NestedStaticLayout, ViewWriteThroughInterleave) {
  using Buf1D = Kokkos::View<float*, Kokkos::LayoutRight, Kokkos::HostSpace>;
  Buf1D buf("buf", 625);
  Kokkos::deep_copy(buf, -1.f);

  View<Buf1D, Interleave> v{buf, Interleave{}};
  for (int i = 0; i < 25; ++i)
    for (int j = 0; j < 25; ++j) v(i, j) = static_cast<float>(25 * i + j);

  for (int i = 0; i < 25; ++i)
    for (int j = 0; j < 25; ++j) {
      const int slot =
          (i % 5) * 1 + ((i / 5) % 5) * 25 + (j % 5) * 5 + ((j / 5) % 5) * 125;
      EXPECT_FLOAT_EQ(buf(slot), static_cast<float>(25 * i + j))
          << "i=" << i << " j=" << j;
    }

  // Every slot written: the map is onto, so no sentinel survives.
  for (int s = 0; s < 625; ++s) EXPECT_NE(buf(s), -1.f) << "slot " << s;
}

// ---------------------------------------------------------------------------
// TEST: the decode loop every tier kernel uses, over a nested layout.
// ---------------------------------------------------------------------------
TEST(NestedStaticLayout, ViewDecodeLoopRagged) {
  using Buf1D = Kokkos::View<float*, Kokkos::LayoutRight, Kokkos::HostSpace>;
  Buf1D buf("buf", 720);
  Kokkos::deep_copy(buf, -1.f);

  View<Buf1D, Ragged> v{buf, Ragged{}};
  Ragged              layout{};

  for (int f = 0; f < layout.size(); ++f) v[layout[f]] = static_cast<float>(f);

  // Decode is a bijection onto the slots, so slot f holds exactly f.
  for (int f = 0; f < 720; ++f)
    EXPECT_FLOAT_EQ(buf(f), static_cast<float>(f)) << "f=" << f;
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
