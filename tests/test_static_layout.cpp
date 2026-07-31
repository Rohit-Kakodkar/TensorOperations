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

// ---------------------------------------------------------------------------
// 4. The generic reshape — reshape(StaticLayout, Tile, Order).
//
// Reinterprets a flat StaticLayout (packed or padded, any memory order) under a
// new shape and a new memory order, with no data movement. It is the producer
// the nested layout above was built for.
//
// Every accepted case asserts the resulting TYPE *and* the offset map it
// produces. Type equality alone is not enough: a planner that emits a plausible
// shape with wrong strides would pass a type check and read the wrong memory.
// ---------------------------------------------------------------------------

// The k-th element of Padded2D in memory order, which is what a reshape of it
// must reproduce. Padded2D is row-major over (4,8), so element k is (k/8, k%8).
constexpr int padded2d_offset_at(int k) noexcept {
  return static_cast<int>(Padded2D{}.flat(k / 8, k % 8));
}

// --- 4a. Packed source, flat result ----------------------------------------
// Every new mode maps to a single contiguous run, so the result collapses to
// the FLAT specialisation — which is what keeps stride(d) available. A flat
// result whose strides are also packed goes one step further and recovers the
// NAMED layout, so that the result stays usable with tile_layout()/tile_view().
using ReshRight = decltype(reshape(StaticTileLayoutRight<32>{},
                                   StaticTile<4, 8>{}, order<1, 0>{}));
static_assert(std::is_same_v<ReshRight, StaticTileLayoutRight<4, 8>>);
// ...whose base is exactly the packed StaticLayout for that shape and order.
static_assert(std::is_base_of_v<Manual2D, ReshRight>);
static_assert(
    std::is_base_of_v<
        StaticLayout<StaticTile<4, 8>, strides<8, 1>, order<1, 0>>, ReshRight>);
static_assert(ReshRight::stride(0) == 8 && ReshRight::stride(1) == 1);
static_assert(ReshRight::mode_arity<0>() == 1 &&
              ReshRight::mode_arity<1>() == 1);

// The same source under the opposite order gives the column-major answer, i.e.
// the Order argument alone selects between the two conventions the packed
// overloads hard-code.
using ReshLeft = decltype(reshape(StaticTileLayoutRight<32>{},
                                  StaticTile<4, 8>{}, order<0, 1>{}));
static_assert(std::is_same_v<ReshLeft, StaticTileLayoutLeft<4, 8>>);
static_assert(
    std::is_base_of_v<
        StaticLayout<StaticTile<4, 8>, strides<1, 4>, order<0, 1>>, ReshLeft>);
static_assert(ReshLeft::stride(0) == 1 && ReshLeft::stride(1) == 4);

// The LayoutRight / LayoutLeft spellings are the same two results.
static_assert(
    std::is_same_v<decltype(reshape(StaticTileLayoutRight<32>{},
                                    StaticTile<4, 8>{}, LayoutRight{})),
                   ReshRight>);
static_assert(
    std::is_same_v<decltype(reshape(StaticTileLayoutRight<32>{},
                                    StaticTile<4, 8>{}, LayoutLeft{})),
                   ReshLeft>);

// A named layout binds to the generic overload by base-class deduction, and a
// packed result comes back named, so the three-argument form is a drop-in
// superset of the two-argument ones: same source, same target order, same type.
static_assert(
    std::is_same_v<decltype(reshape(StaticTileLayoutRight<32>{},
                                    StaticTile<4, 8>{}, LayoutRight{})),
                   decltype(reshape(StaticTileLayoutRight<32>{},
                                    StaticTile<4, 8>{}))>);
static_assert(
    std::is_same_v<decltype(reshape(StaticTileLayoutLeft<32>{},
                                    StaticTile<4, 8>{}, LayoutLeft{})),
                   decltype(reshape(StaticTileLayoutLeft<32>{},
                                    StaticTile<4, 8>{}))>);

// --- 4b. Padded source, merged across the gap ------------------------------
// (4,8) with a row pitch of 9 flattened to rank 1: no single stride can express
// it, so the result is one mode of two leaves — extents (8,4), strides (1,9).
using ReshPaddedFlat =
    decltype(reshape(Padded2D{}, StaticTile<32>{}, order<0>{}));
static_assert(
    std::is_same_v<ReshPaddedFlat,
                   StaticLayout<DT<ext<8, 4>>, DT<ext<1, 9>>, DT<ext<0, 1>>>>);
static_assert(ReshPaddedFlat::rank == 1);
static_assert(ReshPaddedFlat::extent(0) == 32);
static_assert(ReshPaddedFlat::mode_arity<0>() == 2);

constexpr bool padded_flat_offsets_match() noexcept {
  for (int k = 0; k < 32; ++k)
    if (static_cast<int>(ReshPaddedFlat{}.flat(k)) != padded2d_offset_at(k))
      return false;
  return true;
}
static_assert(padded_flat_offsets_match());

// --- 4c. Padded source, ragged result --------------------------------------
// -> (2,16) with mode 1 fastest. Mode 0 is one leaf (stride 18, two whole
// rows); mode 1 spans a row and a half, so it needs two.
using ReshPaddedRagged =
    decltype(reshape(Padded2D{}, StaticTile<2, 16>{}, order<1, 0>{}));
static_assert(
    std::is_same_v<ReshPaddedRagged,
                   StaticLayout<DT<ext<2>, ext<8, 2>>, DT<ext<18>, ext<1, 9>>,
                                DT<ext<1>, ext<2, 0>>>>);
static_assert(ReshPaddedRagged::mode_arity<0>() == 1);
static_assert(ReshPaddedRagged::mode_arity<1>() == 2);

constexpr bool padded_ragged_offsets_match() noexcept {
  for (int m0 = 0; m0 < 2; ++m0)
    for (int m1 = 0; m1 < 16; ++m1)
      if (static_cast<int>(ReshPaddedRagged{}.flat(m0, m1)) !=
          padded2d_offset_at(m0 * 16 + m1))
        return false;
  return true;
}
static_assert(padded_ragged_offsets_match());

// --- 4d. Arbitrary memory order ---------------------------------------------
// Stride3D is {4,8,3} with dim 2 fastest, so its memory stream is
// (3, s=1), (4, s=3), (8, s=12). Flattening to rank 1 merges all three, and
// because the source is packed the three runs are contiguous end to end: the
// merged mode is one leaf of extent 96 and stride 1, i.e. the reshape sees
// through the permuted dimension order to the dense stream underneath.
using ReshStride3D =
    decltype(reshape(Stride3D{}, StaticTile<96>{}, order<0>{}));
static_assert(std::is_same_v<ReshStride3D, StaticTileLayoutRight<96>>);
static_assert(ReshStride3D::mode_arity<0>() == 1);
static_assert(ReshStride3D::stride(0) == 1);

constexpr bool stride3d_flat_offsets_match() noexcept {
  // Element k of the source in memory order sits at offset k (it is packed),
  // and element k of the result must sit at the same place.
  for (int k = 0; k < 96; ++k)
    if (static_cast<int>(ReshStride3D{}.flat(k)) != k) return false;
  return true;
}
static_assert(stride3d_flat_offsets_match());

// Padding is what makes an arbitrary-order source ragged. Padded3D has the same
// dimension order but a row pitch of 16 on the middle axis, so its stream is
// (3, s=1), (4, s=3), (8, s=16): the first two runs coalesce into 12 dense
// slots, the third cannot follow them, and the flattened mode needs two leaves.
using ReshPadded3D =
    decltype(reshape(Padded3D{}, StaticTile<96>{}, order<0>{}));
static_assert(
    std::is_same_v<ReshPadded3D, StaticLayout<DT<ext<12, 8>>, DT<ext<1, 16>>,
                                              DT<ext<0, 1>>>>);
static_assert(ReshPadded3D::mode_arity<0>() == 2);

constexpr bool padded3d_flat_offsets_match() noexcept {
  // Element k of Padded3D in ITS memory order: the stream is dim 2 (extent 3,
  // stride 1), then dim 0 (extent 4, stride 3), then dim 1 (extent 8, str 16).
  for (int k = 0; k < 96; ++k) {
    const int src = (k % 3) * 1 + ((k / 3) % 4) * 3 + (k / 12) * 16;
    if (static_cast<int>(ReshPadded3D{}.flat(k)) != src) return false;
  }
  return true;
}
static_assert(padded3d_flat_offsets_match());

// --- 4e. Identity ------------------------------------------------------------
// Reshaping to the source's own shape and order returns the source type,
// padding and all — the reshape is a no-op rather than a re-derivation.
static_assert(std::is_same_v<decltype(reshape(Padded2D{}, StaticTile<4, 8>{},
                                              order<1, 0>{})),
                             Padded2D>);
static_assert(std::is_same_v<decltype(reshape(Padded3D{}, StaticTile<4, 8, 3>{},
                                              order<2, 0, 1>{})),
                             Padded3D>);

// --- 4f. Rejections ---------------------------------------------------------
// The plan is inspectable without calling reshape(), so the negative cases are
// testable without provoking the compile error they are supposed to provoke.
template <typename SrcExt, typename SrcStr, typename SrcOrd, typename NewExt,
          typename NewOrd>
inline constexpr Impl::ReshapeStatus plan_status =
    Impl::ReshapePlanFor<SrcExt, SrcStr, SrcOrd, NewExt, NewOrd>{}().status;

// A radix-6 stream cannot yield a mode of 8: gcd(8, 6) == 2, then gcd(4, 3)
// == 1.
static_assert(plan_status<StaticTile<6, 4>, strides<4, 1>, order<1, 0>,
                          StaticTile<8, 3>, order<1, 0>> ==
              Impl::ReshapeStatus::NotFactorable);
// Wrong element count.
static_assert(plan_status<StaticTile<4, 8>, strides<9, 1>, order<1, 0>,
                          StaticTile<2, 17>, order<1, 0>> ==
              Impl::ReshapeStatus::CountMismatch);
// Order is not a permutation...
static_assert(plan_status<StaticTile<4, 8>, strides<9, 1>, order<1, 0>,
                          StaticTile<2, 16>, order<1, 1>> ==
              Impl::ReshapeStatus::BadOrder);
// ...including the wrong number of entries, which would otherwise index out of
// bounds inside the planner before any diagnostic could fire.
static_assert(plan_status<StaticTile<4, 8>, strides<9, 1>, order<1, 0>,
                          StaticTile<2, 16>, order<0>> ==
              Impl::ReshapeStatus::BadOrder);

// And the accepted cases above really are accepted, not silently degraded.
static_assert(plan_status<StaticTile<4, 8>, strides<9, 1>, order<1, 0>,
                          StaticTile<32>, order<0>> == Impl::ReshapeStatus::Ok);

// ---------------------------------------------------------------------------
// 5. tile_layout on top of the generic reshape.
//
// Tiling IS a reshape that splits each dimension E into (E/T, T), so the static
// tile_layout overloads name Impl::tiled_layout_t instead of deriving strides a
// second time. Two things need proving: the named overloads still produce
// exactly what they produced before, and the sources that only reshape can
// describe — arbitrary order, and padding — now tile as well.
// ---------------------------------------------------------------------------

// --- 5a. The named overloads are unchanged ---------------------------------
// The load-bearing property of the whole refactor: same source, same tile, same
// type as the hand-rolled interleaving produced. tests/test_tiling.cpp asserts
// the literal types; this asserts the equivalence the implementation rests on.
static_assert(
    std::is_same_v<decltype(tile_layout(StaticTileLayoutRight<8, 12>{},
                                        StaticTile<2, 4>{})),
                   StaticTileLayoutRight<4, 2, 3, 4>>);
static_assert(std::is_same_v<decltype(tile_layout(StaticTileLayoutLeft<8, 12>{},
                                                  StaticTile<2, 4>{})),
                             StaticTileLayoutLeft<2, 4, 4, 3>>);

// ...and each is literally the reshape to the interleaved shape and order.
static_assert(
    std::is_same_v<decltype(tile_layout(StaticTileLayoutRight<8, 12>{},
                                        StaticTile<2, 4>{})),
                   decltype(reshape(StaticTileLayoutRight<8, 12>{},
                                    StaticTile<4, 2, 3, 4>{}, LayoutRight{}))>);
static_assert(
    std::is_same_v<decltype(tile_layout(StaticTileLayoutLeft<8, 12>{},
                                        StaticTile<2, 4>{})),
                   decltype(reshape(StaticTileLayoutLeft<8, 12>{},
                                    StaticTile<2, 4, 4, 3>{}, LayoutLeft{}))>);

// --- 5b. Arbitrary memory order --------------------------------------------
// 8x12 with dim 0 fastest (strides {1,8}). Tiled by (2,4) outer-first the modes
// are <8/2, 2, 12/4, 4> = <4,2,3,4>; dim 0's pair stays fastest and within each
// pair the inner mode leads, so the memory order is (1,0,3,2) and the strides
// are outer0=2, inner0=1, outer1=32, inner1=8.
using Src2DStride = StaticTileLayoutStride<StaticTile<8, 12>, 0, 1>;
static_assert(Src2DStride::stride(0) == 1 && Src2DStride::stride(1) == 8);

using StrideTiled = decltype(tile_layout(Src2DStride{}, StaticTile<2, 4>{}));
static_assert(StrideTiled::rank == 4);
static_assert(StrideTiled::extent(0) == 4 && StrideTiled::extent(1) == 2 &&
              StrideTiled::extent(2) == 3 && StrideTiled::extent(3) == 4);
static_assert(StrideTiled::stride(0) == 2 && StrideTiled::stride(1) == 1 &&
              StrideTiled::stride(2) == 32 && StrideTiled::stride(3) == 8);
// Flat, not nested: every mode is a single contiguous run, so stride(d) — and
// therefore the TensorLike concept — survives.
static_assert(StrideTiled::mode_arity<0>() == 1 &&
              StrideTiled::mode_arity<1>() == 1 &&
              StrideTiled::mode_arity<2>() == 1 &&
              StrideTiled::mode_arity<3>() == 1);
// Packed in its own order, but that order is neither Right's nor Left's, so
// ReshapeFlat's named recovery does not fire and the result is the bare
// StaticLayout the named Stride layout derives from.
static_assert(
    std::is_same_v<StrideTiled,
                   StaticLayout<StaticTile<4, 2, 3, 4>, strides<2, 1, 32, 8>,
                                order<1, 0, 3, 2>>>);
static_assert(std::is_base_of_v<
              StrideTiled,
              StaticTileLayoutStride<StaticTile<4, 2, 3, 4>, 1, 0, 3, 2>>);

// Position-preserving: mode (o0,i0,o1,i1) must land where the source puts
// element (2*o0+i0, 4*o1+i1).
constexpr bool stride_tiling_preserves_positions() noexcept {
  for (int o0 = 0; o0 < 4; ++o0)
    for (int i0 = 0; i0 < 2; ++i0)
      for (int o1 = 0; o1 < 3; ++o1)
        for (int i1 = 0; i1 < 4; ++i1) {
          const int src = Src2DStride{}.flat(2 * o0 + i0, 4 * o1 + i1);
          if (static_cast<int>(StrideTiled{}.flat(o0, i0, o1, i1)) != src)
            return false;
        }
  return true;
}
static_assert(stride_tiling_preserves_positions());

// --- 5c. Padded source — the case no named layout can express --------------
// Padded2D is 4x8 at a row pitch of 9. Tiled by (2,4): modes <2,2,2,4>, order
// (3,2,1,0), strides outer0=18 (two padded rows), inner0=9, outer1=4, inner1=1.
using PaddedTiled = decltype(tile_layout(Padded2D{}, StaticTile<2, 4>{}));
static_assert(
    std::is_same_v<PaddedTiled,
                   StaticLayout<StaticTile<2, 2, 2, 4>, strides<18, 9, 4, 1>,
                                order<3, 2, 1, 0>>>);
static_assert(PaddedTiled::mode_arity<0>() == 1 &&
              PaddedTiled::mode_arity<3>() == 1);
static_assert(PaddedTiled::stride(0) == 18 && PaddedTiled::stride(3) == 1);

constexpr bool padded_tiling_preserves_positions() noexcept {
  for (int o0 = 0; o0 < 2; ++o0)
    for (int i0 = 0; i0 < 2; ++i0)
      for (int o1 = 0; o1 < 2; ++o1)
        for (int i1 = 0; i1 < 4; ++i1) {
          const int src = Padded2D{}.flat(2 * o0 + i0, 4 * o1 + i1);
          if (static_cast<int>(PaddedTiled{}.flat(o0, i0, o1, i1)) != src)
            return false;
        }
  return true;
}
static_assert(padded_tiling_preserves_positions());

// --- 5d. Rejections ---------------------------------------------------------
// Same trick as 4f: the plan is inspectable without provoking the compile error
// it is supposed to provoke.
template <bool InnerFirst, typename SrcExt, typename SrcOrd, typename Tile>
inline constexpr Impl::ReshapeStatus tiling_status =
    Impl::tiled_packed_plan_t<InnerFirst, SrcExt, SrcOrd, Tile>{}().status;

// A tile that does not divide the source extent. This used to truncate E/T and
// silently drop the remainder; it is now a diagnosed compile error.
static_assert(
    tiling_status<false, StaticTile<8, 12>, order<1, 0>, StaticTile<5, 4>> ==
    Impl::ReshapeStatus::CountMismatch);
// The dividing tile next to it really is accepted.
static_assert(
    tiling_status<false, StaticTile<8, 12>, order<1, 0>, StaticTile<2, 4>> ==
    Impl::ReshapeStatus::Ok);

// --- 5e. Rank ceiling -------------------------------------------------------
// A rank-N tiling needs 2N modes, so kReshapeMaxModes is what bounds the source
// rank. At the old value of 8 a rank-5 source would have been rejected, where
// the hand-rolled interleaving it replaced had no limit at all.
static_assert(tiling_status<false, StaticTile<4, 4, 4, 4, 4>,
                            order<4, 3, 2, 1, 0>, StaticTile<2, 2, 2, 2, 2>> ==
              Impl::ReshapeStatus::Ok);
static_assert(
    std::is_same_v<decltype(tile_layout(StaticTileLayoutRight<4, 4, 4, 4, 4>{},
                                        StaticTile<2, 2, 2, 2, 2>{})),
                   StaticTileLayoutRight<2, 2, 2, 2, 2, 2, 2, 2, 2, 2>>);
// Past the ceiling it is reported, not silently truncated: rank 9 needs 18.
static_assert(tiling_status<false, StaticTile<2, 2, 2, 2, 2, 2, 2, 2, 2>,
                            order<8, 7, 6, 5, 4, 3, 2, 1, 0>,
                            StaticTile<1, 1, 1, 1, 1, 1, 1, 1, 1>> ==
              Impl::ReshapeStatus::CapacityExceeded);

// ---------------------------------------------------------------------------
// 6. reshape with a NESTED source.
//
// reshape produces the nested layout, so it should consume one. It does so by
// forwarding to the flat layout over the source's concatenated leaves
// (Impl::flat_view_t), which rests on one claim: a nested layout's mode
// grouping is purely logical, so its element->offset map is identical to that
// flat view's. 6a proves exactly that; everything after it follows.
// ---------------------------------------------------------------------------

// --- 6a. The claim the whole overload rests on ------------------------------
// ReshapeReproducesSource walks every element comparing a layout's real
// decode-and-encode path against the mixed-radix stream of an (ext, str, order)
// triple. Passing the NESTED layout as the "Result" turns the reshape oracle
// into a proof that the flattening is offset-exact — for equal arities and for
// ragged ones.
static_assert(Impl::ReshapeReproducesSource<Interleave, StaticTile<5, 5, 5, 5>,
                                            strides<1, 25, 5, 125>,
                                            order<0, 2, 1, 3>>::value,
              "flat_view_t<Interleave> is not offset-exact");
static_assert(Impl::ReshapeReproducesSource<Ragged, StaticTile<2, 3, 4, 5, 6>,
                                            strides<1, 8, 2, 24, 120>,
                                            order<0, 2, 1, 3, 4>>::value,
              "flat_view_t<Ragged> is not offset-exact");

// ...and flat_view_t really does produce those triples from the type alone.
static_assert(std::is_same_v<
              Impl::flat_view_t<Ragged>,
              StaticLayout<StaticTile<2, 3, 4, 5, 6>, strides<1, 8, 2, 24, 120>,
                           order<0, 2, 1, 3, 4>>>);

// --- 6b. Nested -> flat -----------------------------------------------------
// Ragged is a dense mixed-radix bijection onto [0, 720), so flattening it to
// rank 1 coalesces every leaf into one contiguous run and recovers the NAMED
// layout — the same collapse a packed flat source would get.
using NestedFlattened =
    decltype(reshape(Ragged{}, StaticTile<720>{}, order<0>{}));
static_assert(std::is_same_v<NestedFlattened, StaticTileLayoutRight<720>>);
static_assert(NestedFlattened::mode_arity<0>() == 1);

constexpr bool nested_flatten_is_dense() noexcept {
  for (int k = 0; k < 720; ++k)
    if (static_cast<int>(NestedFlattened{}.flat(k)) != k) return false;
  return true;
}
static_assert(nested_flatten_is_dense());

// --- 6c. Nested -> nested, and agreement with the equivalent flat source ----
// Padded2D expressed as a nested layout: two modes, each of arity 1. Reshaping
// it must give bit-for-bit what reshaping Padded2D itself gives (4b above) —
// the source spelling cannot change the answer.
using NestedPadded2D =
    StaticLayout<DT<ext<4>, ext<8>>, DT<ext<9>, ext<1>>, DT<ext<1>, ext<0>>>;
static_assert(Impl::ReshapeReproducesSource<NestedPadded2D, StaticTile<4, 8>,
                                            strides<9, 1>, order<1, 0>>::value);

static_assert(std::is_same_v<decltype(reshape(NestedPadded2D{},
                                              StaticTile<32>{}, order<0>{})),
                             ReshPaddedFlat>);
static_assert(decltype(reshape(NestedPadded2D{}, StaticTile<32>{},
                               order<0>{}))::mode_arity<0>() == 2);

// --- 6d. Round trips are ASYMMETRIC -----------------------------------------
// Reshaping to the source's LEAF shape and order is the identity: it returns
// the flat view itself.
static_assert(
    std::is_same_v<decltype(reshape(Interleave{}, StaticTile<5, 5, 5, 5>{},
                                    order<0, 2, 1, 3>{})),
                   Impl::flat_view_t<Interleave>>);

// Reshaping to the source's MODE shape is NOT. reshape is defined over memory
// order and the planner carves each mode to completion in memory order, never
// seeing the source's grouping — so Interleave's four leaves, whose memory
// order is (mode0 leaf, mode1 leaf, mode0 leaf, mode1 leaf), get regrouped as
// two contiguous runs. The result is a plain dense row-major 25x25: a valid
// reshape of the same 625 elements in the same memory order, and a completely
// different index map. This is the property callers must not assume; it is
// pinned here so a future change to the carve loop cannot alter it silently.
using InterleaveModeShape =
    decltype(reshape(Interleave{}, StaticTile<25, 25>{}, LayoutRight{}));
static_assert(
    std::is_same_v<InterleaveModeShape, StaticTileLayoutRight<25, 25>>);
static_assert(!std::is_same_v<InterleaveModeShape, Interleave>);
static_assert(InterleaveModeShape::mode_arity<0>() == 1);  // flat, not nested
static_assert(InterleaveModeShape{}.flat(1, 0) == 25);     // Interleave: 1
static_assert(Interleave{}.flat(1, 0) == 1);
static_assert(InterleaveModeShape{}.flat(5, 0) == 125);  // Interleave: 25
static_assert(Interleave{}.flat(5, 0) == 25);

// --- 6e. Diagnostics are inherited from the flat overload -------------------
// The nested overload forwards rather than re-entering the planner, so the
// rejection statuses are the same ones. Checked through the plan, as in 4f.
static_assert(plan_status<StaticTile<2, 3, 4, 5, 6>, strides<1, 8, 2, 24, 120>,
                          order<0, 2, 1, 3, 4>, StaticTile<719>, order<0>> ==
              Impl::ReshapeStatus::CountMismatch);
static_assert(plan_status<StaticTile<2, 3, 4, 5, 6>, strides<1, 8, 2, 24, 120>,
                          order<0, 2, 1, 3, 4>, StaticTile<720>, order<0>> ==
              Impl::ReshapeStatus::Ok);

// A nested source arrives with more LEAVES than a flat source of the same rank
// — Ragged is rank 2 but five leaves — so it eats the kReshapeMaxLeaves budget
// faster. Well inside it here; the ceiling itself is covered by 5e.
static_assert(Impl::flat_view_t<Ragged>::rank == 5);

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
// TEST: write-through on a reshaped padded layout. The rank-1 view merges
// across the padding gap, so an offset map that ignored the pitch would still
// produce a dense, plausible-looking result — the pad slots are what catch it.
// Values are distinct per element, so a transposed map cannot pass either.
// ---------------------------------------------------------------------------
TEST(GenericReshape, ViewWriteThroughPaddedMerge) {
  using Buf1D = Kokkos::View<float*, Kokkos::LayoutRight, Kokkos::HostSpace>;
  Buf1D buf("buf", 36);  // 4 rows x pitch 9
  Kokkos::deep_copy(buf, -1.f);

  View<Buf1D, Padded2D> v{buf, Padded2D{}};
  auto                  flat = reshape(v, StaticTile<32>{}, order<0>{});
  static_assert(std::is_same_v<decltype(flat)::layout_t, ReshPaddedFlat>);
  ASSERT_EQ(flat.extent(0), 32);

  for (int k = 0; k < 32; ++k) flat(k) = static_cast<float>(100 + k);

  // Element k of the reshaped view is element (k/8, k%8) of the source.
  for (int k = 0; k < 32; ++k)
    EXPECT_FLOAT_EQ(buf(9 * (k / 8) + (k % 8)), static_cast<float>(100 + k))
        << "k=" << k;
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(buf(9 * i + 8), -1.f) << "pad of row " << i << " written";

  // Reading back through the original 2-D view agrees element for element.
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j)
      EXPECT_FLOAT_EQ(v(i, j), static_cast<float>(100 + i * 8 + j))
          << "i=" << i << " j=" << j;
}

// ---------------------------------------------------------------------------
// TEST: the ragged result decodes as a bijection onto the source's elements —
// the decode path every tier kernel's flat loop uses.
// ---------------------------------------------------------------------------
TEST(GenericReshape, DecodeRoundTripRagged) {
  ReshPaddedRagged l{};
  ASSERT_EQ(l.size(), 32);

  bool hit[35] = {};  // 32 elements spread over Padded2D's 35 slots
  for (int f = 0; f < l.size(); ++f) {
    const auto c = l[f];
    ASSERT_GE(c[0], 0);
    ASSERT_LT(c[0], 2) << "f=" << f;
    ASSERT_GE(c[1], 0);
    ASSERT_LT(c[1], 16) << "f=" << f;
    const auto off = l.flat(c[0], c[1]);
    ASSERT_LT(off, 35u) << "f=" << f;
    EXPECT_FALSE(hit[off]) << "flat offset " << off << " reached twice";
    hit[off] = true;
  }
  // Injective onto the source's elements, and the three pad slots are
  // untouched.
  EXPECT_FALSE(hit[8]);
  EXPECT_FALSE(hit[17]);
  EXPECT_FALSE(hit[26]);
  for (int f = 0; f < 32; ++f)
    EXPECT_TRUE(hit[padded2d_offset_at(f)]) << "element " << f << " missed";
}

// ---------------------------------------------------------------------------
// TEST: reshape a View whose layout is NESTED. Same source memory and same
// target as GenericReshape.ViewWriteThroughPaddedMerge, but reached through the
// nested spelling of the source — so it proves two things at once: the View
// wrapper picks the new overload up with no change of its own (its
// requires-clause is just "reshape(l,t,o) is valid"), and the answer does not
// depend on how the source was spelled. The pad slots are still the oracle.
// ---------------------------------------------------------------------------
TEST(NestedReshape, ViewWriteThroughFromNestedSource) {
  using Buf1D = Kokkos::View<float*, Kokkos::LayoutRight, Kokkos::HostSpace>;
  Buf1D buf("buf", 36);  // 4 rows x pitch 9
  Kokkos::deep_copy(buf, -1.f);

  View<Buf1D, NestedPadded2D> v{buf, NestedPadded2D{}};
  auto                        flat = reshape(v, StaticTile<32>{}, order<0>{});
  static_assert(std::is_same_v<decltype(flat)::layout_t, ReshPaddedFlat>);
  ASSERT_EQ(flat.extent(0), 32);

  for (int k = 0; k < 32; ++k) flat(k) = static_cast<float>(200 + k);

  for (int k = 0; k < 32; ++k)
    EXPECT_FLOAT_EQ(buf(9 * (k / 8) + (k % 8)), static_cast<float>(200 + k))
        << "k=" << k;
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(buf(9 * i + 8), -1.f) << "pad of row " << i << " written";

  // Reading back through the original nested view agrees element for element.
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j)
      EXPECT_FLOAT_EQ(v(i, j), static_cast<float>(200 + i * 8 + j))
          << "i=" << i << " j=" << j;
}

// ---------------------------------------------------------------------------
// TEST: a nested source flattened to rank 1 decodes as a dense bijection.
// Ragged has unequal mode arities (2 and 3) and no repeated extent, so a
// mode- or leaf-order transposition in the flattening cannot pass.
// ---------------------------------------------------------------------------
TEST(NestedReshape, RaggedFlattensToDenseBijection) {
  NestedFlattened l{};
  ASSERT_EQ(l.size(), 720);

  bool hit[720] = {};
  for (int f = 0; f < l.size(); ++f) {
    const auto off = l.flat(l[f][0]);
    ASSERT_LT(off, 720u) << "f=" << f;
    EXPECT_FALSE(hit[off]) << "flat offset " << off << " reached twice";
    hit[off] = true;
    // The nested source puts its f-th memory-order element at the same place.
    EXPECT_EQ(static_cast<int>(off), f) << "f=" << f;
  }
  for (int s = 0; s < 720; ++s) EXPECT_TRUE(hit[s]) << "slot " << s;
}

// TEST: tile_view over a PADDED source — the tiling no named layout could
// express. The pad slots are the oracle: an offset map that ignored the row
// pitch would still look dense and plausible, and would write over them.
// ---------------------------------------------------------------------------
TEST(TiledStaticLayout, ViewWriteThroughPaddedTiling) {
  using Buf1D = Kokkos::View<float*, Kokkos::LayoutRight, Kokkos::HostSpace>;
  Buf1D buf("buf", 36);  // 4 rows x pitch 9
  Kokkos::deep_copy(buf, -1.f);

  View<Buf1D, Padded2D> v{buf, Padded2D{}};
  auto                  tiled = tile_view(v, StaticTile<2, 4>{});
  static_assert(std::is_same_v<decltype(tiled)::layout_t, PaddedTiled>);
  ASSERT_EQ(tiled.extent(0), 2);
  ASSERT_EQ(tiled.extent(1), 2);
  ASSERT_EQ(tiled.extent(2), 2);
  ASSERT_EQ(tiled.extent(3), 4);

  // Distinct per element, so a transposed or collapsed map cannot pass.
  for (int o0 = 0; o0 < 2; ++o0)
    for (int i0 = 0; i0 < 2; ++i0)
      for (int o1 = 0; o1 < 2; ++o1)
        for (int i1 = 0; i1 < 4; ++i1)
          tiled(o0, i0, o1, i1) =
              static_cast<float>(100 + 8 * (2 * o0 + i0) + (4 * o1 + i1));

  // Tile coordinate (o0,i0,o1,i1) is source element (2*o0+i0, 4*o1+i1), which
  // the padded source puts at 9*row + col.
  for (int o0 = 0; o0 < 2; ++o0)
    for (int i0 = 0; i0 < 2; ++i0)
      for (int o1 = 0; o1 < 2; ++o1)
        for (int i1 = 0; i1 < 4; ++i1) {
          const int row = 2 * o0 + i0, col = 4 * o1 + i1;
          EXPECT_FLOAT_EQ(buf(9 * row + col),
                          static_cast<float>(100 + 8 * row + col))
              << "o0=" << o0 << " i0=" << i0 << " o1=" << o1 << " i1=" << i1;
        }
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(buf(9 * i + 8), -1.f) << "pad of row " << i << " written";

  // Reading back through the untiled view agrees element for element.
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j)
      EXPECT_FLOAT_EQ(v(i, j), static_cast<float>(100 + 8 * i + j))
          << "i=" << i << " j=" << j;
}

// ---------------------------------------------------------------------------
// TEST: an arbitrary-order source tiles to a bijection onto its own slots —
// the decode path every tier kernel's flat loop uses.
// ---------------------------------------------------------------------------
TEST(TiledStaticLayout, StrideTilingDecodeRoundTrip) {
  StrideTiled layout{};
  ASSERT_EQ(layout.size(), 96);

  bool hit[96] = {};
  for (int f = 0; f < layout.size(); ++f) {
    const auto c   = layout[f];
    const auto off = layout.flat(c[0], c[1], c[2], c[3]);
    ASSERT_LT(off, 96u) << "f=" << f;
    EXPECT_FALSE(hit[off]) << "flat offset " << off << " reached twice";
    hit[off] = true;
  }
  // Onto: the source is packed, so every one of its 96 slots is covered.
  for (int s = 0; s < 96; ++s) EXPECT_TRUE(hit[s]) << "slot " << s << " missed";
}

// ---------------------------------------------------------------------------
// TEST: the reshaped layout works inside a device kernel.
//
// The materializer is a two-level pack expansion whose inner length varies per
// mode — the construct most at risk from nvcc's device frontend. Everything
// else in this file is host-only, so without this the risk would be uncovered
// in tree. On a host-only build this still exercises the Serial backend.
// ---------------------------------------------------------------------------
// Kernel body as a named functor: an extended __host__ __device__ lambda
// (KOKKOS_LAMBDA) may not appear inside GoogleTest's private TestBody under
// nvcc — the same reason test_device_tuple.cpp spells its kernel out.
namespace {
struct ReshapeDeviceKernel {
  using BufD = Kokkos::View<float*, Kokkos::LayoutRight,
                            Kokkos::DefaultExecutionSpace::memory_space>;
  View<BufD, ReshPaddedFlat> flat;
  KOKKOS_FUNCTION void       operator()(int k) const {
    flat(k) = static_cast<float>(100 + k);
  }
};
}  // namespace

TEST(GenericReshape, NestedLayoutInDeviceKernel) {
  using Exec = Kokkos::DefaultExecutionSpace;
  using BufD = ReshapeDeviceKernel::BufD;
  BufD buf("buf", 36);
  Kokkos::deep_copy(buf, -1.f);

  Kokkos::parallel_for("reshape_device", Kokkos::RangePolicy<Exec>(0, 32),
                       ReshapeDeviceKernel{{buf, ReshPaddedFlat{}}});
  Kokkos::fence();

  auto host = Kokkos::create_mirror_view(buf);
  Kokkos::deep_copy(host, buf);

  for (int k = 0; k < 32; ++k)
    EXPECT_FLOAT_EQ(host(9 * (k / 8) + (k % 8)), static_cast<float>(100 + k))
        << "k=" << k;
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(host(9 * i + 8), -1.f) << "pad of row " << i << " written";
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
