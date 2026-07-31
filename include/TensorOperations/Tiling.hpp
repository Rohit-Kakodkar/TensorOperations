#pragma once
#include <TensorOperations/DeviceTuple.hpp>
#include <TensorOperations/Permute.hpp>
#include <TensorOperations/TileLayout.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <type_traits>
#include <utility>

namespace TensorOperations {

// Convenience: rank_c<N> == std::integral_constant<int, N>{}
template <int N>
inline constexpr std::integral_constant<int, N> rank_c{};

// ---------------------------------------------------------------------------
// Tiling specification types
//
// A tile carries one extent per participating mode of the operation it tiles
// (for a contraction that is Rank_C + NumContracted). The extent-binding axis
// is "when are the extents known":
//
//   StaticTile<int... Extents> — compile-time extents. Required by the register
//     tier (registers cannot be runtime-indexed) and also usable with scratch
//     (compile-time scratch size, unrolled loops).
//   DynamicTile<int Rank>      — runtime extents. Scratch / team tier only.
//
// Both expose a uniform interface (rank, extent(i), is_static) so evaluators
// read extents the same way regardless of kind. There is no NoTiling: the
// un-staged baseline is just the degenerate unit tile StaticTile<1,...,1>.
// ---------------------------------------------------------------------------

template <int... Extents>
struct StaticTile {
  static constexpr int rank = sizeof...(Extents);
  static_assert(rank > 0 && ((Extents > 0) && ...),
                "StaticTile requires at least one positive extent");
  static constexpr bool                is_static = true;
  KOKKOS_FUNCTION static constexpr int extent(int i) noexcept {
    constexpr int e[] = {Extents...};
    return e[i];
  }
};

template <int Rank>
struct DynamicTile {
  static constexpr int          rank      = Rank;
  static constexpr bool         is_static = false;
  Kokkos::Array<int, Rank>      extents;
  KOKKOS_FUNCTION constexpr int extent(int i) const { return extents[i]; }
};

// ---------------------------------------------------------------------------
// make_tile_layout — factories: Tile → TileLayout
// ---------------------------------------------------------------------------

template <int... E>
KOKKOS_FUNCTION constexpr StaticTileLayoutRight<E...> make_tile_layout(
    StaticTile<E...>, LayoutRight) noexcept {
  return {};
}

template <int N>
KOKKOS_FUNCTION DynamicTileLayoutRight<N> make_tile_layout(
    DynamicTile<N> t, LayoutRight) noexcept {
  return DynamicTileLayoutRight<N>{t.extents};
}

template <int... E>
KOKKOS_FUNCTION constexpr StaticTileLayoutLeft<E...> make_tile_layout(
    StaticTile<E...>, LayoutLeft) noexcept {
  return {};
}

template <int N>
KOKKOS_FUNCTION DynamicTileLayoutLeft<N> make_tile_layout(DynamicTile<N> t,
                                                          LayoutLeft) noexcept {
  return DynamicTileLayoutLeft<N>{t.extents};
}

// StaticTile<E...> + integer_sequence<int, Ord...> → StaticTileLayoutStride
template <int... E, int... Ord>
KOKKOS_FUNCTION constexpr StaticTileLayoutStride<StaticTile<E...>, Ord...>
make_tile_layout(StaticTile<E...>,
                 std::integer_sequence<int, Ord...>) noexcept {
  return {};
}

// ---------------------------------------------------------------------------
// tile_layout — factory: TileLayout × Tile → 2N-dimensional TileLayout
//
// Produces a *position-preserving* 2N-d TileLayout: the per-dimension (outer
// tile count, inner tile extent) pair is **interleaved** — Right (row-major)
// emits <E0/T0, T0, E1/T1, T1, ...> (outer,inner per dim), Left (col-major)
// emits <T0, E0/T0, ...> (inner,outer per dim). With that ordering the plain
// row/col-major strides of the resulting layout reproduce the exact element
// position in the un-tiled source: element (o_d*T_d + i_d) along dim d lands at
// Σ_d (o_d*T_d + i_d)·src_stride(d). Indexing order is therefore interleaved:
// view(o0, i0, o1, i1, ...).
//
// Static src + static tile  → fully constexpr StaticTileLayout (compile-time
//                             extents AND strides, zero runtime cost).
// Dynamic src + static tile → DynamicTileLayout (outer counts are runtime).
// Any src    + dynamic tile → DynamicTileLayout with runtime extents.
//
// The STATIC overloads are the generic reshape (see below) applied to the
// interleaved shape: tiling *is* a reshape that splits each dimension E into
// (E/T, T), so the planner derives the strides and picks the result type rather
// than this file deriving them a second time. Two consequences:
//
//   • Coverage is whatever reshape covers, so a source in an arbitrary memory
//     order (StaticTileLayoutStride) or with padding (a non-packed
//     StaticLayout) tiles too — neither was expressible before. A padded
//     source's result is a flat but non-packed StaticLayout, i.e. it keeps
//     stride(d) and TensorLike but is not one of the four NAMED layout types.
//   • A tile that does not divide the source extent is now a compile error
//     rather than a truncating E/T that silently drops the remainder. This is
//     intentional; Evaluator/Team.hpp already treats that truncation as a
//     wrong-results hazard and guards it separately.
//
// The dynamic overloads keep their hand-rolled interleaving: the planner is
// compile-time-only and cannot serve a runtime extent.
//
// Index order for the new (Stride / padded) overloads is OUTER-FIRST,
// view(o0, i0, o1, i1, ...), matching Right. Left's inner-first spelling is a
// quirk of StaticTileLayoutLeft — it is what makes a Left result packed
// column-major — not a general rule, and there is no such constraint for an
// arbitrary order.
// ---------------------------------------------------------------------------

namespace Impl {

// pack_at<I, V...> — the I-th value of an int pack, as a constant expression.
template <std::size_t I, int... V>
constexpr int pack_at() noexcept {
  constexpr int a[] = {V...};
  return a[I];
}

// TiledExtentAt<P, InnerFirst, StaticTile<E...>, StaticTile<T...>>::value —
// the extent at flat position P of the interleaved (outer,inner)-per-dim tiling
// of source extents E... by tile extents T..., with d = P/2 the source dim.
// InnerFirst=false (Right): even P → outer (E/T), odd P → inner (T).
// InnerFirst=true  (Left):  even P → inner (T),   odd P → outer (E/T).
template <std::size_t P, bool InnerFirst, class Ext, class Tile>
struct TiledExtentAt;

template <std::size_t P, bool InnerFirst, int... E, int... T>
struct TiledExtentAt<P, InnerFirst, StaticTile<E...>, StaticTile<T...>> {
  static constexpr std::size_t d     = P / 2;
  static constexpr bool        inner = ((P % 2 == 0) == InnerFirst);
  static constexpr int         value =
      inner ? pack_at<d, T...>() : pack_at<d, E...>() / pack_at<d, T...>();
};

// tiled_tile_t<InnerFirst, Ext, Tile> — the interleaved extents as a
// StaticTile, in MODE order. This is the target *shape*; the strides and the
// layout type are left to the reshape planner, which is what the
// layout-selecting build_static_tiled used to decide here.
template <bool InnerFirst, class Ext, class Tile, class Ps>
struct TiledTile;
template <bool InnerFirst, class Ext, class Tile, std::size_t... P>
struct TiledTile<InnerFirst, Ext, Tile, std::index_sequence<P...>> {
  using type = StaticTile<TiledExtentAt<P, InnerFirst, Ext, Tile>::value...>;
};

template <bool InnerFirst, class Ext, class Tile>
using tiled_tile_t =
    typename TiledTile<InnerFirst, Ext, Tile,
                       std::make_index_sequence<2 * Ext::rank>>::type;

// TiledOrderAt<InnerFirst, SrcOrderSeq>::get(q) — which of the 2N new modes
// sits at memory position q, fastest first.
//
// Source dimension d splits into modes 2d and 2d+1, and the INNER of that pair
// is always the faster of the two; InnerFirst says which of the two the inner
// mode is. So walking the source's own memory order O and emitting each
// dimension's pair inner-first gives the tiled order for any source — packed or
// padded, in any memory order:
//
//   NO[q] = 2*O[q/2] + (InnerFirst ? q%2 : 1 - q%2)
//
// Right (O = {N-1,...,0}, outer-first) reproduces right_order_t<2N> and Left
// (O = {0,...,N-1}, inner-first) reproduces left_order_t<2N>, so this one rule
// covers the named overloads and the arbitrary-order ones alike.
//
// The result is a permutation of [0, 2N) by construction — 2d + {0,1} over a
// permutation of the dimensions hits every value exactly once — so a tiling can
// never trip the planner's BadOrder.
//
// Two classes for the same reason as PackedStrideAt in TileLayout.hpp: get() is
// consumed in a template argument, which requires an already complete class.
// Deliberately NOT KOKKOS_FUNCTION, like the planner it feeds — see the note
// above Impl::make_reshape_plan.
template <bool InnerFirst, class SrcOrderSeq>
struct TiledOrderAt;

template <bool InnerFirst, int... O>
struct TiledOrderAt<InnerFirst, std::integer_sequence<int, O...>> {
  static constexpr int rank = sizeof...(O);

  static constexpr int get(std::size_t q) noexcept {
    constexpr int ord[] = {O...};
    const int     half  = static_cast<int>(q % 2);
    return 2 * ord[q / 2] + (InnerFirst ? half : 1 - half);
  }
};

template <bool InnerFirst, class SrcOrderSeq, class Qs>
struct TiledOrder;
template <bool InnerFirst, class SrcOrderSeq, std::size_t... Q>
struct TiledOrder<InnerFirst, SrcOrderSeq, std::index_sequence<Q...>> {
  using type =
      std::integer_sequence<int,
                            TiledOrderAt<InnerFirst, SrcOrderSeq>::get(Q)...>;
};

template <bool InnerFirst, class SrcOrderSeq>
using tiled_order_t = typename TiledOrder<
    InnerFirst, SrcOrderSeq,
    std::make_index_sequence<2 * TiledOrderAt<InnerFirst, SrcOrderSeq>::rank>>::
    type;

// The layout a tiling produces, and the plan behind it, named straight from the
// source's own (extents, strides, order) and the tile.
//
// Naming the result through reshaped_layout_t rather than calling reshape() is
// deliberate: reshape() is not constexpr and tile_layout is, so a call would
// silently cost tile_layout its constexpr-ness. The plan alias exists so the
// overloads can report a rejection in TILING terms — a non-dividing tile reads
// far better than "element count must be preserved".
template <bool InnerFirst, class Ext, class Str, class Ord, class Tile>
using tiled_plan_t =
    ReshapePlanFor<Ext, Str, Ord, tiled_tile_t<InnerFirst, Ext, Tile>,
                   tiled_order_t<InnerFirst, Ord>>;

// TiledLayoutOf — the result type, with the plan's rejections reported in
// TILING terms. The diagnostics live here rather than in each tile_layout body
// so that naming the return type is all an overload has to do; a "count
// mismatch" is always a non-dividing tile in this context, and saying so beats
// reshape's necessarily generic wording. The status is read through
// reshape_status_v rather than by calling the host-only planner — see the note
// above it in TileLayout.hpp.
template <bool InnerFirst, class Ext, class Str, class Ord, class Tile>
struct TiledLayoutOf {
  using plan_t = tiled_plan_t<InnerFirst, Ext, Str, Ord, Tile>;
  static constexpr ReshapeStatus status = reshape_status_v<plan_t>;

  static_assert(status != ReshapeStatus::CountMismatch,
                "tile_layout: every tile extent must divide the corresponding "
                "source extent exactly");
  static_assert(status != ReshapeStatus::CapacityExceeded,
                "tile_layout: source rank exceeds Impl::kReshapeMaxModes / 2; "
                "raise Impl::kReshapeMaxModes");
  static_assert(
      status != ReshapeStatus::NotFactorable &&
          status != ReshapeStatus::BadOrder,
      "tile_layout: the tiled shape is not carvable from the source's "
      "memory-order stream — unreachable for a dividing tile, so this "
      "is a planner bug");

  using type = reshape_result_t<plan_t>;
};

template <bool InnerFirst, class Ext, class Str, class Ord, class Tile>
using tiled_layout_t =
    typename TiledLayoutOf<InnerFirst, Ext, Str, Ord, Tile>::type;

// Same, for a PACKED source — which is all three named static layouts, since
// each derives from static_layout_for_t (a packed StaticLayout) for its order.
template <bool InnerFirst, class Ext, class Ord, class Tile>
using tiled_packed_layout_t =
    tiled_layout_t<InnerFirst, Ext, packed_strides_t<Ext, Ord>, Ord, Tile>;

template <bool InnerFirst, class Ext, class Ord, class Tile>
using tiled_packed_plan_t =
    tiled_plan_t<InnerFirst, Ext, packed_strides_t<Ext, Ord>, Ord, Tile>;

// ---------------------------------------------------------------------------
// Tiling a NESTED source.
//
// The extents carry over unchanged — TiledExtentAt splits each MODE SIZE into
// (size/T, T) exactly as it splits a flat dimension's extent. The ORDER does
// not, and this is the whole difficulty:
//
// tiled_order_t reads the source's memory order over DIMENSIONS and works
// because each flat dimension is one contiguous run, so a dimension's two
// halves stay adjacent in memory. A nested mode is not one run — its leaves can
// be interleaved with another mode's — so no ordering OF THE MODES reproduces
// the tiled memory order. For the interleaved 25x25 (leaves (5,5),(5,5),
// strides (1,25),(5,125)) tiled by (5,5), the naive "modes in memory order,
// inner before outer" rule gives (1,0,3,2), which carves mode 0's outer half
// out of mode 1's leaf: silently wrong.
//
// What DOES determine the order uniquely is each tiled mode's own stride.
// Stepping a mode's inner half by one moves that source mode's index by 1 and
// its outer half by T, and Mode::offset turns either into a memory offset. Each
// tiled mode occupies one contiguous run, so sorting the 2N of them by that
// offset ascending IS their memory order. For the example above the deltas are
// outer0=25, inner0=1, outer1=125, inner1=5, giving (1,3,0,2) — each mode's
// inner half first, then the outers — which is the position-preserving answer.
//
// If the sort ever got this wrong the TENSOR_OPS_VERIFY_RESHAPE oracle would
// catch it: the result would not reproduce the source's element->offset map.
// ---------------------------------------------------------------------------
template <bool InnerFirst, class L, class Tile>
struct NestedTiled;

template <bool InnerFirst, typename... ExtSeqs, typename... StrSeqs,
          typename... OrdSeqs, int... T>
struct NestedTiled<
    InnerFirst,
    StaticLayout<DeviceTuple<ExtSeqs...>, DeviceTuple<StrSeqs...>,
                 DeviceTuple<OrdSeqs...>>,
    StaticTile<T...>> {
  static constexpr int N = static_cast<int>(sizeof...(ExtSeqs));
  static_assert(N == static_cast<int>(sizeof...(T)),
                "tile_layout: source rank must match tile rank");

  // The mode sizes, so TiledExtentAt can be reused verbatim on them.
  using mode_sizes_t = StaticTile<Mode<ExtSeqs, StrSeqs>::mode_size...>;
  using tile_t       = tiled_tile_t<InnerFirst, mode_sizes_t, StaticTile<T...>>;

  // Memory offset reached by stepping each tiled mode once. Mode 2m and 2m+1
  // are source mode m's two halves; InnerFirst says which of the pair is the
  // inner.
  static constexpr std::array<int, 2 * N> deltas() noexcept {
    std::array<int, 2 * N> d{};
    constexpr int          t[] = {T...};
    int                    m   = 0;
    ((d[2 * m + (InnerFirst ? 0 : 1)] =
          static_cast<int>(Mode<ExtSeqs, StrSeqs>::offset(1)),
      d[2 * m + (InnerFirst ? 1 : 0)] =
          static_cast<int>(Mode<ExtSeqs, StrSeqs>::offset(t[m])),
      ++m),
     ...);
    return d;
  }

  // Argsort ascending: ord[j] is the tiled mode at memory position j.
  static constexpr std::array<int, 2 * N> ord() noexcept {
    const auto             d = deltas();
    std::array<int, 2 * N> o{};
    for (int j = 0; j < 2 * N; ++j) o[j] = j;
    for (int a = 0; a < 2 * N; ++a)
      for (int b = a + 1; b < 2 * N; ++b)
        if (d[o[b]] < d[o[a]]) {
          const int tmp = o[a];
          o[a]          = o[b];
          o[b]          = tmp;
        }
    return o;
  }
};

template <bool InnerFirst, class L, class Tile, class Js>
struct NestedTiledOrderSeq;
template <bool InnerFirst, class L, class Tile, std::size_t... J>
struct NestedTiledOrderSeq<InnerFirst, L, Tile, std::index_sequence<J...>> {
  using type =
      std::integer_sequence<int, NestedTiled<InnerFirst, L, Tile>::ord()[J]...>;
};

template <bool InnerFirst, class L, class Tile>
using nested_tiled_order_t = typename NestedTiledOrderSeq<
    InnerFirst, L, Tile,
    std::make_index_sequence<2 * static_cast<std::size_t>(
                                     NestedTiled<InnerFirst, L, Tile>::N)>>::
    type;

// The source triple is the flat leaf view, exactly as for reshape.
template <bool InnerFirst, class L, class Tile>
struct NestedTiledLayout;
template <bool InnerFirst, typename... ExtSeqs, typename... StrSeqs,
          typename... OrdSeqs, class Tile>
struct NestedTiledLayout<
    InnerFirst,
    StaticLayout<DeviceTuple<ExtSeqs...>, DeviceTuple<StrSeqs...>,
                 DeviceTuple<OrdSeqs...>>,
    Tile> {
  using layout_t =
      StaticLayout<DeviceTuple<ExtSeqs...>, DeviceTuple<StrSeqs...>,
                   DeviceTuple<OrdSeqs...>>;
  using type = reshaped_layout_t<
      seq_to_tile_t<concat_seq_t<ExtSeqs...>>, concat_seq_t<StrSeqs...>,
      concat_seq_t<OrdSeqs...>,
      typename NestedTiled<InnerFirst, layout_t, Tile>::tile_t,
      nested_tiled_order_t<InnerFirst, layout_t, Tile>>;
};

template <bool InnerFirst, class L, class Tile>
using nested_tiled_layout_t =
    typename NestedTiledLayout<InnerFirst, L, Tile>::type;

// Build the interleaved (outer, inner)-per-dim extent array used by every
// dynamic tile_layout overload. InnerFirst=true (Left layout): position 2*d
// holds the inner (tile) extent, 2*d+1 holds the outer (src/tile) count.
// InnerFirst=false (Right layout): outer comes first.
template <bool InnerFirst, int N>
KOKKOS_FUNCTION Kokkos::Array<int, 2 * N> interleave_tile_extents(
    const Kokkos::Array<int, N>& src_ext,
    const Kokkos::Array<int, N>& tile_ext) noexcept {
  Kokkos::Array<int, 2 * N> exts{};
  for (int d = 0; d < N; ++d) {
    const int outer = src_ext[d] / tile_ext[d];
    const int inner = tile_ext[d];
    exts[2 * d]     = InnerFirst ? inner : outer;
    exts[2 * d + 1] = InnerFirst ? outer : inner;
  }
  return exts;
}

}  // namespace Impl

// ---------------------------------------------------------------------------
// reorder_static_tile — gather-permute a StaticTile's extents.
//
// new tile axis i has extent E[Perm[i]] (gather convention, matching
// reorder_layout / permuted_alias). Compile-time only; used to reorder an
// operand's (or the output's) tile from user axis order into the canonical
// [free.., contracted..] order the team GEMM expects.
//
//   reorder_static_tile(StaticTile<8,32,4>{}, integer_sequence<int,1,0,2>{})
//       -> StaticTile<32, 8, 4>
// ---------------------------------------------------------------------------
template <int... E, int... Perm>
KOKKOS_FUNCTION constexpr auto reorder_static_tile(
    StaticTile<E...>, std::integer_sequence<int, Perm...>) noexcept
    -> StaticTile<Impl::pack_at<static_cast<std::size_t>(Perm), E...>()...> {
  static_assert(sizeof...(Perm) == sizeof...(E),
                "reorder_static_tile: permutation rank must match tile rank");
  return {};
}

// Identity-safe wrapper: the identity permutation returns the tile unchanged
// (so canonical contractions keep their exact original tile type, and
// non-static tiles are never reorder-instantiated).
template <typename Tile, int... Perm>
KOKKOS_FUNCTION auto reorder_tile_value(
    const Tile& tile, std::integer_sequence<int, Perm...> perm) {
  if constexpr (Impl::is_identity_seq(perm))
    return tile;
  else
    return reorder_static_tile(tile, perm);
}

// ---------------------------------------------------------------------------
// OperandAxes<PermSeq> — relates one operand's native axis order to the
// canonical order the evaluators work in.
//
// PermSeq is the gather permutation: canonical axis i comes from native axis
// PermSeq[i], so canon.extent(i) == native.extent(PermSeq[i]). Both directions
// are needed across the evaluators and each was previously open-coded:
//   • the contraction evaluator stores each operand's NATIVE tile and derives
//     the canonical one (permA/permB/permC),
//   • the combine evaluator receives each operand's tile already in CANONICAL
//     (output) order and derives the native one (inverse of the label gather).
// Both then scatter a canonical tile index into the operand's native order to
// drive that operand's own natively-tiled evaluator — the identical operation
// in each case.
//
// Everything here is compile-time or a trivially-inlined value transform; the
// identity permutation short-circuits via reorder_tile_value.
// ---------------------------------------------------------------------------
template <typename PermSeq>
struct OperandAxes {
  using perm_seq         = PermSeq;
  using inverse_perm_seq = Impl::inverse_perm_seq_t<PermSeq>;

  static constexpr bool is_identity = Impl::is_identity_seq(PermSeq{});

  // native tile -> canonical tile, and back.
  template <typename Tile>
  using canon_tile_t =
      decltype(reorder_tile_value(std::declval<Tile>(), PermSeq{}));
  template <typename Tile>
  using native_tile_t =
      decltype(reorder_tile_value(std::declval<Tile>(), inverse_perm_seq{}));

  template <typename Tile>
  KOKKOS_FUNCTION static canon_tile_t<Tile> to_canon_tile(const Tile& t) {
    return reorder_tile_value(t, PermSeq{});
  }
  template <typename Tile>
  KOKKOS_FUNCTION static native_tile_t<Tile> to_native_tile(const Tile& t) {
    return reorder_tile_value(t, inverse_perm_seq{});
  }

  // Canonical tile index -> the operand's own native tile index.
  template <std::size_t R>
  KOKKOS_FUNCTION static auto to_native_idx(const Kokkos::Array<int, R>& idx) {
    return Impl::scatter_index(idx, PermSeq{});
  }
};

template <int... Extents, int... TileE>
KOKKOS_FUNCTION constexpr auto tile_layout(StaticTileLayoutRight<Extents...>,
                                           StaticTile<TileE...>) noexcept
    -> Impl::tiled_packed_layout_t<
        false, StaticTile<Extents...>,
        Impl::right_order_t<static_cast<int>(sizeof...(Extents))>,
        StaticTile<TileE...>> {
  static_assert(sizeof...(Extents) == sizeof...(TileE),
                "tile_layout: source rank must match tile rank");
  return {};
}

template <int... Extents, int... TileE>
KOKKOS_FUNCTION constexpr auto tile_layout(StaticTileLayoutLeft<Extents...>,
                                           StaticTile<TileE...>) noexcept
    -> Impl::tiled_packed_layout_t<
        true, StaticTile<Extents...>,
        Impl::left_order_t<static_cast<int>(sizeof...(Extents))>,
        StaticTile<TileE...>> {
  static_assert(sizeof...(Extents) == sizeof...(TileE),
                "tile_layout: source rank must match tile rank");
  return {};
}

// Arbitrary compile-time memory order. Packed like the two above, so it goes
// through the same packed alias; only the order differs. Outer-first.
//
// An exact match on StaticTileLayoutStride beats the derived-to-base conversion
// to the StaticLayout overload below, so the two never compete even though the
// named layout derives from exactly that StaticLayout.
template <int... Extents, int... Order, int... TileE>
KOKKOS_FUNCTION constexpr auto tile_layout(
    StaticTileLayoutStride<StaticTile<Extents...>, Order...>,
    StaticTile<TileE...>) noexcept
    -> Impl::tiled_packed_layout_t<false, StaticTile<Extents...>,
                                   std::integer_sequence<int, Order...>,
                                   StaticTile<TileE...>> {
  static_assert(sizeof...(Extents) == sizeof...(TileE),
                "tile_layout: source rank must match tile rank");
  return {};
}

// A NESTED source. The tile has one extent per MODE (the layout's presented
// rank), not per leaf — tiling is about the shape a layout presents, and a
// mode's leaf structure is an implementation detail of how it reaches memory.
// Outer-first, like the two above.
//
// Unlike every other overload here this one cannot borrow tiled_order_t; see
// Impl::NestedTiled for why a nested source needs its order derived from the
// tiled modes' strides instead.
template <typename... ExtSeqs, typename... StrSeqs, typename... OrdSeqs,
          int... TileE>
KOKKOS_FUNCTION constexpr auto tile_layout(
    StaticLayout<DeviceTuple<ExtSeqs...>, DeviceTuple<StrSeqs...>,
                 DeviceTuple<OrdSeqs...>>,
    StaticTile<TileE...>) noexcept
    -> Impl::nested_tiled_layout_t<
        false,
        StaticLayout<DeviceTuple<ExtSeqs...>, DeviceTuple<StrSeqs...>,
                     DeviceTuple<OrdSeqs...>>,
        StaticTile<TileE...>> {
  static_assert(sizeof...(ExtSeqs) == sizeof...(TileE),
                "tile_layout: source rank must match tile rank");
  return {};
}

// Any flat StaticLayout, PADDED included — the case no named layout can
// express. A padded source tiles to a flat but non-packed StaticLayout:
// stride(d) and TensorLike survive, the named-type recovery does not.
// Outer-first.
template <int... Extents, int... S, int... O, int... TileE>
KOKKOS_FUNCTION constexpr auto tile_layout(
    StaticLayout<StaticTile<Extents...>, std::integer_sequence<int, S...>,
                 std::integer_sequence<int, O...>>,
    StaticTile<TileE...>) noexcept
    -> Impl::tiled_layout_t<
        false, StaticTile<Extents...>, std::integer_sequence<int, S...>,
        std::integer_sequence<int, O...>, StaticTile<TileE...>> {
  static_assert(sizeof...(Extents) == sizeof...(TileE),
                "tile_layout: source rank must match tile rank");
  return {};
}

template <int N, int... TileE>
KOKKOS_FUNCTION auto tile_layout(DynamicTileLayoutRight<N> src,
                                 StaticTile<TileE...>      tile) noexcept
    -> DynamicTileLayoutRight<2 * N> {
  static_assert(N == static_cast<int>(sizeof...(TileE)),
                "tile_layout: source rank must match tile rank");
  Kokkos::Array<int, N> src_ext{}, tile_ext{};
  for (int d = 0; d < N; ++d) {
    src_ext[d]  = src.extent(d);
    tile_ext[d] = tile.extent(d);
  }
  return DynamicTileLayoutRight<2 * N>{
      Impl::interleave_tile_extents<false, N>(src_ext, tile_ext)};
}

template <int N, int... TileE>
KOKKOS_FUNCTION auto tile_layout(DynamicTileLayoutLeft<N> src,
                                 StaticTile<TileE...>     tile) noexcept
    -> DynamicTileLayoutLeft<2 * N> {
  static_assert(N == static_cast<int>(sizeof...(TileE)),
                "tile_layout: source rank must match tile rank");
  Kokkos::Array<int, N> src_ext{}, tile_ext{};
  for (int d = 0; d < N; ++d) {
    src_ext[d]  = src.extent(d);
    tile_ext[d] = tile.extent(d);
  }
  return DynamicTileLayoutLeft<2 * N>{
      Impl::interleave_tile_extents<true, N>(src_ext, tile_ext)};
}

template <int... Extents, int N>
KOKKOS_FUNCTION auto tile_layout(StaticTileLayoutRight<Extents...>,
                                 DynamicTile<N> tile) noexcept
    -> DynamicTileLayoutRight<2 * N> {
  static_assert(static_cast<int>(sizeof...(Extents)) == N,
                "tile_layout: source rank must match tile rank");
  constexpr int         src_e[] = {Extents...};
  Kokkos::Array<int, N> src_ext{}, tile_ext{};
  for (int d = 0; d < N; ++d) {
    src_ext[d]  = src_e[d];
    tile_ext[d] = tile.extent(d);
  }
  return DynamicTileLayoutRight<2 * N>{
      Impl::interleave_tile_extents<false, N>(src_ext, tile_ext)};
}

template <int... Extents, int N>
KOKKOS_FUNCTION auto tile_layout(StaticTileLayoutLeft<Extents...>,
                                 DynamicTile<N> tile) noexcept
    -> DynamicTileLayoutLeft<2 * N> {
  static_assert(static_cast<int>(sizeof...(Extents)) == N,
                "tile_layout: source rank must match tile rank");
  constexpr int         src_e[] = {Extents...};
  Kokkos::Array<int, N> src_ext{}, tile_ext{};
  for (int d = 0; d < N; ++d) {
    src_ext[d]  = src_e[d];
    tile_ext[d] = tile.extent(d);
  }
  return DynamicTileLayoutLeft<2 * N>{
      Impl::interleave_tile_extents<true, N>(src_ext, tile_ext)};
}

template <int N>
KOKKOS_FUNCTION auto tile_layout(DynamicTileLayoutRight<N> src,
                                 DynamicTile<N>            tile) noexcept
    -> DynamicTileLayoutRight<2 * N> {
  Kokkos::Array<int, N> src_ext{}, tile_ext{};
  for (int d = 0; d < N; ++d) {
    src_ext[d]  = src.extent(d);
    tile_ext[d] = tile.extent(d);
  }
  return DynamicTileLayoutRight<2 * N>{
      Impl::interleave_tile_extents<false, N>(src_ext, tile_ext)};
}

template <int N>
KOKKOS_FUNCTION auto tile_layout(DynamicTileLayoutLeft<N> src,
                                 DynamicTile<N>           tile) noexcept
    -> DynamicTileLayoutLeft<2 * N> {
  Kokkos::Array<int, N> src_ext{}, tile_ext{};
  for (int d = 0; d < N; ++d) {
    src_ext[d]  = src.extent(d);
    tile_ext[d] = tile.extent(d);
  }
  return DynamicTileLayoutLeft<2 * N>{
      Impl::interleave_tile_extents<true, N>(src_ext, tile_ext)};
}

// ---------------------------------------------------------------------------
// reshape — reinterpret a TileLayout under a new shape given by a Tile.
//
// The ordering (Right/Left) is preserved from the source layout; the Tile
// argument supplies the new extents. For static layouts the element-count
// check is a static_assert; for dynamic layouts it fires at runtime when
// KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK is defined.
//
//   reshape(StaticTileLayoutRight<32>{},    StaticTile<4, 8>{})
//       -> StaticTileLayoutRight<4, 8>
//   reshape(StaticTileLayoutLeft<32>{},     StaticTile<4, 8>{})
//       -> StaticTileLayoutLeft<4, 8>
//   reshape(DynamicTileLayoutRight<1>{{32}}, DynamicTile<2>{{4, 8}})
//       -> DynamicTileLayoutRight<2>
//   reshape(DynamicTileLayoutLeft<1>{{32}},  DynamicTile<2>{{4, 8}})
//       -> DynamicTileLayoutLeft<2>
//
// A fifth, three-argument overload below handles the generic static case
// (padded strides and/or arbitrary memory order), where the new memory order
// cannot be inferred and has to be supplied.
// ---------------------------------------------------------------------------

template <int... OldE, int... NewE>
KOKKOS_FORCEINLINE_FUNCTION auto reshape(StaticTileLayoutRight<OldE...>,
                                         StaticTile<NewE...>) noexcept
    -> StaticTileLayoutRight<NewE...> {
  static_assert((static_cast<std::size_t>(OldE) * ...) ==
                    (static_cast<std::size_t>(NewE) * ...),
                "reshape: element count must be preserved");
  return {};
}

template <int... OldE, int... NewE>
KOKKOS_FORCEINLINE_FUNCTION auto reshape(StaticTileLayoutLeft<OldE...>,
                                         StaticTile<NewE...>) noexcept
    -> StaticTileLayoutLeft<NewE...> {
  static_assert((static_cast<std::size_t>(OldE) * ...) ==
                    (static_cast<std::size_t>(NewE) * ...),
                "reshape: element count must be preserved");
  return {};
}

template <int OldRank, int NewRank>
KOKKOS_FUNCTION auto reshape(DynamicTileLayoutRight<OldRank> src,
                             DynamicTile<NewRank>            tile) noexcept
    -> DynamicTileLayoutRight<NewRank> {
#ifdef KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK
  int new_size = 1;
  for (int k = 0; k < NewRank; ++k) new_size *= tile.extent(k);
  if (src.size() != new_size)
    Kokkos::abort("reshape: element count must be preserved");
#else
  (void)src;
#endif
  return DynamicTileLayoutRight<NewRank>{tile.extents};
}

template <int OldRank, int NewRank>
KOKKOS_FUNCTION auto reshape(DynamicTileLayoutLeft<OldRank> src,
                             DynamicTile<NewRank>           tile) noexcept
    -> DynamicTileLayoutLeft<NewRank> {
#ifdef KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK
  int new_size = 1;
  for (int k = 0; k < NewRank; ++k) new_size *= tile.extent(k);
  if (src.size() != new_size)
    Kokkos::abort("reshape: element count must be preserved");
#else
  (void)src;
#endif
  return DynamicTileLayoutLeft<NewRank>{tile.extents};
}

// ---------------------------------------------------------------------------
// reshape(StaticLayout, Tile, Order) — the generic static reshape.
//
// Reinterprets any flat StaticLayout — packed or PADDED, in any memory order —
// under a new shape and a new memory order, with no data movement. Order[j] is
// the new mode at memory position j, fastest-varying first, the same convention
// StaticLayout's own Order and make_tile_layout(tile, order_seq) use.
//
// The third argument is not optional sugar. reshape(Right<32>, Tile<4,8>) lists
// the new extents slowest-first while reshape(Left<32>, Tile<4,8>) lists them
// fastest-first, so no rule inferred from the extents alone reproduces both,
// and a generic Order matches neither. Being a different arity, this overload
// also cannot compete with the four packed ones above — which matters, because
// StaticTileLayoutRight<4,8> and StaticTileLayoutStride<StaticTile<4,8>,1,0>
// are distinct types sharing one StaticLayout base and could otherwise reshape
// under two different conventions.
//
// The result is a FLAT StaticLayout whenever every new mode maps to a single
// contiguous run of the source, and the NESTED StaticLayout when one does not:
//
//   using Padded2D = StaticLayout<StaticTile<4,8>, strides<9,1>, order<1,0>>;
//
//   reshape(StaticTileLayoutRight<32>{}, StaticTile<4,8>{}, LayoutRight{})
//       -> StaticLayout<StaticTile<4,8>, strides<8,1>, order<1,0>>
//          (flat — the base class of StaticTileLayoutRight<4,8>)
//   reshape(Padded2D{}, StaticTile<32>{}, order<0>{})
//       -> StaticLayout<DeviceTuple<ext<8,4>>, DeviceTuple<ext<1,9>>,
//                       DeviceTuple<ext<0,1>>>
//          (nested: one mode of two leaves, merging across the padding gap)
//
// A nested result has no stride(d) — a multi-leaf mode has no single stride —
// so it does not satisfy the TensorLike concept and cannot be fed back in as a
// leaf tensor. That is a compile error on .stride(), never a wrong answer.
//
// Not every shape is reachable: a new mode that straddles source dimensions
// whose extents share no common factor is not an affine reinterpretation of the
// source, and is rejected by static_assert rather than approximated.
// ---------------------------------------------------------------------------

template <int... E, int... S, int... O, int... F, int... NO>
KOKKOS_FORCEINLINE_FUNCTION auto reshape(
    StaticLayout<StaticTile<E...>, std::integer_sequence<int, S...>,
                 std::integer_sequence<int, O...>>,
    StaticTile<F...>, std::integer_sequence<int, NO...>) noexcept
    -> Impl::reshaped_layout_t<
        StaticTile<E...>, std::integer_sequence<int, S...>,
        std::integer_sequence<int, O...>, StaticTile<F...>,
        std::integer_sequence<int, NO...>> {
  using Fn =
      Impl::ReshapePlanFor<StaticTile<E...>, std::integer_sequence<int, S...>,
                           std::integer_sequence<int, O...>, StaticTile<F...>,
                           std::integer_sequence<int, NO...>>;
  // The plan is read through Impl::reshape_status_v rather than invoked here:
  // this function is __host__ __device__ and the planner is host-only, so
  // calling it — even inside a static_assert — would trip nvcc warning #20013.
  constexpr Impl::ReshapeStatus status = Impl::reshape_status_v<Fn>;

  // One assert per status, so the diagnostic names the actual problem.
  static_assert(status != Impl::ReshapeStatus::CountMismatch,
                "reshape: element count must be preserved");
  static_assert(status != Impl::ReshapeStatus::BadOrder,
                "reshape: the new memory order must be a permutation of "
                "[0, new rank), with one entry per new extent");
  static_assert(status != Impl::ReshapeStatus::NotFactorable,
                "reshape: the target shape cannot be carved from the source's "
                "memory-order stream — a new mode straddles source dimensions "
                "whose extents share no common factor, so the result is not an "
                "affine reinterpretation of the source");
  static_assert(status != Impl::ReshapeStatus::CapacityExceeded,
                "reshape: plan capacity exceeded; raise "
                "Impl::kReshapeMaxModes / kReshapeMaxLeaves");

#if TENSOR_OPS_VERIFY_RESHAPE
  static_assert(
      Impl::ReshapeVerifyIf<Impl::reshape_ok_v<Fn>, Impl::reshape_result_t<Fn>,
                            StaticTile<E...>, std::integer_sequence<int, S...>,
                            std::integer_sequence<int, O...>>::value,
      "reshape: the emitted layout does not reproduce the source's "
      "element->offset map (planner bug)");
#endif

  return {};
}

// ---------------------------------------------------------------------------
// reshape(nested StaticLayout, Tile, Order) — a NESTED source.
//
// reshape produces the nested layout, so it should be able to consume one. A
// nested layout's mode grouping is purely logical: the offsets it addresses,
// and their memory order, are fixed by its leaves alone. So reshaping it is
// reshaping the flat layout over its concatenated leaves — see
// Impl::flat_view_t, which is where that equivalence and the ordering
// conventions behind it are written down.
//
// Forwarding to the flat overload rather than re-entering the planner directly
// is what keeps this honest: the status-specific static_asserts, the
// TENSOR_OPS_VERIFY_RESHAPE oracle, and ReshapeFlat's named-type recovery all
// apply unchanged, because it is literally the same call.
//
// ROUND TRIPS ARE ASYMMETRIC, and the difference is not a bug. reshape is
// defined over MEMORY order, and the planner carves each new mode to completion
// in memory order without ever seeing the source's mode grouping. So reshaping
// a nested source back to its LEAF shape and order is the identity, while
// reshaping it to its MODE shape need not be: when a mode's leaves are
// interleaved in memory with another mode's, the carve picks them up in memory
// order and regroups them differently. The result still reproduces the source's
// element->offset stream — it is a valid reshape — but it is not the layout you
// started from. Do not assume the identity property the flat sources have.
// ---------------------------------------------------------------------------

template <typename... ExtSeqs, typename... StrSeqs, typename... OrdSeqs,
          int... F, int... NO>
KOKKOS_FORCEINLINE_FUNCTION auto reshape(
    StaticLayout<DeviceTuple<ExtSeqs...>, DeviceTuple<StrSeqs...>,
                 DeviceTuple<OrdSeqs...>>,
    StaticTile<F...> tile, std::integer_sequence<int, NO...> order) noexcept
    -> decltype(reshape(
        Impl::flat_view_t<
            StaticLayout<DeviceTuple<ExtSeqs...>, DeviceTuple<StrSeqs...>,
                         DeviceTuple<OrdSeqs...>>>{},
        tile, order)) {
  return reshape(
      Impl::flat_view_t<
          StaticLayout<DeviceTuple<ExtSeqs...>, DeviceTuple<StrSeqs...>,
                       DeviceTuple<OrdSeqs...>>>{},
      tile, order);
}

// LayoutRight / LayoutLeft spellings of the new memory order, so the common
// cases read like the rest of the API instead of spelling out an
// integer_sequence. Same convention as make_tile_layout(tile, LayoutRight{}).
template <int... E, int... S, int... O, int... F>
KOKKOS_FORCEINLINE_FUNCTION auto reshape(
    StaticLayout<StaticTile<E...>, std::integer_sequence<int, S...>,
                 std::integer_sequence<int, O...>>
                     src,
    StaticTile<F...> tile, LayoutRight) noexcept
    -> decltype(reshape(
        src, tile, Impl::right_order_t<static_cast<int>(sizeof...(F))>{})) {
  return reshape(src, tile,
                 Impl::right_order_t<static_cast<int>(sizeof...(F))>{});
}

template <int... E, int... S, int... O, int... F>
KOKKOS_FORCEINLINE_FUNCTION auto reshape(
    StaticLayout<StaticTile<E...>, std::integer_sequence<int, S...>,
                 std::integer_sequence<int, O...>>
                     src,
    StaticTile<F...> tile, LayoutLeft) noexcept
    -> decltype(reshape(src, tile,
                        Impl::left_order_t<static_cast<int>(sizeof...(F))>{})) {
  return reshape(src, tile,
                 Impl::left_order_t<static_cast<int>(sizeof...(F))>{});
}

// The same two spellings for a nested source. Needed separately because the two
// above pattern-match the flat specialisation; without them a nested source
// would be stuck with the explicit integer_sequence form.
template <typename... ExtSeqs, typename... StrSeqs, typename... OrdSeqs,
          int... F>
KOKKOS_FORCEINLINE_FUNCTION auto reshape(
    StaticLayout<DeviceTuple<ExtSeqs...>, DeviceTuple<StrSeqs...>,
                 DeviceTuple<OrdSeqs...>>
                     src,
    StaticTile<F...> tile, LayoutRight) noexcept
    -> decltype(reshape(
        src, tile, Impl::right_order_t<static_cast<int>(sizeof...(F))>{})) {
  return reshape(src, tile,
                 Impl::right_order_t<static_cast<int>(sizeof...(F))>{});
}

template <typename... ExtSeqs, typename... StrSeqs, typename... OrdSeqs,
          int... F>
KOKKOS_FORCEINLINE_FUNCTION auto reshape(
    StaticLayout<DeviceTuple<ExtSeqs...>, DeviceTuple<StrSeqs...>,
                 DeviceTuple<OrdSeqs...>>
                     src,
    StaticTile<F...> tile, LayoutLeft) noexcept
    -> decltype(reshape(src, tile,
                        Impl::left_order_t<static_cast<int>(sizeof...(F))>{})) {
  return reshape(src, tile,
                 Impl::left_order_t<static_cast<int>(sizeof...(F))>{});
}

// ---------------------------------------------------------------------------
// prefix_product — collapse a tile to a 2D [before, after] shape at a split.
//
// Given a split point S, returns a 2-extent tile whose first extent is the
// product of dims [0, S) and whose second is the product of dims [S, rank).
// Kind-preserving (StaticTile -> StaticTile, DynamicTile -> DynamicTile<2>) so
// the result is a valid reshape target for the matching TileLayout. Used by the
// team contraction evaluator to view each operand's scratch as a 2D GEMM
// matrix.
//
//   prefix_product(StaticTile<2,4,8>{},     rank_c<1>) -> StaticTile<2, 32>
//   prefix_product(StaticTile<2,4,8>{},     rank_c<0>) -> StaticTile<1, 64>
//   prefix_product(DynamicTile<3>{{2,4,8}}, rank_c<1>) -> DynamicTile<2>{{2,
//   32}}
// ---------------------------------------------------------------------------

namespace Impl {

template <int Split, int... E>
KOKKOS_FUNCTION constexpr int product_before() noexcept {
  int i = 0, p = 1;
  ((p *= (i++ < Split ? E : 1)), ...);
  return p;
}

template <int Split, int... E>
KOKKOS_FUNCTION constexpr int product_from() noexcept {
  int i = 0, p = 1;
  ((p *= (i++ >= Split ? E : 1)), ...);
  return p;
}

}  // namespace Impl

template <int... E, int Split>
KOKKOS_FUNCTION constexpr auto prefix_product(
    StaticTile<E...>, std::integral_constant<int, Split>) noexcept {
  static_assert(Split >= 0 && Split <= static_cast<int>(sizeof...(E)),
                "prefix_product: split point out of range");
  return StaticTile<Impl::product_before<Split, E...>(),
                    Impl::product_from<Split, E...>()>{};
}

template <int N, int Split>
KOKKOS_FUNCTION constexpr DynamicTile<2> prefix_product(
    DynamicTile<N> tile, std::integral_constant<int, Split>) noexcept {
  static_assert(Split >= 0 && Split <= N,
                "prefix_product: split point out of range");
  int before = 1, after = 1;
  for (int i = 0; i < Split; ++i) before *= tile.extent(i);
  for (int i = Split; i < N; ++i) after *= tile.extent(i);
  return DynamicTile<2>{{before, after}};
}

// ---------------------------------------------------------------------------
// Tile<TileA, TileB, TileC> — bundles the three per-operand tiles of a
// team-tier contraction (A's modes, B's modes, output C's modes). Stored by
// value so the runtime extents of dynamic tiles are carried into the evaluator.
// ---------------------------------------------------------------------------

template <typename TileA, typename TileB, typename TileC>
struct Tile {
  TileA a{};
  TileB b{};
  TileC c{};
};

// ---------------------------------------------------------------------------
// output_tile — the free-mode (output C) tile of a tiling spec.
//
// A contraction tiling is a Tile<A,B,C> bundle whose output is the C tile; a
// plain tile (register / input tier) is its own output tile. Lets a driver read
// the free-mode tile uniformly from either form to size work items and decode
// tile indices (Tile<A,B,C> itself has no extent()).
// ---------------------------------------------------------------------------

template <typename TileA, typename TileB, typename TileC>
KOKKOS_FUNCTION TileC output_tile(const Tile<TileA, TileB, TileC>& t) noexcept {
  return t.c;
}
template <int... E>
KOKKOS_FUNCTION constexpr StaticTile<E...> output_tile(
    StaticTile<E...> t) noexcept {
  return t;
}
template <int R>
KOKKOS_FUNCTION DynamicTile<R> output_tile(DynamicTile<R> t) noexcept {
  return t;
}

// ---------------------------------------------------------------------------
// CombineTile — per-operand tile bundle for a combine node.
//
// A combine node's output tile fixes both its own output and every operand's
// tile IF every operand is a leaf (input) node: operand K stages its input
// slab with the same output tile. To support a contraction node as a combine
// operand, operand K needs a `Tile<A,B,C>` bundle instead — the inner
// contraction's A/B/C tile spec — whose C tile equals the combine's output
// tile. CombineTile carries the output tile plus one per-operand tile spec:
//
//   OpTile == OutTile        for input operands (staged in output order)
//   OpTile == Tile<A,B,C>    for contraction operands. C is in that operand's
//                             own USER order and need not match OutTile
//                             directly: the combine evaluator reads it through
//                             the operand's permC and gathers it into the
//                             combine's output order, which is what must equal
//                             OutTile. Both collapse to C == OutTile when the
//                             operand is canonical and in the output's order.
//
// `output_tile(CombineTile)` returns the shared output tile so the graph
// driver keeps sizing work items / decoding tile indices uniformly. For a
// plain combine (all input operands), callers can keep passing the plain
// output tile directly; the combine evaluator accepts either form.
// ---------------------------------------------------------------------------
template <typename OutTile, typename... OpTiles>
struct CombineTile {
  static constexpr int    rank      = OutTile::rank;
  static constexpr bool   is_static = OutTile::is_static;
  static constexpr int    num_ops   = static_cast<int>(sizeof...(OpTiles));
  OutTile                 out{};
  DeviceTuple<OpTiles...> ops{};

  KOKKOS_FUNCTION constexpr int extent(int i) const noexcept {
    return out.extent(i);
  }
};

namespace Impl {
template <typename T>
struct is_combine_tile : std::false_type {};
template <typename OutTile, typename... OpTiles>
struct is_combine_tile<CombineTile<OutTile, OpTiles...>> : std::true_type {};
template <typename T>
inline constexpr bool is_combine_tile_v = is_combine_tile<T>::value;
}  // namespace Impl

template <typename OutTile, typename... OpTiles>
KOKKOS_FUNCTION OutTile
output_tile(const CombineTile<OutTile, OpTiles...>& t) noexcept {
  return t.out;
}

// Host-side factory: assemble a CombineTile from an output tile and one tile
// spec per operand. For input operands pass the output tile again (or its
// equivalent); for contraction operands pass the corresponding Tile<A,B,C>.
template <typename OutTile, typename... OpTiles>
CombineTile<OutTile, OpTiles...> make_combine_tile(const OutTile& out,
                                                   const OpTiles&... ops) {
  return CombineTile<OutTile, OpTiles...>{out, DeviceTuple<OpTiles...>(ops...)};
}

// ---------------------------------------------------------------------------
// StageTile<SourceTile, PermSeq> — target spec for staging a global-view
// interm node into scratch.
//
// Bundles the operand's own (native-order) tile shape with the gather
// permutation (native -> canonical) needed to bring it into the order the
// destination scratch tile expects. The scratch tile's actual shape is
// derived on demand via reorder_tile_value(source_tile, PermSeq{}) — the same
// helper the contraction evaluator already uses to build its canonical
// per-operand tiles. PermSeq carries no runtime data; it's a template
// parameter only.
// ---------------------------------------------------------------------------
template <typename SourceTile, typename PermSeq>
struct StageTile {
  using source_tile_t = SourceTile;
  using perm_seq      = PermSeq;

  SourceTile source_tile{};
};

}  // namespace TensorOperations
