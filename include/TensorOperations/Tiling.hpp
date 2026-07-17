#pragma once
#include <TensorOperations/DeviceTuple.hpp>
#include <TensorOperations/Permute.hpp>
#include <TensorOperations/TileLayout.hpp>

#include <Kokkos_Core.hpp>

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

// Build a StaticTileLayout{Right,Left} from the interleaved extents (declared
// only; used in the trailing-return decltype of tile_layout).
template <bool InnerFirst, class Ext, class Tile, std::size_t... P>
auto build_static_tiled(std::index_sequence<P...>) -> std::conditional_t<
    InnerFirst,
    StaticTileLayoutLeft<TiledExtentAt<P, InnerFirst, Ext, Tile>::value...>,
    StaticTileLayoutRight<TiledExtentAt<P, InnerFirst, Ext, Tile>::value...>>;

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

template <int... Extents, int... TileE>
KOKKOS_FUNCTION constexpr auto tile_layout(StaticTileLayoutRight<Extents...>,
                                           StaticTile<TileE...>) noexcept
    -> decltype(Impl::build_static_tiled<false, StaticTile<Extents...>,
                                         StaticTile<TileE...>>(
        std::make_index_sequence<2 * sizeof...(Extents)>{})) {
  static_assert(sizeof...(Extents) == sizeof...(TileE),
                "tile_layout: source rank must match tile rank");
  return {};
}

template <int... Extents, int... TileE>
KOKKOS_FUNCTION constexpr auto tile_layout(StaticTileLayoutLeft<Extents...>,
                                           StaticTile<TileE...>) noexcept
    -> decltype(Impl::build_static_tiled<true, StaticTile<Extents...>,
                                         StaticTile<TileE...>>(
        std::make_index_sequence<2 * sizeof...(Extents)>{})) {
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
//   OpTile == Tile<A,B,C>    for contraction operands (C must == OutTile in
//                             canonical/combine order)
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
  SourceTile source_tile{};
};

}  // namespace TensorOperations
