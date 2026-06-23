#pragma once
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

}  // namespace Impl

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
  Kokkos::Array<int, 2 * N> exts{};
  for (int d = 0; d < N; ++d) {
    exts[2 * d]     = src.extent(d) / tile.extent(d);  // outer (Right: first)
    exts[2 * d + 1] = tile.extent(d);                  // inner
  }
  return DynamicTileLayoutRight<2 * N>{exts};
}

template <int N, int... TileE>
KOKKOS_FUNCTION auto tile_layout(DynamicTileLayoutLeft<N> src,
                                 StaticTile<TileE...>     tile) noexcept
    -> DynamicTileLayoutLeft<2 * N> {
  static_assert(N == static_cast<int>(sizeof...(TileE)),
                "tile_layout: source rank must match tile rank");
  Kokkos::Array<int, 2 * N> exts{};
  for (int d = 0; d < N; ++d) {
    exts[2 * d]     = tile.extent(d);                  // inner (Left: first)
    exts[2 * d + 1] = src.extent(d) / tile.extent(d);  // outer
  }
  return DynamicTileLayoutLeft<2 * N>{exts};
}

template <int... Extents, int N>
KOKKOS_FUNCTION auto tile_layout(StaticTileLayoutRight<Extents...>,
                                 DynamicTile<N> tile) noexcept
    -> DynamicTileLayoutRight<2 * N> {
  static_assert(static_cast<int>(sizeof...(Extents)) == N,
                "tile_layout: source rank must match tile rank");
  constexpr int             src_e[] = {Extents...};
  Kokkos::Array<int, 2 * N> exts{};
  for (int d = 0; d < N; ++d) {
    exts[2 * d]     = src_e[d] / tile.extent(d);  // outer (Right: first)
    exts[2 * d + 1] = tile.extent(d);             // inner
  }
  return DynamicTileLayoutRight<2 * N>{exts};
}

template <int... Extents, int N>
KOKKOS_FUNCTION auto tile_layout(StaticTileLayoutLeft<Extents...>,
                                 DynamicTile<N> tile) noexcept
    -> DynamicTileLayoutLeft<2 * N> {
  static_assert(static_cast<int>(sizeof...(Extents)) == N,
                "tile_layout: source rank must match tile rank");
  constexpr int             src_e[] = {Extents...};
  Kokkos::Array<int, 2 * N> exts{};
  for (int d = 0; d < N; ++d) {
    exts[2 * d]     = tile.extent(d);             // inner (Left: first)
    exts[2 * d + 1] = src_e[d] / tile.extent(d);  // outer
  }
  return DynamicTileLayoutLeft<2 * N>{exts};
}

template <int N>
KOKKOS_FUNCTION auto tile_layout(DynamicTileLayoutRight<N> src,
                                 DynamicTile<N>            tile) noexcept
    -> DynamicTileLayoutRight<2 * N> {
  Kokkos::Array<int, 2 * N> exts{};
  for (int d = 0; d < N; ++d) {
    exts[2 * d]     = src.extent(d) / tile.extent(d);  // outer (Right: first)
    exts[2 * d + 1] = tile.extent(d);                  // inner
  }
  return DynamicTileLayoutRight<2 * N>{exts};
}

template <int N>
KOKKOS_FUNCTION auto tile_layout(DynamicTileLayoutLeft<N> src,
                                 DynamicTile<N>           tile) noexcept
    -> DynamicTileLayoutLeft<2 * N> {
  Kokkos::Array<int, 2 * N> exts{};
  for (int d = 0; d < N; ++d) {
    exts[2 * d]     = tile.extent(d);                  // inner (Left: first)
    exts[2 * d + 1] = src.extent(d) / tile.extent(d);  // outer
  }
  return DynamicTileLayoutLeft<2 * N>{exts};
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

}  // namespace TensorOperations
