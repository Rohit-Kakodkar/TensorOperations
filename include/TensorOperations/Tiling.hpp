#pragma once
#include <TensorOperations/TileLayout.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

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
  static constexpr Kokkos::Array<int, rank> extents{Extents...};
  static constexpr bool                     is_static = true;
  KOKKOS_FUNCTION static constexpr int      extent(int i) { return extents[i]; }
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
    StaticTile<E...>) noexcept {
  return {};
}

template <int N>
KOKKOS_FUNCTION DynamicTileLayoutRight<N> make_tile_layout(
    DynamicTile<N> t) noexcept {
  return DynamicTileLayoutRight<N>{t.extents};
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

}  // namespace TensorOperations
