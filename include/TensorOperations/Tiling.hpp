#pragma once
#include <TensorOperations/TileLayout.hpp>

#include <Kokkos_Core.hpp>

namespace TensorOperations {

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
KOKKOS_FUNCTION constexpr StaticTileLayout<E...> make_tile_layout(
    StaticTile<E...>) noexcept {
  return {};
}

template <int N>
KOKKOS_FUNCTION DynamicTileLayout<N> make_tile_layout(
    DynamicTile<N> t) noexcept {
  return DynamicTileLayout<N>{t.extents};
}

}  // namespace TensorOperations
