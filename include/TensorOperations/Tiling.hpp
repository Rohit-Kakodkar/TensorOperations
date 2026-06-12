#pragma once
#include <array>

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
  static constexpr std::array<int, rank> extents{Extents...};
  static constexpr bool                  is_static = true;
  static constexpr int extent(int i) { return extents[i]; }  // constexpr
};

template <int Rank>
struct DynamicTile {
  static constexpr int  rank      = Rank;
  static constexpr bool is_static = false;
  std::array<int, Rank> extents;
  constexpr int         extent(int i) const { return extents[i]; }  // runtime
};

}  // namespace TensorOperations
