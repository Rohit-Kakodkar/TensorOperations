#pragma once
#include <TensorOperations/Permute.hpp>
#include <TensorOperations/Tiling.hpp>

#include <cstdint>
#include <utility>

namespace TensorOperations {

// ---------------------------------------------------------------------------
// LabelTiles — one tile extent per LABEL, for the whole graph.
//
// A tile is currently a property of a NODE: stage() takes one, and a
// contraction's output tile is derived from its operands' layouts by
// CanonTileOf. Two mechanisms, both node-local, and neither says the thing that
// is actually true of a level graph -- a label has ONE tile size everywhere it
// appears. Say that once, and every node's tile becomes a lookup.
//
// What that buys beyond saying it once: the GRID stops being a designated node.
// The grid node is today whichever tensor was staged LAST, which sets the
// league size and every member's tile index, and nothing at the call site says
// so. With tiles keyed on labels, a label is gridded iff its extent exceeds its
// tile, and there is nothing to designate and nothing to get wrong by
// reordering.
//
// The cost, chosen deliberately: a label cannot be tiled coarsely in one node
// and finely in another. Nothing in the SEM pipeline does that, and a single
// source of truth is the point.
// ---------------------------------------------------------------------------
template <int32_t Label, int Extent>
struct LabelTile {
  static constexpr int32_t label  = Label;
  static constexpr int     extent = Extent;
  static_assert(Extent > 0, "LabelTile: a tile extent must be positive");
};

template <typename... Entries>
struct LabelTiles {
  static constexpr std::size_t size = sizeof...(Entries);
};

namespace Impl {

template <typename... Entries>
constexpr bool label_tiles_distinct() {
  constexpr std::size_t N = sizeof...(Entries);
  if constexpr (N < 2) {
    return true;
  } else {
    const int32_t ls[] = {Entries::label...};
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = 0; j < i; ++j)
        if (ls[i] == ls[j]) return false;
    return true;
  }
}

// -1 when the label is absent, which the guard below turns into a diagnosable
// error rather than a silently wrong extent.
template <typename... Entries>
constexpr int label_tile_lookup(int32_t l) {
  const int32_t ls[] = {Entries::label..., int32_t{0}};
  const int     es[] = {Entries::extent..., 0};
  for (std::size_t i = 0; i < sizeof...(Entries); ++i)
    if (ls[i] == l) return es[i];
  return -1;
}

// Namespace-scope variable templates rather than calls, for the reason spelled
// out in Permute.hpp: the functions above are host-only constexpr, and naming
// one from a KOKKOS_FUNCTION -- including inside a static_assert -- trips nvcc
// #20013 while Kokkos_ENABLE_CUDA_CONSTEXPR is off, as it is here.
template <typename LT>
inline constexpr bool label_tiles_distinct_v = false;
template <typename... Entries>
inline constexpr bool label_tiles_distinct_v<LabelTiles<Entries...>> =
    label_tiles_distinct<Entries...>();

template <typename LT, int32_t Label>
inline constexpr int label_tile_v = -1;
template <typename... Entries, int32_t Label>
inline constexpr int label_tile_v<LabelTiles<Entries...>, Label> =
    label_tile_lookup<Entries...>(Label);

template <typename LT, typename ModesSeq>
struct TileFromLabels;

template <typename LT, int32_t... Modes>
struct TileFromLabels<LT, std::integer_sequence<int32_t, Modes...>> {
  static_assert(label_tiles_distinct_v<LT>,
                "LabelTiles: a label may appear at most once in the map");
  static_assert(
      ((label_tile_v<LT, Modes> > 0) && ...),
      "LabelTiles: every label a node carries must appear in the graph's tile "
      "map -- an absent label has no tile extent to give, and defaulting one "
      "would silently change the iteration space");
  using type = StaticTile<label_tile_v<LT, Modes>...>;
};

}  // namespace Impl

// The tile a node with these labels gets, in the label sequence's own order.
// Order matters: the same label set in a different order is a different tile,
// which is what lets the declared and canonical output tiles of one contraction
// both come from this map.
template <typename LT, typename ModesSeq>
using tile_from_labels_t = typename Impl::TileFromLabels<LT, ModesSeq>::type;

}  // namespace TensorOperations
