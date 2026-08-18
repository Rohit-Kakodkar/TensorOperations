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
// A BLOCKED label: the tensor is longer than the tile along it, so it is cut
// into tiles and becomes a grid mode -- one team per tile.
template <int32_t Label, int Extent>
struct LabelTile {
  static constexpr int32_t label   = Label;
  static constexpr int     extent  = Extent;
  static constexpr bool    gridded = true;
  static_assert(Extent > 0, "LabelTile: a tile extent must be positive");
};

// A WHOLE label: the tile spans the axis, so there is exactly one tile and
// every team's index along it is 0.
//
// Declaring that is worth a distinct entry because it keeps the label OUT of
// the grid. A grid mode costs a live register for its index and a division to
// decode it, for every team, whether or not it ever varies -- and most labels
// of a real graph never do. The SEM3D pipeline blocks one axis out of six, and
// gridding the other five measured a 25% occupancy loss.
//
// Getting this wrong is a silent wrong answer -- a genuinely blocked label
// marked Whole would leave every team computing the first tile -- so the extent
// is checked against the tile at launch, unconditionally.
template <int32_t Label, int Extent>
struct LabelWhole {
  static constexpr int32_t label   = Label;
  static constexpr int     extent  = Extent;
  static constexpr bool    gridded = false;
  static_assert(Extent > 0, "LabelWhole: an extent must be positive");
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

// The map's labels, in declaration order. This doubles as the GRID's mode list:
// a level graph iterates over every label it knows about, and a label whose
// extent equals its tile simply contributes one tile and a constant index 0.
// That is why no node has to be designated as the grid -- the label set is the
// grid, and it cannot be got wrong by reordering a call.
template <typename LT>
struct LabelSeq;
template <typename... Entries>
struct LabelSeq<LabelTiles<Entries...>> {
  using type = std::integer_sequence<int32_t, Entries::label...>;
};

// Where a label sits in that list, so a per-label quantity gathered at runtime
// (an extent, say) lands in the slot the grid tile expects. -1 if absent, which
// the TileFromLabels guard has already ruled out for any label a node carries.
template <typename... Entries>
constexpr int label_index_lookup(int32_t l) {
  const int32_t ls[] = {Entries::label..., int32_t{0}};
  for (std::size_t i = 0; i < sizeof...(Entries); ++i)
    if (ls[i] == l) return static_cast<int>(i);
  return -1;
}

// The same lookup with the label as a RUNTIME value, for gathering a per-label
// quantity while walking a node's modes. Host-only, like every constexpr
// function here; label_index_v is the form to name from device code.
template <typename LT>
struct LabelIndexOf;
template <typename... Entries>
struct LabelIndexOf<LabelTiles<Entries...>> {
  static constexpr int get(int32_t l) {
    return label_index_lookup<Entries...>(l);
  }
};
template <typename LT>
constexpr int label_index_of(int32_t l) {
  return LabelIndexOf<LT>::get(l);
}

// Whether a label is a grid mode, and its tile, by runtime label value.
template <typename... Entries>
constexpr bool label_gridded_lookup(int32_t l) {
  const int32_t ls[] = {Entries::label..., int32_t{0}};
  const bool    gs[] = {Entries::gridded..., false};
  for (std::size_t i = 0; i < sizeof...(Entries); ++i)
    if (ls[i] == l) return gs[i];
  return false;
}
template <typename LT>
struct LabelQuery;
template <typename... Entries>
struct LabelQuery<LabelTiles<Entries...>> {
  static constexpr bool gridded(int32_t l) {
    return label_gridded_lookup<Entries...>(l);
  }
  static constexpr int tile(int32_t l) {
    return label_tile_lookup<Entries...>(l);
  }
};
template <typename LT>
constexpr bool label_gridded_of(int32_t l) {
  return LabelQuery<LT>::gridded(l);
}
template <typename LT>
constexpr int label_tile_of(int32_t l) {
  return LabelQuery<LT>::tile(l);
}

template <typename LT, int32_t Label>
inline constexpr int label_index_v = -1;
template <typename... Entries, int32_t Label>
inline constexpr int label_index_v<LabelTiles<Entries...>, Label> =
    label_index_lookup<Entries...>(Label);

}  // namespace Impl

// The map's labels as a mode sequence -- the grid's modes.
template <typename LT>
using label_seq_t = typename Impl::LabelSeq<LT>::type;

// The tile of the grid itself: one extent per label, in map order.
template <typename LT>
using grid_tile_t = typename Impl::TileFromLabels<LT, label_seq_t<LT>>::type;

// The tile a node with these labels gets, in the label sequence's own order.
// Order matters: the same label set in a different order is a different tile,
// which is what lets the declared and canonical output tiles of one contraction
// both come from this map.
template <typename LT, typename ModesSeq>
using tile_from_labels_t = typename Impl::TileFromLabels<LT, ModesSeq>::type;

}  // namespace TensorOperations
