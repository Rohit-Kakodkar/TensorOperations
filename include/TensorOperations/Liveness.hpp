#pragma once
// Slot liveness analysis and tile-index gathering — the compile-time half of a
// graph driver, with no execution mode attached.
//
// Extracted from DagGraph.hpp when the TeamPolicyTag v1 path was removed. Every
// function here is a constexpr pass over integer arrays and node TYPES: nothing
// allocates, launches, or names an evaluator. LevelPlan.hpp builds a LEVEL
// timeline on top of it; LevelGraph.hpp uses the gather half.
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Permute.hpp>

#include <array>
#include <cstddef>
#include <utility>

#include <Kokkos_Core.hpp>

namespace TensorOperations {
namespace Impl {

// --- liveness --------------------------------------------------------------
//
// A slot's LIVE RANGE is [the member that writes it, the last member that reads
// it]. Both ends are compile-time: definition order IS topological order, and
// an operand naming a slot carries that slot's index in its TYPE
// (SlotTag::SlotIdx). So the whole analysis is a constexpr pass over the
// member list and the device only ever sees integer constants.
//
// Two slots whose ranges do not overlap can share one buffer, which is what
// takes the SEM stiffness graph from 14 buffers to 8 -- the four gradients and
// four integrands that are genuinely live at once, and nothing more. Without
// this every member's output is allocated for the whole kernel, including the
// four divergences and two weighted sums that die immediately into their
// consumers.
//
// RANGES ARE CLOSED AT BOTH ENDS, and that is what makes reuse safe at a single
// step rather than merely likely: a member reads its operands and writes its
// own outputs during ONE evaluation, so a slot last read at step K and a slot
// defined at step K overlap at K and can never be pooled together.

// The slot an operand names, or -1 for an operand that is not a slot.
template <typename Op>
constexpr int operand_slot() {
  if constexpr (has_node_tag_v<SlotTag, Op>)
    return static_cast<int>(Op::SlotIdx);
  else
    return -1;
}

// Record step k as a reader of slot s. Callers walk k in ascending order, so a
// plain assignment leaves the LAST reader behind and no max() is needed.
template <std::size_t NS>
constexpr void note_read(std::array<std::size_t, NS>& last, int s,
                         std::size_t k) {
  if (s >= 0) last[static_cast<std::size_t>(s)] = k;
}

template <typename Node, std::size_t NS, std::size_t... Is>
constexpr void note_combine_reads(std::array<std::size_t, NS>& last,
                                  std::size_t k, std::index_sequence<Is...>) {
  (note_read<NS>(
       last, operand_slot<tuple_element_t<Is, typename Node::ops_tuple_t>>(),
       k),
   ...);
}

// One member's readers. A SCAN of its operands, not a traversal: in the flat
// form an operand is a leaf input or a NAME, never a subtree.
template <typename Node, std::size_t NS>
constexpr void note_node_reads(std::array<std::size_t, NS>& last,
                               std::size_t                  k) {
  if constexpr (has_node_tag_v<ContractionTag, Node>) {
    note_read<NS>(last, operand_slot<typename Node::node_a_type>(), k);
    note_read<NS>(last, operand_slot<typename Node::node_b_type>(), k);
  } else if constexpr (has_node_tag_v<CombineTag, Node> ||
                       has_node_tag_v<EinsumTag, Node>) {
    note_combine_reads<Node, NS>(
        last, k,
        std::make_index_sequence<static_cast<std::size_t>(Node::NumOps)>{});
  } else if constexpr (has_node_tag_v<StagedTag, Node>) {
    note_read<NS>(last, operand_slot<typename Node::operand_type>(), k);
  }
}

// Which pool each slot lives in: the lowest-numbered pool whose occupants all
// have live ranges disjoint from this slot's.
//
// Slots are visited in index order, which IS ascending definition order because
// the caller's slot numbering is a prefix sum over its member list. That makes
// this the LEFT-EDGE algorithm on an interval graph, and left-edge is OPTIMAL
// in the number of pools there -- not a heuristic that happens to do well.
//
// A pool costs the MAX of its occupants, so a graph whose members all emit one
// tile of the same shape (the common case, and the SEM graph exactly) wastes
// nothing. Where sizes differ, a small slot sharing a pool with a large one
// leaves a tail that no other slot can enter unless the coloring puts it there.
// Ordering by DECREASING SIZE packs that better and is the generalization to
// reach for if a real graph ever shows the spread; it also gives up left-edge's
// optimality guarantee, and it needs sizes as compile-time constants, which is
// why it is not what is here.
//
// The coloring knows nothing about members, levels or tiles -- it is a function
// of two integer arrays. Callers must supply CLOSED ranges; the overlap test
// below relies on it.
template <std::size_t NS>
constexpr std::array<std::size_t, NS> left_edge_colour(
    const std::array<std::size_t, NS>& def,
    const std::array<std::size_t, NS>& last) {
  std::array<std::size_t, NS> pool{};
  for (std::size_t s = 0; s < NS; ++s) {
    std::size_t p = 0;
    while (true) {
      bool clash = false;
      for (std::size_t t = 0; t < s; ++t)
        if (pool[t] == p && def[s] <= last[t] && def[t] <= last[s]) {
          clash = true;
          break;
        }
      if (!clash) break;
      ++p;
    }
    pool[s] = p;
  }
  return pool;
}

// --- tile-index gathering ---------------------------------------------------

// For each of Node's modes, the position of that label among the GRID's modes,
// or -1 if the grid does not carry it. This is how every member's own tile
// index is derived from the ONE index the team actually decodes.
template <typename NodeModes, typename RootModes>
constexpr auto compute_gather() {
  constexpr auto            n = seq_to_array(NodeModes{});
  constexpr auto            r = seq_to_array(RootModes{});
  std::array<int, n.size()> g{};
  for (std::size_t i = 0; i < n.size(); ++i) {
    g[i] = -1;
    for (std::size_t j = 0; j < r.size(); ++j)
      if (r[j] == n[i]) {
        g[i] = static_cast<int>(j);
        break;
      }
  }
  return g;
}
template <typename NodeModes, typename RootModes>
using gather_seq_t = array_to_seq_t<compute_gather<NodeModes, RootModes>()>;

// Gather the grid's tile index into a member's own mode order. A mode the grid
// does not carry gets index 0 -- sound only if that mode has exactly one tile,
// which LevelGraph::index_consistent() checks host-side before launching.
template <int NodeRank, std::size_t RootRank, int... G>
KOKKOS_FUNCTION Kokkos::Array<int, NodeRank> node_index(
    const Kokkos::Array<int, RootRank>& root_idx,
    std::integer_sequence<int, G...>) {
  const int                    g[NodeRank] = {G...};
  Kokkos::Array<int, NodeRank> out{};
  for (int i = 0; i < NodeRank; ++i)
    out[i] = (g[i] >= 0) ? root_idx[static_cast<std::size_t>(g[i])] : 0;
  return out;
}

// The team's league rank -> its tile index along each grid mode, row-major over
// the grid (last mode fastest).
//
// A mode with exactly one tile is skipped rather than divided: the compiler
// cannot prove a runtime tile count is 1, and most modes of a real grid are
// un-tiled (the SEM3D graph grids one axis out of six). Skipping them turns
// twenty instructions into a compare the whole team takes the same way.
template <int Rank, typename Tile>
KOKKOS_FUNCTION Kokkos::Array<int, Rank> decode_tile_index(
    int idx, const Kokkos::Array<int, Rank>& shape, const Tile& tile) {
  Kokkos::Array<int, Rank> tidx{};
  for (int d = Rank - 1; d >= 0; --d) {
    const int n = (shape[d] + tile.extent(d) - 1) / tile.extent(d);
    if (n == 1) {
      tidx[d] = 0;
      continue;
    }
    tidx[d] = static_cast<int>(idx % static_cast<int>(n));
    idx /= static_cast<int>(n);
  }
  return tidx;
}

}  // namespace Impl
}  // namespace TensorOperations
