#pragma once
#include <TensorOperations/DeviceTuple.hpp>
#include <TensorOperations/Evaluator.hpp>

#include <cstddef>
#include <utility>

#include <Kokkos_Core.hpp>

namespace TensorOperations {

// ---------------------------------------------------------------------------
// SlotView<ValueType, ExecSpace, Tile>
//
// The team-scratch view type holding one node's output tile.
//
// THE LOAD-BEARING PROPERTY OF THE WHOLE DAG DESIGN IS THAT THIS ALIAS EXISTS.
// A slot's storage type is a function of (value type, exec space, tile) ALONE
// -- the node that produced the data does not appear in it. So every node in a
// graph can be typed from its own tile, independently, with no recursion into
// whatever computed it. If this were not true, the node list would have to be
// built as a left fold (node K+1's type depending on node K's), and the flat
// driver of stage 5 would be a substantially harder piece of work.
//
// It is also what a caller needs to spell a slot node HOST-side, before any
// buffer exists: `make_slot_node<'i','k'>(SlotView<float, ES, Tile>{}, shape)`
// declares the node with a placeholder, and the driver assigns the real buffer
// on device (node factories are host-only, so the graph cannot be assembled
// where team scratch lives -- see tests/test_slot_node.cpp).
// ---------------------------------------------------------------------------
template <typename ValueType, typename ExecSpace, typename Tile>
using SlotView = decltype(Impl::alloc_scratch_tile<ValueType, ExecSpace>(
    std::declval<Impl::team_member_t<ExecSpace>>(), std::declval<Tile>()));

// ---------------------------------------------------------------------------
// SlotStore<Views...>
//
// One team-scratch buffer per node OUTPUT in a DAG (so one per node, except for
// a multi-output combine), carved in a single pass and owned by the driver
// rather than by the evaluators that fill them.
//
// This is the inversion that makes sharing possible. An evaluator that carves
// its own output decides that buffer's identity by CONSTRUCTION ORDER, so a
// result cannot outlive its evaluator and no other node can name it. Here the
// driver carves everything up front, hands node K its own buffer to adopt (see
// the adopting constructors in Evaluator/Team.hpp), and hands the SAME buffer
// to every later node that names K as an operand (SlotTag). One buffer, one
// evaluation, N readers.
//
// The invariant this type carries, and which its tests check rather than
// assume: the buffers are pairwise DISJOINT, and together they occupy no more
// than slot_store_bytes(). A cursor bug that overlapped two slots would not
// crash -- it would silently feed one node another node's data.
//
// Thin on purpose. It is a DeviceTuple plus a name for that invariant.
// ---------------------------------------------------------------------------
template <typename... Views>
struct SlotStore {
  static constexpr std::size_t size = sizeof...(Views);

  DeviceTuple<Views...> views;

  template <std::size_t K>
  KOKKOS_FUNCTION const auto& get() const {
    return views.template get<K>();
  }
};

// ---------------------------------------------------------------------------
// slot_store_bytes<V, ES>(tiles...) — team scratch the store needs.
//
// Host-side sizing, called before the parallel_for that carves anything, in
// exactly the role Evaluator::scratch_size_per_team plays for a tree. A driver
// requests this PLUS each node's operand_scratch_size_per_team() -- the
// complement that adoption leaves the evaluators still carving for themselves.
// Requesting scratch_size_per_team() instead would charge every output twice.
// ---------------------------------------------------------------------------
template <typename ValueType, typename ExecSpace, typename... Tiles>
std::size_t slot_store_bytes(const Tiles&... tiles) {
  return (std::size_t{0} + ... +
          Impl::scratch_tile_bytes<ValueType, ExecSpace>(tiles));
}

// ---------------------------------------------------------------------------
// carve_slot_store<V, ES>(team, tiles...) — allocate the store, device-side.
//
// BRACES, NOT PARENTHESES, and that is load-bearing. Each alloc_scratch_tile
// bumps the team's scratch cursor, so the calls must not race to it; function
// argument evaluation order is UNSPECIFIED in C++, but the elements of a
// braced-init-list are evaluated left to right even when the braces select a
// constructor ([dcl.init.list]). DeviceTuple has a user-declared constructor,
// so this is list-initialization, not aggregate initialization -- the guarantee
// still holds, and it is the same one make_op_allocs already relies on in
// Evaluator/Team.hpp.
//
// (Distinctness alone does not actually need the ordering -- a bump allocator
// hands out disjoint regions whatever order it is called in. What the ordering
// buys is that slot K lands at a PREDICTABLE offset, which is the difference
// between a scratch layout you can reason about in a profile and one you
// cannot.)
// ---------------------------------------------------------------------------
template <typename ValueType, typename ExecSpace, typename Team,
          typename... Tiles>
KOKKOS_FUNCTION auto carve_slot_store(const Team& team, const Tiles&... tiles)
    -> SlotStore<SlotView<ValueType, ExecSpace, Tiles>...> {
  return {DeviceTuple<SlotView<ValueType, ExecSpace, Tiles>...>{
      Impl::alloc_scratch_tile<ValueType, ExecSpace>(team, tiles)...}};
}

}  // namespace TensorOperations
