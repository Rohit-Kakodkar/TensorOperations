#pragma once
#include <TensorOperations/DeviceTuple.hpp>
#include <TensorOperations/Evaluator.hpp>

#include <cstddef>
#include <type_traits>
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
// than arena_slot_store_bytes(). An offset bug that overlapped two slots would
// not crash -- it would silently feed one node another node's data.
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
// place_slot_store<V, ES>(base, seq, tiles...) — the same store, over buffers
// the CALLER chose.
//
// The store with NO allocator at all: slot I is built at base[I], and NOTHING
// requires those pointers to be distinct -- a liveness plan (DagGraph.hpp) may
// point two slots at one buffer when their live ranges do not overlap.
//
// THE INVARIANT IS THEREFORE WEAKER THAN carve_arena_slot_store's, and
// deliberately. The arena lays slots end to end and so guarantees them pairwise
// disjoint; this guarantees only what its caller's plan guarantees, which is
// that slots whose live ranges OVERLAP are disjoint. A plan that got that wrong
// would not crash -- it would silently feed one node another node's data -- so
// the plan is checked directly in tests/test_slot_liveness.cpp rather than
// trusted.
//
// The returned type is identical to the arena's for the same tiles: a slot's
// view type is a function of its tile alone, never of where it lives.
template <typename ValueType, typename ExecSpace, typename... Tiles,
          std::size_t... Is>
KOKKOS_FUNCTION auto place_slot_store(
    const Kokkos::Array<ValueType*, sizeof...(Tiles)>& base,
    std::index_sequence<Is...>, const Tiles&... tiles)
    -> SlotStore<SlotView<ValueType, ExecSpace, Tiles>...> {
  return {DeviceTuple<SlotView<ValueType, ExecSpace, Tiles>...>{
      Impl::alloc_scratch_tile_at<ValueType, ExecSpace>(base[Is], tiles)...}};
}

namespace Impl {

template <typename ValueType, typename ExecSpace>
KOKKOS_FUNCTION constexpr std::size_t slot_arena_align() {
  std::size_t a = sizeof(ValueType);
  if (alignof(ValueType) > a) a = alignof(ValueType);
  const std::size_t k =
      static_cast<std::size_t>(ExecSpace::scratch_memory_space::ALIGN);
  if (k > a) a = k;
  return a;
}

template <typename Tile>
KOKKOS_FUNCTION constexpr std::size_t slot_tile_elems() {
  static_assert(Tile::is_static,
                "arena slot store: slot tiles must have compile-time extents, "
                "because the per-slot offsets are compile-time constants");
  using layout_t = std::decay_t<decltype(make_tile_layout(std::declval<Tile>(),
                                                          LayoutRight{}))>;
  return static_cast<std::size_t>(layout_t::num_elements);
}

template <typename ValueType, typename ExecSpace>
KOKKOS_FUNCTION constexpr std::size_t slot_arena_step(std::size_t elems) {
  constexpr std::size_t a = slot_arena_align<ValueType, ExecSpace>();
  static_assert(a % sizeof(ValueType) == 0,
                "arena slot store: the scratch alignment must be a whole "
                "number of elements for offsets to be expressible in elements");
  const std::size_t b = elems * sizeof(ValueType);
  return ((b + a - 1) / a * a) / sizeof(ValueType);
}

template <typename ValueType, typename ExecSpace, typename... Tiles>
KOKKOS_FUNCTION constexpr std::size_t slot_arena_prefix(std::size_t n) {
  const std::size_t steps[] = {
      slot_arena_step<ValueType, ExecSpace>(slot_tile_elems<Tiles>())..., 0};
  std::size_t o = 0;
  for (std::size_t k = 0; k < n; ++k) o += steps[k];
  return o;
}

template <typename... Tiles>
struct SlotTiles {};

// A pool assignment, one entry per slot, as a type so it can travel as one
// template argument. Same reason SlotTiles exists.
template <std::size_t... Pools>
struct SlotPools {};

// The arena laid out by POOL rather than by slot: pool P is as big as its
// largest occupant, pools sit end to end, and every slot assigned to P starts
// at P's base. Slots sharing a pool therefore ALIAS, which is the point -- the
// caller's plan is what guarantees that no two of them are ever live at once.
//
// prefix(i) is the offset of slot i in elements; prefix(N) is the whole arena.
// ONE function for both, called from the host to size the policy and from the
// device to carve, because those two numbers agreeing is the invariant.
//
// pelems is sized N rather than the pool count because the pool count is not a
// constant expression here and cannot exceed the number of slots.
template <typename ValueType, typename ExecSpace, typename PoolsList,
          typename TilesList>
struct slot_pool_arena;

template <typename ValueType, typename ExecSpace, std::size_t... Pools,
          typename... Tiles>
struct slot_pool_arena<ValueType, ExecSpace, SlotPools<Pools...>,
                       SlotTiles<Tiles...>> {
  static constexpr std::size_t N = sizeof...(Tiles);
  static_assert(sizeof...(Pools) == N,
                "pooled slot store: the plan must assign exactly one pool per "
                "slot");

  static constexpr std::size_t prefix(std::size_t i) {
    const std::size_t steps[] = {
        slot_arena_step<ValueType, ExecSpace>(slot_tile_elems<Tiles>())..., 0};
    const std::size_t pools[] = {Pools..., 0};

    std::size_t pelems[N > 0 ? N : 1] = {};
    std::size_t np                    = 0;
    for (std::size_t k = 0; k < N; ++k) {
      if (steps[k] > pelems[pools[k]]) pelems[pools[k]] = steps[k];
      if (pools[k] + 1 > np) np = pools[k] + 1;
    }

    const std::size_t upto = (i >= N) ? np : pools[i];
    std::size_t       off  = 0;
    for (std::size_t p = 0; p < upto; ++p) off += pelems[p];
    return off;
  }

  static constexpr std::size_t total() { return prefix(N); }
};

template <typename ValueType, typename ExecSpace, typename PoolsList,
          typename TilesList, std::size_t I>
struct slot_pool_offset {
  static constexpr std::size_t value =
      slot_pool_arena<ValueType, ExecSpace, PoolsList, TilesList>::prefix(I);
};

template <typename ValueType, typename ExecSpace, typename TilesList,
          std::size_t I>
struct slot_arena_offset;

template <typename ValueType, typename ExecSpace, typename... Tiles,
          std::size_t I>
struct slot_arena_offset<ValueType, ExecSpace, SlotTiles<Tiles...>, I> {
  static constexpr std::size_t value =
      slot_arena_prefix<ValueType, ExecSpace, Tiles...>(I);
};

}  // namespace Impl

// ---------------------------------------------------------------------------
// arena_slot_store_bytes / carve_arena_slot_store — the whole store over ONE
// bump allocation.
//
// The store used to take one alloc_scratch_tile per slot, and each of those is
// a full Kokkos get_shmem_common -- alignment fixup, capacity check, cursor
// bump, serially dependent through m_iter. That is a PROLOGUE cost paid N times
// per team: on the SEM3D level graph (35 slots) it measured 5,718,016 warp
// instructions against the hand-written kernel's 753,664, i.e. 70% of that
// kernel's whole instruction deficit. One carve plus 35 compile-time offsets
// replaces all of it. This is DagGraph's dag_carve_pools with the pool count
// fixed at one -- and unlike pooling, it needs no liveness analysis, because it
// is not trying to make the store SMALLER.
//
// Pairwise disjointness is still guaranteed, now by the prefix sum being
// strictly increasing rather than by the allocator's cursor.
//
// Sizing and carving share slot_arena_prefix so the host figure and the device
// figure cannot drift; under-requesting by one rounding step overruns the team
// allocation on GPU only, since Serial's 32 KB absorbs it.
// ---------------------------------------------------------------------------
template <typename ValueType, typename ExecSpace, typename... Tiles>
std::size_t arena_slot_store_bytes(const Tiles&...) {
  constexpr std::size_t elems =
      Impl::slot_arena_prefix<ValueType, ExecSpace, Tiles...>(sizeof...(Tiles));
  return Impl::scratch_backing_t<ValueType, ExecSpace>::shmem_size(elems);
}

template <typename ValueType, typename ExecSpace, typename Team,
          typename... Tiles, std::size_t... Is>
KOKKOS_FUNCTION auto place_arena_slot_store(const Team& team,
                                            std::index_sequence<Is...>,
                                            const Tiles&... tiles)
    -> SlotStore<SlotView<ValueType, ExecSpace, Tiles>...> {
  constexpr std::size_t elems =
      Impl::slot_arena_prefix<ValueType, ExecSpace, Tiles...>(sizeof...(Tiles));
  Impl::scratch_backing_t<ValueType, ExecSpace> arena(team.team_scratch(0),
                                                      elems);
  ValueType*                                    base = arena.data();
  return {DeviceTuple<SlotView<ValueType, ExecSpace, Tiles>...>{
      Impl::alloc_scratch_tile_at<ValueType, ExecSpace>(
          base + Impl::slot_arena_offset<ValueType, ExecSpace,
                                         Impl::SlotTiles<Tiles...>, Is>::value,
          tiles)...}};
}

template <typename ValueType, typename ExecSpace, typename Team,
          typename... Tiles>
KOKKOS_FUNCTION auto carve_arena_slot_store(const Team& team,
                                            const Tiles&... tiles)
    -> SlotStore<SlotView<ValueType, ExecSpace, Tiles>...> {
  return place_arena_slot_store<ValueType, ExecSpace>(
      team, std::index_sequence_for<Tiles...>{}, tiles...);
}

// ---------------------------------------------------------------------------
// The pooled forms. Same store, same slot types, same single allocation -- the
// only difference is that slot I lands at its POOL's base instead of its own,
// so slots whose live ranges do not overlap share memory.
//
// The disjointness guarantee is correspondingly WEAKER and deliberately so: the
// arena forms guarantee every slot pairwise disjoint, these guarantee only what
// the caller's plan guarantees. A plan that got that wrong would not crash --
// it would silently feed one node another node's data -- so the plan is checked
// directly (tests/test_level_liveness.cpp) rather than trusted.
// ---------------------------------------------------------------------------
template <typename ValueType, typename ExecSpace, typename PoolsList,
          typename... Tiles>
std::size_t pooled_arena_slot_store_bytes(const Tiles&...) {
  constexpr std::size_t elems =
      Impl::slot_pool_arena<ValueType, ExecSpace, PoolsList,
                            Impl::SlotTiles<Tiles...>>::total();
  return Impl::scratch_backing_t<ValueType, ExecSpace>::shmem_size(elems);
}

template <typename ValueType, typename ExecSpace, typename PoolsList,
          typename Team, typename... Tiles, std::size_t... Is>
KOKKOS_FUNCTION auto place_pooled_arena_slot_store(const Team& team,
                                                   std::index_sequence<Is...>,
                                                   const Tiles&... tiles)
    -> SlotStore<SlotView<ValueType, ExecSpace, Tiles>...> {
  constexpr std::size_t elems =
      Impl::slot_pool_arena<ValueType, ExecSpace, PoolsList,
                            Impl::SlotTiles<Tiles...>>::total();
  Impl::scratch_backing_t<ValueType, ExecSpace> arena(team.team_scratch(0),
                                                      elems);
  ValueType*                                    base = arena.data();
  return {DeviceTuple<SlotView<ValueType, ExecSpace, Tiles>...>{
      Impl::alloc_scratch_tile_at<ValueType, ExecSpace>(
          base + Impl::slot_pool_offset<ValueType, ExecSpace, PoolsList,
                                        Impl::SlotTiles<Tiles...>, Is>::value,
          tiles)...}};
}

template <typename ValueType, typename ExecSpace, typename PoolsList,
          typename Team, typename... Tiles>
KOKKOS_FUNCTION auto carve_pooled_arena_slot_store(const Team& team,
                                                   const Tiles&... tiles)
    -> SlotStore<SlotView<ValueType, ExecSpace, Tiles>...> {
  return place_pooled_arena_slot_store<ValueType, ExecSpace, PoolsList>(
      team, std::index_sequence_for<Tiles...>{}, tiles...);
}

}  // namespace TensorOperations
