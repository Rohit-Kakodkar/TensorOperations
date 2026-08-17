#pragma once
#include <TensorOperations/DeviceTuple.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/Graph.hpp>
#include <TensorOperations/SlotStore.hpp>

#include <array>
#include <cstddef>
#include <utility>

#include <Kokkos_Core.hpp>

namespace TensorOperations {

// ---------------------------------------------------------------------------
// DagGraph — a FLAT, topologically-ordered node list whose operands NAME
// earlier results instead of nesting them.
//
// The tree driver in Graph.hpp evaluates a node by recursing into its operands,
// so a subtree reachable from two consumers is built, allocated and evaluated
// ONCE PER CONSUMER. For the SEM stiffness kernel that is 16 gradient sums
// where 4 suffice, and it is the largest measured term in that benchmark's gap
// (2.23x, measured by CONTROL-C).
//
// Here every node owns a CONTIGUOUS RUN of slots in a driver-carved store --
// one per output, so one slot for all but a multi-output combine. Node K writes
// into its own slots (adopting them), and any later node needing one of those
// results NAMES that slot as an operand. One buffer, one evaluation, N readers.
//
// The slot index is therefore NOT the node index once any node emits more than
// one output; Impl::dag_slot_base / dag_slot_owner are the prefix-sum that
// relates them, and every `Roots...` pack and SlotTag operand carries a SLOT
// index, never a node index.
//
// Nothing is deduplicated automatically. The CALLER declares the sharing by
// naming a handle twice, and that is exactly what makes this tractable: no node
// identity up to alpha-renaming ever has to be inferred, which is the part that
// made this an XL problem when the library was expected to work it out.
//
// USAGE. add() returns the new graph and a handle, so definition order IS the
// topological order and no index is written by hand:
//
//   auto [g0, grad]   = make_dag<float, ES>().add(grad_node, grad_tile);
//   auto [g1, stress] = g0.add(stress_node(grad.as<'q','e','j'>()), s_tile);
//   g1.outputs(stress).execute(TeamPolicyTag<ES>{}, force);
//
// `handle.as<labels...>()` mints the operand. Calling it twice with DIFFERENT
// labels is the relabel mechanism: one buffer, two names, no copy.
//
// BUILD ON THE HOST, BIND ON THE DEVICE. The node factories are host-only (they
// run extent assertions and touch std::array), but a slot's storage is team
// scratch and exists only inside the kernel. So a handle mints its node with a
// value-initialized PLACEHOLDER view and the launcher assigns the real buffer
// on device before constructing each evaluator. Only the pointer is deferred.
// Calling a host-only factory from a kernel instead is diagnosed ONLY as nvcc
// warning #20011: it compiles, Serial stays green, and CUDA traps at runtime.
//
// MULTI-OUTPUT NODES. A combine whose fn returns Kokkos::Array<V,M> emits M
// tensors, and add() hands back M handles:
//
//   auto [g, fxi_0, fxi_1] = g0.add(two_output_combine, tile);
//
// All M share one shape, one tile and one layout (make_out_allocs builds every
// slot from the same output_tile), so they differ only in which buffer they
// name -- which is exactly why M consumers of M different outputs still cost
// ONE evaluation of everything underneath them. That is the redundancy a DAG
// cannot remove by sharing alone: sharing collapses duplicate SUBTREES, and
// multi-output collapses duplicate WORK INSIDE ONE NODE (in the SEM stiffness
// kernel, four stress-tensor evaluations down to two).
//
// V1 RESTRICTIONS, each static_asserted or checked where it is relied on:
//   * a node's modes that the root does not carry must be single-tiled;
//   * shared modes must be tiled identically in node and root.
// ---------------------------------------------------------------------------

namespace Impl {

// --- slot arithmetic -------------------------------------------------------
//
// Node K owns output_arity<Node_K> consecutive slots. These three functions are
// the whole of the node-index <-> slot-index relation; nothing else may assume
// the two coincide. They mirror Graph.hpp's node_arities/node_offset, restated
// over a DeviceTuple because that is what a DagGraph stores its nodes in.

template <typename NodesT, std::size_t... Ks>
constexpr std::array<std::size_t, sizeof...(Ks)> dag_arities_impl(
    std::index_sequence<Ks...>) {
  return {static_cast<std::size_t>(
      output_arity<tuple_element_t<Ks, NodesT>>::value)...};
}
template <typename NodesT>
constexpr auto dag_arities() {
  return dag_arities_impl<NodesT>(
      std::make_index_sequence<tuple_size_v<NodesT>>{});
}

// Total slots the store must carve.
template <typename NodesT>
constexpr std::size_t dag_num_slots() {
  std::size_t n = 0;
  for (std::size_t a : dag_arities<NodesT>()) n += a;
  return n;
}

// First slot owned by node K.
template <typename NodesT>
constexpr std::size_t dag_slot_base(std::size_t k) {
  const auto  a   = dag_arities<NodesT>();
  std::size_t off = 0;
  for (std::size_t i = 0; i < k; ++i) off += a[i];
  return off;
}

// The node that owns slot S -- the inverse of dag_slot_base, needed wherever a
// slot index arrives from outside (a designated output) and the producing
// node's type is what the work requires.
template <typename NodesT>
constexpr std::size_t dag_slot_owner(std::size_t slot) {
  const auto  a   = dag_arities<NodesT>();
  std::size_t end = 0;
  for (std::size_t k = 0; k < a.size(); ++k) {
    end += a[k];
    if (slot < end) return k;
  }
  return a.size();  // unreachable for a slot minted by add()
}

// Variable-template forms, and they are not a convenience. The three functions
// above are constexpr but HOST-only (std::array is), so naming one inside a
// KOKKOS_FUNCTION -- even to initialize a constexpr local -- draws nvcc warning
// #20013 and would need --expt-relaxed-constexpr. As namespace-scope variable
// templates the initializer runs on the host at instantiation and device code
// only ever names a constant.
template <typename NodesT, std::size_t K>
inline constexpr std::size_t dag_slot_base_v = dag_slot_base<NodesT>(K);
template <typename NodesT, std::size_t S>
inline constexpr std::size_t dag_slot_owner_v = dag_slot_owner<NodesT>(S);

// --- liveness --------------------------------------------------------------
//
// A slot's LIVE RANGE is [the node that writes it, the last node that reads
// it]. Both ends are compile-time: definition order IS topological order, and
// an operand naming a slot carries that slot's index in its TYPE
// (SlotTag::SlotIdx). So the whole analysis is a constexpr pass over the node
// list and the device only ever sees integer constants.
//
// Two slots whose ranges do not overlap can share one buffer, which is what
// takes the SEM stiffness graph from 14 buffers to 8 -- the four gradients and
// four integrands that are genuinely live at once, and nothing more. Without
// this every node's output is allocated for the whole kernel, including the
// four divergences and two weighted sums that die immediately into their
// consumers.
//
// RANGES ARE CLOSED AT BOTH ENDS, and that is what makes reuse safe at a single
// node rather than merely likely: a node reads its operands and writes its own
// outputs during ONE evaluation, so a slot last read at node K and a slot
// defined at node K overlap at K and can never be pooled together.

// The slot an operand names, or -1 for an operand that is not a slot.
template <typename Op>
constexpr int dag_operand_slot() {
  if constexpr (has_node_tag_v<SlotTag, Op>)
    return static_cast<int>(Op::SlotIdx);
  else
    return -1;
}

// Record node k as a reader of slot s. Callers walk k in ascending order, so a
// plain assignment leaves the LAST reader behind and no max() is needed.
template <std::size_t NS>
constexpr void dag_note_read(std::array<std::size_t, NS>& last, int s,
                             std::size_t k) {
  if (s >= 0) last[static_cast<std::size_t>(s)] = k;
}

template <typename Node, std::size_t NS, std::size_t... Is>
constexpr void dag_note_combine_reads(std::array<std::size_t, NS>& last,
                                      std::size_t                  k,
                                      std::index_sequence<Is...>) {
  (dag_note_read<NS>(
       last,
       dag_operand_slot<tuple_element_t<Is, typename Node::ops_tuple_t>>(), k),
   ...);
}

// One node's readers. A SCAN of its operands, not a traversal: in the flat form
// an operand is a leaf input or a NAME, never a subtree.
template <typename Node, std::size_t NS>
constexpr void dag_note_node_reads(std::array<std::size_t, NS>& last,
                                   std::size_t                  k) {
  if constexpr (has_node_tag_v<ContractionTag, Node>) {
    dag_note_read<NS>(last, dag_operand_slot<typename Node::node_a_type>(), k);
    dag_note_read<NS>(last, dag_operand_slot<typename Node::node_b_type>(), k);
  } else if constexpr (has_node_tag_v<CombineTag, Node>) {
    dag_note_combine_reads<Node, NS>(
        last, k,
        std::make_index_sequence<static_cast<std::size_t>(Node::NumOps)>{});
  } else if constexpr (has_node_tag_v<StagedTag, Node>) {
    dag_note_read<NS>(last, dag_operand_slot<typename Node::operand_type>(), k);
  }
}

template <typename NodesT, std::size_t NS, std::size_t... Ks>
constexpr void dag_note_all_reads(std::array<std::size_t, NS>& last,
                                  std::index_sequence<Ks...>) {
  (dag_note_node_reads<tuple_element_t<Ks, NodesT>, NS>(last, Ks), ...);
}

// Which pool each slot lives in: the lowest-numbered pool whose occupants all
// have live ranges disjoint from this slot's.
//
// Slots are visited in index order, which IS ascending definition order because
// dag_slot_base is a prefix sum over the node list. That makes this the
// LEFT-EDGE algorithm on an interval graph, and left-edge is OPTIMAL in the
// number of pools there -- not a heuristic that happens to do well.
//
// A pool costs the MAX of its occupants, so a graph whose nodes all emit one
// tile of the same shape (the common case, and the SEM graph exactly) wastes
// nothing. Where sizes differ, a small slot sharing a pool with a large one
// leaves a tail that no other slot can enter unless the coloring puts it there.
// Ordering by DECREASING SIZE packs that better and is the generalization to
// reach for if a real graph ever shows the spread; it also gives up left-edge's
// optimality guarantee, and it needs sizes as compile-time constants, which is
// why it is not what is here.
template <typename NodesT, std::size_t... Roots>
constexpr auto dag_pool_of_slot() {
  constexpr std::size_t NS = dag_num_slots<NodesT>();
  constexpr std::size_t NN = tuple_size_v<NodesT>;

  std::array<std::size_t, NS> def{}, last{}, pool{};
  for (std::size_t s = 0; s < NS; ++s) {
    def[s] = dag_slot_owner<NodesT>(s);
    // A slot nobody reads is still live where it is written.
    last[s] = def[s];
  }
  dag_note_all_reads<NodesT, NS>(last, std::make_index_sequence<NN>{});

  // A designated output is read by dag_store_roots AFTER every node has run, so
  // it outlives the whole list. NN is one past the last node index.
  const std::array<std::size_t, sizeof...(Roots)> roots{Roots...};
  for (std::size_t i = 0; i < sizeof...(Roots); ++i) last[roots[i]] = NN;

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

template <typename NodesT, std::size_t... Roots>
constexpr std::size_t dag_pool_count() {
  const auto  p = dag_pool_of_slot<NodesT, Roots...>();
  std::size_t n = 0;
  for (std::size_t s = 0; s < p.size(); ++s)
    if (p[s] + 1 > n) n = p[s] + 1;
  return n;
}

// Variable-template forms, for the same reason dag_slot_base_v exists: the
// functions above are HOST-only constexpr (std::array is), so device code must
// name a constant rather than call one. Keyed on an index_sequence of the roots
// so the root set travels as one type.
template <typename NodesT, typename RootsSeq>
struct dag_plan;
template <typename NodesT, std::size_t... Roots>
struct dag_plan<NodesT, std::index_sequence<Roots...>> {
  static constexpr auto of_slot() {
    return dag_pool_of_slot<NodesT, Roots...>();
  }
  static constexpr std::size_t count() {
    return dag_pool_count<NodesT, Roots...>();
  }
};

template <typename NodesT, typename RootsSeq, std::size_t S>
inline constexpr std::size_t dag_slot_pool_v =
    dag_plan<NodesT, RootsSeq>::of_slot()[S];
template <typename NodesT, typename RootsSeq>
inline constexpr std::size_t dag_pool_count_v =
    dag_plan<NodesT, RootsSeq>::count();

// For each of Node's modes, the position of that label among the ROOT's modes,
// or -1 if the root does not carry it. This is how every node's own tile index
// is derived from the ONE index the team actually decodes.
template <typename NodeModes, typename RootModes>
constexpr auto compute_dag_gather() {
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
using dag_gather_seq_t =
    array_to_seq_t<compute_dag_gather<NodeModes, RootModes>()>;

// Gather the root's tile index into a node's own mode order. A mode the root
// does not carry gets index 0 -- sound only if that mode has exactly one tile,
// which DagGraph::index_consistent() checks host-side before launching.
template <int NodeRank, std::size_t RootRank, int... G>
KOKKOS_FUNCTION Kokkos::Array<int, NodeRank> dag_node_index(
    const Kokkos::Array<int, RootRank>& root_idx,
    std::integer_sequence<int, G...>) {
  const int                    g[NodeRank] = {G...};
  Kokkos::Array<int, NodeRank> out{};
  for (int i = 0; i < NodeRank; ++i)
    out[i] = (g[i] >= 0) ? root_idx[static_cast<std::size_t>(g[i])] : 0;
  return out;
}

// --- binding ---------------------------------------------------------------
//
// Replace a slot operand's placeholder view with the real buffer. Everything
// else passes through: an input operand has nothing to bind, and a DAG has no
// nested subtrees to recurse into -- an operand is a leaf input or a NAME,
// never a subtree, which is the whole point of the flat form.
template <typename Op, typename Store>
KOKKOS_FUNCTION Op bind_operand(Op op, const Store& store) {
  if constexpr (has_node_tag_v<SlotTag, Op>)
    op.storage_ = store.template get<Op::SlotIdx>();
  return op;
}

// Plain function templates, not template lambdas: nvcc is unreliable with
// generic lambdas in device code.
template <typename Node, typename Store, std::size_t... Ks>
KOKKOS_FUNCTION void bind_combine_ops(Node& n, const Store& store,
                                      std::index_sequence<Ks...>) {
  ((n.operands.template get<Ks>() =
        bind_operand(n.operands.template get<Ks>(), store)),
   ...);
}

template <typename Node, typename Store>
KOKKOS_FUNCTION Node bind_slots(Node n, const Store& store) {
  if constexpr (has_node_tag_v<ContractionTag, Node>) {
    n.node_a = bind_operand(n.node_a, store);
    n.node_b = bind_operand(n.node_b, store);
  } else if constexpr (has_node_tag_v<CombineTag, Node>) {
    bind_combine_ops(n, store, std::make_index_sequence<Node::NumOps>{});
  }
  return n;
}

// A node's canonical output tile -- the shape its slot buffer really has. For a
// contraction the bundle's `.c` is in USER order while the C scratch is
// canonical, so this is not simply output_tile().
template <typename Node, typename Tile>
using dag_out_tile_t = operand_leaf_tile_t<Node, Tile>;

template <typename ES, typename Node, typename Tile>
using dag_eval_t = Evaluator<TeamPolicyTag<ES>, Node, Tile>;

// The buffer type node K's slot must hold: exactly what its evaluator would
// have carved for itself, taken from the evaluator rather than re-derived so
// the two cannot disagree.
template <typename ES, typename Node, typename Tile>
using dag_slot_view_t = typename dag_eval_t<ES, Node, Tile>::scratch_view_t;

// The LAST node in the list -- the one whose canonical output tile is the grid
// every other node's index is gathered from.
//
// A FREE alias, not a member one. A member alias template whose right-hand side
// does not depend on its own parameter gets resolved eagerly with the class,
// and `N - 1` underflows for the empty graph make_dag() returns. This form
// depends on NodesT, so it is instantiated only where it is named.
template <typename NodesT>
using dag_grid_node_t = tuple_element_t<tuple_size_v<NodesT> - 1, NodesT>;

}  // namespace Impl

// ---------------------------------------------------------------------------
// SlotHandle — what add() hands back: a compile-time slot index plus what is
// needed to mint an operand naming it.
//
// It carries the producing node's SHAPE because a consumer computes its k-tile
// counts against the operand's global extents, not the tile's.
// ---------------------------------------------------------------------------
template <std::size_t Idx, typename ScratchViewT, int Rank>
struct SlotHandle {
  static constexpr std::size_t slot_index = Idx;
  static constexpr int         rank       = Rank;
  using scratch_view_t                    = ScratchViewT;

  Kokkos::Array<int, Rank> shape;

  // Mint an operand naming this slot, labelled as the CONSUMER wants it.
  // Calling this twice with different labels over one handle is the relabel
  // mechanism: the buffer is named twice rather than copied.
  template <int32_t... Modes>
  auto as() const {
    return make_slot_node<Idx, Modes...>(ScratchViewT{}, shape);
  }
};

namespace Impl {

// The shape the adopting constructor wants. A combine owns one buffer PER
// OUTPUT and takes an array of its M consecutive slots; a contraction owns one
// and takes the bare view. The two constructors differ in type, and this is the
// one place that has to know.
template <typename Node, std::size_t Base, typename Store, std::size_t... Ms>
KOKKOS_FUNCTION auto dag_adopted_arg(const Store& store,
                                     std::index_sequence<Ms...>) {
  if constexpr (has_node_tag_v<CombineTag, Node>) {
    using View = std::decay_t<decltype(store.template get<Base>())>;
    return Kokkos::Array<View, sizeof...(Ms)>{
        store.template get<Base + Ms>()...};
  } else {
    return store.template get<Base>();
  }
}

// Evaluate node K: bind its slot operands, adopt its own slots, run it at the
// index gathered from the root's.
template <std::size_t K, typename ES, typename RootNode, typename ValueT,
          typename Team, typename NodesT, typename TilesT, typename Store,
          std::size_t RootR>
KOKKOS_FUNCTION void dag_run_node(const Team& team, const NodesT& nodes,
                                  const TilesT& tiles, const Store& store,
                                  const Kokkos::Array<int, RootR>& root_idx,
                                  ValueT* operand_base) {
  using Node = tuple_element_t<K, NodesT>;
  using Gather =
      dag_gather_seq_t<typename Node::modes_seq, typename RootNode::modes_seq>;
  constexpr std::size_t Base = dag_slot_base_v<NodesT, K>;
  constexpr std::size_t M    = output_arity<Node>::value;

  const auto idx   = dag_node_index<Node::Rank, RootR>(root_idx, Gather{});
  auto       bound = bind_slots(nodes.template get<K>(), store);
  // A FRESH arena per node is the reset: operand staging is dead at this node's
  // barrier, so every node starts at the region's base and the driver only ever
  // reserved the largest node's worth. `arena` is a local, so the cursor lives
  // in registers and no thread has to publish it.
  OperandArena<ValueT> arena{operand_base, 0};
  auto                 eval = make_evaluator<TeamPolicyTag<ES>>(
      bound, tiles.template get<K>(), team,
      dag_adopted_arg<Node, Base>(store, std::make_index_sequence<M>{}), arena);
  eval(team, idx);
  // Node K's result must be visible before any later node reads it. The
  // contraction evaluator ends its k-loop with a barrier, but the combine
  // evaluator does not, so this cannot be left to them.
  team.team_barrier();
}

template <typename ES, typename RootNode, typename ValueT, typename Team,
          typename NodesT, typename TilesT, typename Store, std::size_t RootR,
          std::size_t... Ks>
KOKKOS_FUNCTION void dag_run_all(const Team& team, const NodesT& nodes,
                                 const TilesT& tiles, const Store& store,
                                 const Kokkos::Array<int, RootR>& root_idx,
                                 ValueT*                          operand_base,
                                 std::index_sequence<Ks...>) {
  (dag_run_node<Ks, ES, RootNode, ValueT>(team, nodes, tiles, store, root_idx,
                                          operand_base),
   ...);
}

// Write one root SLOT out to its global view, through the existing store
// evaluator (Specialization 6) -- unchanged from the tree path.
//
// R is a slot index, so the producing node has to be looked up: a multi-output
// combine's outputs 0 and 1 are two different slots of the SAME node, sharing
// its tile, its modes and its output permutation.
template <std::size_t R, typename ES, typename RootNode, typename Team,
          typename NodesT, typename TilesT, typename Store, typename ViewT,
          std::size_t RootR>
KOKKOS_FUNCTION void dag_store_root(const Team& team, const TilesT& tiles,
                                    const Store&                     store,
                                    const Kokkos::Array<int, RootR>& root_idx,
                                    const ViewT&                     view) {
  constexpr std::size_t K = dag_slot_owner_v<NodesT, R>;
  using Node              = tuple_element_t<K, NodesT>;
  // Gathered, not passed through: an output need not be the grid node, and the
  // index it is written at is its own. Identity whenever it carries the grid
  // node's modes in the grid node's order, which is every case today.
  using Gather =
      dag_gather_seq_t<typename Node::modes_seq, typename RootNode::modes_seq>;
  const auto idx = dag_node_index<Node::Rank, RootR>(root_idx, Gather{});

  auto seval = make_evaluator<TeamPolicyTag<ES>>(
      make_interm_node(store.template get<R>()),
      output_tile(tiles.template get<K>()));
  seval(team, idx, view, output_perm_seq<Node>());
}

template <typename ES, typename RootNode, typename Team, typename NodesT,
          typename TilesT, typename Store, typename ViewArr, std::size_t RootR,
          std::size_t... Rs>
KOKKOS_FUNCTION void dag_store_roots(const Team& team, const TilesT& tiles,
                                     const Store&                     store,
                                     const Kokkos::Array<int, RootR>& root_idx,
                                     const ViewArr&                   views,
                                     std::index_sequence<Rs...>) {
  int i = 0;
  ((dag_store_root<Rs, ES, RootNode, Team, NodesT>(team, tiles, store, root_idx,
                                                   views[i++])),
   ...);
}

// The tile slot S's buffer has: its owning node's CANONICAL output tile, which
// every one of that node's outputs shares.
template <typename NodesT, typename TilesT, std::size_t S>
KOKKOS_FUNCTION auto dag_slot_tile(const TilesT& tiles) {
  constexpr std::size_t K = dag_slot_owner_v<NodesT, S>;
  return canonical_c_tile<tuple_element_t<K, NodesT>>(tiles.template get<K>());
}

template <typename NodesT, typename TilesT, std::size_t S>
KOKKOS_FUNCTION std::size_t dag_slot_elems(const TilesT& tiles) {
  return static_cast<std::size_t>(
      make_tile_layout(dag_slot_tile<NodesT, TilesT, S>(tiles), LayoutRight{})
          .size());
}

// A pool is as big as its largest occupant. Sizes come from the TILES, which
// are runtime values, so this is a runtime max over a compile-time membership
// test -- the plan is constexpr, the sizing is not.
template <typename NodesT, typename RootsSeq, typename TilesT, std::size_t S>
KOKKOS_FUNCTION void dag_pool_accum(std::size_t& m, std::size_t p,
                                    const TilesT& tiles) {
  if (dag_slot_pool_v<NodesT, RootsSeq, S> == p) {
    const std::size_t n = dag_slot_elems<NodesT, TilesT, S>(tiles);
    if (n > m) m = n;
  }
}

template <typename NodesT, typename RootsSeq, typename TilesT,
          std::size_t... Ss>
KOKKOS_FUNCTION std::size_t dag_pool_elems(const TilesT& tiles, std::size_t p,
                                           std::index_sequence<Ss...>) {
  std::size_t m = 0;
  (dag_pool_accum<NodesT, RootsSeq, TilesT, Ss>(m, p, tiles), ...);
  return m;
}

// Team scratch the POOLED store needs: the sum of the pool maxima, against
// slot_store_bytes()'s sum over every slot. This is the number the launcher
// requests, and it must be computed the same way on the host (to size the
// policy) and on the device (to carve), which is why both go through
// dag_pool_elems.
template <typename V, typename ES, typename NodesT, typename RootsSeq,
          typename TilesT, std::size_t... Ps>
std::size_t dag_pool_bytes(const TilesT& tiles, std::index_sequence<Ps...>) {
  using slots_seq = std::make_index_sequence<dag_num_slots<NodesT>()>;
  return (
      std::size_t{0} + ... +
      scratch_backing_t<V, ES>::shmem_size(
          dag_pool_elems<NodesT, RootsSeq, TilesT>(tiles, Ps, slots_seq{})));
}

// One node's operand staging, named so the max fold below can call it twice
// without spelling the evaluator type out each time.
// HOST-ONLY: operand_scratch_size_per_team is a host function, so naming it in
// device code buys warning #20011 and a runtime trap. The kernel never asks for
// a size -- it is handed the element count the host computed.
template <typename ES, typename NodesT, typename TilesT, std::size_t K>
std::size_t dag_node_operand_bytes(const TilesT& tiles) {
  return dag_eval_t<ES, tuple_element_t<K, NodesT>,
                    tuple_element_t<K, TilesT>>::
      operand_scratch_size_per_team(tiles.template get<K>());
}

// Operand staging for the whole graph: the LARGEST node's requirement, not the
// sum over nodes.
//
// Every node's operand buffers die at that node's team_barrier, so no two
// nodes' are ever live together, and the largest node's total is a hard lower
// bound -- one this reaches exactly.
//
// That is also why this is a max of SUMS rather than, as the slot store does, a
// sum of per-pool maxima. Pooling would pin each operand buffer to one offset
// across every node, and max-of-sums <= sum-of-maxes always: for a graph whose
// nodes want their large operand in different positions the gap approaches 2x.
// Operand live ranges are all [K,K], so the interval graph is a disjoint union
// of cliques and the optimal plan degenerates to "reset a cursor per node" --
// there is nothing for a colouring pass to discover.
template <typename V, typename ES, typename NodesT, typename TilesT,
          std::size_t... Ks>
std::size_t dag_operand_bytes_max(const TilesT& tiles,
                                  std::index_sequence<Ks...>) {
  std::size_t m = 0;
  ((m = dag_node_operand_bytes<ES, NodesT, TilesT, Ks>(tiles) > m
            ? dag_node_operand_bytes<ES, NodesT, TilesT, Ks>(tiles)
            : m),
   ...);
  return m;
}

// The same in ELEMENTS, for the device-side carve; rounded up so a partial
// element cannot truncate the region.
template <typename V, typename ES, typename NodesT, typename TilesT,
          std::size_t... Ks>
std::size_t dag_operand_elems(const TilesT& tiles, std::index_sequence<Ks...>) {
  return (dag_operand_bytes_max<V, ES, NodesT, TilesT>(
              tiles, std::index_sequence<Ks...>{}) +
          sizeof(V) - 1) /
         sizeof(V);
}

// What the arena carve actually CONSUMES from the team cursor, which is what
// the launcher must request -- not dag_operand_bytes_max, because shmem_size
// rounds up and the difference is real scratch. Host and device must agree here
// exactly: under-requesting by even one rounding step overruns the team's
// allocation and faults the kernel, and it faults ONLY on GPU, because Serial's
// 32 KB is slack enough to absorb it. Learned the hard way.
template <typename V, typename ES, typename NodesT, typename TilesT,
          std::size_t... Ks>
std::size_t dag_arena_bytes(const TilesT& tiles, std::index_sequence<Ks...>) {
  return scratch_backing_t<V, ES>::shmem_size(
      dag_operand_elems<V, ES, NodesT, TilesT>(tiles,
                                               std::index_sequence<Ks...>{}));
}

// Carve the pools, then place every slot in the one its live range earned it.
//
// Two bump allocations where there used to be one per slot: the POOLS come off
// the team cursor (so they are disjoint from each other and from the operand
// scratch the evaluators still carve), and the slots are then placed inside
// them at no further cost. A slot's view type is unchanged -- only where it
// points is.
template <typename V, typename ES, typename NodesT, typename RootsSeq,
          typename Team, typename TilesT, std::size_t... Ps>
KOKKOS_FUNCTION auto dag_carve_pools(const Team& team, const TilesT& tiles,
                                     std::index_sequence<Ps...>) {
  using slots_seq = std::make_index_sequence<dag_num_slots<NodesT>()>;
  // Braces, not parentheses: each backing view bumps the team cursor and the
  // elements of a braced-init-list are evaluated left to right, so the pools
  // land at predictable offsets. Same guarantee carve_slot_store relies on.
  return Kokkos::Array<V*, sizeof...(Ps)>{
      scratch_backing_t<V, ES>(
          team.team_scratch(0),
          dag_pool_elems<NodesT, RootsSeq, TilesT>(tiles, Ps, slots_seq{}))
          .data()...};
}

template <typename V, typename ES, typename NodesT, typename RootsSeq,
          typename Team, typename TilesT, std::size_t... Ss>
KOKKOS_FUNCTION auto dag_carve_store(const Team& team, const TilesT& tiles,
                                     std::index_sequence<Ss...>) {
  constexpr std::size_t P     = dag_pool_count_v<NodesT, RootsSeq>;
  const auto            pools = dag_carve_pools<V, ES, NodesT, RootsSeq>(
      team, tiles, std::make_index_sequence<P>{});
  const Kokkos::Array<V*, sizeof...(Ss)> base{
      pools[dag_slot_pool_v<NodesT, RootsSeq, Ss>]...};
  return place_slot_store<V, ES>(base, std::index_sequence<Ss...>{},
                                 dag_slot_tile<NodesT, TilesT, Ss>(tiles)...);
}

// Launch the whole graph. A FREE function template whose parameters are all
// TYPES -- deduced index_sequences rather than non-type packs -- because nvcc
// cannot generate a registration stub for an extended (KOKKOS_LAMBDA) lambda
// sitting inside a member function template with a non-type parameter pack. It
// fails only on the CUDA build, and only at stub generation, so the diagnostic
// names a synthesized .stub.c file rather than anything a reader wrote.
// Graph.hpp's execute_one_output_team is a free function for the same reason.
template <typename ValueType, typename ES, typename NodesT, typename TilesT,
          typename RootsSeq, typename... ViewTs>
int execute_dag_team(const NodesT& nodes, const TilesT& tiles,
                     std::size_t bytes, std::size_t arena_elems_in,
                     int team_size, RootsSeq roots, const ViewTs&... views) {
  using member_t               = team_member_t<ES>;
  constexpr std::size_t NNodes = tuple_size_v<NodesT>;
  using all_seq                = std::make_index_sequence<NNodes>;
  using slots_seq     = std::make_index_sequence<dag_num_slots<NodesT>()>;
  using GridNode      = dag_grid_node_t<NodesT>;
  constexpr int RootR = GridNode::Rank;

  // Copies, not references into the caller's graph: a device lambda capturing a
  // reference would dereference a host pointer on the device.
  const NodesT nd        = nodes;
  const TilesT td        = tiles;
  const auto   grid_node = nodes.template get<NNodes - 1>();
  const auto   grid_tile =
      canonical_c_tile<GridNode>(tiles.template get<NNodes - 1>());

  const int wk = work_items(grid_node, grid_tile);
  // Captured by value into the kernel: the arena's size is a HOST computation
  // (see dag_node_operand_bytes) and the device is only told how big it is.
  const std::size_t arena_elems = arena_elems_in;

  // Kokkos::AUTO is a poor default HERE and measurably so. It sizes the team
  // from OCCUPANCY alone, which is the wrong objective for a graph whose work
  // per team is a fixed small tile: on an H100 it picks 512 threads for a tile
  // of TE*N*N = 256 points, so every TeamVectorRange leaves half the team idle,
  // and the GEMM's TeamThreadRange (a few dozen work items) leaves far more.
  //
  // Measured on the SEM graph (E=2.5M): AUTO 25.94 ms, 512 -> 42.5, 256
  // -> 24.0, 128 -> 16.8, **64 -> 15.0**, 32 -> 18.3. The optimum is interior
  // and it is LOW-occupancy, which is the counterintuitive part and is worth
  // stating outright because "raise occupancy" is the reflex this defeats:
  //
  //   block   occupancy   warp-inst   fp32-inst   barrier stall   time
  //      64      21.6%      143.9 M      574 M        1.01       800.8 us
  //     128      42.8%      190.3 M      577 M        3.61       843.8 us
  //
  // (ncu, H100, E=65536.) Doubling the team doubles resident warps and buys
  // 0.5% more FLOATING-POINT work -- the fp32 count is flat because the useful
  // work is fixed by the tile. What the extra warps do is execute the loop
  // scaffolding and the index arithmetic anyway (+32% warp-instructions) and
  // participate in every one of the graph's ~14 team_barrier()s (3.6x the
  // barrier stall ratio). Occupancy is a measure of RESIDENT warps, not of
  // useful ones, and past the tile's parallelism the two diverge.
  //
  // Not hardcoded, because the right size depends on the tile and a library
  // cannot know the caller's: exposed as DagOutputs::team_size(n), defaulting
  // to AUTO so nothing changes for a caller that does not care.
  Kokkos::TeamPolicy<ES> policy =
      team_size > 0 ? Kokkos::TeamPolicy<ES>(wk, team_size)
                    : Kokkos::TeamPolicy<ES>(wk, Kokkos::AUTO);
  policy.set_scratch_size(0, Kokkos::PerTeam(static_cast<int>(bytes)));

  using ViewT = std::tuple_element_t<0, std::tuple<ViewTs...>>;
  static_assert((std::is_same_v<ViewT, ViewTs> && ...),
                "DagGraph::execute: output views must share one type");
  const Kokkos::Array<ViewT, sizeof...(ViewTs)> varr{views...};

  Kokkos::parallel_for(
      "TensorOperations::execute_dag", policy,
      KOKKOS_LAMBDA(const member_t& team) {
        const auto grid_idx = decode_tile_index<RootR>(
            static_cast<int>(team.league_rank()), grid_node.shape(), grid_tile);

        // Every node's output buffer, carved once and owned here rather than by
        // the evaluators that fill them.
        const auto store = dag_carve_store<ValueType, ES, NodesT, RootsSeq>(
            team, td, slots_seq{});

        // ONE operand-staging region for the whole graph, sized by the largest
        // node. Carved after the slot pools so both come off the team cursor in
        // a predictable order.
        ValueType* const operand_base =
            scratch_backing_t<ValueType, ES>(team.team_scratch(0), arena_elems)
                .data();

        dag_run_all<ES, GridNode, ValueType>(team, nd, td, store, grid_idx,
                                             operand_base, all_seq{});
        dag_store_roots<ES, GridNode, member_t, NodesT>(team, td, store,
                                                        grid_idx, varr, roots);
      });
  return wk;
}

}  // namespace Impl

// ---------------------------------------------------------------------------
// DagOutputs — a DagGraph plus the slots designated as graph outputs.
// ---------------------------------------------------------------------------
template <typename Dag, std::size_t... Roots>
struct DagOutputs {
  Dag dag;
  // Threads per team; <= 0 means Kokkos::AUTO. See execute_dag_team for why
  // AUTO is worth overriding on a graph whose tiles are small.
  int team = -1;

  DagOutputs team_size(int n) const { return {dag, n}; }

  // What the launch actually requests. These live HERE and not on DagGraph
  // because the liveness plan needs the root set: a designated output is read
  // after every node has run, so it is live to the end of the kernel and cannot
  // share a pool with anything.
  std::size_t slot_bytes() const {
    return dag.template pooled_slot_bytes<Roots...>();
  }
  // The arena, not the sum: this is what the launch requests.
  // dag.operand_bytes() is still the un-pooled bound if you want the ratio.
  std::size_t operand_bytes() const { return dag.arena_bytes(); }
  std::size_t scratch_bytes() const { return slot_bytes() + operand_bytes(); }

  // Pools the store carves, against dag.slot_bytes()'s one-per-slot. The ratio
  // of the two is what liveness bought, and both are printable before anything
  // runs -- scratch is the number that decides whether a tile size is viable at
  // all, so it must be answerable on the host.
  static constexpr std::size_t num_pools =
      Impl::dag_pool_count_v<typename Dag::nodes_type,
                             std::index_sequence<Roots...>>;

  template <typename ES, TensorLike... Ts>
  int execute(const TeamPolicyTag<ES>&, const Ts&... ts) const {
    return dag.template launch<ES, Roots...>(team, ts...);
  }
};

// ---------------------------------------------------------------------------
// DagGraph<ValueType, ExecSpace, DeviceTuple<Nodes...>, DeviceTuple<Tiles...>>
//
// Nodes and tiles live in DeviceTuples rather than std::tuples because the
// whole pack is captured into the kernel by value, and std::get is unreliable
// in CUDA device code -- the reason DeviceTuple exists at all.
// ---------------------------------------------------------------------------
template <typename ValueType, typename ExecSpace, typename NodesT,
          typename TilesT>
struct DagGraph {
  static constexpr std::size_t N = tuple_size_v<NodesT>;
  using nodes_type               = NodesT;
  using tiles_type               = TilesT;

  NodesT nodes;
  TilesT tiles;

  // Append a node. Returns the new graph followed by ONE HANDLE PER OUTPUT, so
  // DEFINITION ORDER IS THE TOPOLOGICAL ORDER -- the caller never writes an
  // index, and a node can only name slots that already exist.
  //
  //   auto [g, c]      = g0.add(contraction, tile);   // single output
  //   auto [g, f0, f1] = g0.add(two_out_combine, tile);
  //
  // The structured binding's arity is the node's output arity; getting it wrong
  // is a compile error at the binding, which is where a reader is looking.
  template <typename Node, typename Tile>
  auto add(const Node& node, const Tile& tile) const {
    constexpr std::size_t M =
        static_cast<std::size_t>(Impl::output_arity<Node>::value);
    return add_impl<Node, Tile>(node, tile, std::make_index_sequence<M>{});
  }

  // Designate graph outputs. Order matches the views passed to execute().
  template <typename... Handles>
  auto outputs(const Handles&...) const {
    return DagOutputs<DagGraph, Handles::slot_index...>{*this};
  }

  // The LAST node declared fixes the team's work decomposition: its canonical
  // output tile is the grid, and every other node's tile index is gathered from
  // that one by label. It must therefore be a graph output, and must carry
  // every mode that is tiled more than once.
  // See Impl::dag_grid_node_t for why it is not a member typedef.

  // Team scratch for the slot store WITHOUT liveness: one canonical output tile
  // per node OUTPUT, every one of them allocated for the whole kernel. This is
  // the upper bound the launcher used to request; it is kept because it is the
  // honest denominator for "what did liveness buy", and pooled_slot_bytes is
  // what actually gets requested. See DagOutputs::slot_bytes.
  std::size_t slot_bytes() const {
    return slot_bytes_impl(std::make_index_sequence<N>{});
  }

  // The same store with slots POOLED by live range: the sum of the pool maxima.
  // Depends on the root set -- a designated output outlives every node, so
  // which slots are roots changes the plan -- hence a template rather than the
  // plain member above.
  template <std::size_t... Roots>
  std::size_t pooled_slot_bytes() const {
    using RootsSeq = std::index_sequence<Roots...>;
    return Impl::dag_pool_bytes<ValueType, ExecSpace, NodesT, RootsSeq>(
        tiles,
        std::make_index_sequence<Impl::dag_pool_count_v<NodesT, RootsSeq>>{});
  }
  // What the evaluators still carve once their outputs are adopted -- the
  // complement of the store, never scratch_size_per_team(), which would charge
  // every output twice.
  // The un-pooled bound: every node's operand staging charged at once. Kept as
  // the honest denominator for what the arena buys; arena_bytes() is what the
  // launcher actually requests.
  std::size_t operand_bytes() const {
    return operand_bytes_impl(std::make_index_sequence<N>{});
  }

  // What the OPERAND ARENA needs: the largest single node's staging, because no
  // two nodes' operand buffers are ever live at once. Independent of the root
  // set, unlike the slot pools.
  std::size_t arena_bytes() const {
    return Impl::dag_arena_bytes<ValueType, ExecSpace, NodesT, TilesT>(
        tiles, std::make_index_sequence<N>{});
  }
  // The same region measured in elements, which is what the device carve wants.
  std::size_t arena_elems() const {
    return Impl::dag_operand_elems<ValueType, ExecSpace, NodesT, TilesT>(
        tiles, std::make_index_sequence<N>{});
  }
  std::size_t scratch_bytes() const { return slot_bytes() + operand_bytes(); }

  // PRECONDITION for the label-gathered index scheme, checked host-side where a
  // false answer is a clear error rather than a wrong number.
  //
  // Every node is evaluated at ONE index per team, gathered from the grid
  // node's by label. That is sound only if, for each node mode:
  //   * the grid node carries it, and tiles it identically (same tile extent
  //     and same global extent, hence the same tile count), or
  //   * the grid node does not carry it and it has exactly ONE tile, so index
  //     0 is the only index there is.
  // A mode tiled differently, or multi-tiled and absent from the grid, would
  // silently evaluate the node at the wrong tile.
  bool index_consistent() const {
    return index_consistent_impl(std::make_index_sequence<N>{});
  }

  // --- the launcher --------------------------------------------------------
  //
  // One kernel for the whole graph. Structurally Graph.hpp's
  // execute_one_output_team -- decode a tile index per team, evaluate, store --
  // with the recursion replaced by a fold over the flat node list.
  template <typename ES, std::size_t... Roots, typename... ViewTs>
  int launch(int team_size, const ViewTs&... views) const {
    static_assert(sizeof...(Roots) == sizeof...(ViewTs),
                  "DagGraph::execute needs one view per designated output");
    static_assert(std::is_same_v<ES, ExecSpace>,
                  "DagGraph::execute policy tag must match the graph's "
                  "execution space");
    assert(index_consistent() &&
           "DagGraph: a node's mode is either tiled differently from the grid "
           "node's or is multi-tiled and absent from it, so its tile index "
           "cannot be gathered from the grid node's. Retile so every shared "
           "mode matches and every unshared mode has one tile.");
    return Impl::execute_dag_team<ValueType, ES>(
        nodes, tiles, pooled_slot_bytes<Roots...>() + arena_bytes(),
        arena_elems(), team_size, std::index_sequence<Roots...>{}, views...);
  }

 private:
  // The handle pack is minted here rather than in add() so the M-fold expansion
  // has an index_sequence to expand over.
  template <typename Node, typename Tile, std::size_t... Ms>
  auto add_impl(const Node& node, const Tile& tile,
                std::index_sequence<Ms...>) const {
    using NewNodes             = decltype(tuple_append(nodes, node));
    using NewTiles             = decltype(tuple_append(tiles, tile));
    constexpr std::size_t Base = Impl::dag_num_slots<NodesT>();
    return std::make_tuple(
        DagGraph<ValueType, ExecSpace, NewNodes, NewTiles>{
            tuple_append(nodes, node), tuple_append(tiles, tile)},
        SlotHandle<Base + Ms, Impl::dag_slot_view_t<ExecSpace, Node, Tile>,
                   Node::Rank>{node.shape()}...);
  }

  // One slot per node OUTPUT; all of a node's outputs share its canonical
  // output tile, so the size is that tile's bytes times the arity.
  template <std::size_t... Ks>
  std::size_t slot_bytes_impl(std::index_sequence<Ks...>) const {
    return (std::size_t{0} + ... +
            (static_cast<std::size_t>(
                 Impl::output_arity<tuple_element_t<Ks, NodesT>>::value) *
             Impl::scratch_tile_bytes<ValueType, ExecSpace>(
                 Impl::canonical_c_tile<tuple_element_t<Ks, NodesT>>(
                     tiles.template get<Ks>()))));
  }
  template <std::size_t... Ks>
  std::size_t operand_bytes_impl(std::index_sequence<Ks...>) const {
    return (std::size_t{0} + ... +
            Impl::dag_eval_t<ExecSpace, tuple_element_t<Ks, NodesT>,
                             tuple_element_t<Ks, TilesT>>::
                operand_scratch_size_per_team(tiles.template get<Ks>()));
  }

  template <std::size_t K>
  bool node_index_consistent() const {
    using Node       = tuple_element_t<K, NodesT>;
    using GridNode   = Impl::dag_grid_node_t<NodesT>;
    using Gather     = Impl::dag_gather_seq_t<typename Node::modes_seq,
                                              typename GridNode::modes_seq>;
    constexpr auto g = Impl::seq_to_array(Gather{});

    const auto n_tile  = Impl::canonical_c_tile<Node>(tiles.template get<K>());
    const auto n_shape = nodes.template get<K>().shape();
    const auto g_tile =
        Impl::canonical_c_tile<GridNode>(tiles.template get<N - 1>());
    const auto g_shape = nodes.template get<N - 1>().shape();

    for (int i = 0; i < Node::Rank; ++i) {
      if (g[i] < 0) {
        if (Impl::tile_count_along(n_tile, i, n_shape[i]) != 1) return false;
      } else {
        if (n_tile.extent(i) != g_tile.extent(g[i])) return false;
        if (n_shape[i] != g_shape[g[i]]) return false;
      }
    }
    return true;
  }
  template <std::size_t... Ks>
  bool index_consistent_impl(std::index_sequence<Ks...>) const {
    return (node_index_consistent<Ks>() && ...);
  }
};

// An empty graph. ValueType is the scalar every slot is typed on; a DAG is
// homogeneous in it, since one store holds every node's output.
template <typename ValueType,
          typename ExecSpace = Kokkos::DefaultExecutionSpace>
auto make_dag() {
  return DagGraph<ValueType, ExecSpace, DeviceTuple<>, DeviceTuple<>>{{}, {}};
}

}  // namespace TensorOperations
