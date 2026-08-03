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
// Here every node owns one slot in a driver-carved store. Node K writes into
// slot K (adopting it), and any later node needing K's result NAMES slot K as
// an operand. One buffer, one evaluation, N readers.
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
// V1 RESTRICTIONS, each static_asserted or checked where it is relied on:
//   * every node emits ONE output (NumOut == 1), since a node owns one slot;
//   * a node's modes that the root does not carry must be single-tiled;
//   * shared modes must be tiled identically in node and root.
// ---------------------------------------------------------------------------

namespace Impl {

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
// OUTPUT and takes an array; a contraction takes the bare view. v1 pins
// NumOut == 1, so the array is always one element -- but the two constructors
// still differ in type, and this is the one place that has to know.
template <typename Node, typename View>
KOKKOS_FUNCTION auto dag_adopted_arg(const View& v) {
  if constexpr (has_node_tag_v<CombineTag, Node>)
    return Kokkos::Array<View, 1>{v};
  else
    return v;
}

// Evaluate node K: bind its slot operands, adopt its own slot, run it at the
// index gathered from the root's.
template <std::size_t K, typename ES, typename RootNode, typename Team,
          typename NodesT, typename TilesT, typename Store, std::size_t RootR>
KOKKOS_FUNCTION void dag_run_node(const Team& team, const NodesT& nodes,
                                  const TilesT& tiles, const Store& store,
                                  const Kokkos::Array<int, RootR>& root_idx) {
  using Node = tuple_element_t<K, NodesT>;
  using Gather =
      dag_gather_seq_t<typename Node::modes_seq, typename RootNode::modes_seq>;

  const auto idx   = dag_node_index<Node::Rank, RootR>(root_idx, Gather{});
  auto       bound = bind_slots(nodes.template get<K>(), store);
  auto       eval  = make_evaluator<TeamPolicyTag<ES>>(
      bound, tiles.template get<K>(), team,
      dag_adopted_arg<Node>(store.template get<K>()));
  eval(team, idx);
  // Node K's result must be visible before any later node reads it. The
  // contraction evaluator ends its k-loop with a barrier, but the combine
  // evaluator does not, so this cannot be left to them.
  team.team_barrier();
}

template <typename ES, typename RootNode, typename Team, typename NodesT,
          typename TilesT, typename Store, std::size_t RootR, std::size_t... Ks>
KOKKOS_FUNCTION void dag_run_all(const Team& team, const NodesT& nodes,
                                 const TilesT& tiles, const Store& store,
                                 const Kokkos::Array<int, RootR>& root_idx,
                                 std::index_sequence<Ks...>) {
  (dag_run_node<Ks, ES, RootNode>(team, nodes, tiles, store, root_idx), ...);
}

// Write one root slot out to its global view, through the existing store
// evaluator (Specialization 6) -- unchanged from the tree path.
template <std::size_t R, typename ES, typename Team, typename NodesT,
          typename TilesT, typename Store, typename ViewT, std::size_t RootR>
KOKKOS_FUNCTION void dag_store_root(const Team& team, const TilesT& tiles,
                                    const Store&                     store,
                                    const Kokkos::Array<int, RootR>& root_idx,
                                    const ViewT&                     view) {
  using Node = tuple_element_t<R, NodesT>;
  static_assert(
      output_arity<Node>::value == 1,
      "DagGraph: a node owns exactly one slot, so a multi-output combine "
      "cannot yet be a DAG node. Register it through the tree driver instead");

  auto seval = make_evaluator<TeamPolicyTag<ES>>(
      make_interm_node(store.template get<R>()),
      output_tile(tiles.template get<R>()));
  seval(team, root_idx, view, output_perm_seq<Node>());
}

template <typename ES, typename Team, typename NodesT, typename TilesT,
          typename Store, typename ViewArr, std::size_t RootR,
          std::size_t... Rs>
KOKKOS_FUNCTION void dag_store_roots(const Team& team, const TilesT& tiles,
                                     const Store&                     store,
                                     const Kokkos::Array<int, RootR>& root_idx,
                                     const ViewArr&                   views,
                                     std::index_sequence<Rs...>) {
  int i = 0;
  ((dag_store_root<Rs, ES, Team, NodesT>(team, tiles, store, root_idx,
                                         views[i++])),
   ...);
}

// Carve one slot per node, sized by that node's CANONICAL output tile.
template <typename V, typename ES, typename NodesT, typename Team,
          typename TilesT, std::size_t... Ks>
KOKKOS_FUNCTION auto dag_carve_store(const Team& team, const TilesT& tiles,
                                     std::index_sequence<Ks...>) {
  return carve_slot_store<V, ES>(team,
                                 canonical_c_tile<tuple_element_t<Ks, NodesT>>(
                                     tiles.template get<Ks>())...);
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
                     std::size_t bytes, RootsSeq roots,
                     const ViewTs&... views) {
  using member_t               = team_member_t<ES>;
  constexpr std::size_t NNodes = tuple_size_v<NodesT>;
  using all_seq                = std::make_index_sequence<NNodes>;
  using GridNode               = dag_grid_node_t<NodesT>;
  constexpr int RootR          = GridNode::Rank;

  // Copies, not references into the caller's graph: a device lambda capturing a
  // reference would dereference a host pointer on the device.
  const NodesT nd        = nodes;
  const TilesT td        = tiles;
  const auto   grid_node = nodes.template get<NNodes - 1>();
  const auto   grid_tile =
      canonical_c_tile<GridNode>(tiles.template get<NNodes - 1>());

  const int wk = work_items(grid_node, grid_tile);

  Kokkos::TeamPolicy<ES> policy(wk, Kokkos::AUTO);
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
        const auto store =
            dag_carve_store<ValueType, ES, NodesT>(team, td, all_seq{});

        dag_run_all<ES, GridNode>(team, nd, td, store, grid_idx, all_seq{});
        dag_store_roots<ES, member_t, NodesT>(team, td, store, grid_idx, varr,
                                              roots);
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

  template <typename ES, TensorLike... Ts>
  int execute(const TeamPolicyTag<ES>&, const Ts&... ts) const {
    return dag.template launch<ES, Roots...>(ts...);
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

  NodesT nodes;
  TilesT tiles;

  // Append a node. Returns the new graph and a handle naming its slot, so
  // DEFINITION ORDER IS THE TOPOLOGICAL ORDER -- the caller never writes an
  // index, and a node can only name slots that already exist.
  template <typename Node, typename Tile>
  auto add(const Node& node, const Tile& tile) const {
    static_assert(Impl::output_arity<Node>::value == 1,
                  "DagGraph: a node owns exactly one slot, so a multi-output "
                  "combine cannot yet be a DAG node");
    using NewNodes = decltype(tuple_append(nodes, node));
    using NewTiles = decltype(tuple_append(tiles, tile));
    using Handle =
        SlotHandle<N, Impl::dag_slot_view_t<ExecSpace, Node, Tile>, Node::Rank>;
    return std::make_tuple(
        DagGraph<ValueType, ExecSpace, NewNodes, NewTiles>{
            tuple_append(nodes, node), tuple_append(tiles, tile)},
        Handle{node.shape()});
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

  // Team scratch for the slot store: one canonical output tile per node.
  std::size_t slot_bytes() const {
    return slot_bytes_impl(std::make_index_sequence<N>{});
  }
  // What the evaluators still carve once their outputs are adopted -- the
  // complement of the store, never scratch_size_per_team(), which would charge
  // every output twice.
  std::size_t operand_bytes() const {
    return operand_bytes_impl(std::make_index_sequence<N>{});
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
  int launch(const ViewTs&... views) const {
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
        nodes, tiles, scratch_bytes(), std::index_sequence<Roots...>{},
        views...);
  }

 private:
  template <std::size_t... Ks>
  std::size_t slot_bytes_impl(std::index_sequence<Ks...>) const {
    return (std::size_t{0} + ... +
            Impl::scratch_tile_bytes<ValueType, ExecSpace>(
                Impl::canonical_c_tile<tuple_element_t<Ks, NodesT>>(
                    tiles.template get<Ks>())));
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
