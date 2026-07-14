#pragma once
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace TensorOperations {

// A bound handle for the m-th output of a multi-output node. ops() returns one
// per output, so the structured binding lines up 1:1 with the views passed to
// execute (e.g. `auto [g, p0, p1] = graph.ops(combine2);`). It carries the node
// for introspection but is a terminal output only — not usable as an operand (a
// fused operand needs a single value).
template <typename Node, int M>
struct CombineOutputHandle {
  Node                 node;
  static constexpr int Rank        = Node::Rank;
  static constexpr int OutputIndex = M;
  using value_type                 = typename Node::value_type;
  using modes_seq                  = typename Node::modes_seq;
  KOKKOS_FUNCTION Kokkos::Array<int, Rank> shape() const {
    return node.shape();
  }
};

namespace Impl {

// Detect a ContractionTag node (the only node type with operands to recurse
// into); all other node types are leaves.
template <typename Node>
using is_contraction = has_node_tag<ContractionTag, Node>;

// Number of output tensors a node emits: 1 for every node except a multi-output
// combine, which exposes `NumOut`.
template <typename Node, typename = void>
struct output_arity : std::integral_constant<int, 1> {};
template <typename Node>
struct output_arity<Node, std::void_t<decltype(Node::NumOut)>>
    : std::integral_constant<int, Node::NumOut> {};

// Per-node output arities of a stored-node tuple, their sum, and the running
// view offset before node I — used to slice the flat view pack per node.
template <typename Tuple, std::size_t... I>
constexpr std::array<int, sizeof...(I)> node_arities_impl(
    std::index_sequence<I...>) {
  return {output_arity<std::tuple_element_t<I, Tuple>>::value...};
}
template <typename Tuple>
constexpr auto node_arities() {
  return node_arities_impl<Tuple>(
      std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}
template <typename Tuple>
constexpr int total_arity() {
  int s = 0;
  for (int a : node_arities<Tuple>()) s += a;
  return s;
}
template <typename Tuple, std::size_t I>
constexpr int node_offset() {
  const auto a   = node_arities<Tuple>();
  int        off = 0;
  // int loop index: `k < I` on the I == 0 instantiation trips nvcc's
  // pointless-unsigned-comparison warning (#186) in every including TU.
  for (int k = 0; k < static_cast<int>(I); ++k) off += a[k];
  return off;
}

// Number of ContractionTag nodes in a node's subtree (itself + both operands).
// Fixes the pre-order layout of a flat, per-contraction tile list.
template <typename Node>
constexpr std::size_t count_contractions() {
  if constexpr (is_contraction<Node>::value)
    return 1 + count_contractions<typename Node::node_a_type>() +
           count_contractions<typename Node::node_b_type>();
  else
    return 0;
}

// Assemble the nested per-operand tile structure the contraction evaluator
// consumes from a FLAT, pre-order list of per-contraction Tile<A,B,C> bundles
// (one bundle per ContractionTag node; order: node, then A-subtree, then
// B-subtree). Each contraction's own bundle.c is its output tile; an operand
// that is itself a contraction contributes a nested Tile whose bundle governs,
// so the parent bundle's slot for that operand must match (checked below). Leaf
// (input) operands take the parent bundle's .a/.b tile directly.
template <std::size_t Offset, typename Node, typename Tuple>
auto assemble_tile(const Node& node, const Tuple& tiles);

template <std::size_t Offset, typename OpNode, typename LeafTile,
          typename Tuple>
auto assemble_operand(const OpNode& op, const LeafTile& leaf,
                      const Tuple& tiles) {
  if constexpr (is_contraction<OpNode>::value) {
    auto nested = assemble_tile<Offset>(op, tiles);
    static_assert(
        std::is_same_v<LeafTile, decltype(output_tile(nested))>,
        "flat tile list: a parent's operand tile must equal the fused "
        "sub-contraction's output tile (bundle.c)");
    return nested;
  } else {
    return leaf;
  }
}

template <std::size_t Offset, typename Node, typename Tuple>
auto assemble_tile(const Node& node, const Tuple& tiles) {
  static_assert(is_contraction<Node>::value,
                "assemble_tile expects a contraction node");
  const auto&           bundle = std::get<Offset>(tiles);
  constexpr std::size_t offB =
      Offset + 1 + count_contractions<typename Node::node_a_type>();
  auto a_spec = assemble_operand<Offset + 1>(node.node_a, bundle.a, tiles);
  auto b_spec = assemble_operand<offB>(node.node_b, bundle.b, tiles);
  return Tile<decltype(a_spec), decltype(b_spec), decltype(bundle.c)>{
      a_spec, b_spec, bundle.c};
}

// Row-major decode of a linear work-item index into per-mode tile indices.
// Returns 0-based tile coordinates (not element offsets) along each free mode.
template <int Rank, typename Tile>
KOKKOS_FUNCTION Kokkos::Array<int, Rank> decode_tile_index(
    int idx, const Kokkos::Array<int, Rank>& shape, const Tile& tile) {
  Kokkos::Array<int, Rank> tidx{};
  for (int d = Rank - 1; d >= 0; --d) {
    int n   = (shape[d] + tile.extent(d) - 1) / tile.extent(d);
    tidx[d] = static_cast<int>(idx % static_cast<int>(n));
    idx /= static_cast<int>(n);
  }
  return tidx;
}

template <typename Node, typename Tile>
int work_items(const Node& n, const Tile& tile) {
  constexpr int R = Node::Rank;
  static_assert(Tile::rank >= R,
                "tile rank must cover at least the node's output (free) modes");
  const auto out   = n.shape();
  int        total = 1;
  for (int d = 0; d < R; ++d) {
    const int t = tile.extent(d);
    total *= static_cast<int>((out[d] + t - 1) / t);  // ceil-div
  }
  return total;
}

// The output permutation (permC) as a full-rank sequence: maps canonical output
// mode i -> its position in the user output order. The store evaluator subviews
// the native output on the ordered path and reorders it into canonical order
// via this sequence. Identity seq for canonical contractions and
// non-contraction outputs (so the store's scatter + reorder collapse to
// no-ops).
template <typename NodeType>
KOKKOS_FUNCTION auto output_perm_seq() {
  if constexpr (is_contraction<NodeType>::value)
    return typename NodeType::permC_seq{};
  else
    return std::make_integer_sequence<int, NodeType::Rank>{};
}

// The output (C) tile in the node's canonical (freeA ++ freeB) mode order:
// the user-order output tile gathered by permC. Identity permC (canonical
// contractions, non-contraction outputs) passes the tile through unchanged.
template <typename NodeType, typename Tile>
KOKKOS_FUNCTION auto canonical_c_tile(const Tile& tile) {
  return reorder_tile_value(output_tile(tile), output_perm_seq<NodeType>());
}

// Launches one team-policy kernel for one output node (one view per output
// tensor) and returns the work-item count.
//
// node.shape() is canonical, so the output tile is presented canonically for
// the contraction evaluator (which canonicalizes A/B/C internally from the same
// user-order tile bundle). The store, however, tiles the NATIVE user output
// (u_tile) so subview_tile stays on the compile-time-ordered path, then
// reorders that ordered subview into canonical order before writing.
//
// One launcher covers every output arity: as_result_array normalizes the
// evaluator's result (a single interm handle, or a Kokkos::Array of them for a
// multi-output combine) to M components, and the constexpr-M store loop fully
// unrolls — for M == 1 this is exactly the former single-output kernel. Kept a
// single PLAIN function template (no SFINAE / no in-lambda `if constexpr`):
// nvcc's extended-lambda tagging chokes both on `enable_if` non-type params on
// the enclosing template AND on a lambda first-capturing inside an internal
// `if constexpr`. One kernel is instantiated per node type, as before.
template <typename NodeType, typename Tile, typename... ViewTs>
int execute_one_output_team(const NodeType& node, const Tile& tile,
                            const ViewTs&... views) {
  using exec_space = typename NodeType::exec_space;
  using member_t   = typename Kokkos::TeamPolicy<exec_space>::member_type;
  using eval_type  = Evaluator<TeamPolicyTag<exec_space>, NodeType, Tile>;
  constexpr int M  = output_arity<NodeType>::value;
  static_assert(sizeof...(ViewTs) == static_cast<std::size_t>(M),
                "execute: one view must be supplied per node output");

  const auto c_tile = canonical_c_tile<NodeType>(tile);
  const auto u_tile = output_tile(tile);
  const int  wk     = work_items(node, c_tile);
  const int  bytes  = eval_type::scratch_size_per_team(tile);

  Kokkos::TeamPolicy<exec_space> policy(static_cast<int>(wk), Kokkos::AUTO);
  policy.set_scratch_size(0, Kokkos::PerTeam(bytes));

  // The M output views share a type (all outputs share the node's modes), so
  // pack them into an array captured by value in the kernel.
  using ViewT = std::tuple_element_t<0, std::tuple<ViewTs...>>;
  static_assert((std::is_same_v<ViewT, ViewTs> && ...),
                "execute: all outputs of a multi-output node share its modes; "
                "the corresponding views must have one common type");
  const Kokkos::Array<ViewT, M> varr{views...};

  Kokkos::parallel_for(
      "TensorOperations::execute_team", policy,
      KOKKOS_LAMBDA(const member_t& team) {
        const auto shape      = node.shape();
        const auto c_tile_idx = decode_tile_index<NodeType::Rank>(
            static_cast<int>(team.league_rank()), shape, c_tile);
        auto eval = make_evaluator<TeamPolicyTag<exec_space>>(node, tile, team);
        const auto interms = as_result_array(eval(team, c_tile_idx));
        for (int m = 0; m < M; ++m) {
          auto seval =
              make_evaluator<TeamPolicyTag<exec_space>>(interms[m], u_tile);
          seval(team, c_tile_idx, varr[m], output_perm_seq<NodeType>());
        }
      });
  return wk;
}

// Expand a node into its return handles for ops(): the node itself for a
// single-output node (so it stays usable as a downstream operand), or one
// CombineOutputHandle per output for a multi-output node.
template <typename Node>
auto expand_output_handles(const Node& n) {
  constexpr int M = output_arity<Node>::value;
  if constexpr (M == 1) {
    return std::make_tuple(n);
  } else {
    return [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
      return std::make_tuple(
          CombineOutputHandle<Node, static_cast<int>(Ms)>{n}...);
    }(std::make_index_sequence<static_cast<std::size_t>(M)>{});
  }
}

// Run output node I: slice its `output_arity` views out of the flat view tuple
// (starting at its prefix-sum offset) and launch its kernel.
template <std::size_t I, typename OutputsTuple, typename ViewsTuple,
          typename Tile>
int run_output(const OutputsTuple& outs, const ViewsTuple& views,
               const Tile& tile) {
  using Node           = std::tuple_element_t<I, OutputsTuple>;
  constexpr int Offset = node_offset<OutputsTuple, I>();
  constexpr int Count  = output_arity<Node>::value;
  return [&]<std::size_t... Ks>(std::index_sequence<Ks...>) {
    return execute_one_output_team(std::get<I>(outs), tile,
                                   std::get<Offset + Ks>(views)...);
  }(std::make_index_sequence<static_cast<std::size_t>(Count)>{});
}

}  // namespace Impl

// Compile-time computational graph builder.
// OutputTuple tracks the nodes produced by the most recent ops() call.
template <typename OutputTuple = std::tuple<>>
struct Graph {
  Graph() = default;

  // Register any mix of node handles (InputNode, ContractionNode, etc.).
  // Returns a flattened tuple {new Graph, handle_0, ...} so callers bind the
  // graph and every output in one structured-binding step:
  //   auto [g1, T1, T2] = g.ops(con1, con2);
  // A multi-output node is EXPANDED into one handle per output, so a 2-output
  // combine yields two handles:  auto [g1, p0, p1] = g.ops(combine2);
  // The new Graph stores the same (unexpanded) node tuple as its outputs — each
  // node is evaluated once; the returned handles are lightweight (no data
  // copy).
  template <typename... Nodes>
  auto ops(Nodes&&... nodes) const {
    auto node_tuple = std::make_tuple(nodes...);  // stored: evaluated once each
    using G         = Graph<decltype(node_tuple)>;
    auto handles    = std::tuple_cat(Impl::expand_output_handles(nodes)...);
    return std::tuple_cat(std::make_tuple(G(node_tuple)), std::move(handles));
  }

  // Evaluate the graph and write each output into the corresponding view.
  // Exactly one view per output (= sum of the nodes' output arities). Views are
  // consumed positionally in node order, so they line up 1:1 with the handles
  // returned by ops().
  template <typename ES, typename Tile, TensorLike... Ts>
  int execute(const TeamPolicyTag<ES>&, const Tile& tile,
              const Ts&... ts) const {
    auto                  views = std::forward_as_tuple(ts...);
    constexpr std::size_t N     = std::tuple_size_v<OutputTuple>;
    static_assert(
        Impl::total_arity<OutputTuple>() == static_cast<int>(sizeof...(Ts)),
        "execute needs one view per output (sum of output arities)");
    int wk_items = 0;
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      ((wk_items += Impl::run_output<I>(outputs, views, tile)), ...);
    }(std::make_index_sequence<N>{});

    return wk_items;
  }

  // Flat tile-list form for fused chains: one Tile<A,B,C> bundle per
  // contraction node, in pre-order (node, then A-subtree, then B-subtree). The
  // bundles are assembled into the nested per-operand tile the evaluator
  // consumes, so the caller writes a flat list rather than a hand-nested Tile.
  // Single output.
  template <typename ES, typename... Bundles, TensorLike... Ts>
  int execute(const TeamPolicyTag<ES>&, const std::tuple<Bundles...>& tiles,
              const Ts&... ts) const {
    static_assert(std::tuple_size_v<OutputTuple> == 1,
                  "flat tile-list execute supports a single output node");
    static_assert(sizeof...(Ts) == 1, "exactly one output view is required");
    const auto& out    = std::get<0>(outputs);
    const auto  nested = Impl::assemble_tile<0>(out, tiles);
    auto        views  = std::tie(ts...);
    return Impl::execute_one_output_team(out, nested, std::get<0>(views));
  }

 private:
  template <typename>
  friend struct Graph;

  explicit Graph(OutputTuple o) : outputs(std::move(o)) {}

  OutputTuple outputs;
};

inline Graph<> make_graph() { return {}; }

}  // namespace TensorOperations
