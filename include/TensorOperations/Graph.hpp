#pragma once
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <array>
#include <cstddef>
#include <tuple>
#include <utility>

namespace TensorOperations {

namespace Impl {

// Detect a ContractionTag node (the only node type with operands to recurse
// into); all other node types are leaves.
template <typename Node>
struct is_contraction : std::false_type {};
template <typename NA, typename NB, typename R, typename S, typename ES,
          typename H>
struct is_contraction<NodeHandle<ContractionTag, NA, NB, R, S, ES, H>>
    : std::true_type {};

// Total contracted modes in a node subtree, summed over every contraction node.
// Leaves (InputTag / IntermTag) contribute 0.
template <typename Node, bool = is_contraction<Node>::value>
struct contracted_count : std::integral_constant<int, 0> {};  // leaf
template <typename Node>
struct contracted_count<Node, true>
    : std::integral_constant<
          int, Node::NumContracted +
                   contracted_count<typename Node::node_a_type>::value +
                   contracted_count<typename Node::node_b_type>::value> {};

// Number of participating modes of a node's own op: free (output) modes plus
// contracted modes. Leaves (input/interm) have no contracted modes.
template <typename Node>
constexpr int participating_rank() {
  if constexpr (is_contraction<Node>::value)
    return Node::Rank + Node::NumContracted;
  else
    return Node::Rank;
}

// Build the participating-mode extents [free.., contracted..] for a node.
// Free-mode extents come from the node's own shape(). Contracted-mode extents
// are recovered from operand A under the GEMM/Option-A convention (A is stored
// [freeA.., contracted..], so the contracted extents are the last NumContracted
// entries of node_a.shape()). Extents are runtime (Kokkos View extents).
template <typename Node>
std::array<int, participating_rank<Node>()> participating_extents(
    const Node& n) {
  constexpr int      P = participating_rank<Node>();
  std::array<int, P> ext{};
  const auto         out = n.shape();
  for (int d = 0; d < Node::Rank; ++d) ext[d] = out[d];
  if constexpr (is_contraction<Node>::value) {
    constexpr int NumK  = Node::NumContracted;
    constexpr int FreeA = Node::node_a_type::Rank - NumK;
    const auto    a     = n.node_a.shape();
    for (int i = 0; i < NumK; ++i) ext[Node::Rank + i] = a[FreeA + i];
  }
  return ext;
}

// Row-major decode of a linear work-item index into per-mode tile indices.
// Returns 0-based tile coordinates (not element offsets) along each free mode.
template <int Rank, typename Tile>
KOKKOS_FUNCTION Kokkos::Array<int, Rank> decode_tile_index(
    std::size_t idx, const Kokkos::Array<int, Rank>& shape, const Tile& tile) {
  Kokkos::Array<int, Rank> tidx{};
  for (int d = Rank - 1; d >= 0; --d) {
    int n   = (shape[d] + tile.extent(d) - 1) / tile.extent(d);
    tidx[d] = static_cast<int>(idx % static_cast<std::size_t>(n));
    idx /= static_cast<std::size_t>(n);
  }
  return tidx;
}

}  // namespace Impl

// Compile-time computational graph builder.
// OutputTuple tracks the nodes produced by the most recent ops() call.
template <typename OutputTuple = std::tuple<>>
struct Graph {
  Graph() = default;

  // Register any mix of node handles (InputNode, ContractionNode, etc.).
  // Returns a new Graph storing the outputs and the same tuple for structured
  // binding decomposition.
  template <typename... Nodes>
  auto ops(Nodes&&... nodes) const {
    auto node_tuple = std::make_tuple(nodes...);
    return std::pair{Graph<decltype(node_tuple)>(node_tuple),
                     std::move(node_tuple)};
  }

  // Number of free indices: the modes that survive to the result, i.e. the sum
  // of each output node's Rank. Pure compile-time arithmetic.
  static constexpr int num_free_indices() {
    return []<std::size_t... I>(std::index_sequence<I...>) {
      return (0 + ... + std::tuple_element_t<I, OutputTuple>::Rank);
    }(std::make_index_sequence<std::tuple_size_v<OutputTuple>>{});
  }

  // Number of contraction indices: the modes summed over across every
  // contraction node reachable from the outputs, summed across all outputs.
  static constexpr int num_contraction_indices() {
    return []<std::size_t... I>(std::index_sequence<I...>) {
      return (
          0 + ... +
          Impl::contracted_count<std::tuple_element_t<I, OutputTuple>>::value);
    }(std::make_index_sequence<std::tuple_size_v<OutputTuple>>{});
  }

  // Total work items across all outputs for the given tiling shape.
  template <typename Tile>
  std::size_t tile_count(const Tile& tile) const {
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
      return (std::size_t{0} + ... + work_items(std::get<I>(outputs), tile));
    }(std::make_index_sequence<std::tuple_size_v<OutputTuple>>{});
  }

  // Evaluate the graph and write each output into the corresponding view.
  // Exactly one view must be provided per output node.
  template <typename PolicyTag, typename Tile, TensorLike... Ts>
  std::size_t execute(const PolicyTag&, const Tile& tile,
                      const Ts&... ts) const {
    static_assert(sizeof...(Ts) == std::tuple_size_v<OutputTuple>,
                  "Number of runtime tensor arguments must match number of "
                  "output nodes in the graph");
    static_assert(std::is_same_v<PolicyTag, RangePolicyTag>,
                  "Currently only RangePolicyTag is supported");

    const auto wk_items = tile_count(tile);

    auto                  ts_tuple = std::tie(ts...);
    constexpr std::size_t N        = std::tuple_size_v<OutputTuple>;
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      (execute_one_output(std::get<I>(outputs), std::get<I>(ts_tuple), tile),
       ...);
    }(std::make_index_sequence<N>{});

    return wk_items;
  }

 private:
  template <typename>
  friend struct Graph;

  explicit Graph(OutputTuple o) : outputs(std::move(o)) {}

  OutputTuple outputs;

  template <typename Node, typename Tile>
  static std::size_t work_items(const Node& n, const Tile& tile) {
    constexpr int R = Node::Rank;
    static_assert(
        Tile::rank >= R,
        "tile rank must cover at least the node's output (free) modes");
    const auto  out   = n.shape();
    std::size_t total = 1;
    for (int d = 0; d < R; ++d) {
      const int t = tile.extent(d);
      total *= static_cast<std::size_t>((out[d] + t - 1) / t);  // ceil-div
    }
    return total;
  }

  // Launch a Kokkos::RangePolicy kernel for a single output node, writing
  // results into `view`. One kernel per output avoids tuple access on device.
  template <typename NodeType, typename ViewT, typename Tile>
  static void execute_one_output(const NodeType& node, const ViewT& view,
                                 const Tile& tile) {
    const std::size_t wk = work_items(node, tile);
    Kokkos::parallel_for(
        "TensorOperations::execute",
        Kokkos::RangePolicy<typename NodeType::exec_space>(0, wk),
        KOKKOS_LAMBDA(std::size_t local_idx) {
          const auto shape = node.shape();
          const auto c_tile_idx =
              Impl::decode_tile_index<NodeType::Rank>(local_idx, shape, tile);

          auto eval   = make_evaluator<RangePolicyTag>(node, tile);
          auto interm = eval(c_tile_idx);

          auto seval = make_evaluator<RangePolicyTag>(interm, tile);
          seval(c_tile_idx, view);
        });
  }
};

inline Graph<> make_graph() { return {}; }

}  // namespace TensorOperations
