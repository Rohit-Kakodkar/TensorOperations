#pragma once
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

}  // namespace Impl

// Compile-time computational graph builder.
// OutputTuple tracks the nodes produced by the most recent ops() call.
template <typename OutputTuple = std::tuple<>>
struct Graph {
  OutputTuple outputs;

  // Register any mix of node handles (InputNode, ContractionNode, etc.).
  // Returns a new Graph storing the outputs and the same tuple for structured
  // binding decomposition.
  template <typename... Nodes>
  auto ops(Nodes&&... nodes) const {
    auto node_tuple = std::make_tuple(nodes...);
    return std::pair{Graph<decltype(node_tuple)>{node_tuple},
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

  // Work items needed to produce one output node `n` with `tile`: the product
  // over all participating modes (free ++ contracted) of the ceil-division of
  // the mode extent by the tile extent. One work item == one evaluator
  // tile-step (output tiles x contracted blocks).
  template <typename Node, typename Tile>
  static std::size_t work_items(const Node& n, const Tile& tile) {
    constexpr int P = Impl::participating_rank<Node>();
    static_assert(Tile::rank == P,
                  "tile rank must equal the node's participating-mode count "
                  "(output Rank + NumContracted)");
    const auto  ext   = Impl::participating_extents(n);
    std::size_t total = 1;
    for (int d = 0; d < P; ++d) {
      const int t = tile.extent(d);
      total *= static_cast<std::size_t>((ext[d] + t - 1) / t);  // ceil-div
    }
    return total;
  }

  // Total work items across all outputs for the given policy and tiling shape.
  // PolicyTag is part of the interface and will drive evaluator selection
  // later; it does not affect the count yet.
  template <typename PolicyTag, typename Tile, TensorLike... Ts>
  std::size_t execute(const PolicyTag&, const Tile& tile,
                      const Ts&... ts) const {
    const auto wk_items = [&]<std::size_t... I>(std::index_sequence<I...>) {
      return (std::size_t{0} + ... + work_items(std::get<I>(outputs), tile));
    }(std::make_index_sequence<std::tuple_size_v<OutputTuple>>{});

    Kokkos::parallel_for("GraphExecution", Kokkos::RangePolicy<>(0, wk_items),
                         KOKKOS_LAMBDA(std::size_t){
                             // Placeholder for actual execution logic.
                         });

    return wk_items;
  }
};

inline Graph<> make_graph() { return {}; }

}  // namespace TensorOperations
