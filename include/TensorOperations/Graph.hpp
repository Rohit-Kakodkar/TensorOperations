#pragma once
#include <TensorOperations/NodeHandle.hpp>
#include <tuple>
#include <utility>

namespace TensorOperations {

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

  // Placeholder: triggers execution over stored outputs.
  // No-op until the kernel dispatch engine is implemented.
  void execute() const {}
};

inline Graph<> make_graph() { return {}; }

}  // namespace TensorOperations
