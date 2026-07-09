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
using is_contraction = has_node_tag<ContractionTag, Node>;

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

// Launches one team-policy kernel for one output node and returns the number
// of work items (output tiles) it covered.
template <typename NodeType, typename ViewT, typename Tile>
int execute_one_output_team(const NodeType& node, const ViewT& view,
                            const Tile& tile) {
  using exec_space = typename NodeType::exec_space;
  using member_t   = typename Kokkos::TeamPolicy<exec_space>::member_type;
  using eval_type  = Evaluator<TeamPolicyTag<exec_space>, NodeType, Tile>;

  // node.shape() is canonical, so the output tile is presented canonically for
  // the contraction evaluator (which canonicalizes A/B/C internally from the
  // same user-order tile bundle). The store, however, tiles the NATIVE user
  // output (u_tile) so subview_tile stays on the compile-time-ordered path,
  // then reorders that ordered subview into canonical order before writing.
  const auto c_tile = canonical_c_tile<NodeType>(tile);
  const auto u_tile = output_tile(tile);
  const int  wk     = work_items(node, c_tile);
  const int  bytes  = eval_type::scratch_size_per_team(tile);

  Kokkos::TeamPolicy<exec_space> policy(static_cast<int>(wk), Kokkos::AUTO);
  policy.set_scratch_size(0, Kokkos::PerTeam(bytes));

  Kokkos::parallel_for(
      "TensorOperations::execute_team", policy,
      KOKKOS_LAMBDA(const member_t& team) {
        const auto shape      = node.shape();
        const auto c_tile_idx = decode_tile_index<NodeType::Rank>(
            static_cast<int>(team.league_rank()), shape, c_tile);

        auto eval = make_evaluator<TeamPolicyTag<exec_space>>(node, tile, team);
        auto interm = eval(team, c_tile_idx);

        auto seval = make_evaluator<TeamPolicyTag<exec_space>>(interm, u_tile);
        seval(team, c_tile_idx, view, output_perm_seq<NodeType>());
      });
  return wk;
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

  // Evaluate the graph and write each output into the corresponding view.
  // Exactly one view must be provided per output node.
  template <typename ES, typename Tile, TensorLike... Ts>
  int execute(const TeamPolicyTag<ES>&, const Tile& tile,
              const Ts&... ts) const {
    auto                  ts_tuple = std::tie(ts...);
    constexpr std::size_t N        = std::tuple_size_v<OutputTuple>;
    int                   wk_items = 0;
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      ((wk_items += Impl::execute_one_output_team(std::get<I>(outputs),
                                                  std::get<I>(ts_tuple), tile)),
       ...);
    }(std::make_index_sequence<N>{});

    return wk_items;
  }

 private:
  template <typename>
  friend struct Graph;

  explicit Graph(OutputTuple o) : outputs(std::move(o)) {}

  OutputTuple outputs;
};

inline Graph<> make_graph() { return {}; }

}  // namespace TensorOperations
