#pragma once
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/TiledLayout.hpp>
#include <TensorOperations/Tiling.hpp>
#include <TensorOperations/Macros.hpp>
#include <TensorOperations/TimingInstrumentation.hpp>
#include <array>
#include <utility>

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
namespace TensorOperations {

// ---------------------------------------------------------------------------
// Policy tags
// ---------------------------------------------------------------------------
template <typename ES = Kokkos::DefaultExecutionSpace>
struct TeamPolicyTag {
  using execution_space = ES;
};

// Tiling specs (StaticTile / DynamicTile) live in Tiling.hpp.

// ---------------------------------------------------------------------------
// Impl helpers
// ---------------------------------------------------------------------------
namespace Impl {

// Apply a hook (input load-time or contraction store-time) to the element at
// a given global coordinate. Every hook takes one index per rank followed by
// the element by mutable reference: op(idx[0], ..., idx[Rank-1], v). NoHook
// is the no-op identity.
template <typename Op, std::size_t Rank, typename V, std::size_t... Is>
KOKKOS_FORCEINLINE_FUNCTION void apply_hook_expand(
    const Op& op, const Kokkos::Array<int, Rank>& idx, V& v,
    std::index_sequence<Is...>) {
  op(idx[Is]..., v);
}

template <typename Op, std::size_t Rank, typename V>
KOKKOS_FORCEINLINE_FUNCTION void apply_hook(const Op&                       op,
                                            const Kokkos::Array<int, Rank>& idx,
                                            V&                              v) {
  apply_hook_expand(op, idx, v, std::make_index_sequence<Rank>{});
}

template <std::size_t Rank, typename V>
KOKKOS_FORCEINLINE_FUNCTION void apply_hook(const NoHook&,
                                            const Kokkos::Array<int, Rank>&,
                                            V&) {}

}  // namespace Impl

// ---------------------------------------------------------------------------
// Primary template — undefined; must use a specialization
// ---------------------------------------------------------------------------
template <typename PolicyTag, typename NodeType, typename Tiling>
struct Evaluator;

template <typename PolicyTag, typename NodeType, typename Tile>
KOKKOS_FUNCTION auto make_evaluator(NodeType node, Tile tile)
    -> Evaluator<PolicyTag, NodeType, Tile> {
  return Evaluator<PolicyTag, NodeType, Tile>(node, tile);
}

// Team-tier overload: evaluators that allocate from team scratch must be built
// inside the kernel, so they take the team member at construction.
template <typename PolicyTag, typename NodeType, typename Tile,
          typename TeamMember>
KOKKOS_FUNCTION auto make_evaluator(NodeType node, Tile tile,
                                    const TeamMember& team)
    -> Evaluator<PolicyTag, NodeType, Tile> {
  return Evaluator<PolicyTag, NodeType, Tile>(node, tile, team);
}

#include <TensorOperations/Evaluator/Team.hpp>

}  // namespace TensorOperations
