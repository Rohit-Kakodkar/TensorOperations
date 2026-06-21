#pragma once
#include <TensorOperations/Layout.hpp>
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
struct TeamPolicyTag {};

// Tiling specs (StaticTile / DynamicTile) live in Tiling.hpp.

// ---------------------------------------------------------------------------
// Impl helpers
// ---------------------------------------------------------------------------
namespace Impl {

// Apply an input node's hook at load time. NoHook is the identity.
template <typename Op, typename V>
KOKKOS_FORCEINLINE_FUNCTION V apply_hook(const Op& op, V v) {
  return op(v);
}
template <typename V>
KOKKOS_FORCEINLINE_FUNCTION V apply_hook(const NoHook&, V v) {
  return v;
}

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
