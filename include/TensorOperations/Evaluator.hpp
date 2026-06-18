#pragma once
#include <TensorOperations/Layout.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/RegisterArray.hpp>
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
struct RangePolicyTag {};
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

// Project a tiling type onto a set of participating-mode positions, yielding a
// StaticTile / RegisterArray whose extents are Tile::extent(Pos[i])... .
template <typename Tile, auto Pos, std::size_t... I>
StaticTile<Tile::extent(static_cast<int>(Pos[I]))...> project_tile_fn(
    std::index_sequence<I...>);
template <typename Tile, auto Pos>
using project_tile_t = decltype(project_tile_fn<Tile, Pos>(
    std::make_index_sequence<Pos.size()>{}));

template <typename V, typename Tile, auto Pos, std::size_t... I>
RegisterArray<V, Tile::extent(static_cast<int>(Pos[I]))...> project_regs_fn(
    std::index_sequence<I...>);
template <typename V, typename Tile, auto Pos>
using project_regs_t = decltype(project_regs_fn<V, Tile, Pos>(
    std::make_index_sequence<Pos.size()>{}));

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

#include <TensorOperations/Evaluator/Range.hpp>
#include <TensorOperations/Evaluator/Team.hpp>

}  // namespace TensorOperations
