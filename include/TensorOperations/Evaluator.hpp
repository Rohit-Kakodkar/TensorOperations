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

// Apply a hook (input load-time or contraction store-time) to staged scratch.
// Every hook takes one index per rank followed by the element by mutable
// reference: op(idx[0], ..., idx[Rank-1], v). NoHook is the no-op identity.
template <typename Op, std::size_t Rank, typename V, std::size_t... Is>
KOKKOS_FORCEINLINE_FUNCTION void apply_hook_at(
    const Op& op, const Kokkos::Array<int, Rank>& idx, V& v,
    std::index_sequence<Is...>) {
  op(idx[Is]..., v);
}

template <typename Op, typename TeamMember, typename Scratch, std::size_t Rank>
KOKKOS_FUNCTION void apply_hook(const Op& op, const TeamMember& team,
                                const Kokkos::Array<int, Rank>& tile_idx,
                                const Scratch&                  scratch) {
  const auto layout = scratch.layout();
  const auto total  = scratch.size();
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, total), [=](int i) {
    const auto               coord = layout[i];
    Kokkos::Array<int, Rank> gidx{};
    for (std::size_t d = 0; d < Rank; ++d)
      gidx[d] = tile_idx[d] * scratch.extent(static_cast<int>(d)) + coord[d];
    auto v = scratch[coord];
    apply_hook_at(op, gidx, v, std::make_index_sequence<Rank>{});
    scratch[coord] = v;
  });
}

template <typename TeamMember, typename Scratch, std::size_t Rank>
KOKKOS_FORCEINLINE_FUNCTION void apply_hook(const NoHook&, const TeamMember&,
                                            const Kokkos::Array<int, Rank>&,
                                            const Scratch&) {}

// Apply a pointwise combine op to N (homogeneous) operand values at a given
// global coordinate, returning the combined result. Mirrors apply_hook's index
// expansion but takes N values and yields fn's result — either a scalar or a
// Kokkos::Array<V, M> for a multi-output combine:
// fn(idx[0], ..., idx[Rank-1], vals[0], ..., vals[N-1]).
template <typename Fn, std::size_t Rank, typename V, std::size_t N,
          std::size_t... Is, std::size_t... Ks>
KOKKOS_FORCEINLINE_FUNCTION auto apply_combine_expand(
    const Fn& fn, const Kokkos::Array<int, Rank>& idx,
    const Kokkos::Array<V, N>& vals, std::index_sequence<Is...>,
    std::index_sequence<Ks...>) {
  return fn(idx[Is]..., vals[Ks]...);
}

template <typename Fn, std::size_t Rank, typename V, std::size_t N>
KOKKOS_FORCEINLINE_FUNCTION auto apply_combine(
    const Fn& fn, const Kokkos::Array<int, Rank>& idx,
    const Kokkos::Array<V, N>& vals) {
  return apply_combine_expand(fn, idx, vals, std::make_index_sequence<Rank>{},
                              std::make_index_sequence<N>{});
}

// Normalize a combine result to a Kokkos::Array<V, M> so the evaluator can
// write output m uniformly. A scalar becomes a 1-element array; an array passes
// through. V is the output scalar (the array element type).
template <typename V, typename R>
KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<V, 1> as_output_array(const R& r) {
  return {static_cast<V>(r)};
}
template <typename V, typename U, std::size_t M>
KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<V, M> as_output_array(
    const Kokkos::Array<U, M>& r) {
  Kokkos::Array<V, M> out{};
  for (std::size_t m = 0; m < M; ++m) out[m] = static_cast<V>(r[m]);
  return out;
}

// Normalize an evaluator result to Kokkos::Array<T, M> so the store loop can
// index output m uniformly: a single interm handle (single-output evaluators)
// becomes a 1-element array; a multi-output combine's array passes through.
// Returns BY VALUE in both overloads — a const& pass-through would dangle when
// called on the temporary returned by eval().
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<T, 1> as_result_array(const T& r) {
  return {r};
}
template <typename T, std::size_t M>
KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<T, M> as_result_array(
    const Kokkos::Array<T, M>& r) {
  return r;
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
