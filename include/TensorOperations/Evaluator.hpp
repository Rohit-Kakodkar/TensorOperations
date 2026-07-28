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

// The team handle for a team-tier kernel. Spelled once here so the evaluators,
// the scratch allocators and the graph driver don't each repeat the nested
// dependent-typename form.
template <typename ES>
using team_member_t = typename Kokkos::TeamPolicy<ES>::member_type;

// Iterate a tile's coordinate space team-parallel, calling f(coord) once per
// element. `iter_view` supplies the traversal order -- any other view indexed
// inside f must share its extents. Where two views of different layouts are
// involved (a strided/ordered global subview and a contiguous LayoutRight
// scratch tile), pass whichever one should drive the access pattern.
template <typename Team, typename IterView, typename F>
KOKKOS_FORCEINLINE_FUNCTION void team_for_each_coord(const Team&     team,
                                                     const IterView& iter_view,
                                                     F               f) {
  const auto layout = iter_view.layout();
  const auto total  = iter_view.size();
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, total),
                       [=](int i) { f(layout[i]); });
}

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
  team_for_each_coord(team, scratch, [=](auto coord) {
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

// In-place reorder of a scratch tile by a gather permutation, expressed as a
// sequence of parallel axis-pair transpositions (no auxiliary buffer). A gather
// permutation perm (canonical.extent(i) == native.extent(perm[i])) is
// decomposed at compile time into transpositions that sort it to identity; each
// transposition is an involution on the fixed physical buffer, applied as one
// embarrassingly-parallel TeamVectorRange pass. See reorder_scratch_in_place.
//
// PRECONDITION: every transposed axis pair must have equal extent, else a swap
// is not a self-map of the buffer. Enforced by a static_assert at the call site
// (transpositions_equal_extent) where the scratch Layout type is concrete.

// A compile-time list of axis-pair transpositions. count is the number of
// pairs used; pairs[t] = {a, b} is the t-th axis swap. The capacity
// Rank*(Rank-1)/2 bounds the transpositions produced by a selection sort.
template <int Rank>
struct TranspositionPlan {
  int                                                       count = 0;
  Kokkos::Array<Kokkos::Array<int, 2>, Rank*(Rank - 1) / 2> pairs{};
};

// Decompose a gather permutation into transpositions that sort perm ->
// identity. Selection sort: whenever dim i currently sits at position j, swap
// positions (i, j) and record the pair. Fully constexpr.
//
// The recorded pairs sort perm to identity when read front-to-back; to REALIZE
// perm as a product of buffer axis-swaps they must be APPLIED back-to-front
// (see reorder_scratch_in_place), because sequential buffer swaps compose as
// gather functions in the opposite order to the sort.
template <int... Perm>
constexpr TranspositionPlan<sizeof...(Perm)> transposition_plan(
    std::integer_sequence<int, Perm...>) {
  constexpr int           Rank    = sizeof...(Perm);
  int                     p[Rank] = {Perm...};
  TranspositionPlan<Rank> plan{};
  for (int i = 0; i < Rank; ++i)
    for (int j = i + 1; j < Rank; ++j)
      if (p[j] == i) {  // dim i currently at position j -> swap into place
        plan.pairs[plan.count++] = {i, j};
        const int t              = p[i];
        p[i]                     = p[j];
        p[j]                     = t;
      }
  return plan;
}

// constexpr predicate: every transposed axis pair has equal extent under the
// (static) scratch layout. Layout::extent(k) is static constexpr on the
// StaticTileLayout* scratch layouts, so this is fully compile-time.
template <typename Layout, int... Perm>
constexpr bool transpositions_equal_extent(
    std::integer_sequence<int, Perm...> perm) {
  const auto plan = transposition_plan(perm);
  for (int t = 0; t < plan.count; ++t)
    if (Layout::extent(plan.pairs[t][0]) != Layout::extent(plan.pairs[t][1]))
      return false;
  return true;
}

// Swap axes (a, b) of the scratch tile in place: one parallel pass, each
// unordered coordinate pair handled exactly once (guard c[a] < c[b]); the
// diagonal c[a] == c[b] is a no-op. Requires extent(a) == extent(b).
template <typename TeamMember, typename View>
KOKKOS_FUNCTION void swap_axes(const TeamMember& team, const View& view, int a,
                               int b) {
  const auto layout = view.layout();
  const int  total  = view.size();
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, total), [=](int s) {
    auto c = layout[s];
    if (c[a] < c[b]) {
      auto c2        = c;
      c2[a]          = c[b];
      c2[b]          = c[a];
      const int d    = layout.flat_offset(c2);
      auto      t    = view.data()[s];
      view.data()[s] = view.data()[d];
      view.data()[d] = t;
    }
  });
  team.team_barrier();  // successive transpositions are dependent
}

// In-place reorder of a scratch tile by a gather permutation, as a sequence of
// parallel axis-pair transpositions (no auxiliary buffer). The equal-extent
// precondition is static_asserted at the call site.
template <typename TeamMember, typename View, int... Perm>
KOKKOS_FUNCTION void reorder_scratch_in_place(
    const TeamMember& team, const View& view,
    std::integer_sequence<int, Perm...> perm) {
  constexpr auto plan = transposition_plan(perm);
  // Apply back-to-front: the plan sorts perm -> identity front-to-back, so the
  // buffer swaps that reproduce perm run in reverse (gather functions compose
  // in the opposite order to the positional sort).
  for (int t = plan.count - 1; t >= 0; --t)
    swap_axes(team, view, plan.pairs[t][0], plan.pairs[t][1]);
}

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
