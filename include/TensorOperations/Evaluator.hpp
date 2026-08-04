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

// The inverse of as_result_array, for the one context that needs a SINGLE value
// rather than a normalized pack: an operand's evaluated result, which a
// consumer stages into one scratch slot. A single-output evaluator's interm
// handle passes through; a combine's Kokkos::Array is unwrapped.
//
// This is where the phase-1 restriction on fused combine operands lives, and
// the one place a phase-2 output selector would take an index instead of
// asserting M == 1. Stated here rather than at the ScratchAllocator call site
// so the restriction is one declaration, not a condition duplicated per
// consumer.
//
// Returns BY VALUE for the same reason as_result_array does: the argument is
// the temporary returned by eval(), so a const& pass-through would dangle.
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION T single_result(const T& r) {
  return r;
}
template <typename T, std::size_t M>
KOKKOS_FORCEINLINE_FUNCTION T single_result(const Kokkos::Array<T, M>& r) {
  static_assert(M == 1,
                "a multi-output combine node cannot yet be consumed as an "
                "operand: selecting one output of several is phase 2 (a bound "
                "output handle as an operand). Register the multi-output "
                "combine as its own graph output instead");
  return r[0];
}

// --- node output order ------------------------------------------------------
//
// A contraction node's modes_seq, shape() and C scratch are all CANONICAL
// (freeA ++ freeB); permC records where each canonical mode sits in the USER's
// requested output order. A node's Tile<A,B,C> bundle, by contrast, is written
// by the user, so its `.c` slot is in USER order. The helpers below are the
// single place that reconciles the two, and they are used both by the graph
// driver (which presents the user's output view canonically) and by the
// contraction evaluator (which must know the axis order a fused operand's
// scratch actually comes back in). Every one of them collapses to a no-op for a
// non-contraction node.

// permC as a full-rank sequence: canonical output mode i -> its position in the
// user output order. Identity for canonical contractions and for every
// non-contraction node.
template <typename NodeType>
KOKKOS_FUNCTION auto output_perm_seq() {
  if constexpr (has_node_tag_v<ContractionTag, NodeType>)
    return typename NodeType::permC_seq{};
  else
    return std::make_integer_sequence<int, NodeType::Rank>{};
}

// The output (C) tile of a tiling spec in the node's canonical mode order: the
// user-order output tile gathered by permC.
//
// Read as an OPERAND's tile (see operand_leaf_tile_t), this is the axis order
// that operand's own evaluator actually produces: an input operand's evaluator
// yields the operand's declared order, so its tile passes straight through,
// while a fused contraction hands back its CANONICAL C scratch even though the
// bundle it was tiled with carries `.c` in that sub-contraction's USER order.
// Going through here keeps a parent's tile shapes, its k-tile counts (which
// index node.shape(), also canonical) and its permA/permB (computed from
// modes_seq, also canonical) in one consistent axis order.
template <typename NodeType, typename Tile>
KOKKOS_FUNCTION auto canonical_c_tile(const Tile& tile) {
  return reorder_tile_value(output_tile(tile), output_perm_seq<NodeType>());
}

template <typename Node, typename Tile>
using operand_leaf_tile_t =
    decltype(canonical_c_tile<Node>(std::declval<Tile>()));

// --- operand kinds ----------------------------------------------------------
//
// Does this node's evaluator hand back its OWN team-scratch tile, rather than
// being read out of global memory into a staging buffer the consumer carves?
//
// This is the distinction every operand-staging decision below actually turns
// on, and it is NOT "is it a contraction": a fused combine produces its output
// scratch exactly like a fused contraction does, so a consumer stages both by
// reordering that buffer in place (or consumes it zero-copy) and neither costs
// a byte of the consumer's own scratch. Only an InputTag operand is copied.
//
// Spelled once, here, because the alternative -- open-coding the tag
// disjunction at each site -- is what let the two predicates below silently
// treat a combine operand as "copied, therefore unconstrained".
template <typename Node>
inline constexpr bool produces_own_scratch_v =
    has_node_tag_v<ContractionTag, Node> || has_node_tag_v<CombineTag, Node>;

// Does this node READ a team-scratch buffer that some OTHER node owns, rather
// than producing one itself or being copied out of global memory into a staging
// buffer the consumer carves?
//
// Exactly the SlotTag nodes, and the distinction is a third thing, not a
// rewording of either predicate beside it. A slot is consumed ZERO-COPY, like a
// fused operand -- but unlike a fused operand its buffer may have OTHER
// consumers, so nothing may write to it. Both halves matter: treating a slot as
// "produces its own scratch" would charge it that subtree's recursive bytes and
// try to evaluate it, and treating it as "not produces_own_scratch, therefore
// copied, therefore shape-unconstrained" is exactly the mistake
// operand_stageable_v's comment below warns about, now with a shared buffer
// behind it.
//
// produces_own_scratch_v deliberately does NOT admit it: that predicate means
// "hands back its own scratch BY EVALUATING A SUBTREE", and a slot evaluates
// nothing -- its buffer is already filled by the time any consumer runs.
template <typename Node>
inline constexpr bool reads_foreign_scratch_v = has_node_tag_v<SlotTag, Node>;

// Every node kind an evaluator can consume as an operand: a leaf input, a fused
// node that produces its own scratch, or a slot naming a buffer some other node
// already filled.
template <typename Node>
inline constexpr bool fusable_operand_v =
    has_node_tag_v<InputTag, Node> || produces_own_scratch_v<Node> ||
    reads_foreign_scratch_v<Node>;

// Can operand `Node`, tiled by `Tile`, be staged into the axis order `PermSeq`
// gathers into?
//
// A fused operand (contraction or combine) hands back its own output scratch
// and is reordered IN PLACE, which is a self-map of that fixed buffer only when
// every transposed axis pair has equal extents -- a 16x32 tile cannot be
// transposed inside its own storage. An input operand is copied into a fresh
// staging buffer on the way in, so nothing constrains it.
//
// A SLOT operand is the case the equal-extent rule does not cover, because the
// question there is not whether the reorder FITS but whether it is permitted at
// all: the buffer may have other consumers, and an in-place reorder would
// permute it under them. So a slot admits ONLY the identity -- the true
// zero-copy passthrough -- and a differently-ordered consumer must take the
// relabel path (operand_relabelable_v, below) or name the buffer through a
// second slot node carrying its own labels. This is why the branch leads: the
// `!produces_own_scratch_v` test that follows would otherwise answer
// "unconstrained" for a slot, on the reasoning that it is copied. It is not.
//
// Deliberately spelled with transpositions_equal_extent, the SAME predicate the
// in-place reorder asserts on the scratch layout at its call site (see
// Specialization 8 in Evaluator/Team.hpp): stating the rule once means an
// evaluator's early, friendly rejection cannot drift from what the reorder
// actually requires. Naming it also makes the rule testable -- a violation is a
// hard error from inside an evaluator's class body, which no `requires` can
// detect, so a test can only check it by asserting this predicate directly (see
// the static_asserts above FusedPermutedOperandATeam in test_graph.cpp).
// The branches are if constexpr rather than a `||` chain for the same reason
// Specialization 8 guards its own static_assert that way: transpositions_equal_
// extent reads extents statically off its argument, which only a StaticTile
// offers, and a `||` would instantiate it even for the cases that never reach
// the reorder.
template <typename Node, typename Tile, typename PermSeq>
inline constexpr bool operand_stageable_v = [] {
  if constexpr (reads_foreign_scratch_v<Node>)
    return is_identity_seq(PermSeq{});  // shared: never reordered in place
  else if constexpr (!produces_own_scratch_v<Node>)
    return true;  // copied into its own staging buffer; shape unconstrained
  else if constexpr (is_identity_seq(PermSeq{}))
    return true;  // zero-copy passthrough; nothing is reordered
  else
    return transpositions_equal_extent<operand_leaf_tile_t<Node, Tile>>(
        PermSeq{});
}();

// Should operand `Node` instead be RELABELED into the order `PermSeq` gathers
// into -- consumed as a strided view over its own scratch, with no copy, no
// reorder pass, and no constraint relating the permuted axes' extents?
//
// The complement of operand_stageable_v, and stated here beside it so the two
// staging rules are read together and for the same testability reason: a
// consumer's choice between them is otherwise buried in an evaluator class
// body, where a test can only observe it by instantiating the whole evaluator.
//
// A fused operand (contraction or combine) qualifies, and so does a SLOT: what
// they have in common is that their storage is per-team scratch, so relabeling
// it aliases nothing observable -- unlike an input operand, whose storage is
// the user's global tensor and whose reads would also lose their coalescing if
// driven by the consumer's traversal order rather than the source's memory
// order.
//
// For a slot this path is not merely permitted but REQUIRED: it is the only way
// a differently-ordered consumer can read a shared buffer at all, since
// operand_stageable_v forbids reordering one in place. It stays zero-copy and
// read-only -- reorder_view retypes the layout and touches no data -- which is
// exactly what sharing needs.
//
// Restricted further to the permuted, unhooked case:
//   • an identity permutation already costs nothing on the staging path (true
//     zero-copy passthrough), and relabeling would needlessly retype it;
//   • a hook is applied by writing back into the operand's own C scratch, so
//     hooked operands stay on the staging path rather than growing a second
//     place that mutates a sub-contraction's buffer.
// Whether the consumer CAN relabel at all is a separate question this predicate
// does not answer: a contraction's GEMM needs a contiguous LayoutRight source
// and must always stage. Only the combine evaluator consults this.
//
// hook_type is declared only on the ContractionTag node specialization, hence
// the if constexpr chain rather than a `&&` one. A CombineTag node carries no
// hook member at all (its fn already sees every coordinate and operand value,
// so there is no separate store hook to apply), and a SlotTag node carries none
// by deliberate design (NodeHandle.hpp: a hook writes back into the buffer it
// rides on, which a shared buffer cannot allow). Both branches therefore drop
// the hook half of the rule as vacuously satisfied rather than naming a
// hook_type they do not have.
template <typename Node, typename PermSeq>
inline constexpr bool operand_relabelable_v = [] {
  if constexpr (has_node_tag_v<ContractionTag, Node>)
    return !is_identity_seq(PermSeq{}) &&
           std::is_same_v<typename Node::hook_type, NoHook>;
  else if constexpr (has_node_tag_v<CombineTag, Node>)
    return !is_identity_seq(PermSeq{});
  else if constexpr (has_node_tag_v<SlotTag, Node>)
    return !is_identity_seq(PermSeq{});
  else
    return false;
}();

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

// Adopting overload: the evaluator writes its output into `adopted`, a buffer
// the caller already carved, instead of taking one from the team cursor. A
// contraction adopts a single scratch view; a multi-output combine adopts a
// Kokkos::Array of them, one per output.
//
// The caller must then request the evaluator's operand_scratch_size_per_team()
// -- NOT scratch_size_per_team() -- on top of whatever it allocated for
// `adopted`, or the output is charged twice.
template <typename PolicyTag, typename NodeType, typename Tile,
          typename TeamMember, typename Adopted>
KOKKOS_FUNCTION auto make_evaluator(NodeType node, Tile tile,
                                    const TeamMember& team,
                                    const Adopted&    adopted)
    -> Evaluator<PolicyTag, NodeType, Tile> {
  return Evaluator<PolicyTag, NodeType, Tile>(node, tile, team, adopted);
}

// Adopting overload with an OPERAND ARENA: the output is adopted as above, and
// the operand staging comes from a driver-owned region instead of the team
// cursor. A driver using this requests the arena ONCE, sized by its largest
// node, rather than operand_scratch_size_per_team() summed over every node --
// operand staging is dead at the node's own barrier, so no two nodes' buffers
// are ever live together. See Impl::OperandArena.
template <typename PolicyTag, typename NodeType, typename Tile,
          typename TeamMember, typename Adopted, typename Arena>
KOKKOS_FUNCTION auto make_evaluator(NodeType node, Tile tile,
                                    const TeamMember& team,
                                    const Adopted& adopted, Arena& arena)
    -> Evaluator<PolicyTag, NodeType, Tile> {
  return Evaluator<PolicyTag, NodeType, Tile>(node, tile, team, adopted, arena);
}

#include <TensorOperations/Evaluator/Team.hpp>

}  // namespace TensorOperations
