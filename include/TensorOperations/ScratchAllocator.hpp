#pragma once
// Included from within TensorOperations namespace by Evaluator.hpp (after
// Impl::scratch_tile_bytes is defined in Evaluator/Team.hpp).

// ---------------------------------------------------------------------------
// ScratchAllocator<PolicyTag, OuterOpTag, NodeTypeOrTag [, TileType]>
//
// Per-slot scratch allocation coordinator. Three or four template parameters:
//   PolicyTag      — memory tier: TeamPolicyTag<ES>
//   OuterOpTag     — containing operation: ContractionTag, CombineTag
//   NodeTypeOrTag  — either:
//     • IntermTag                    — the output (C) slot; no operand node
//     • a full operand NodeHandle    — InputTag or ContractionTag node
//   TileType — required for every operand-bearing specialization (IntermTag
//     is the only 3-param form, since it has no operand and needs no tile to
//     build an inner evaluator from). Storing the inner Evaluator directly
//     means its scratch is allocated exactly once (in the constructor) rather
//     than on every k-tile/combine call, and stage() never rebuilds it.
//
// Static interface (host-side sizing):
//   ScratchAllocator<P,O,IntermTag>::bytes<V>(canon_tile)
//   ScratchAllocator<P,O,N,T>::bytes<V>(native_tile, perm)
//
// The operand-bearing forms take the operand's NATIVE tile (the same spec the
// allocator was keyed on) plus the gather permutation that brings it into the
// consuming operation's canonical order — the two pieces of information the
// specializations need to disagree about. A ContractionTag operand recurses
// into its inner evaluator, which is keyed on the native tile BUNDLE and
// accepts nothing else; an InputTag operand sizes a staging buffer, whose shape
// is the CANONICAL one. Callers therefore cannot supply a single tile that
// serves both, and the perm is what lets each form derive its own. It defaults
// to the identity sequence, which is the whole story for an unpermuted operand.
//
// Instance interface (device-side allocation):
//   ScratchAllocator<P,O,IntermTag> sa{};            // output (C) slot only
//   ScratchAllocator<P,O,N,T> sa{node, tile, team};  // operand (Input/Contr.)
//   auto view   = sa.alloc<V>(team, canon_tile);
//   auto staged = sa.stage(team, idx);               // operand-bearing only
//
// alloc() keeps taking the canonical tile directly: unlike bytes(), both
// operand forms agree there (the ContractionTag one ignores the argument
// entirely, since it hands back the inner evaluator's own scratch).
//
// Uniform factory (3-arg form selects the InputTag- or ContractionTag-operand
// specialization via requires constraints on the operand node type):
//   make_scratch_allocator<PolicyTag, OuterOpTag>(node, tile, team)
// ---------------------------------------------------------------------------

// Primary template — undefined; must specialize.
template <typename PolicyTag, typename OuterOpTag, typename NodeTypeOrTag,
          typename TileType = void>
struct ScratchAllocator;

// ---------------------------------------------------------------------------
// Operand-bearing specializations (4-param).
//
// These apply uniformly across every OuterOpTag (ContractionTag, CombineTag):
// the containing operation has no bearing on how an operand's own scratch is
// staged, only on how the IntermTag output slot below is used by the caller.
// Each specialization stores an inner Evaluator built at construction time
// (tile/team supplied then), so scratch is allocated exactly once and
// stage() never rebuilds the evaluator.
// ---------------------------------------------------------------------------

// ContractionTag operand: stores the inner Evaluator so its scratch is
// allocated exactly once during construction, not on every k-tile/combine
// call. alloc() returns the inner C scratch directly (zero-copy, no cursor
// bump) -- valid for either OuterOpTag, and now also for a PERMUTED operand,
// which the consumer stages by reordering that same buffer in place rather
// than by copying it elsewhere. bytes() returns the full recursive
// scratch_size_per_team of the inner eval and ignores `perm`: neither the
// zero-copy nor the in-place-reorder path costs a byte beyond it.
template <typename ES, typename OuterOpTag, typename NA, typename TileA>
  requires(Impl::has_node_tag_v<ContractionTag, NA>)
struct ScratchAllocator<TeamPolicyTag<ES>, OuterOpTag, NA, TileA> {
  using inner_eval_t = Evaluator<TeamPolicyTag<ES>, NA, TileA>;
  inner_eval_t eval_;

  template <typename Team>
  KOKKOS_FUNCTION ScratchAllocator(const NA& node, const TileA& tile,
                                   const Team& team)
      : eval_(make_evaluator<TeamPolicyTag<ES>>(node, tile, team)) {}

  // Variadic perm: this form genuinely has nothing to do with it, so accepting
  // zero or one says so more plainly than a defaulted parameter it discards.
  template <typename V, typename... PermSeq>
  static std::size_t bytes(const TileA& tile, const PermSeq&...) {
    return inner_eval_t::scratch_size_per_team(tile);
  }
  template <typename V, typename Team, typename CanonTile>
  KOKKOS_FUNCTION auto alloc(const Team&, const CanonTile&) const {
    return eval_.scratch_;  // inner C scratch, already allocated in ctor
  }
  KOKKOS_FUNCTION const inner_eval_t& inner_eval() const { return eval_; }
  template <typename Team, typename Idx>
  KOKKOS_FUNCTION auto stage(const Team& team, const Idx& idx) const {
    return eval_(team, idx);
  }
};

// InputTag operand: stores the inner Evaluator so its tiled view is set up
// once during construction rather than recreated on every call. alloc()
// carves a new scratch tile from the team cursor. bytes() is the operand
// staging cost, not a recursive evaluation cost -- and the staging buffer has
// the CANONICAL shape, so this is the one form that actually applies `perm`.
// Also valid for either OuterOpTag -- an input operand is staged identically
// regardless of the containing operation.
template <typename ES, typename OuterOpTag, typename NA, typename TileA>
  requires(Impl::has_node_tag_v<InputTag, NA>)
struct ScratchAllocator<TeamPolicyTag<ES>, OuterOpTag, NA, TileA> {
  using inner_eval_t = Evaluator<TeamPolicyTag<ES>, NA, TileA>;
  inner_eval_t eval_;

  template <typename Team>
  KOKKOS_FUNCTION ScratchAllocator(const NA& node, const TileA& tile,
                                   const Team& team)
      : eval_(make_evaluator<TeamPolicyTag<ES>>(node, tile, team)) {}

  // TileA is a leaf tile here (only a fused operand is tiled with a bundle), so
  // the identity default reads straight off its rank.
  template <typename V,
            typename PermSeq = std::make_integer_sequence<int, TileA::rank>>
  static std::size_t bytes(const TileA& tile, PermSeq perm = {}) {
    return Impl::scratch_tile_bytes<V, ES>(reorder_tile_value(tile, perm));
  }
  template <typename V, typename Team, typename CanonTile>
  KOKKOS_FUNCTION auto alloc(const Team& team, const CanonTile& tile) const {
    return Impl::alloc_scratch_tile<V, ES>(team, tile);
  }
  template <typename Team, typename Idx>
  KOKKOS_FUNCTION auto stage(const Team& team, const Idx& idx) const {
    return eval_(team, idx);
  }
};

// IntermTag = the output (C) accumulator slot: no operand node, so this is
// the only 3-param form. With no operand there is no native-vs-canonical
// distinction to reconcile, so bytes() keeps taking the canonical tile alone
// (no perm parameter). stage() has no source to stage from, so it's a trivial
// alias for alloc() (kept for interface uniformity with the operand-bearing
// specializations above). Uniform across every OuterOpTag.
template <typename ES, typename OuterOpTag>
struct ScratchAllocator<TeamPolicyTag<ES>, OuterOpTag, IntermTag> {
  KOKKOS_DEFAULTED_FUNCTION ScratchAllocator() = default;

  template <typename V, typename CanonTile>
  static std::size_t bytes(const CanonTile& tile) {
    return Impl::scratch_tile_bytes<V, ES>(tile);
  }
  template <typename V, typename Team, typename CanonTile>
  KOKKOS_FUNCTION auto alloc(const Team& team, const CanonTile& tile) const {
    return Impl::alloc_scratch_tile<V, ES>(team, tile);
  }
  template <typename V, typename Team, typename CanonTile>
  KOKKOS_FUNCTION auto stage(const Team& team, const CanonTile& tile) const {
    return alloc<V>(team, tile);
  }
};

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

// Constructs the 4-param specialization (InputTag- or ContractionTag-operand)
// that stores the inner Evaluator built from the native tile, so stage()
// avoids recreating it and (for a ContractionTag operand) its scratch is
// allocated exactly once.
template <typename PolicyTag, typename OuterOpTag, typename NA, typename TileA,
          typename Team>
  requires(Impl::has_node_tag_v<InputTag, NA> ||
           Impl::has_node_tag_v<ContractionTag, NA>)
KOKKOS_FUNCTION auto make_scratch_allocator(const NA& node, const TileA& tile,
                                            const Team& team) {
  return ScratchAllocator<PolicyTag, OuterOpTag, NA, TileA>{node, tile, team};
}
