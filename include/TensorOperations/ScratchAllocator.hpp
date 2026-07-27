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
//   ScratchAllocator<P,O,N[,T]>::bytes<V>(tile)
//
// Instance interface (device-side allocation):
//   ScratchAllocator<P,O,IntermTag> sa{};            // output (C) slot only
//   ScratchAllocator<P,O,N,T> sa{node, tile, team};  // operand
//   (Input/Contraction) auto view = sa.alloc<V>(team, tile); auto staged =
//   sa.stage(team, idx);  // operand-bearing forms only
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
// bump) -- valid for either OuterOpTag, since a ContractionTag operand's
// canonicality (identity permC, matching output order) is statically
// asserted at the use site regardless of which operation contains it.
// bytes() returns the full recursive scratch_size_per_team of the inner eval.
template <typename ES, typename OuterOpTag, typename NA, typename TileA>
  requires(Impl::has_node_tag_v<ContractionTag, NA>)
struct ScratchAllocator<TeamPolicyTag<ES>, OuterOpTag, NA, TileA> {
  using inner_eval_t = Evaluator<TeamPolicyTag<ES>, NA, TileA>;
  inner_eval_t eval_;

  template <typename Team>
  KOKKOS_FUNCTION ScratchAllocator(const NA& node, const TileA& tile,
                                   const Team& team)
      : eval_(make_evaluator<TeamPolicyTag<ES>>(node, tile, team)) {}

  template <typename V, typename CanonTile>
  static std::size_t bytes(const CanonTile& tile) {
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
// staging cost, not a recursive evaluation cost. Also valid for either
// OuterOpTag -- an input operand is staged identically regardless of the
// containing operation.
template <typename ES, typename OuterOpTag, typename NA, typename TileA>
  requires(Impl::has_node_tag_v<InputTag, NA>)
struct ScratchAllocator<TeamPolicyTag<ES>, OuterOpTag, NA, TileA> {
  using inner_eval_t = Evaluator<TeamPolicyTag<ES>, NA, TileA>;
  inner_eval_t eval_;

  template <typename Team>
  KOKKOS_FUNCTION ScratchAllocator(const NA& node, const TileA& tile,
                                   const Team& team)
      : eval_(make_evaluator<TeamPolicyTag<ES>>(node, tile, team)) {}

  template <typename V, typename CanonTile>
  static std::size_t bytes(const CanonTile& tile) {
    return Impl::scratch_tile_bytes<V, ES>(tile);
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
// the only 3-param form. stage() has no source to stage from, so it's a
// trivial alias for alloc() (kept for interface uniformity with the
// operand-bearing specializations above). Uniform across every OuterOpTag.
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
