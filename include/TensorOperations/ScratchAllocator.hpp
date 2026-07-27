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
// TeamPolicyTag × ContractionTag specializations
//
// Every operand-bearing specialization is 4-param only: it stores an inner
// Evaluator built at construction time (tile/team supplied then), so scratch
// is allocated exactly once and stage() never rebuilds the evaluator.
// ---------------------------------------------------------------------------

// ContractionTag operand (4-param): stores the inner Evaluator so its scratch
// is allocated exactly once during construction, not on every k-tile call.
// alloc() returns the inner C scratch directly (zero-copy, no cursor bump).
// bytes() returns the full recursive scratch_size_per_team of the inner eval.
template <typename ES, typename NA, typename TileA>
  requires(Impl::has_node_tag_v<ContractionTag, NA>)
struct ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, NA, TileA> {
  using inner_eval_t = Evaluator<TeamPolicyTag<ES>, NA, TileA>;
  inner_eval_t eval_;

  template <typename Team>
  KOKKOS_FUNCTION explicit ScratchAllocator(const NA& node, const TileA& tile,
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

// InputTag operand (4-param): stores the inner Evaluator so its tiled view is
// set up once during construction rather than recreated on every call.
// alloc() carves a new scratch tile from the team cursor (same as 3-param).
// bytes() is the operand staging cost, not a recursive evaluation cost.
template <typename ES, typename NA, typename TileA>
  requires(Impl::has_node_tag_v<InputTag, NA>)
struct ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, NA, TileA> {
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

// IntermTag = the output (C) accumulator slot: no operand node. stage() has
// no source to stage from, so it's a trivial alias for alloc() (kept for
// interface uniformity with the operand-bearing specializations above).
template <typename ES>
struct ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, IntermTag> {
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
// TeamPolicyTag × CombineTag specializations
//
// Mirrors the ContractionTag family above: operand-bearing specializations
// are 4-param only, storing an inner Evaluator built at construction so
// stage() never rebuilds it. Unlike ContractionTag's own ContractionTag-
// operand form, alloc() here intentionally keeps allocating a fresh
// destination scratch tile (not zero-copy) for both operand kinds; bytes()
// stays non-recursive (Impl::scratch_tile_bytes) since the CombineTag
// evaluator adds a nested contraction operand's recursive scratch separately.
// ---------------------------------------------------------------------------

template <typename ES, typename NA, typename TileA>
  requires(Impl::has_node_tag_v<InputTag, NA>)
struct ScratchAllocator<TeamPolicyTag<ES>, CombineTag, NA, TileA> {
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

template <typename ES, typename NA, typename TileA>
  requires(Impl::has_node_tag_v<ContractionTag, NA>)
struct ScratchAllocator<TeamPolicyTag<ES>, CombineTag, NA, TileA> {
  using inner_eval_t = Evaluator<TeamPolicyTag<ES>, NA, TileA>;
  inner_eval_t eval_;

  template <typename Team>
  KOKKOS_FUNCTION explicit ScratchAllocator(const NA& node, const TileA& tile,
                                            const Team& team)
      : eval_(make_evaluator<TeamPolicyTag<ES>>(node, tile, team)) {}

  template <typename V, typename CanonTile>
  static std::size_t bytes(const CanonTile& tile) {
    return Impl::scratch_tile_bytes<V, ES>(tile);
  }
  template <typename V, typename Team, typename CanonTile>
  KOKKOS_FUNCTION auto alloc(const Team& team, const CanonTile& tile) const {
    return Impl::alloc_scratch_tile<V, ES>(team, tile);
    // future: return node_.result_.storage_; (zero-copy, mirroring
    // ContractionTag's ContractionTag-operand form) is intentionally deferred.
  }
  template <typename Team, typename Idx>
  KOKKOS_FUNCTION auto stage(const Team& team, const Idx& idx) const {
    return eval_(team, idx);
  }
};

template <typename ES>
struct ScratchAllocator<TeamPolicyTag<ES>, CombineTag, IntermTag> {
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

// 3-arg form (InputTag): constructs the 4-param specialization that stores the
// inner Evaluator built from the native tile, so stage() avoids recreating it.
template <typename PolicyTag, typename OuterOpTag, typename NA, typename TileA,
          typename Team>
  requires(Impl::has_node_tag_v<InputTag, NA>)
KOKKOS_FUNCTION auto make_scratch_allocator(const NA& node, const TileA& tile,
                                            const Team& team) {
  return ScratchAllocator<PolicyTag, OuterOpTag, NA, TileA>{node, tile, team};
}

// 3-arg form (ContractionTag): constructs the 4-param specialization that
// stores the inner Evaluator and allocates its scratch exactly once.
template <typename PolicyTag, typename OuterOpTag, typename NA, typename TileA,
          typename Team>
  requires(Impl::has_node_tag_v<ContractionTag, NA>)
KOKKOS_FUNCTION auto make_scratch_allocator(const NA& node, const TileA& tile,
                                            const Team& team) {
  return ScratchAllocator<PolicyTag, OuterOpTag, NA, TileA>{node, tile, team};
}
