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
//   TileType (optional, default void) — when NodeTypeOrTag is a
//     ContractionTag node, the canonical tile passed to the inner evaluator.
//     Selecting the 4-param form stores the inner Evaluator directly, so its
//     scratch is allocated exactly once (in the constructor) rather than on
//     every k-tile call.
//
// Static interface (host-side sizing):
//   ScratchAllocator<P,O,N[,T]>::bytes<V>(tile)
//
// Instance interface (device-side allocation):
//   ScratchAllocator<P,O,N> sa{node};               // InputTag / IntermTag
//   ScratchAllocator<P,O,N,T> sa{node, tile, team}; // ContractionTag operand
//   auto view = sa.alloc<V>(team, tile);
//
// Uniform factory (3-arg form selects 3- or 4-param specialization via
// requires constraints on the operand node type):
//   make_scratch_allocator<PolicyTag, OuterOpTag>(node, tile, team)
// ---------------------------------------------------------------------------

// Primary template — undefined; must specialize.
template <typename PolicyTag, typename OuterOpTag, typename NodeTypeOrTag,
          typename TileType = void>
struct ScratchAllocator;

// ---------------------------------------------------------------------------
// TeamPolicyTag × ContractionTag specializations
// ---------------------------------------------------------------------------

// InputTag operand: always staged from global → scratch. Node is accepted in
// the constructor but not stored — InputTag never enables zero-copy.
template <typename ES, typename NA>
  requires(Impl::has_node_tag_v<InputTag, NA>)
struct ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, NA> {
  KOKKOS_FUNCTION explicit ScratchAllocator(const NA& /*node*/) {}

  template <typename V, typename CanonTile>
  static std::size_t bytes(const CanonTile& tile) {
    return Impl::scratch_tile_bytes<V, ES>(tile);
  }
  template <typename V, typename Team, typename CanonTile>
  KOKKOS_FUNCTION auto alloc(const Team& team, const CanonTile& tile) const {
    return Impl::alloc_scratch_tile<V, ES>(team, tile);
  }
};

// ContractionTag operand (3-param / TileType=void): identity re-stage fallback.
template <typename ES, typename NA>
  requires(Impl::has_node_tag_v<ContractionTag, NA>)
struct ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, NA> {
  NA node_;

  KOKKOS_FUNCTION explicit ScratchAllocator(const NA& node) : node_(node) {}

  template <typename V, typename CanonTile>
  static std::size_t bytes(const CanonTile& tile) {
    return Impl::scratch_tile_bytes<V, ES>(tile);
  }
  template <typename V, typename Team, typename CanonTile>
  KOKKOS_FUNCTION auto alloc(const Team& team, const CanonTile& tile) const {
    return Impl::alloc_scratch_tile<V, ES>(team, tile);
  }
};

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

// IntermTag = the output (C) accumulator slot: no operand node.
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
};

// ---------------------------------------------------------------------------
// TeamPolicyTag × CombineTag specializations
// ---------------------------------------------------------------------------

template <typename ES, typename NA>
  requires(Impl::has_node_tag_v<InputTag, NA>)
struct ScratchAllocator<TeamPolicyTag<ES>, CombineTag, NA> {
  KOKKOS_FUNCTION explicit ScratchAllocator(const NA& /*node*/) {}

  template <typename V, typename CanonTile>
  static std::size_t bytes(const CanonTile& tile) {
    return Impl::scratch_tile_bytes<V, ES>(tile);
  }
  template <typename V, typename Team, typename CanonTile>
  KOKKOS_FUNCTION auto alloc(const Team& team, const CanonTile& tile) const {
    return Impl::alloc_scratch_tile<V, ES>(team, tile);
  }
};

template <typename ES, typename NA>
  requires(Impl::has_node_tag_v<ContractionTag, NA>)
struct ScratchAllocator<TeamPolicyTag<ES>, CombineTag, NA> {
  NA node_;

  KOKKOS_FUNCTION explicit ScratchAllocator(const NA& node) : node_(node) {}

  template <typename V, typename CanonTile>
  static std::size_t bytes(const CanonTile& tile) {
    return Impl::scratch_tile_bytes<V, ES>(tile);
  }
  template <typename V, typename Team, typename CanonTile>
  KOKKOS_FUNCTION auto alloc(const Team& team, const CanonTile& tile) const {
    return Impl::alloc_scratch_tile<V, ES>(team, tile);
    // future: return node_.result_.storage_;
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
};

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

// 1-arg form: for InputTag/ContractionTag nodes where no tile context is
// needed (CombineTag operand staging, or the legacy ContractionTag 3-param
// path).
template <typename PolicyTag, typename OuterOpTag, typename NA>
KOKKOS_FUNCTION auto make_scratch_allocator(const NA& node) {
  return ScratchAllocator<PolicyTag, OuterOpTag, NA>{node};
}

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
