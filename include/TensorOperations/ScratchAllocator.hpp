#pragma once
// Included from within TensorOperations namespace by Evaluator.hpp (after
// Impl::scratch_tile_bytes is defined in Evaluator/Team.hpp).

// ---------------------------------------------------------------------------
// ScratchAllocator<PolicyTag, OuterOpTag, NodeTypeOrTag, ValueType, TileType
//                  [, PermSeq]>
//
// Per-slot scratch owner. Everything a slot needs is fixed when the allocator
// is built -- the scalar type and the permutation in the TYPE, the node/tile/
// team in the CONSTRUCTOR -- so the slot's scratch view is carved exactly once,
// there, and handed back by a zero-argument get().
//
//   PolicyTag      — memory tier: TeamPolicyTag<ES>
//   OuterOpTag     — containing operation: ContractionTag, CombineTag
//   NodeTypeOrTag  — either:
//     • IntermTag                    — the output (C) slot; no operand node
//     • a full operand NodeHandle    — InputTag or ContractionTag node
//   ValueType      — the scalar the slot's scratch is typed on. Chosen by the
//     consuming evaluator (its own value_type), never derived per operand: a
//     staging buffer holds the consumer's scalar regardless of the input
//     tensor's own element type.
//   TileType       — the operand's NATIVE tile (operand forms) or the output's
//     CANONICAL tile (IntermTag).
//   PermSeq        — the gather permutation bringing the operand's native axis
//     order into the consuming operation's canonical one. Only the InputTag
//     form consumes it (its staging buffer has the CANONICAL shape); the
//     ContractionTag-operand and IntermTag forms accept and ignore it. Defaults
//     to the sentinel `void`, resolved to the identity sequence where used.
//
// Static interface (host-side sizing):
//   ScratchAllocator<...>::bytes(tile)
//
// bytes() is the one thing that stays static and keeps a runtime tile argument:
// it is the PRE-INSTANCE sizing query. Graph.hpp calls it to decide how much
// scratch to request via policy.set_scratch_size() strictly before the
// parallel_for that constructs any evaluator, so there is no instance to read a
// stored tile from -- and a DynamicTile's extents aren't in the type either.
//
// Instance interface (device-side, one team):
//   ScratchAllocator<P,O,IntermTag,V,CT> sa{canon_tile, team};  // output slot
//   ScratchAllocator<P,O,N,V,T,Perm>     sa{node, tile, team};  // operand
//   auto view   = sa.get();               // the slot's own scratch view
//   auto staged = sa.stage(team, idx);    // operand-bearing forms only
//
// stage() keeps its team argument rather than storing one: its sole caller
// (Impl::stage_operand_into) needs a team for its own stager evaluator anyway,
// so a stored copy would save nothing at the call site while adding a member to
// every operand allocator, recursively, one per operand per nesting level.
// ---------------------------------------------------------------------------

// Primary template — undefined; must specialize.
template <typename PolicyTag, typename OuterOpTag, typename NodeTypeOrTag,
          typename ValueType, typename TileType, typename PermSeq = void>
struct ScratchAllocator;

// ---------------------------------------------------------------------------
// Operand-bearing specializations.
//
// These apply uniformly across every OuterOpTag (ContractionTag, CombineTag):
// the containing operation has no bearing on how an operand's own scratch is
// staged, only on how the IntermTag output slot below is used by the caller.
// Each stores an inner Evaluator built at construction time, so its scratch is
// allocated exactly once and stage() never rebuilds the evaluator.
// ---------------------------------------------------------------------------

// ContractionTag operand: the inner evaluator already owns a C scratch tile
// (allocated in its own constructor), so this form carves nothing of its own --
// get() hands that buffer back directly (zero-copy, no cursor bump), valid for
// either OuterOpTag and also for a PERMUTED operand, which the consumer stages
// by reordering that same buffer in place rather than by copying it elsewhere.
// bytes() is the inner evaluator's full recursive scratch_size_per_team and
// ignores PermSeq: neither the zero-copy nor the in-place-reorder path costs a
// byte beyond it.
template <typename ES, typename OuterOpTag, typename NA, typename V,
          typename TileA, typename PermSeq>
  requires(Impl::has_node_tag_v<ContractionTag, NA>)
struct ScratchAllocator<TeamPolicyTag<ES>, OuterOpTag, NA, V, TileA, PermSeq> {
  using inner_eval_t   = Evaluator<TeamPolicyTag<ES>, NA, TileA>;
  using team_member_t  = Impl::team_member_t<ES>;
  using scratch_view_t = typename inner_eval_t::scratch_view_t;

  inner_eval_t eval_;

  KOKKOS_FUNCTION ScratchAllocator(const NA& node, const TileA& tile,
                                   const team_member_t& team)
      : eval_(make_evaluator<TeamPolicyTag<ES>>(node, tile, team)) {}

  static std::size_t bytes(const TileA& tile) {
    return inner_eval_t::scratch_size_per_team(tile);
  }
  KOKKOS_FUNCTION scratch_view_t      get() const { return eval_.scratch(); }
  KOKKOS_FUNCTION const inner_eval_t& inner_eval() const { return eval_; }
  template <typename Team, typename Idx>
  KOKKOS_FUNCTION auto stage(const Team& team, const Idx& idx) const {
    return eval_(team, idx);
  }
};

// InputTag operand: carves a staging buffer of the CANONICAL shape from the
// team cursor, in the constructor -- this is the one form that actually applies
// PermSeq, since an input operand's own tile is in its native order and the
// staged copy is in the consumer's. The stored inner Evaluator sets its tiled
// view up once here rather than on every stage() call. Valid for either
// OuterOpTag -- an input operand is staged identically regardless of the
// containing operation.
template <typename ES, typename OuterOpTag, typename NA, typename V,
          typename TileA, typename PermSeq>
  requires(Impl::has_node_tag_v<InputTag, NA>)
struct ScratchAllocator<TeamPolicyTag<ES>, OuterOpTag, NA, V, TileA, PermSeq> {
  using inner_eval_t  = Evaluator<TeamPolicyTag<ES>, NA, TileA>;
  using team_member_t = Impl::team_member_t<ES>;
  // TileA is a leaf tile here (only a fused operand is tiled with a rank-less
  // Tile<A,B,C> bundle, and that is the specialization above), so the identity
  // fallback for an unpermuted operand reads straight off its rank.
  using perm_seq_t =
      std::conditional_t<std::is_void_v<PermSeq>,
                         std::make_integer_sequence<int, TileA::rank>, PermSeq>;
  using axes_t         = OperandAxes<perm_seq_t>;
  using scratch_view_t = decltype(Impl::alloc_scratch_tile<V, ES>(
      std::declval<team_member_t>(),
      axes_t::to_canon_tile(std::declval<TileA>())));

  inner_eval_t   eval_;
  scratch_view_t scratch_;

  KOKKOS_FUNCTION ScratchAllocator(const NA& node, const TileA& tile,
                                   const team_member_t& team)
      : eval_(make_evaluator<TeamPolicyTag<ES>>(node, tile, team)),
        scratch_(Impl::alloc_scratch_tile<V, ES>(
            team, axes_t::to_canon_tile(tile))) {}

  static std::size_t bytes(const TileA& tile) {
    return Impl::scratch_tile_bytes<V, ES>(axes_t::to_canon_tile(tile));
  }
  KOKKOS_FUNCTION scratch_view_t get() const { return scratch_; }
  template <typename Team, typename Idx>
  KOKKOS_FUNCTION auto stage(const Team& team, const Idx& idx) const {
    return eval_(team, idx);
  }
};

// IntermTag = the output (C) accumulator slot: no operand node, hence no
// native-vs-canonical distinction to reconcile (PermSeq is left at its `void`
// default by every instantiation) and no source to stage from, hence no
// stage(). It carves its tile in the constructor like the InputTag form above.
// Uniform across every OuterOpTag.
template <typename ES, typename OuterOpTag, typename V, typename CanonTile>
struct ScratchAllocator<TeamPolicyTag<ES>, OuterOpTag, IntermTag, V,
                        CanonTile> {
  using team_member_t  = Impl::team_member_t<ES>;
  using scratch_view_t = decltype(Impl::alloc_scratch_tile<V, ES>(
      std::declval<team_member_t>(), std::declval<CanonTile>()));

  scratch_view_t scratch_;

  KOKKOS_FUNCTION ScratchAllocator(const CanonTile&     tile,
                                   const team_member_t& team)
      : scratch_(Impl::alloc_scratch_tile<V, ES>(team, tile)) {}

  static std::size_t bytes(const CanonTile& tile) {
    return Impl::scratch_tile_bytes<V, ES>(tile);
  }
  KOKKOS_FUNCTION scratch_view_t get() const { return scratch_; }
};
