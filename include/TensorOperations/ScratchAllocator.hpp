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
//     • a full operand NodeHandle    — an InputTag leaf, a fused node that
//       produces its own scratch, or a SlotTag naming a buffer another node
//       owns (ContractionTag / CombineTag)
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

// Fused operand (ContractionTag or CombineTag -- see produces_own_scratch_v):
// the inner evaluator already owns its output scratch tile (allocated in its
// own constructor), so this form carves nothing of its own -- get() hands that
// buffer back directly (zero-copy, no cursor bump), valid for either OuterOpTag
// and also for a PERMUTED operand, which the consumer stages by reordering that
// same buffer in place rather than by copying it elsewhere. bytes() is the
// inner evaluator's full recursive scratch_size_per_team and ignores PermSeq:
// neither the zero-copy nor the in-place-reorder path costs a byte beyond it.
//
// One specialization covers both fused kinds because the containing operation
// only ever asks three things of an operand -- its scratch view type, that
// buffer, and "evaluate yourself at this tile" -- and a combine answers all
// three the same way a contraction does. The single asymmetry is that a
// combine's operator() returns one interm PER OUTPUT, normalized by
// Impl::single_result (which is also where the phase-1 NumOut == 1 restriction
// is stated).
template <typename ES, typename OuterOpTag, typename NA, typename V,
          typename TileA, typename PermSeq>
  requires(Impl::produces_own_scratch_v<NA>)
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
  KOKKOS_FUNCTION scratch_view_t get() const { return eval_.scratch(); }
  template <typename Team, typename Idx>
  KOKKOS_FUNCTION auto stage(const Team& team, const Idx& idx) const {
    return Impl::single_result(eval_(team, idx));
  }
};

// SlotTag operand: names a buffer some OTHER node owns and already filled.
// Carves nothing, evaluates nothing, and costs nothing -- bytes() is 0, because
// the buffer is charged once by whoever allocates the slot store, not once per
// consumer. That zero is the whole point of the node kind: the fused form above
// charges its operand's full recursive scratch_size_per_team PER CONSUMER, and
// holds a whole inner Evaluator by value to do it, which is what makes a shared
// subtree cost N times over.
//
// stage() hands back an IntermTag node over that same buffer rather than the
// slot node itself, for a specific reason: Specialization 8 dispatches on
// whether the SOURCE's storage_type equals the destination scratch type
// (Evaluator/Team.hpp), and equality is what selects its zero-copy passthrough
// branch over its copy branch. A slot's storage IS a destination-typed scratch
// view -- alloc_scratch_tile builds both -- so the passthrough fires and no
// data moves. The NoHook is not a default but a guarantee: see NodeHandle.hpp
// on why a slot carries no hook, and Impl::operand_stageable_v on the matching
// prohibition against reordering one in place.
//
// PermSeq is accepted and ignored, as in the fused form. A slot is only ever
// presented in its own order -- a differently-ordered consumer takes the
// relabel path (Impl::operand_relabelable_v) or names the buffer through a
// second slot node -- and operand_stageable_v rejects anything else before it
// reaches here.
template <typename ES, typename OuterOpTag, typename NA, typename V,
          typename TileA, typename PermSeq>
  requires(Impl::has_node_tag_v<SlotTag, NA>)
struct ScratchAllocator<TeamPolicyTag<ES>, OuterOpTag, NA, V, TileA, PermSeq> {
  using team_member_t  = Impl::team_member_t<ES>;
  using scratch_view_t = typename NA::storage_type;
  using interm_type =
      NodeHandle<IntermTag, scratch_view_t,
                 std::integral_constant<int, NA::Rank>, ES, NoHook>;

  scratch_view_t storage_;

  KOKKOS_FUNCTION ScratchAllocator(const NA& node, const TileA&,
                                   const team_member_t&)
      : storage_(node.storage_) {}

  static std::size_t             bytes(const TileA&) { return 0; }
  KOKKOS_FUNCTION scratch_view_t get() const { return storage_; }
  template <typename Team, typename Idx>
  KOKKOS_FUNCTION interm_type stage(const Team&, const Idx&) const {
    return {storage_, NoHook{}};
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
//
// ADOPTION. The second constructor takes an already-carved buffer and carves
// nothing, so an evaluator can write into storage someone ELSE allocated. That
// is what lets a DAG driver own every intermediate: it carves one buffer per
// node up front and hands each evaluator its own, instead of each evaluator
// bump-allocating from the team cursor at construction. Without it the driver
// would have to carve a second buffer per node and copy.
//
// Adoption is a CONSTRUCTOR choice, not a type-level one, and deliberately so:
// both forms produce the same scratch_view_t, so an adopting evaluator is
// type-identical to a carving one and nothing downstream has to know which it
// got. The one thing that cannot follow from the constructor is bytes(), which
// is static and has no instance to ask -- an adopting evaluator must simply not
// have its output charged, which is why the sizing split lives in the
// evaluators (scratch_size_per_team vs operand_scratch_size_per_team) rather
// than as a flag here.
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

  // Adopt a buffer carved elsewhere. Takes no team and bumps no cursor.
  KOKKOS_FUNCTION explicit ScratchAllocator(const scratch_view_t& adopted)
      : scratch_(adopted) {}

  static std::size_t bytes(const CanonTile& tile) {
    return Impl::scratch_tile_bytes<V, ES>(tile);
  }
  KOKKOS_FUNCTION scratch_view_t get() const { return scratch_; }
};
