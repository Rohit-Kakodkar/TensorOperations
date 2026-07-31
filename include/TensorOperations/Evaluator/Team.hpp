#pragma once
// Included from within TensorOperations namespace by Evaluator.hpp

namespace Impl {

// Extract extent D from a scratch View: compile-time when the layout is static
// (avoids a runtime `.extent()` call), runtime otherwise. Replaces the three
// repeated if-constexpr lambdas that extract SA, SK, SB in accumulate_block.
template <int D, typename V>
KOKKOS_FORCEINLINE_FUNCTION int scratch_extent(const V& v) noexcept {
  if constexpr (V::layout_t::is_static)
    return V::layout_t::extent(D);
  else
    return v.extent(D);
}

// View a staged scratch tile as a 2D GEMM matrix: collapse the first `Split`
// tile dims into rows and the rest into columns. Split is the free-mode count
// for A / the contracted-mode count for B / the free-A count for C.
//
// The THREE-argument reshape, with the target order stated rather than inferred
// from the source layout's kind. For the row-major scratch this always gets --
// alloc_scratch_tile builds make_tile_layout(tile, LayoutRight{}) and nothing
// else supplies a staged layout -- LayoutRight reproduces the two-argument
// form's result type exactly, so this is behaviour- and codegen-neutral. It was
// checked before the switch by asserting the two spellings agree at every call
// site in the tree, and by diffing the emitted SASS.
//
// The order is fixed here only because the source's is. If a staged operand
// ever reaches the GEMM in a non-row-major order -- the reason to want an
// explicit order at all -- this argument has to be DERIVED from that layout,
// not left as LayoutRight, and the collapse then needs a guard that it still
// enumerates rows the way B and C do. Passing a constant here does not make
// that case work; it only puts the parameter where it will go.
template <int Split, typename View, typename Tile>
KOKKOS_FORCEINLINE_FUNCTION auto as_matrix(const View& v, const Tile& t) {
  return reshape(v, prefix_product(t, rank_c<Split>), LayoutRight{});
}

// Ceil-division. The template form binds the divisor at compile time so the
// compiler lowers it to a multiply-shift instead of a runtime integer division
// (expensive on GPU); the runtime form is used only when the divisor genuinely
// isn't known (dynamic tiles).
template <int Divisor>
KOKKOS_FORCEINLINE_FUNCTION int ceil_div(int n) noexcept {
  static_assert(Divisor > 0, "ceil_div divisor must be positive");
  return (n + Divisor - 1) / Divisor;
}
KOKKOS_FORCEINLINE_FUNCTION int ceil_div(int n, int divisor) noexcept {
  return (n + divisor - 1) / divisor;
}

// Number of tiles that cover extent `ext` along dim `d` of `tile`. For a static
// tile the divisor is compile-time, so we pick the matching dim's constexpr
// extent (keeping the cheap multiply-shift); dynamic tiles use a runtime
// divide.
template <typename Tile>
KOKKOS_FUNCTION int tile_count_along(const Tile& tile, int d,
                                     int ext) noexcept {
  if constexpr (Tile::is_static) {
    int n = 0;
    [&]<int... Ds>(std::integer_sequence<int, Ds...>) {
      ((d == Ds ? (n = ceil_div<Tile::extent(Ds)>(ext)) : 0), ...);
    }(std::make_integer_sequence<int, Tile::rank>{});
    return n;
  } else {
    return ceil_div(ext, tile.extent(d));
  }
}

// --- compile-time index folds ------------------------------------------------
//
// Fold f over the index range [0, N). f is invoked as
// f(std::integral_constant<std::size_t, K>{}) rather than f(K), because the
// per-operand aliases these fold over (op_is_relabeled_v<K>, op_alloc_t<K>,
// native_tile_t<K>, ...) are templates on K -- a runtime index would not
// compile. Each callable recovers it with
// `constexpr std::size_t K = decltype(k)::value;`.
//
// Host-side only: these serve scratch_size_per_team's sizing pass and
// compile-time asserts. The device-side per-operand builders deliberately keep
// their explicit index_sequence form, because they rely on braced-init
// left-to-right evaluation order to advance the team's scratch cursor.
template <std::size_t N, typename F>
std::size_t sum_over_index(F f) {
  return [&]<std::size_t... Ks>(std::index_sequence<Ks...>) {
    return (std::size_t{0} + ... +
            f(std::integral_constant<std::size_t, Ks>{}));
  }(std::make_index_sequence<N>{});
}

template <std::size_t N, typename F>
constexpr bool all_of_index(F f) {
  return [&]<std::size_t... Ks>(std::index_sequence<Ks...>) {
    return (f(std::integral_constant<std::size_t, Ks>{}) && ...);
  }(std::make_index_sequence<N>{});
}

// --- team-scratch tile allocation -------------------------------------------
//
// Every scratch-tier evaluator stages tiles as LayoutRight scratch views sized
// by a tile spec. alloc_scratch_tile carves one such tile out of team scratch
// (each call advances the team's scratch cursor); scratch_tile_bytes is the
// matching per-tile contribution to scratch_size_per_team.
template <typename ValueType, typename ES>
using scratch_backing_t =
    Kokkos::View<ValueType*, typename ES::scratch_memory_space,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <typename ValueType, typename ES, typename Team, typename Tile>
KOKKOS_FORCEINLINE_FUNCTION auto alloc_scratch_tile(const Team& team,
                                                    const Tile& tile) {
  const auto layout   = make_tile_layout(tile, LayoutRight{});
  using tile_layout_t = std::decay_t<decltype(layout)>;
  scratch_backing_t<ValueType, ES> backing(
      team.team_scratch(0), static_cast<std::size_t>(layout.size()));
  return ScratchView<ValueType, ES, tile_layout_t>{backing, layout};
}

template <typename ValueType, typename ES, typename Tile>
std::size_t scratch_tile_bytes(const Tile& tile) {
  return scratch_backing_t<ValueType, ES>::shmem_size(
      static_cast<std::size_t>(make_tile_layout(tile, LayoutRight{}).size()));
}

// CRTP mix-in supplying the proxy-assignment calling convention
// `evaluator(team, tile_idx) = src`, shared by the relabel and stage
// evaluators (Specializations 7 and 8). A plain operator= never sees tile_idx,
// so operator() returns a proxy binding (this, tile_idx) and the assignment
// forwards to the derived evaluator's assign(tile_idx, src) -- the derived
// class supplies only that one method.
//
// Parameterized on the derived evaluator's own template arguments rather than
// on the derived type directly, so Rank can be read off the node and the
// derived type reconstructed as Evaluator<PolicyTag, NodeType, Tiling>.
//
// The usual dependent-base lookup pitfall does not apply here: nothing inside
// a derived evaluator names operator() or the proxy unqualified -- callers
// invoke operator() on the complete derived type, where ordinary member lookup
// finds the inherited one.
template <typename PolicyTag, typename NodeType, typename Tiling>
struct TileAssignable {
  using derived_t           = Evaluator<PolicyTag, NodeType, Tiling>;
  static constexpr int Rank = NodeType::Rank;

  struct AssignProxy {
    const derived_t*         self;
    Kokkos::Array<int, Rank> tile_idx;

    template <typename SrcNode>
    KOKKOS_FUNCTION auto operator=(const SrcNode& src) const {
      return self->assign(tile_idx, src);
    }
  };

  template <typename Team>
  KOKKOS_FUNCTION AssignProxy
  operator()(const Team&, Kokkos::Array<int, Rank> tile_idx) const {
    return AssignProxy{static_cast<const derived_t*>(this), tile_idx};
  }
};

// Stage one operand tile into the allocator's own scratch and return the
// resulting interm handle. `stage_tile` carries both the operand's native tile
// shape and the native->canonical permutation, so the native tile index is
// derived here from canon_idx rather than being scattered by hand at each call
// site; the destination likewise comes from the same allocator that produces
// the source, so neither pair can disagree. The assignment drives the
// stage-or-passthrough evaluator (Specialization 8), whose own dispatch on the
// source's storage type picks copy+reorder, in-place reorder, or zero-copy
// passthrough.
template <typename PolicyTag, typename Team, typename StageTileT, std::size_t R,
          typename Alloc>
KOKKOS_FUNCTION auto stage_operand_into(const Team&                  team,
                                        const StageTileT&            stage_tile,
                                        const Kokkos::Array<int, R>& canon_idx,
                                        const Alloc&                 alloc) {
  using axes_t = OperandAxes<typename StageTileT::perm_seq>;
  auto stager  = make_evaluator<PolicyTag>(make_interm_node(alloc.get()),
                                           stage_tile, team);
  return stager(team, canon_idx) =
             alloc.stage(team, axes_t::to_native_idx(canon_idx));
}

}  // namespace Impl

// ScratchAllocator is defined here (after Impl::scratch_tile_bytes) because
// its TeamPolicyTag specializations call scratch_tile_bytes internally.
#include <TensorOperations/ScratchAllocator.hpp>

// ---------------------------------------------------------------------------
// Specialization 2: TeamPolicyTag + InputTag + Tile_  (global-view tier)
//
// Returns the operand's own tile as a pure, unstaged reinterpretation of its
// existing (global-memory) storage: no reordering, no data movement, no hook
// application. The tile stays in the operand's own declared mode order — a
// compile-time "reshape" for static tiles via subview_tile's
// OrderedSubviewLayout path (register-resident, no local-memory spill).
// Reordering into a canonical axis order and staging into scratch (applying
// the hook along the way) is the caller's explicit responsibility.
// ---------------------------------------------------------------------------
template <typename ES, TensorLike T, typename ModesSeq, typename HookOp,
          typename Tile_>
struct Evaluator<TeamPolicyTag<ES>, NodeHandle<InputTag, T, ModesSeq, HookOp>,
                 Tile_> {
  using node_type           = NodeHandle<InputTag, T, ModesSeq, HookOp>;
  using tiling_type         = Tile_;
  using policy_tag          = TeamPolicyTag<ES>;
  static constexpr int Rank = tiling_type::rank;
  using value_type          = typename node_type::value_type;
  using exec_space          = ES;
  using team_member_t       = Impl::team_member_t<exec_space>;

  using tiled_input_t = TiledView<TensorHandle<T, ModesSeq>, tiling_type>;
  using global_view_t = decltype(subview_tile(
      std::declval<tiled_input_t>(), std::declval<Kokkos::Array<int, Rank>>()));
  using interm_type =
      NodeHandle<IntermTag, global_view_t, std::integral_constant<int, Rank>,
                 exec_space, HookOp>;
  using result_type = interm_type;

  static_assert(node_type::Rank == Rank,
                "input staging tile must carry one extent per input mode");

  node_type     node;
  tiling_type   tiling;
  tiled_input_t tiled_input_;

  // team is accepted (unused) to keep this constructor's call signature
  // identical to every other team-tier evaluator's — existing call sites
  // construct all of them uniformly as (node, tile, team).
  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type t,
                            const team_member_t& team)
      : node(n), tiling(t), tiled_input_(tile_view(n.handle, t)) {}

  // Same (team, tile_idx) call signature as before; team is unused — this is
  // now pure pointer/layout arithmetic, no Kokkos::parallel_for.
  KOKKOS_FUNCTION result_type operator()(
      const team_member_t& team, Kokkos::Array<int, Rank> tile_idx) const {
    auto sv = subview_tile(tiled_input_, tile_idx);
    return result_type{sv, node.hook_op};
  }

  // Outer tile count along native dimension d (the operand's own declared
  // order; no permutation applied here).
  KOKKOS_FUNCTION int outer_extent(int d) const noexcept {
    return tiled_input_.extent(d);
  }
};

// ---------------------------------------------------------------------------
// Specialization 4: TeamPolicyTag + ContractionTag + Tile_  (scratch tier)
// ---------------------------------------------------------------------------
template <typename NA, typename NB, typename IntCRank, typename S, typename ES,
          typename HookOp, typename CModesSeq, typename PermCSeq,
          typename TileA, typename TileB, typename TileC>
struct Evaluator<TeamPolicyTag<ES>,
                 NodeHandle<ContractionTag, NA, NB, IntCRank, S, ES, HookOp,
                            CModesSeq, PermCSeq>,
                 TensorOperations::Tile<TileA, TileB, TileC>> {
  using node_type = NodeHandle<ContractionTag, NA, NB, IntCRank, S, ES, HookOp,
                               CModesSeq, PermCSeq>;
  using tiling_type         = TensorOperations::Tile<TileA, TileB, TileC>;
  using policy_tag          = TeamPolicyTag<ES>;
  static constexpr int Rank = node_type::Rank;
  using value_type          = S;
  using exec_space          = ES;

 private:
  static constexpr int RankC = Rank;
  static constexpr int NumK  = node_type::NumContracted;
  static constexpr int RankA = NA::Rank;
  static constexpr int RankB = NB::Rank;
  static constexpr int FreeA = RankA - NumK;
  static constexpr int FreeB = RankB - NumK;
  static_assert(FreeA + FreeB == RankC,
                "free-mode counts must sum to the output rank");

  // The free-mode (output) tile of each operand, in the axis order that
  // operand's own evaluator hands back: the tile itself for an input operand,
  // or the fused sub-contraction's CANONICAL output tile for a contraction
  // operand (its bundle's `.c` slot is written in the sub-contraction's user
  // order, which permC reconciles -- see Impl::canonical_c_tile). Everything
  // below that describes A or B "natively" -- the GEMM sizing, the k-tile
  // counts, permA/permB -- is expressed against these leaf tiles, and all three
  // agree only because they are all canonical for a fused operand.
  using a_leaf_t = Impl::operand_leaf_tile_t<NA, TileA>;
  using b_leaf_t = Impl::operand_leaf_tile_t<NB, TileB>;

  // Check operand output-tile ranks match node ranks
  static_assert(a_leaf_t::rank == RankA, "TileA rank must match node A");
  static_assert(b_leaf_t::rank == RankB, "TileB rank must match node B");
  static_assert(TileC::rank == RankC, "TileC rank must match output rank");

  // Per-operand permutations from the operand's own axis order into the GEMM's
  // canonical order (freeA ++ contracted for A, contracted ++ freeB for B).
  // They are computed from each operand's modes_seq, so "the operand's own
  // order" means the input's declared labels for a leaf and the CANONICAL ones
  // for a fused sub-contraction (whose modes_seq is canonical) -- exactly the
  // order Impl::operand_leaf_tile_t presents that operand's tile in. permC maps
  // this contraction's canonical output back to the user order. All three are
  // identity for a canonical contraction, in which case every canonicalization
  // below is a no-op passthrough.
  using permA_seq =
      Impl::permA_seq_t<typename NA::modes_seq, typename NB::modes_seq>;
  using permB_seq =
      Impl::permB_seq_t<typename NA::modes_seq, typename NB::modes_seq>;
  using permC_seq = PermCSeq;

  // Native <-> canonical axis relation per operand. This evaluator holds each
  // operand's own-order leaf tile and derives the canonical one (the combine
  // evaluator runs the same vocabulary in the opposite direction).
  using axes_a = OperandAxes<permA_seq>;
  using axes_b = OperandAxes<permB_seq>;
  using axes_c = OperandAxes<permC_seq>;

  static_assert(Impl::fusable_operand_v<NA> && Impl::fusable_operand_v<NB>,
                "contraction operands must be input nodes or fused nodes "
                "(contraction / combine); a multi-output slice is a terminal "
                "output, not an operand");

  // Each operand's leaf tile brought into this contraction's canonical order.
  // Only the LEAF is permuted, never the whole Tile<A,B,C> bundle a fused
  // operand is tiled with: that bundle belongs to the sub-contraction, whose
  // evaluator is keyed on it and canonicalizes its own A/B/C internally.
  using a_out_canon_t  = typename axes_a::template canon_tile_t<a_leaf_t>;
  using b_out_canon_t  = typename axes_b::template canon_tile_t<b_leaf_t>;
  using tile_c_canon_t = typename axes_c::template canon_tile_t<TileC>;

  // A fused contraction operand may sit in ANY position: it is staged by
  // reordering its own C scratch in place (see Impl::operand_stageable_v for
  // the rule and why a shape-changing permutation is excluded). Rejecting one
  // here, early, is what keeps it from reaching Specialization 8 as a scratch
  // type that no longer matches the one alloc_a_.get()/alloc_b_.get() produces
  // -- which would silently take the COPY branch, whose source and destination
  // are the very same buffer for a fused operand.
  static_assert(Impl::operand_stageable_v<NA, TileA, permA_seq>,
                "permA must preserve fused operand A's tile extents (it is "
                "staged in place); see Impl::operand_stageable_v");
  static_assert(Impl::operand_stageable_v<NB, TileB, permB_seq>,
                "permB must preserve fused operand B's tile extents (it is "
                "staged in place); see Impl::operand_stageable_v");

  // One allocator per slot, each owning the scratch it carves. Every operand
  // passes its real perm: the InputTag form stages a canonically-shaped copy
  // with it, the ContractionTag form ignores it (it reorders its own C scratch
  // in place downstream instead).
  using alloc_a_t = ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, NA,
                                     value_type, TileA, permA_seq>;
  using alloc_b_t = ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, NB,
                                     value_type, TileB, permB_seq>;
  using alloc_c_t = ScratchAllocator<TeamPolicyTag<ES>, ContractionTag,
                                     IntermTag, value_type, tile_c_canon_t>;

 public:
  // The C slot's allocator already derives this from tile_c_canon_t; naming it
  // here keeps interm_type/result_type spelled in terms of one layout.
  using scratch_view_t = typename alloc_c_t::scratch_view_t;
  using interm_type =
      NodeHandle<IntermTag, scratch_view_t, std::integral_constant<int, Rank>,
                 exec_space, HookOp>;
  using result_type   = interm_type;
  using team_member_t = Impl::team_member_t<exec_space>;

  // The operands' full tiling specs (a nested Tile<A,B,C> bundle, when fused)
  // are not retained: only the allocators, built from them here, still need
  // them. What the evaluator itself works in is the leaf tiles below.
  node_type      node;
  a_leaf_t       a_leaf_;  // A's leaf tile, in A's own axis order
  b_leaf_t       b_leaf_;  // B's leaf tile, in B's own axis order
  a_out_canon_t  a_tile_;  // ... and in this contraction's canonical order
  b_out_canon_t  b_tile_;
  tile_c_canon_t c_tile_;
  alloc_a_t      alloc_a_;
  alloc_b_t      alloc_b_;
  alloc_c_t      alloc_c_;
  result_type    result_;

  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type t,
                            const team_member_t& team)
      : node(n),
        a_leaf_(Impl::canonical_c_tile<NA>(t.a)),
        b_leaf_(Impl::canonical_c_tile<NB>(t.b)),
        a_tile_(axes_a::to_canon_tile(a_leaf_)),
        b_tile_(axes_b::to_canon_tile(b_leaf_)),
        c_tile_(axes_c::to_canon_tile(t.c)),
        alloc_a_(n.node_a, t.a, team),
        alloc_b_(n.node_b, t.b, team),
        alloc_c_(c_tile_, team) {
    result_.hook_op  = node.hook_op;
    result_.storage_ = alloc_c_.get();
  }

  // This evaluator's output (C) scratch tile. Read by a PARENT contraction or
  // combine that fuses this node as an operand: its ScratchAllocator stages
  // this buffer directly rather than copying it elsewhere.
  KOKKOS_FUNCTION scratch_view_t scratch() const { return alloc_c_.get(); }

  KOKKOS_FUNCTION result_type operator()(
      const team_member_t& team, Kokkos::Array<int, Rank> tile_idx) const {
    TIMING_SCOPE_ENTER(g_timing_stats.contraction_accum_time,
                       g_timing_stats.contraction_accum_count);
    // Zero the output scratch at the start of every evaluation so this operator
    // is re-runnable. A parent contraction re-invokes a fused operand once per
    // contracted tile (recompute) on the same evaluator instance, and each run
    // must accumulate from a clean C.
    auto c = alloc_c_.get();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, c.size()),
                         [c](int i) { c.data()[i] = S{0}; });
    team.team_barrier();
    // Per-mode k-tile counts (from A's own leaf tile/shape, read through
    // permA_seq) and their product. a_leaf_ and node.node_a.shape() are both
    // in A's own axis order -- the operand's declared order for an input leaf,
    // canonical for a fused sub-contraction (whose modes_seq, and hence
    // permA_seq, is canonical too) -- so one indexing scheme covers both and
    // this needs no dispatch on the operand's tag. Contracted tiles share
    // single-buffered scratch, so the walk over the linearized contracted-tile
    // space is serialized by team barriers.
    const auto               pA = Impl::seq_to_karray(permA_seq{});
    Kokkos::Array<int, NumK> n_k_tiles{};
    int                      total_k_tiles = 1;
    for (int i = 0; i < NumK; ++i) {
      int native_dim = pA[FreeA + i];
      n_k_tiles[i]   = Impl::tile_count_along(a_leaf_, native_dim,
                                              node.node_a.shape()[native_dim]);
      total_k_tiles *= n_k_tiles[i];
    }
    // Mixed-radix counter (LSB fastest); carry-increment avoids per-step
    // div/mod.
    Kokkos::Array<int, NumK> k_tile_idx{};
    for (int lin = 0; lin < total_k_tiles; ++lin) {
      accumulate_block(team, tile_idx, k_tile_idx);
      for (int i = NumK - 1; i >= 0; --i) {
        if (++k_tile_idx[i] < n_k_tiles[i]) break;
        k_tile_idx[i] = 0;
      }
    }
    TIMING_SCOPE_EXIT(g_timing_stats.contraction_accum_time,
                      g_timing_stats.contraction_accum_count);
    return result_;
  }

  KOKKOS_FUNCTION int outer_extent_canon(int d) const noexcept {
    return Impl::tile_count_along(c_tile_, d, node.shape()[d]);
  }

  // Each operand allocator is handed its own NATIVE tiling spec and derives its
  // own cost from that plus the perm already baked into its type: a fused
  // operand recurses into the sub-contraction's full scratch_size_per_team
  // (which only accepts that native bundle), an input operand sizes a staging
  // buffer of the canonical shape. The output (C) slot has no operand and so
  // no perm to apply.
  static std::size_t scratch_size_per_team(const tiling_type& t) {
    return alloc_c_t::bytes(axes_c::to_canon_tile(t.c)) +
           alloc_a_t::bytes(t.a) + alloc_b_t::bytes(t.b);
  }

 private:
  // Validate that each GEMM dimension (SA, SK, SB) is a multiple of its
  // register-block factor (MT, NT, NR). Static tiles are checked at compile
  // time via static_assert; dynamic tiles fall back to runtime assert.
  template <int MT, int NT, int NR, typename AView, typename BView>
  KOKKOS_FUNCTION static void validate_gemm_tile_dims(const AView&,
                                                      const BView&, int SA,
                                                      int SK, int SB) {
    if constexpr (AView::layout_t::is_static) {
      constexpr int SAs = AView::layout_t::extent(0);
      constexpr int SKs = AView::layout_t::extent(1);
      static_assert(
          SAs >= MT && SAs % MT == 0,
          "team GEMM: operand row dim (SA) must be a multiple of the register "
          "row block MT; enlarge the output C tile's free extent");
      static_assert(
          SKs >= NT && SKs % NT == 0,
          "team GEMM: operand contracted dim (SK) must be a multiple of the "
          "register depth NT; enlarge the contracted tile extents");
    } else {
      assert(SA >= MT && SA % MT == 0 &&
             "team GEMM: SA must be a multiple of the register row block MT");
      assert(SK >= NT && SK % NT == 0 &&
             "team GEMM: SK must be a multiple of the register depth NT");
    }
    if constexpr (BView::layout_t::is_static) {
      constexpr int SBs = BView::layout_t::extent(1);
      static_assert(
          SBs >= NR && SBs % NR == 0,
          "team GEMM: operand free dim (SB) must be a multiple of the register "
          "column block (2*W on CPU); enlarge the output C tile's free extent");
    } else {
      assert(
          SB >= NR && SB % NR == 0 &&
          "team GEMM: SB must be a multiple of the register column block NR");
    }
  }

  // Stage one contracted tile of A and B into scratch and accumulate the block
  // product into C. The barriers bracket the shared-scratch reads/writes;
  // consecutive calls reuse the same scratch (no double buffering).
  KOKKOS_FUNCTION void accumulate_block(
      const team_member_t& team, const Kokkos::Array<int, Rank>& c_tile_idx,
      const Kokkos::Array<int, NumK>& k_tile_idx) const {
    constexpr int         P = RankC + NumK;
    Kokkos::Array<int, P> part{};
    for (int d = 0; d < RankC; ++d) part[d] = c_tile_idx[d];
    for (int i = 0; i < NumK; ++i) part[RankC + i] = k_tile_idx[i];

    Kokkos::Array<int, RankA> a_tile_idx{};
    for (int j = 0; j < RankA; ++j)
      a_tile_idx[j] = part[j < FreeA ? j : RankC + (j - FreeA)];
    Kokkos::Array<int, RankB> b_tile_idx{};
    for (int j = 0; j < RankB; ++j)
      b_tile_idx[j] = part[j < NumK ? RankC + j : FreeA + (j - NumK)];

    // a_tile_idx/b_tile_idx are canonical (free++contracted) order -- what
    // a_tile_/b_tile_ and the staged scratch use. stage_operand_into scatters
    // each into the operand's own axis order to drive that operand's evaluator,
    // then hands the perm to Specialization 8, which reorders an input
    // operand's copy on the way into scratch and a fused operand's own scratch
    // in place.
    auto staged_a = Impl::stage_operand_into<TeamPolicyTag<ES>>(
        team, StageTile<a_leaf_t, permA_seq>{a_leaf_}, a_tile_idx, alloc_a_);
    auto staged_b = Impl::stage_operand_into<TeamPolicyTag<ES>>(
        team, StageTile<b_leaf_t, permB_seq>{b_leaf_}, b_tile_idx, alloc_b_);
    team.team_barrier();

    // View each operand's scratch as a 2D GEMM matrix: A[SA,SK], B[SK,SB],
    // C[SA,SB] (all row-major). For static tiles these carry compile-time
    // extents, and the register-tiled views below carry compile-time strides.
    // The shape comes from the CANONICAL leaf tile even when the staged view's
    // own layout type is the operand's native one (a fused operand keeps its
    // buffer's type across an in-place reorder): reshape only checks the total
    // element count, and after staging the buffer holds canonically ordered
    // data contiguously row-major, which is all the GEMM reads.
    auto a = Impl::as_matrix<FreeA>(staged_a.storage_, a_tile_);  // [SA,SK]
    auto b = Impl::as_matrix<NumK>(staged_b.storage_, b_tile_);   // [SK,SB]
    auto c = Impl::as_matrix<FreeA>(alloc_c_.get(), c_tile_);     // [SA,SB]

    const int SA = Impl::scratch_extent<0>(a);
    const int SK = Impl::scratch_extent<1>(a);
    const int SB = Impl::scratch_extent<1>(b);

    namespace KE    = Kokkos::Experimental;
    using simd_t    = KE::simd<value_type>;
    constexpr int W = static_cast<int>(simd_t::size());

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    constexpr int MT = 4;  // output rows / item
    constexpr int NT = 2;  // output cols / item
    // NR=2 (not 4): the shared-load bank is (col_block*NR + n) mod 32, so a
    // power-of-two NR > 2 makes register-blocked lanes alias banks on rows
    // wider than 32 elems. NR=2 halves the conflicts AND drops register
    // pressure (64->48 regs, occupancy 48->60%): measured +20% matmul / +28%
    // contraction on A100 vs NR=4. (Tuning NR beats address swizzling, which
    // only regressed.)
    constexpr int NR = 2;  // C columns / item
#else
    constexpr int MT = 8;  // output rows / item
    constexpr int NT = 8;  // output cols / item
    // NR = 2*W (two SIMD vectors, not one): gives MT*(NR/W)=16 independent FMA
    // accumulators, enough to hide the FMA latency on AVX-512 (2 FMA units)
    // where NR=W left only 8 and under-fed them. Measured ~+8% on serial matmul
    // N=512. Requires the operand free-dim SB to be a multiple of 2*W; static
    // tiles narrower than that are rejected below rather than silently
    // mis-tiled.
    constexpr int NR = 2 * W;  // simd vectors spanning the columns
#endif

    // The register kernel has no remainder path: each staged GEMM dim must be a
    // multiple of its block factor (SA%MT, SK%NT, SB%NR), else tile_view either
    // yields a zero outer extent (a hard error) or silently drops the remainder
    // (wrong results). Static tiles are rejected at compile time with clear
    // messages; dynamic tiles get the same guard at runtime in debug builds.
    validate_gemm_tile_dims<MT, NT, NR>(a, b, SA, SK, SB);

    using RegA = StaticTile<MT, NT>;
    using RegB = StaticTile<NT, NR>;
    using RegC = StaticTile<MT, NR>;

    // Position-preserving, compile-time-strided tiled views over the row-major
    // scratch. The index order is interleaved (outer_d, inner_d) per dim:
    //   a_reg(bi, i, k0, k)   b_reg(k0, k, bj, n)   c_reg(bi, i, bj, n)
    // The W-wide inner column is the last dim (stride 1), so the simd
    // loads/stores below are unit-stride.
    const auto a_reg = tile_view(a, RegA{});  // [SA/MT, MT, SK/NT, NT]
    const auto b_reg = tile_view(b, RegB{});  // [SK/NT, NT, SB/NR, NR]
    auto       c_reg = tile_view(c, RegC{});  // [SA/MT, MT, SB/NR, NR]

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, (SA / MT) * (SB / NR)), [=](int t) {
          const int bi = t / (SB / NR);  // C tile-block row
          const int bj = t % (SB / NR);  // C tile-block column
          simd_t    acc[MT][NR / W];
          TENSOR_PRAGMA_UNROLL
          for (int i = 0; i < MT; ++i) TENSOR_PRAGMA_UNROLL
          for (int n = 0; n < NR; n += W)
            acc[i][n / W] = simd_t(&c_reg(bi, i, bj, n), KE::simd_flag_default);

          // One contracted register-block (NT-deep rank-1 updates).
          for (int k0 = 0; k0 < SK / NT; ++k0) {
            for (int k = 0; k < NT; ++k) {
              simd_t br[NR / W];
              TENSOR_PRAGMA_UNROLL
              for (int n = 0; n < NR; n += W)
                br[n / W] = simd_t(&b_reg(k0, k, bj, n), KE::simd_flag_default);
              TENSOR_PRAGMA_UNROLL
              for (int i = 0; i < MT; ++i) {
                const simd_t a_i(a_reg(bi, i, k0, k));  // broadcast A
                TENSOR_PRAGMA_UNROLL
                for (int n = 0; n < NR; n += W)
                  acc[i][n / W] += a_i * br[n / W];
              }
            }
          }

          TENSOR_PRAGMA_UNROLL
          for (int i = 0; i < MT; ++i) TENSOR_PRAGMA_UNROLL
          for (int n = 0; n < NR; n += W)
            KE::simd_unchecked_store(acc[i][n / W], &c_reg(bi, i, bj, n),
                                     KE::simd_flag_default);
        });
    team.team_barrier();
  }
};

// ---------------------------------------------------------------------------
// Specialization 5: TeamPolicyTag + CombineTag + Tile_  (pointwise scratch
// tier)
//
// P{modes} = fn(A{modes}, B{modes}, ...), a pure per-coordinate combine over N
// operands (no mode reduced). Structurally a stripped-down contraction
// evaluator: stage each operand tile into scratch (reusing the input evaluator,
// Specialization 2, which already handles permuted operands; the contraction
// evaluator, Specialization 4, for a contraction operand; or this very
// specialization, recursively, for a nested combine operand), then a single
// TeamVectorRange applies fn per element into the output scratch tile
// — no GEMM, no k-loop, no zeroing at this tier. Operands may be in any axis
// order: an input operand is gathered into the output (canonical) order via
// label_perm_seq, exactly as the contraction evaluator canonicalizes A/B, and a
// FUSED operand (contraction or combine -- see produces_own_scratch_v) is
// either gathered the same way or relabeled in place of a copy (see
// op_is_relabeled_v). A contraction operand may also sit in a non-canonical
// position of its own (non-identity permC); its tile bundle is read through
// that permC by leaf_tile_t. A combine node's modes are canonical by
// construction, so a combine operand has no permC to reconcile.
//
// Tile_ is either:
//   (a) a plain output tile (StaticTile/DynamicTile) — requires every operand
//       to be an input node (all operand tiles == output tile), or
//   (b) a CombineTile<OutTile, OpTile_0, ..., OpTile_{N-1}> — carries the
//       output tile plus a per-operand tile spec. An input operand's OpTile
//       must equal OutTile; a fused operand's OpTile is that sub-node's own
//       nested bundle (a Tile<A,B,C> for a contraction, whose C slot is in
//       that operand's own USER order; a CombineTile for a combine), and must
//       equal OutTile once read through its permC and gathered into the
//       combine's output order (see the static_assert below).
//
// Every operand presents the combine's output SHAPE, so all operand tiles
// align element-for-element with the output and one coordinate indexes them
// all. They need not share a LAYOUT: a permuted, unhooked fused operand is
// consumed as a zero-copy relabel of that sub-node's own output scratch (a
// strided view over the same bytes), while every other operand is staged into
// a LayoutRight buffer. Both the per-operand stagers and the resulting views
// therefore live in DeviceTuples, read by compile-time folds. fn receives the
// global output coordinate: fn(i_0, ..., i_{Rank-1}, v_0, ..., v_{N-1}).
//
// Multi-output: when fn returns a Kokkos::Array<V, NumOut>, the evaluator
// allocates NumOut output scratch tiles (all sharing the output tile layout)
// and writes result component m into output m; operator() returns a
// Kokkos::Array<interm_type, NumOut>. A scalar-returning fn is the NumOut == 1
// case. Every output shares the node's modes, so there is a single output tile.
// ---------------------------------------------------------------------------
namespace Impl {
// Extract operand K's tile spec from either form (host + device):
//   • plain tile form: the output tile itself, for every K,
//   • CombineTile form: the K-th OpTile in the bundle.
template <std::size_t K, typename Tile>
KOKKOS_FUNCTION auto combine_op_tile(const Tile& t) {
  if constexpr (is_combine_tile_v<Tile>) {
    return t.ops.template get<K>();
  } else {
    return t;  // plain tile: every operand uses the output tile
  }
}
template <std::size_t K, typename Tile>
using combine_op_tile_t =
    decltype(combine_op_tile<K>(std::declval<const Tile&>()));

// The tile a combine operand's OWN evaluator is constructed with (host +
// device), in the same shape as combine_op_tile above.
//
// An input operand's tile is supplied in the combine's output order, so the
// label gather has to be undone to recover the operand's native order. A FUSED
// operand's tile is instead a nested bundle -- a Tile<A,B,C> for a contraction,
// a CombineTile for a combine -- already spelled in that sub-node's own terms,
// and must pass through untouched: gathering it would be a hard error, since
// reorder_tile_value has no overload taking a bundle. The condition is
// produces_own_scratch_v rather than a tag test because "has a nested bundle"
// and "produces its own scratch" name the same set of operands. The discarded
// if-constexpr branch is never instantiated, so only the selected one has to be
// well-formed.
template <typename Node, typename Tile, typename PermSeq>
KOKKOS_FUNCTION auto combine_native_tile(const Tile& t, PermSeq) {
  if constexpr (produces_own_scratch_v<Node>)
    return t;  // bundle: already native
  else
    return OperandAxes<PermSeq>::to_native_tile(t);
}
template <typename Node, typename Tile, typename PermSeq>
using combine_native_tile_t =
    decltype(combine_native_tile<Node>(std::declval<const Tile&>(), PermSeq{}));

// The view type a combine reads one operand through: the allocator's own
// scratch view, or -- for a relabeled operand -- that same view retyped into
// the combine's axis order. Both staging paths hand back the allocator's view
// (Specialization 8's interm wraps the destination the allocator carved), so
// one parameter covers both. Selected by partial specialization rather than
// if-constexpr because this is a type, not a value, and reorder_view is
// SFINAE-constrained on the layout family: the relabel expression must not be
// formed at all for an operand that does not take that path.
template <bool Relabeled, typename View, typename PermSeq>
struct combine_op_view {
  using type = View;
};
template <typename View, typename PermSeq>
struct combine_op_view<true, View, PermSeq> {
  // Same expression Specialization 7 publishes as its dest_view_t; the combine
  // applies it inline rather than constructing that evaluator, because a
  // relabeled operand is NoHook by construction (see operand_relabelable_v) and
  // the hook pass is the only thing Specialization 7 would add.
  using type = decltype(reorder_view(std::declval<View>(), PermSeq{}));
};
template <bool Relabeled, typename View, typename PermSeq>
using combine_op_view_t =
    typename combine_op_view<Relabeled, View, PermSeq>::type;
}  // namespace Impl

template <typename CombineFn, typename IntCRank, typename S, typename ES,
          typename CModesSeq, typename IntNumOut, typename... Ops,
          typename Tile_>
struct Evaluator<TeamPolicyTag<ES>,
                 NodeHandle<CombineTag, CombineFn, IntCRank, S, ES, CModesSeq,
                            IntNumOut, Ops...>,
                 Tile_> {
  using node_type   = NodeHandle<CombineTag, CombineFn, IntCRank, S, ES,
                                 CModesSeq, IntNumOut, Ops...>;
  using tiling_type = Tile_;
  using policy_tag  = TeamPolicyTag<ES>;
  // The output tile: either Tile_ itself (plain form) or Tile_::out (bundle).
  using out_tile_t            = decltype(output_tile(std::declval<Tile_>()));
  static constexpr int Rank   = node_type::Rank;
  static constexpr int NumOps = node_type::NumOps;
  static constexpr int NumOut = node_type::NumOut;
  using value_type            = S;
  using exec_space            = ES;
  using tile_layout_t = decltype(make_tile_layout(std::declval<out_tile_t>(),
                                                  std::declval<LayoutRight>()));
  using scratch_view_t = ScratchView<value_type, exec_space, tile_layout_t>;

  // Every operand must be an input node or a fused node (contraction /
  // combine). A fused operand requires the CombineTile form, since it needs its
  // own nested tiling spec -- a Tile<A,B,C> bundle for a contraction, a
  // CombineTile for a combine -- rather than sharing the output tile.
  static_assert((Impl::fusable_operand_v<Ops> && ...),
                "combine evaluator: operands must be input nodes or fused "
                "nodes (contraction / combine); a multi-output slice is a "
                "terminal output, not an operand");
  static_assert(
      Impl::is_combine_tile_v<Tile_> ||
          (Impl::has_node_tag_v<InputTag, Ops> && ...),
      "combine node with any fused (non-input) operand requires a CombineTile "
      "bundle (use make_combine_tile(out_tile, op_tile_0, ..., "
      "op_tile_{N-1}))");
  // Verify the per-operand tile count matches the operand count when a
  // CombineTile bundle is supplied. Guarded by if constexpr because a direct
  // `Tile_::num_ops` reference is ill-formed on plain tiles even when
  // short-circuited by ||.
  static_assert(
      [] {
        if constexpr (Impl::is_combine_tile_v<Tile_>)
          return Tile_::num_ops == NumOps;
        else
          return true;
      }(),
      "CombineTile must carry one per-operand tile spec per operand");
  static_assert(out_tile_t::rank == Rank,
                "combine output tile must carry one extent per output mode");

  using ops_seq  = std::make_index_sequence<NumOps>;
  using outs_seq = std::make_index_sequence<NumOut>;

  // Per-operand (K) type helpers.
  template <std::size_t K>
  using op_node_t = tuple_element_t<K, typename node_type::ops_tuple_t>;

  // A fused operand's scratch tile is written in that sub-node's own canonical
  // axis order (freeA ++ freeB for a contraction, the output labels for a
  // combine), LayoutRight, which need not be this combine's output order. Two
  // independent things can differ:
  //   (1) the label gather from the operand's canonical modes into the
  //       combine's output modes (perm_seq<K>), and
  //   (2) the operand contraction's own permC, which relates its canonical
  //       order to the USER order its tile bundle's `.c` slot is spelled in.
  // (2) is purely a shape derivation -- Impl::operand_leaf_tile_t reads the
  // bundle's `.c` through permC to recover the tile the C scratch is actually
  // written with -- and is handled by leaf_tile_t<K> below; it is the identity
  // for a combine operand, whose modes are canonical by construction. (1) is a
  // real axis permutation, resolved at staging time: see op_is_relabeled_v.
  //
  // Note perm_seq<K> gathers from the operand's CANONICAL modes, not its user
  // ones: a ContractionTag node's modes_seq is its canonical output labels
  // (NodeHandle.hpp), which is also the order its C scratch is laid out in, and
  // a CombineTag node's modes_seq is canonical outright. So the two directions
  // never need composing -- perm_seq<K> already maps operand-canonical ->
  // combine-output.
  template <std::size_t K>
  using perm_seq =
      Impl::label_perm_seq_t<CModesSeq, typename op_node_t<K>::modes_seq>;

  // Native <-> canonical axis relation for operand K. Unlike the contraction
  // evaluator (which holds native tiles and derives canonical), a combine's
  // per-operand tile arrives in the combine's canonical output order, so this
  // evaluator runs the same vocabulary in the opposite direction.
  template <std::size_t K>
  using axes = OperandAxes<perm_seq<K>>;

  // Per-operand tile spec (identity for plain-tile form; the K-th slot of the
  // CombineTile bundle otherwise).
  template <std::size_t K>
  using op_tile_t = Impl::combine_op_tile_t<K, Tile_>;

  // The tile each operand's own evaluator is constructed with: an input
  // operand's native (label-gather-undone) tile, or a fused operand's own
  // nested bundle (Tile<A,B,C> / CombineTile) passed through.
  template <std::size_t K>
  using native_tile_t =
      Impl::combine_native_tile_t<op_node_t<K>, op_tile_t<K>, perm_seq<K>>;

  // The free-mode tile operand K's own evaluator actually hands back, in that
  // operand's own axis order: its native tile for an input operand, or the
  // fused sub-node's canonical output tile (its bundle's output slot, read
  // through permC for a contraction) for a fused operand. Same helper the
  // contraction evaluator uses for its A/B leaves.
  template <std::size_t K>
  using leaf_tile_t = Impl::operand_leaf_tile_t<op_node_t<K>, native_tile_t<K>>;

  // Every operand must present the combine's output shape, however it got
  // there (staged copy, in-place reorder, or zero-copy relabel). Reading the
  // leaf tile off the NATIVE tile is what lets one condition cover both operand
  // kinds, since it puts them in a common frame: leaf_tile_t is in the
  // operand's own order, and gathering it into the combine's output order must
  // land on out_tile_t. For an input operand the two transforms are exact
  // inverses and this reduces to op_tile_t == out_tile_t; for a fused operand
  // it is the real condition, and it subsumes both of the canonicality
  // restrictions this evaluator used to impose -- leaf_tile_t consumes the
  // operand's permC, canon_tile_t consumes the label gather. Stated here rather
  // than left to surface as an opaque type mismatch deep in stage_all().
  static_assert(
      Impl::all_of_index<NumOps>([](auto k) {
        constexpr std::size_t K = decltype(k)::value;
        return std::is_same_v<
            typename axes<K>::template canon_tile_t<leaf_tile_t<K>>,
            out_tile_t>;
      }),
      "combine evaluator: every operand must present the combine's output "
      "tile -- an input operand's tile must equal it directly, and a fused "
      "operand's nested bundle's output slot (read through that operand's own "
      "permC) must equal it once gathered into the combine's output order");

  // Reconstruct operand K's tile in its true native axis order from the
  // combine's tiling spec. Shared by the host-side sizing path
  // (operand_scratch_size) and the device-side construction path
  // (make_op_allocs), which previously duplicated this expression.
  template <std::size_t K>
  KOKKOS_FUNCTION static native_tile_t<K> native_op_tile(const tiling_type& t) {
    return Impl::combine_native_tile<op_node_t<K>>(Impl::combine_op_tile<K>(t),
                                                   perm_seq<K>{});
  }

  // Is operand K consumed as a zero-copy relabel of the fused sub-node's own
  // output scratch, rather than staged through Specialization 8? The rule
  // itself lives beside operand_stageable_v (the staging path it opts out of)
  // so the two are read -- and tested -- together; the combine is its only
  // consumer, because a contraction's GEMM needs a contiguous source and must
  // stage.
  template <std::size_t K>
  static constexpr bool op_is_relabeled_v =
      Impl::operand_relabelable_v<op_node_t<K>, perm_seq<K>>;

  // reorder_view's scratch overload is reorder_tile, defined only for
  // StaticTileLayoutRight/Left -- there is no dynamic-stride tile layout to
  // retype into. Say so here; otherwise a dynamically-tiled permuted operand
  // fails as an unexplained "no matching reorder_view" inside stage_operand.
  static_assert(
      Impl::all_of_index<NumOps>([](auto k) {
        constexpr std::size_t K = decltype(k)::value;
        return !op_is_relabeled_v<K> || leaf_tile_t<K>::is_static;
      }),
      "combine evaluator: a permuted fused operand must be statically "
      "tiled (the zero-copy relabel retypes its scratch layout, which only the "
      "StaticTileLayout* family supports)");

  // Per-operand allocator, keyed on the operand's native tile and its label
  // gather, so it owns both an inner Evaluator and the scratch tile it stages
  // into, built once at construction (mirroring the ContractionTag evaluator's
  // alloc_a_/alloc_b_) rather than in stage_operand() on every call.
  //
  // A combine has no k-loop of its own, so it still calls stage_operand()
  // exactly once per operand per evaluation. What building here buys is that
  // this evaluator is now itself re-invokable AS an operand: a parent
  // contraction runs a fused combine once per contracted tile, and each of
  // those runs would otherwise rebuild every operand allocator -- and with it
  // every nested evaluator underneath. Constructing once is what keeps that
  // re-invocation to the staging work alone, exactly as it already does for
  // ContractionTag's A/B operands.
  //
  // Read-once is also what makes the relabel path a clear win: a relabeled
  // operand re-reads its source per use where a staged copy amortizes. That
  // holds per evaluation, which is the granularity a combine reads at.
  template <std::size_t K>
  using op_alloc_t =
      ScratchAllocator<TeamPolicyTag<ES>, CombineTag, op_node_t<K>, value_type,
                       native_tile_t<K>, perm_seq<K>>;
  using out_alloc_t = ScratchAllocator<TeamPolicyTag<ES>, CombineTag, IntermTag,
                                       value_type, out_tile_t>;

  // Heterogeneous tuple of per-operand allocators (deduced via an unevaluated
  // helper so we can name the type before the private helpers are defined).
  template <std::size_t... Ks>
  static auto op_allocs_type_helper(std::index_sequence<Ks...>)
      -> DeviceTuple<op_alloc_t<Ks>...>;
  using op_allocs_t = decltype(op_allocs_type_helper(ops_seq{}));

  // The view type the combine loop reads operand K through. A staged operand
  // lands in the combine's own scratch_view_t; a relabeled one is a retype of
  // the sub-contraction's C scratch into a StaticTileLayoutStride view, which
  // is a DIFFERENT type carrying the same bytes. Hence the operand views form
  // a DeviceTuple read by a compile-time fold rather than a Kokkos::Array read
  // by a runtime loop -- the same reason op_allocs_ is a tuple.
  template <std::size_t K>
  using op_view_t =
      Impl::combine_op_view_t<op_is_relabeled_v<K>,
                              typename op_alloc_t<K>::scratch_view_t,
                              perm_seq<K>>;

  template <std::size_t... Ks>
  static auto op_views_type_helper(std::index_sequence<Ks...>)
      -> DeviceTuple<op_view_t<Ks>...>;
  using op_views_t = decltype(op_views_type_helper(ops_seq{}));

  using interm_type =
      NodeHandle<IntermTag, scratch_view_t, std::integral_constant<int, Rank>,
                 exec_space, NoHook>;
  // One interm per output; a scalar-returning fn is simply NumOut == 1.
  using result_type   = Kokkos::Array<interm_type, NumOut>;
  using team_member_t = Impl::team_member_t<exec_space>;

  node_type   node;
  tiling_type tiling;      // the tile spec (plain output tile or CombineTile)
  op_allocs_t op_allocs_;  // one allocator / operand, each owning its scratch
  Kokkos::Array<out_alloc_t, NumOut> out_allocs_;  // one allocator / output

  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type t,
                            const team_member_t& team)
      : node(n),
        tiling(t),
        op_allocs_(make_op_allocs(node, tiling, team, ops_seq{})),
        out_allocs_(make_out_allocs(tiling, team, outs_seq{})) {}

  KOKKOS_FUNCTION result_type operator()(
      const team_member_t& team, Kokkos::Array<int, Rank> tile_idx) const {
    // Bring every operand tile into the combine's output order -- by staging a
    // copy, or by relabeling a fused operand's own scratch (op_is_relabeled_v).
    // Those two yield different view types, so the results collect into a
    // tuple keyed by operand rather than a homogeneous array. The barrier makes
    // every staged write visible before the combine reads it.
    const op_views_t sv = stage_all(team, tile_idx, ops_seq{});
    team.team_barrier();

    // The interm handles wrapping each output allocator's scratch: built once
    // here, written through below, and returned as-is.
    const result_type results = make_results(outs_seq{});
    const auto        f = node.fn;  // local copy: lambda captures no `this`
    // All outputs share one layout, so the first drives the traversal.
    const scratch_view_t out0 = results[0].storage_;
    Impl::team_for_each_coord(team, out0, [=](auto coord) {
      Kokkos::Array<int, Rank> gidx{};
      TENSOR_PRAGMA_UNROLL
      for (int d = 0; d < Rank; ++d)
        gidx[d] = tile_idx[d] * out0.extent(d) + coord[d];
      const Kokkos::Array<value_type, NumOps> vals =
          gather_vals(sv, coord, ops_seq{});
      // Normalize fn's result (scalar or Kokkos::Array) to NumOut components
      // and scatter each into its output tile.
      const Kokkos::Array<value_type, NumOut> r =
          Impl::as_output_array<value_type>(Impl::apply_combine(f, gidx, vals));
      TENSOR_PRAGMA_UNROLL
      for (int m = 0; m < NumOut; ++m) results[m].storage_[coord] = r[m];
    });
    return results;
  }

  // This evaluator's output scratch tile. Read by a PARENT contraction or
  // combine that fuses this node as an operand: its ScratchAllocator hands this
  // buffer straight back rather than carving one of its own, exactly as it does
  // for a fused contraction (whose scratch() this mirrors).
  //
  // Phase 1 restricts a fused combine operand to NumOut == 1, so the first
  // output slot is the only one; Impl::single_result is where that restriction
  // is stated, and a phase-2 output selector would turn this into scratch<M>().
  KOKKOS_FUNCTION scratch_view_t scratch() const {
    return out_allocs_[0].get();
  }

  // Output tile count along canonical dim d (output order is canonical here).
  // Lets a parent treat this evaluator as an operand stager for fused chaining.
  KOKKOS_FUNCTION int outer_extent_canon(int d) const noexcept {
    return Impl::tile_count_along(output_tile(tiling), d, node.shape()[d]);
  }

  static std::size_t scratch_size_per_team(const tiling_type& t) {
    const out_tile_t out = output_tile(t);
    return static_cast<std::size_t>(NumOut) * out_alloc_t::bytes(out) +
           operand_bytes(t);
  }

 private:
  // Stage operand K into its allocator's own scratch tile: the
  // already-constructed allocator (op_allocs_) holds both that tile and an
  // inner Evaluator built once at construction time, so this just runs that
  // evaluator at the native-order tile index
  // (scattered from the combine's canonical tile_idx via perm_seq<K>) instead
  // of rebuilding an evaluator inline (as op_alloc_t's doc comment above
  // notes, this is called exactly once per operand per team today, so the
  // benefit is future-proofing rather than avoiding a current rebuild loop --
  // unlike accumulate_block's A/B operands, which really do loop over
  // k-tiles).
  //
  // Two ways out, chosen at compile time:
  //   • RELABEL (op_is_relabeled_v): run the fused sub-node, then retype its
  //     output scratch into the combine's axis order with reorder_view. No
  //     copy, no data movement, no barrier of its own -- and no constraint
  //     relating the permuted axes' extents, because nothing has to fit back
  //     into a fixed buffer. This is what Specialization 7 does; the
  //     contraction evaluator cannot use it (its GEMM needs a contiguous
  //     LayoutRight destination), but a combine reads its operands one element
  //     at a time and does not care about their memory order.
  //   • STAGE (everything else): hand off to the merged stage-or-passthrough
  //     evaluator (Specialization 8), exactly like accumulate_block does for A
  //     and B. Its own if constexpr on the source's storage type picks
  //     copy+reorder (an input operand), in-place reorder (a permuted fused
  //     operand that did not qualify for the relabel, which keeps the
  //     equal-extent restriction), or true zero-copy passthrough (identity
  //     perm).
  template <std::size_t K>
  KOKKOS_FUNCTION op_view_t<K> stage_operand(
      const team_member_t&            team,
      const Kokkos::Array<int, Rank>& tile_idx) const {
    const auto& alloc = op_allocs_.template get<K>();
    if constexpr (op_is_relabeled_v<K>) {
      // NoHook is part of op_is_relabeled_v, so there is no hook to apply over
      // the relabeled view -- and hence nothing here writes to the operand's
      // scratch, which stays exactly as its own evaluator left it.
      return reorder_view(
          alloc.stage(team, axes<K>::to_native_idx(tile_idx)).storage_,
          perm_seq<K>{});
    } else {
      return Impl::stage_operand_into<TeamPolicyTag<ES>>(
                 team, StageTile<native_tile_t<K>, perm_seq<K>>{}, tile_idx,
                 alloc)
          .storage_;
    }
  }

  template <std::size_t... Ks>
  KOKKOS_FUNCTION op_views_t stage_all(const team_member_t&            team,
                                       const Kokkos::Array<int, Rank>& tile_idx,
                                       std::index_sequence<Ks...>) const {
    return op_views_t{stage_operand<Ks>(team, tile_idx)...};  // left-to-right
  }

  // Read one element of every operand at the output coordinate. The views are
  // heterogeneously typed (a relabeled operand is a strided view over the same
  // bytes), so operand indexing is a pack expansion rather than a runtime loop.
  // Every view -- staged or relabeled -- presents the combine's output shape,
  // so a single coordinate indexes them all.
  template <typename Coord, std::size_t... Ks>
  KOKKOS_FORCEINLINE_FUNCTION static Kokkos::Array<value_type, NumOps>
  gather_vals(const op_views_t& sv, const Coord& coord,
              std::index_sequence<Ks...>) {
    return {sv.template get<Ks>()[coord]...};
  }

  // Total per-operand scratch cost. Each allocator is handed operand K's own
  // NATIVE tile and derives its own cost from that plus the label gather baked
  // into its type, so the two operand kinds no longer need separate folds here:
  // an InputTag operand carves a staging buffer of the canonical (output)
  // shape, while a fused operand is consumed zero-copy (its
  // ScratchAllocator::get() returns the inner evaluator's own output scratch
  // directly) and instead charges that sub-node's real recursive bytes. A
  // permuted fused operand costs the same: relabeling retypes that buffer
  // without carving a second one.
  static std::size_t operand_bytes(const tiling_type& t) {
    return Impl::sum_over_index<NumOps>([&](auto k) {
      constexpr std::size_t K = decltype(k)::value;
      return op_alloc_t<K>::bytes(native_op_tile<K>(t));
    });
  }

  // Construct one allocator per operand from the node's operand pack, each
  // built from its native-order tile + team. Constructing the allocator IS the
  // allocation now, so these braced-init packs are the only place the team's
  // scratch cursor is advanced -- and their left-to-right evaluation order
  // gives each operand a distinct tile.
  template <std::size_t... Ks>
  KOKKOS_FUNCTION static op_allocs_t make_op_allocs(
      const node_type& n, const tiling_type& t, const team_member_t& team,
      std::index_sequence<Ks...>) {
    return op_allocs_t{op_alloc_t<Ks>{n.operands.template get<Ks>(),
                                      native_op_tile<Ks>(t), team}...};
  }

  // One IntermTag allocator per output slot, each carving its own tile.
  template <std::size_t... Ms>
  KOKKOS_FUNCTION static Kokkos::Array<out_alloc_t, NumOut> make_out_allocs(
      const tiling_type& t, const team_member_t& team,
      std::index_sequence<Ms...>) {
    return {(static_cast<void>(Ms), out_alloc_t{output_tile(t), team})...};
  }

  // Wrap each output allocator's scratch tile in an interm handle (NoHook: the
  // store just writes; fn already applied any per-coordinate transform).
  template <std::size_t... Ms>
  KOKKOS_FUNCTION result_type make_results(std::index_sequence<Ms...>) const {
    result_type r{};
    ((r[Ms].storage_ = out_allocs_[Ms].get()), ...);
    return r;
  }
};

// ---------------------------------------------------------------------------
// Specialization 6: TeamPolicyTag + IntermTag(scratch View) — store-evaluator
//
// Writes a computed scratch tile (the result_ produced by Specialization 2 / 4)
// back to the global output view, team-parallel. The exact reverse of
// fill_team: same tile_view / subview_tile + TeamVectorRange structure, writing
// instead of reading. Tile_ is the output (C) tile; Tiles are assumed to evenly
// divide the view extents (no boundary guard), matching the rest of the team
// tier.
// ---------------------------------------------------------------------------
template <typename BackingVT, typename Layout, typename IntRank, typename ES,
          typename HookOp, typename Tile_>
struct Evaluator<
    TeamPolicyTag<ES>,
    NodeHandle<IntermTag, View<BackingVT, Layout>, IntRank, ES, HookOp>,
    Tile_> {
  using node_type =
      NodeHandle<IntermTag, View<BackingVT, Layout>, IntRank, ES, HookOp>;
  using tiling_type         = Tile_;
  using policy_tag          = TeamPolicyTag<ES>;
  static constexpr int Rank = node_type::Rank;
  using value_type          = typename node_type::value_type;
  using exec_space          = ES;
  using team_member_t       = Impl::team_member_t<exec_space>;

  static_assert(Layout::rank == Rank,
                "scratch layout rank must equal node rank");
  static_assert(tiling_type::rank == Rank,
                "store tile must carry one extent per output mode");

  node_type   node;
  tiling_type tiling;  // the output (C) tile

  // No scratch allocation: the scratch storage is already live in
  // node.storage_.
  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type t) : node(n), tiling(t) {}

  // `view` is the NATIVE (user-order) global output and `tiling` its native
  // tile, so subview_tile hits the compile-time-ordered OrderedSubviewLayout
  // path (registers, no local-memory spill). The canonical result is written by
  // reordering the ordered subview into canonical order via reorder_view
  // instead of presenting the output as a strided PermutedView. `perm` is permC
  // (maps canonical output mode i -> user position perm[i]); a full-rank
  // identity seq for canonical / non-permuted outputs makes every step below a
  // no-op.
  template <typename ViewT, int... Perm>
  KOKKOS_FUNCTION void operator()(
      const team_member_t& team, Kokkos::Array<int, Rank> tile_idx,
      const ViewT& view, std::integer_sequence<int, Perm...> perm) const {
    static_assert(sizeof...(Perm) == Rank,
                  "store permutation must have one entry per output mode");
    TIMING_SCOPE_ENTER(g_timing_stats.store_write_time,
                       g_timing_stats.store_write_count);
    team.team_barrier();  // ensure the producer's scratch is fully visible

    // Canonical tile index -> native (user-order) tile index: scatter by perm,
    // since perm[i] = user position of canonical mode i.
    const auto u_idx = Impl::scatter_index(tile_idx, perm);

    const auto tv  = tile_view(view, tiling);  // native -> ordered backing
    const auto sv0 = subview_tile(tv, u_idx);  // OrderedSubviewLayout (fast)
    const auto sv  = reorder_view(sv0, perm);  // canonical order, still ordered
    auto       scratch = node.storage_;
    Impl::apply_hook(node.hook_op, team, tile_idx, scratch);

    // Traversal follows sv, the ordered global destination, so the global
    // writes stay coalesced (scratch is contiguous either way).
    Impl::team_for_each_coord(team, sv,
                              [=](auto coord) { sv[coord] = scratch[coord]; });
    TIMING_SCOPE_EXIT(g_timing_stats.store_write_time,
                      g_timing_stats.store_write_count);
  }
};

// ---------------------------------------------------------------------------
// Specialization 7: TeamPolicyTag + IntermTag(any View) + Perm — relabel
//
// Converts an interm node into another interm node in a different
// (compile-time) axis order. Zero-copy in both storage families: reorder_view
// only relabels the layout, keeping the same backing, with no
// team-collaborative work otherwise.
//
// Generic in the View's backing and layout, because the layout-family
// difference is already resolved one level down by reorder_view's two
// SFINAE-constrained overloads (TiledLayout.hpp):
//   - global/subview tiles (OrderedSubviewLayout) -> reorder_layout, which
//     permutes the independent per-axis extents/strides;
//   - scratch tiles (StaticTileLayoutRight/Left), which have no per-axis Order
//     template parameter, -> reorder_tile, a same-bytes retype into a
//     StaticTileLayoutStride that reindexes strides instead of moving data.
// Nothing below needs to know which one it got.
//
// operator() returns a small proxy binding (this, tile_idx), matching
// Specialization 8's calling convention (`evaluator(team, tile_idx) = src`):
// assigning applies the source's hook over the reordered view via the
// whole-scratch Impl::apply_hook -- same as Specialization 8's copy branch,
// just with no copy (the reordered view still aliases src's own backing).
//
// For scratch storage this is distinct from Specialization 8's in-place
// scratch-to-scratch branch (Impl::reorder_scratch_in_place): that physically
// permutes data because its destination is a fixed, pre-allocated LayoutRight
// buffer that must keep its exact type for downstream GEMM contiguity. This
// evaluator has no such fixed destination -- it produces a new
// differently-typed view over the same bytes.
//
// More specialized than Specialization 6 (same node pattern, generic Tile_),
// so a perm-sequence tiling argument unambiguously selects this relabel
// evaluator over the store evaluator.
// ---------------------------------------------------------------------------
template <typename ES, typename BackingVT, typename Layout, typename IntRank,
          typename HookOp, int... Perm>
struct Evaluator<
    TeamPolicyTag<ES>,
    NodeHandle<IntermTag, View<BackingVT, Layout>, IntRank, ES, HookOp>,
    std::integer_sequence<int, Perm...>>
    : Impl::TileAssignable<
          TeamPolicyTag<ES>,
          NodeHandle<IntermTag, View<BackingVT, Layout>, IntRank, ES, HookOp>,
          std::integer_sequence<int, Perm...>> {
  using node_type =
      NodeHandle<IntermTag, View<BackingVT, Layout>, IntRank, ES, HookOp>;
  static constexpr int Rank = node_type::Rank;
  using perm_seq            = std::integer_sequence<int, Perm...>;
  using team_member_t       = Impl::team_member_t<ES>;
  using dest_view_t         = decltype(reorder_view(
      std::declval<typename node_type::storage_type>(), perm_seq{}));
  using interm_type = NodeHandle<IntermTag, dest_view_t, IntRank, ES, NoHook>;

  static_assert(Layout::rank == Rank, "view layout rank must equal node rank");
  static_assert(sizeof...(Perm) == Rank,
                "relabel permutation must have one entry per mode");

  // node accepted (unused) for constructor-call parity with every other
  // evaluator in this file; team is captured for apply_hook inside assign().
  team_member_t team_;

  KOKKOS_FUNCTION Evaluator(perm_seq, const team_member_t& team)
      : team_(team) {}

  // Invoked by the TileAssignable base on assignment; public for that reason.
  // operator() is inherited from that base.
  KOKKOS_FUNCTION interm_type assign(const Kokkos::Array<int, Rank>& tile_idx,
                                     const node_type& src) const {
    auto dst = reorder_view(src.storage_, perm_seq{});  // zero-copy relabel
    Impl::apply_hook(src.hook_op, team_, tile_idx, dst);
    return {dst, NoHook{}};
  }

  static std::size_t scratch_size_per_team(const perm_seq&) { return 0; }
};

// ---------------------------------------------------------------------------
// Specialization 8: TeamPolicyTag + IntermTag(scratch view) + StageTile —
// stage-or-passthrough operand assignment
//
// Keyed on the DESTINATION scratch tile: node.storage_ is a pre-allocated,
// parent-owned scratch buffer to fill (node's own hook_op is unused — the
// hook always rides on the SOURCE node's hook_op instead, same as every other
// operand-consuming step in this file). operator() returns a small proxy
// binding (this, tile_idx); assigning an operand's evaluated result into that
// proxy drives the reorder+copy (or, for a source whose storage is already
// this same scratch type — e.g. a fused contraction operand — an in-place
// reorder of that scratch buffer, or a true zero-copy passthrough when the
// permutation is identity), followed by applying the source's hook over the
// resulting scratch tile via the whole-scratch Impl::apply_hook. That needs
// tile_idx, which plain operator= never sees — hence the proxy.
//
// The scratch-resident reorder is done in place via a sequence of parallel
// axis-pair transpositions (Impl::reorder_scratch_in_place); it requires every
// transposed axis pair to have equal extent (static_asserted below), since a
// pairwise axis swap is only a self-map of the fixed buffer for equal extents.
// ---------------------------------------------------------------------------
template <typename ES, typename ValueType, typename Layout, int Rank,
          typename HookOp, typename SourceTile, int... Perm>
struct Evaluator<TeamPolicyTag<ES>,
                 NodeHandle<IntermTag, ScratchView<ValueType, ES, Layout>,
                            std::integral_constant<int, Rank>, ES, HookOp>,
                 StageTile<SourceTile, std::integer_sequence<int, Perm...>>>
    : Impl::TileAssignable<
          TeamPolicyTag<ES>,
          NodeHandle<IntermTag, ScratchView<ValueType, ES, Layout>,
                     std::integral_constant<int, Rank>, ES, HookOp>,
          StageTile<SourceTile, std::integer_sequence<int, Perm...>>> {
  using node_type = NodeHandle<IntermTag, ScratchView<ValueType, ES, Layout>,
                               std::integral_constant<int, Rank>, ES, HookOp>;
  using perm_seq  = std::integer_sequence<int, Perm...>;
  using team_member_t  = Impl::team_member_t<ES>;
  using scratch_view_t = ScratchView<ValueType, ES, Layout>;
  using interm_type = NodeHandle<IntermTag, scratch_view_t,
                                 std::integral_constant<int, Rank>, ES, NoHook>;

  node_type     node;  // node.storage_ is the pre-allocated scratch to fill
  team_member_t team_;

  KOKKOS_FUNCTION Evaluator(node_type n, StageTile<SourceTile, perm_seq>,
                            const team_member_t& team)
      : node(n), team_(team) {}

  // Invoked by the TileAssignable base on assignment; public for that reason.
  // operator() is inherited from that base.
  //
  // TODO: both branches apply the source's hook AFTER the reorder, against the
  // CANONICAL tile_idx, so for a permuted operand the hook sees canonical
  // rather than operand-native coordinates. Long-standing for permuted input
  // operands, and now reachable for permuted fused ones too; fixing it means
  // applying the hook to the source before reordering, at
  // OperandAxes<perm_seq>::to_native_idx(tile_idx).
  template <typename SrcNode>
  KOKKOS_FUNCTION interm_type assign(const Kokkos::Array<int, Rank>& tile_idx,
                                     const SrcNode& src) const {
    if constexpr (std::is_same_v<typename SrcNode::storage_type,
                                 scratch_view_t>) {
      // Source already lives in this destination scratch type (e.g. a fused
      // contraction operand). Physically reorder the scratch buffer in place
      // via parallel axis-pair transpositions when the operand is differently
      // ordered; identity is a true zero-copy passthrough.
      if constexpr (!Impl::is_identity_seq(perm_seq{})) {
        // In-place pairwise swaps require equal extents on every transposed
        // axis pair, else the swap is not a self-map of the fixed buffer.
        static_assert(
            Impl::transpositions_equal_extent<Layout>(perm_seq{}),
            "in-place scratch reorder requires equal extents on every "
            "transposed axis pair; stage a differently-ordered copy upstream "
            "for unequal extents");
        Impl::reorder_scratch_in_place(team_, src.storage_, perm_seq{});
      }
      Impl::apply_hook(src.hook_op, team_, tile_idx, src.storage_);
      return {src.storage_, NoHook{}};
    } else {
      auto sv  = reorder_view(src.storage_, perm_seq{});  // zero-copy relabel
      auto dst = node.storage_;
      // Traversal follows sv, the reordered source (dst is contiguous).
      Impl::team_for_each_coord(team_, sv,
                                [=](auto coord) { dst[coord] = sv[coord]; });
      Impl::apply_hook(src.hook_op, team_, tile_idx, dst);
      return {dst, NoHook{}};
    }
  }

  static std::size_t scratch_size_per_team(
      const StageTile<SourceTile, perm_seq>&) {
    return 0;  // destination scratch is owned/sized by the parent evaluator
  }
};
