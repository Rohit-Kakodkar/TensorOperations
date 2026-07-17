#pragma once
// Included from within TensorOperations namespace by Evaluator.hpp

namespace Impl {
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

}  // namespace Impl

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
  using team_member_t = typename Kokkos::TeamPolicy<exec_space>::member_type;

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

  // The free-mode (output) tile of each operand: the tile itself for an input
  // operand, or the fused sub-contraction's output tile for a contraction
  // operand. GEMM sizing and the reshape below read these leaf tiles uniformly
  // via output_tile() (identity for a leaf, `.c` for a nested Tile bundle).
  using a_out_tile_t = decltype(output_tile(std::declval<TileA>()));
  using b_out_tile_t = decltype(output_tile(std::declval<TileB>()));

  // Check operand output-tile ranks match node ranks
  static_assert(a_out_tile_t::rank == RankA, "TileA rank must match node A");
  static_assert(b_out_tile_t::rank == RankB, "TileB rank must match node B");
  static_assert(TileC::rank == RankC, "TileC rank must match output rank");

  // Per-operand permutations from user axis order into the GEMM's canonical
  // order (freeA ++ contracted for A, contracted ++ freeB for B). permC maps
  // the canonical output back to the user order. Identity for canonical
  // contractions (in which case every canonicalization below is a no-op
  // passthrough).
  using permA_seq =
      Impl::permA_seq_t<typename NA::modes_seq, typename NB::modes_seq>;
  using permB_seq =
      Impl::permB_seq_t<typename NA::modes_seq, typename NB::modes_seq>;
  using permC_seq = PermCSeq;

  static_assert((Impl::has_node_tag_v<InputTag, NA> ||
                 Impl::has_node_tag_v<ContractionTag, NA>) &&
                    (Impl::has_node_tag_v<InputTag, NB> ||
                     Impl::has_node_tag_v<ContractionTag, NB>),
                "contraction evaluator v1: operands must be input or fused "
                "contraction nodes (combine operands are a follow-up)");

  // v1 fused chaining: a contraction operand must already sit in the parent's
  // canonical position (identity permA/permB), so no permuted-intermediate
  // handling is needed yet. Relabel so the parent's contracted modes come last
  // in the sub-contraction's output. (Permuting a nested operand is a later
  // follow-up; input operands may still be permuted as before.)
  static_assert(!Impl::has_node_tag_v<ContractionTag, NA> ||
                    Impl::is_identity_seq(permA_seq{}),
                "fused contraction operand A must be in canonical position "
                "(identity permA)");
  static_assert(!Impl::has_node_tag_v<ContractionTag, NB> ||
                    Impl::is_identity_seq(permB_seq{}),
                "fused contraction operand B must be in canonical position "
                "(identity permB)");

  using tile_a_canon_t =
      decltype(reorder_tile_value(std::declval<TileA>(), permA_seq{}));
  using tile_b_canon_t =
      decltype(reorder_tile_value(std::declval<TileB>(), permB_seq{}));
  using tile_c_canon_t =
      decltype(reorder_tile_value(std::declval<TileC>(), permC_seq{}));

  using c_layout_t =
      decltype(make_tile_layout(tile_c_canon_t{}, LayoutRight{}));

  using a_scratch_t =
      ScratchView<value_type, exec_space,
                  decltype(make_tile_layout(output_tile(tile_a_canon_t{}),
                                            LayoutRight{}))>;

  using b_scratch_t =
      ScratchView<value_type, exec_space,
                  decltype(make_tile_layout(output_tile(tile_b_canon_t{}),
                                            LayoutRight{}))>;

 public:
  using scratch_view_t = ScratchView<value_type, exec_space, c_layout_t>;
  using interm_type =
      NodeHandle<IntermTag, scratch_view_t, std::integral_constant<int, Rank>,
                 exec_space, HookOp>;
  using result_type   = interm_type;
  using team_member_t = typename Kokkos::TeamPolicy<exec_space>::member_type;

  node_type      node;
  TileA          tile_a_native_;
  TileB          tile_b_native_;
  tile_a_canon_t a_tile_;
  tile_b_canon_t b_tile_;
  tile_c_canon_t c_tile_;
  a_scratch_t    scratch_a_;
  b_scratch_t    scratch_b_;
  scratch_view_t scratch_;
  result_type    result_;

  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type t,
                            const team_member_t& team)
      : node(n),
        tile_a_native_(t.a),
        tile_b_native_(t.b),
        a_tile_(reorder_tile_value(t.a, permA_seq{})),
        b_tile_(reorder_tile_value(t.b, permB_seq{})),
        c_tile_(reorder_tile_value(t.c, permC_seq{})),
        scratch_a_(Impl::alloc_scratch_tile<value_type, exec_space>(
            team, output_tile(a_tile_))),
        scratch_b_(Impl::alloc_scratch_tile<value_type, exec_space>(
            team, output_tile(b_tile_))),
        scratch_(
            Impl::alloc_scratch_tile<value_type, exec_space>(team, c_tile_)) {
    result_.hook_op  = node.hook_op;
    result_.storage_ = scratch_;
  }

  KOKKOS_FUNCTION result_type operator()(
      const team_member_t& team, Kokkos::Array<int, Rank> tile_idx) const {
    TIMING_SCOPE_ENTER(g_timing_stats.contraction_accum_time,
                       g_timing_stats.contraction_accum_count);
    // Zero the output scratch at the start of every evaluation so this operator
    // is re-runnable. A parent contraction re-invokes a fused operand once per
    // contracted tile (recompute) on the same evaluator instance, and each run
    // must accumulate from a clean C.
    auto c = scratch_;
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, c.size()),
                         [c](int i) { c.data()[i] = S{0}; });
    team.team_barrier();
    // Per-mode k-tile counts (from A's own native tile/shape, read through
    // permA_seq) and their product. output_tile() normalizes "leaf tile"
    // uniformly whether A is an input leaf or a fused sub-contraction (same
    // precedent used for GEMM sizing below); permA_seq is identity in the
    // fused case, so this needs no dispatch on the operand's tag. Contracted
    // tiles share single-buffered scratch, so the walk over the linearized
    // contracted-tile space is serialized by team barriers.
    const auto               pA = Impl::seq_to_karray(permA_seq{});
    Kokkos::Array<int, NumK> n_k_tiles{};
    int                      total_k_tiles = 1;
    for (int i = 0; i < NumK; ++i) {
      int native_dim = pA[FreeA + i];
      n_k_tiles[i] =
          Impl::tile_count_along(output_tile(tile_a_native_), native_dim,
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

  static std::size_t scratch_size_per_team(const tiling_type& t) {
    return Impl::scratch_tile_bytes<value_type, exec_space>(
               reorder_tile_value(t.c, permC_seq{})) +
           Impl::scratch_tile_bytes<value_type, exec_space>(
               output_tile(reorder_tile_value(t.a, permA_seq{}))) +
           Impl::scratch_tile_bytes<value_type, exec_space>(
               output_tile(reorder_tile_value(t.b, permB_seq{})));
  }

 private:
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
    // a_tile_/b_tile_ and the staged scratch use. Scatter into each operand's
    // own native axis order to drive its native-tiled evaluator below
    // (identity scatter for a fused contraction operand, whose permA/permB
    // are asserted identity above).
    const auto a_native_idx = Impl::scatter_index(a_tile_idx, permA_seq{});
    const auto b_native_idx = Impl::scatter_index(b_tile_idx, permB_seq{});

    auto eval_a =
        make_evaluator<TeamPolicyTag<ES>>(node.node_a, tile_a_native_, team);
    auto eval_b =
        make_evaluator<TeamPolicyTag<ES>>(node.node_b, tile_b_native_, team);

    auto stage_a = make_evaluator<TeamPolicyTag<ES>>(
        make_interm_node(scratch_a_),
        StageTile<TileA, permA_seq>{tile_a_native_}, team);
    auto stage_b = make_evaluator<TeamPolicyTag<ES>>(
        make_interm_node(scratch_b_),
        StageTile<TileB, permB_seq>{tile_b_native_}, team);

    auto staged_a = stage_a(team, a_tile_idx) = eval_a(team, a_native_idx);
    auto staged_b = stage_b(team, b_tile_idx) = eval_b(team, b_native_idx);
    team.team_barrier();

    // View each operand's scratch as a 2D GEMM matrix: A[SA,SK], B[SK,SB],
    // C[SA,SB] (all row-major). For static tiles these carry compile-time
    // extents, and the register-tiled views below carry compile-time strides.
    auto a = reshape(
        staged_a.storage_,
        prefix_product(output_tile(a_tile_), rank_c<FreeA>));  // [SA,SK]
    auto b =
        reshape(staged_b.storage_,
                prefix_product(output_tile(b_tile_), rank_c<NumK>));  // [SK,SB]
    auto c =
        reshape(scratch_, prefix_product(c_tile_, rank_c<FreeA>));  // [SA,SB]

    int SA = [&]() {
      if constexpr (a_out_tile_t::is_static)
        return decltype(a)::layout_t::extent(0);
      else
        return a.extent(0);
    }();
    int SK = [&]() {
      if constexpr (a_out_tile_t::is_static)
        return decltype(a)::layout_t::extent(1);
      else
        return a.extent(1);
    }();
    int SB = [&]() {
      if constexpr (b_out_tile_t::is_static)
        return decltype(b)::layout_t::extent(1);
      else
        return b.extent(1);
    }();

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
    if constexpr (a_out_tile_t::is_static) {
      constexpr int SAs = decltype(a)::layout_t::extent(0);
      constexpr int SKs = decltype(a)::layout_t::extent(1);
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
    if constexpr (b_out_tile_t::is_static) {
      constexpr int SBs = decltype(b)::layout_t::extent(1);
      static_assert(
          SBs >= NR && SBs % NR == 0,
          "team GEMM: operand free dim (SB) must be a multiple of the register "
          "column block (2*W on CPU); enlarge the output C tile's free extent");
    } else {
      assert(
          SB >= NR && SB % NR == 0 &&
          "team GEMM: SB must be a multiple of the register column block NR");
    }

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
// Specialization 2, which already handles permuted operands; or the
// contraction evaluator, Specialization 4, for a contraction operand), then a
// single TeamVectorRange applies fn per element into the output scratch tile
// — no GEMM, no k-loop, no zeroing at this tier. Operands may be in any axis
// order; each input operand is gathered into the output (canonical) order via
// label_perm_seq + canonicalize_input, exactly as the contraction evaluator
// canonicalizes A/B.
//
// Tile_ is either:
//   (a) a plain output tile (StaticTile/DynamicTile) — requires every operand
//       to be an input node (all operand tiles == output tile), or
//   (b) a CombineTile<OutTile, OpTile_0, ..., OpTile_{N-1}> — carries the
//       output tile plus a per-operand tile spec. An input operand's OpTile
//       must equal OutTile; a contraction operand's OpTile is the inner
//       Tile<A,B,C> bundle (with C == OutTile in the operand's canonical
//       order).
//
// Every operand and the output share the same LayoutRight scratch layout, so
// all scratch tiles align element-for-element. The heterogeneous per-operand
// stagers live in a DeviceTuple; because all their scratch views have the
// SAME type (same output tile, same scalar), the combine loop reads them from
// a homogeneous Kokkos::Array. fn receives the global output coordinate:
// fn(i_0, ..., i_{Rank-1}, v_0, ..., v_{N-1}).
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

  // Every operand must be an input node or a contraction node. Contraction
  // operand support requires the CombineTile form (each contraction operand
  // needs its own Tile<A,B,C> bundle).
  static_assert(((Impl::has_node_tag_v<InputTag, Ops> ||
                  Impl::has_node_tag_v<ContractionTag, Ops>) &&
                 ...),
                "combine evaluator: operands must be input or contraction "
                "nodes (combine operands are a follow-up)");
  static_assert(
      Impl::is_combine_tile_v<Tile_> ||
          (Impl::has_node_tag_v<InputTag, Ops> && ...),
      "combine node with any contraction operand requires a CombineTile bundle "
      "(use make_combine_tile(out_tile, op_tile_0, ..., op_tile_{N-1}))");
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
  template <std::size_t K>
  using perm_seq =
      Impl::label_perm_seq_t<CModesSeq, typename op_node_t<K>::modes_seq>;
  // A contraction operand must already be in the combine's canonical output
  // order (identity gather from operand modes to output modes). Combined with
  // the check further below that the contraction's own permC is identity, this
  // guarantees the operand's canonical C tile == the combine's output tile in
  // the same LayoutRight layout, so its scratch view matches scratch_view_t.
  template <std::size_t K>
  static constexpr bool op_is_contraction_v =
      Impl::has_node_tag_v<ContractionTag, op_node_t<K>>;
  // A contraction operand's scratch tile is written in the operand's own
  // canonical (freeA ++ freeB) axis order, LayoutRight. For it to match the
  // combine's scratch_view_t exactly (same layout type, so all operand scratch
  // views collapse into a homogeneous Kokkos::Array in the combine loop), we
  // require the operand to be *fully canonical against the combine output*:
  //   (1) the operand's user-output labels match the combine's output labels
  //       in the same order (identity label gather from operand → combine), and
  //   (2) the operand contraction's own permC is identity (freeA++freeB ==
  //       operand user output).
  // Together these mirror the existing "fused contraction operand must be in
  // canonical position" restriction of Specialization 4, so combine and
  // fused-contraction chaining share the same operand-canonicality rules.
  template <std::size_t K>
  static constexpr bool op_contraction_permC_identity() {
    if constexpr (op_is_contraction_v<K>)
      return Impl::is_identity_seq(typename op_node_t<K>::permC_seq{});
    else
      return true;
  }
  static_assert(
      []<std::size_t... Ks>(std::index_sequence<Ks...>) {
        return ((!op_is_contraction_v<Ks> ||
                 Impl::is_identity_seq(perm_seq<Ks>{})) &&
                ...);
      }(ops_seq{}),
      "combine evaluator: a contraction operand must already be in the "
      "combine's output order (identity label gather); relabel the contraction "
      "output to match the combine output modes");
  static_assert(
      []<std::size_t... Ks>(std::index_sequence<Ks...>) {
        return (op_contraction_permC_identity<Ks>() && ...);
      }(ops_seq{}),
      "combine evaluator: a contraction operand must be in canonical position "
      "(identity permC); construct it with output modes == freeA++freeB");

  // Per-operand tile spec (identity for plain-tile form; the K-th slot of the
  // CombineTile bundle otherwise). combine_op_tile<K> presents this in the
  // combine's canonical (output) order; native_perm_seq<K> undoes perm_seq<K>
  // to recover operand K's true native axis order (identity for every
  // ContractionTag operand, per the asserts above), mirroring how
  // accumulate_block derives tile_a_native_/a_tile_ from permA_seq.
  template <std::size_t K>
  using op_tile_t = Impl::combine_op_tile_t<K, Tile_>;
  template <std::size_t K>
  using native_perm_seq = Impl::inverse_perm_seq_t<perm_seq<K>>;
  template <std::size_t K>
  using native_tile_t = decltype(reorder_tile_value(
      std::declval<op_tile_t<K>>(), native_perm_seq<K>{}));

  using interm_type =
      NodeHandle<IntermTag, scratch_view_t, std::integral_constant<int, Rank>,
                 exec_space, NoHook>;
  // One interm per output; a scalar-returning fn is simply NumOut == 1.
  using result_type   = Kokkos::Array<interm_type, NumOut>;
  using team_member_t = typename Kokkos::TeamPolicy<exec_space>::member_type;

  node_type   node;
  tiling_type tiling;  // the tile spec (plain output tile or CombineTile)
  Kokkos::Array<scratch_view_t, NumOps>
      op_scratch_;  // one scratch tile / operand
  Kokkos::Array<scratch_view_t, NumOut>
      outs_;  // one output scratch tile / output

  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type t,
                            const team_member_t& team)
      : node(n),
        tiling(t),
        op_scratch_(alloc_op_scratch(team, ops_seq{})),
        outs_(alloc_out_scratch(team, outs_seq{})) {}

  KOKKOS_FUNCTION result_type operator()(
      const team_member_t& team, Kokkos::Array<int, Rank> tile_idx) const {
    // Stage every operand tile into its scratch (in output order); the
    // per-operand scratch views are all the same type (same output tile,
    // LayoutRight, same scalar), so this collects into a homogeneous array
    // indexed by operand. The barrier makes the staged reads visible before
    // the combine.
    const Kokkos::Array<scratch_view_t, NumOps> sv =
        stage_all(team, tile_idx, ops_seq{});
    team.team_barrier();

    const Kokkos::Array<scratch_view_t, NumOut> outs =
        outs_;                             // capture by value
    const auto layout = outs[0].layout();  // all outputs share this layout
    const auto total  = outs[0].size();
    const auto f      = node.fn;  // local copy: lambda captures no `this`
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, total), [=](int i) {
      const auto               coord = layout[i];
      Kokkos::Array<int, Rank> gidx{};
      TENSOR_PRAGMA_UNROLL
      for (int d = 0; d < Rank; ++d)
        gidx[d] = tile_idx[d] * outs[0].extent(d) + coord[d];
      Kokkos::Array<value_type, NumOps> vals{};
      TENSOR_PRAGMA_UNROLL
      for (int k = 0; k < NumOps; ++k) vals[k] = sv[k][coord];
      // Normalize fn's result (scalar or Kokkos::Array) to NumOut components
      // and scatter each into its output tile.
      const Kokkos::Array<value_type, NumOut> r =
          Impl::as_output_array<value_type>(Impl::apply_combine(f, gidx, vals));
      TENSOR_PRAGMA_UNROLL
      for (int m = 0; m < NumOut; ++m) outs[m][coord] = r[m];
    });
    return make_results(outs_seq{});
  }

  // Output tile count along canonical dim d (output order is canonical here).
  // Lets a parent treat this evaluator as an operand stager for fused chaining.
  KOKKOS_FUNCTION int outer_extent_canon(int d) const noexcept {
    return Impl::tile_count_along(output_tile(tiling), d, node.shape()[d]);
  }

  static std::size_t scratch_size_per_team(const tiling_type& t) {
    return static_cast<std::size_t>(NumOut) *
               Impl::scratch_tile_bytes<value_type, exec_space>(
                   output_tile(t)) +
           static_cast<std::size_t>(NumOps) *
               Impl::scratch_tile_bytes<value_type, exec_space>(
                   output_tile(t)) +
           operand_scratch_size(t, ops_seq{});
  }

 private:
  // Stage operand K into op_scratch_[K]: tile its raw, native-order node
  // (Specialization 2 for an InputTag operand -- a cheap OrderedSubviewLayout
  // view, no scratch/hook of its own; Specialization 4 for a ContractionTag
  // operand -- self-contained, unchanged) at the native-order tile index
  // (scattered from the combine's canonical tile_idx via perm_seq<K>), then
  // hand that off to the merged stage-or-passthrough evaluator (Specialization
  // 8), exactly like accumulate_block does for A and B. That evaluator's own
  // if constexpr on the source's storage type picks copy+reorder (InputTag)
  // vs. true zero-copy passthrough (ContractionTag, asserted-identity perm)
  // -- no tag branching needed here.
  template <std::size_t K>
  KOKKOS_FUNCTION scratch_view_t
  stage_operand(const team_member_t&            team,
                const Kokkos::Array<int, Rank>& tile_idx) const {
    const native_tile_t<K> native_tile = reorder_tile_value(
        Impl::combine_op_tile<K>(tiling), native_perm_seq<K>{});
    const auto native_idx = Impl::scatter_index(tile_idx, perm_seq<K>{});

    auto eval_k = make_evaluator<TeamPolicyTag<ES>>(
        node.operands.template get<K>(), native_tile, team);
    auto stage_k = make_evaluator<TeamPolicyTag<ES>>(
        make_interm_node(op_scratch_[K]),
        StageTile<native_tile_t<K>, perm_seq<K>>{native_tile}, team);
    return (stage_k(team, tile_idx) = eval_k(team, native_idx)).storage_;
  }

  template <std::size_t... Ks>
  KOKKOS_FUNCTION Kokkos::Array<scratch_view_t, NumOps> stage_all(
      const team_member_t& team, const Kokkos::Array<int, Rank>& tile_idx,
      std::index_sequence<Ks...>) const {
    return {stage_operand<Ks>(team, tile_idx)...};  // left-to-right
  }

  // Each operand's own internal scratch cost: 0 for an InputTag operand
  // (Specialization 2 owns no scratch of its own now); the nested
  // contraction's real a+b+c bytes for a ContractionTag operand.
  template <std::size_t K>
  static std::size_t operand_native_scratch_bytes(const tiling_type& t) {
    if constexpr (op_is_contraction_v<K>) {
      const native_tile_t<K> native_tile =
          reorder_tile_value(Impl::combine_op_tile<K>(t), native_perm_seq<K>{});
      return Evaluator<TeamPolicyTag<ES>, op_node_t<K>,
                       native_tile_t<K>>::scratch_size_per_team(native_tile);
    } else {
      return 0;
    }
  }

  template <std::size_t... Ks>
  static std::size_t operand_scratch_size(const tiling_type& t,
                                          std::index_sequence<Ks...>) {
    std::size_t total = 0;
    ((total += operand_native_scratch_bytes<Ks>(t)), ...);
    return total;
  }

  // Allocate NumOps distinct per-operand scratch tiles (each alloc bumps team
  // scratch; braced-init evaluation order is left-to-right).
  template <std::size_t... Ks>
  KOKKOS_FUNCTION Kokkos::Array<scratch_view_t, NumOps> alloc_op_scratch(
      const team_member_t& team, std::index_sequence<Ks...>) const {
    return {(static_cast<void>(Ks),
             Impl::alloc_scratch_tile<value_type, exec_space>(
                 team, output_tile(tiling)))...};
  }

  // Allocate NumOut distinct output scratch tiles (each alloc bumps team
  // scratch; braced-init evaluation order is left-to-right).
  template <std::size_t... Ms>
  KOKKOS_FUNCTION Kokkos::Array<scratch_view_t, NumOut> alloc_out_scratch(
      const team_member_t& team, std::index_sequence<Ms...>) const {
    return {(static_cast<void>(Ms),
             Impl::alloc_scratch_tile<value_type, exec_space>(
                 team, output_tile(tiling)))...};
  }

  // Wrap each output scratch tile in an interm handle (NoHook: the store just
  // writes; fn already applied any per-coordinate transform).
  template <std::size_t... Ms>
  KOKKOS_FUNCTION result_type make_results(std::index_sequence<Ms...>) const {
    result_type r{};
    ((r[Ms].storage_ = outs_[Ms]), ...);
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
  using team_member_t = typename Kokkos::TeamPolicy<exec_space>::member_type;

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

    const auto sv_layout = sv.layout();
    const auto total     = sv.size();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, total), [=](int i) {
      const auto coord = sv_layout[i];
      sv[coord]        = scratch[coord];
    });
    TIMING_SCOPE_EXIT(g_timing_stats.store_write_time,
                      g_timing_stats.store_write_count);
  }
};

// ---------------------------------------------------------------------------
// Specialization 7: TeamPolicyTag + IntermTag(global view) + Perm  — relabel
//
// Converts a global-view interm node into another global-view interm node in
// a different (compile-time) axis order. Zero-copy: reorder_view just
// relabels the same OrderedSubviewLayout backing (independent per-axis
// extents/strides), no team-collaborative work otherwise.
//
// operator() returns a small proxy binding (this, tile_idx), matching
// Specialization 8's calling convention (`evaluator(team, tile_idx) = src`):
// assigning applies the source's hook over the reordered view via the
// whole-scratch Impl::apply_hook -- same as Specialization 8's copy branch,
// just with no copy (the reordered view still aliases src's own backing).
// ---------------------------------------------------------------------------
template <typename ES, typename BackingVT, int Rank, int... Order,
          typename HookOp, int... Perm>
struct Evaluator<
    TeamPolicyTag<ES>,
    NodeHandle<IntermTag, View<BackingVT, OrderedSubviewLayout<Rank, Order...>>,
               std::integral_constant<int, Rank>, ES, HookOp>,
    std::integer_sequence<int, Perm...>> {
  using node_type =
      NodeHandle<IntermTag,
                 View<BackingVT, OrderedSubviewLayout<Rank, Order...>>,
                 std::integral_constant<int, Rank>, ES, HookOp>;
  using perm_seq      = std::integer_sequence<int, Perm...>;
  using team_member_t = typename Kokkos::TeamPolicy<ES>::member_type;
  using dest_view_t   = decltype(reorder_view(
      std::declval<typename node_type::storage_type>(), perm_seq{}));
  using interm_type = NodeHandle<IntermTag, dest_view_t,
                                 std::integral_constant<int, Rank>, ES, NoHook>;

  // node accepted (unused) for constructor-call parity with every other
  // evaluator in this file; team is captured for apply_hook inside
  // AssignProxy::operator=.
  team_member_t team_;

  KOKKOS_FUNCTION Evaluator(perm_seq, const team_member_t& team)
      : team_(team) {}

  struct AssignProxy {
    const Evaluator*         self;
    Kokkos::Array<int, Rank> tile_idx;

    KOKKOS_FUNCTION interm_type operator=(const node_type& src) const {
      auto dst = reorder_view(src.storage_, perm_seq{});  // zero-copy relabel
      Impl::apply_hook(src.hook_op, self->team_, tile_idx, dst);
      return {dst, NoHook{}};
    }
  };

  KOKKOS_FUNCTION AssignProxy
  operator()(const team_member_t&, Kokkos::Array<int, Rank> tile_idx) const {
    return AssignProxy{this, tile_idx};
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
// this same scratch type — a fused contraction operand, whose permA/permB
// are asserted identity by the caller — a true zero-copy passthrough),
// followed by applying the source's hook over the resulting scratch tile via
// the whole-scratch Impl::apply_hook. That needs tile_idx, which plain
// operator= never sees — hence the proxy.
// ---------------------------------------------------------------------------
template <typename ES, typename ValueType, typename Layout, int Rank,
          typename HookOp, typename SourceTile, int... Perm>
struct Evaluator<TeamPolicyTag<ES>,
                 NodeHandle<IntermTag, ScratchView<ValueType, ES, Layout>,
                            std::integral_constant<int, Rank>, ES, HookOp>,
                 StageTile<SourceTile, std::integer_sequence<int, Perm...>>> {
  using node_type = NodeHandle<IntermTag, ScratchView<ValueType, ES, Layout>,
                               std::integral_constant<int, Rank>, ES, HookOp>;
  using perm_seq  = std::integer_sequence<int, Perm...>;
  using team_member_t  = typename Kokkos::TeamPolicy<ES>::member_type;
  using scratch_view_t = ScratchView<ValueType, ES, Layout>;
  using interm_type = NodeHandle<IntermTag, scratch_view_t,
                                 std::integral_constant<int, Rank>, ES, NoHook>;

  node_type     node;  // node.storage_ is the pre-allocated scratch to fill
  team_member_t team_;

  KOKKOS_FUNCTION Evaluator(node_type n, StageTile<SourceTile, perm_seq>,
                            const team_member_t& team)
      : node(n), team_(team) {}

  struct AssignProxy {
    const Evaluator*         self;
    Kokkos::Array<int, Rank> tile_idx;

    template <typename SrcNode>
    KOKKOS_FUNCTION interm_type operator=(const SrcNode& src) const {
      if constexpr (std::is_same_v<typename SrcNode::storage_type,
                                   scratch_view_t>) {
        static_assert(
            Impl::is_identity_seq(perm_seq{}),
            "scratch-resident operand cannot be reordered (no "
            "reorder_layout overload for scratch tile layouts); stage a "
            "differently-ordered copy upstream instead");
        Impl::apply_hook(src.hook_op, self->team_, tile_idx, src.storage_);
        return {src.storage_, NoHook{}};
      } else {
        auto sv  = reorder_view(src.storage_, perm_seq{});  // zero-copy relabel
        auto dst = self->node.storage_;
        Kokkos::parallel_for(Kokkos::TeamVectorRange(self->team_, sv.size()),
                             [=](int i) {
                               auto coord = sv.layout()[i];
                               dst[coord] = sv[coord];
                             });
        Impl::apply_hook(src.hook_op, self->team_, tile_idx, dst);
        return {dst, NoHook{}};
      }
    }
  };

  KOKKOS_FUNCTION AssignProxy
  operator()(const team_member_t&, Kokkos::Array<int, Rank> tile_idx) const {
    return AssignProxy{this, tile_idx};
  }

  static std::size_t scratch_size_per_team(
      const StageTile<SourceTile, perm_seq>&) {
    return 0;  // destination scratch is owned/sized by the parent evaluator
  }
};
