#pragma once
// Included from within TensorOperations namespace by Evaluator.hpp

namespace Impl {
// --- staging operand canonicalization (native subview -> reorder) ----------
//
// A permuted input operand is presented to the contraction as a PermutedView
// (canonical labels, strided). Rather than tile that strided view (runtime
// SubviewLayout, local-memory spill), the input stage evaluator tiles the
// PermutedView's *native* inner view (ordered subview) and reorders the taken
// subview into canonical order. This trait destructures the operand's handle
// type T to recover that native view (`view_t` / `get()`) and the gather
// permutation (`perm_seq`, native -> canonical); it degrades to identity /
// passthrough when the operand isn't permuted. Handle = TensorHandle<T,
// ModesSeq>.
template <typename Handle, typename T, int Rank, bool = is_permuted_view_v<T>>
struct staging_operand {
  using perm_seq = std::make_integer_sequence<int, Rank>;  // identity
  using view_t   = Handle;
  KOKKOS_FUNCTION static const Handle& get(const Handle& h) { return h; }
};
template <typename Handle, typename T, int Rank>
struct staging_operand<Handle, T, Rank, true> {
  using perm_seq = typename T::perm_seq;
  using view_t   = typename T::inner_view_t;
  KOKKOS_FUNCTION static const view_t& get(const Handle& h) { return h.v; }
};
}  // namespace Impl

// ---------------------------------------------------------------------------
// Specialization 2: TeamPolicyTag + InputTag + Tile_  (scratch tier)
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
  using scratch_space       = typename exec_space::scratch_memory_space;
  using tile_layout_t = decltype(make_tile_layout(std::declval<tiling_type>(),
                                                  std::declval<LayoutRight>()));
  using backing_t     = Kokkos::View<value_type*, scratch_space,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using scratch_view_t = ScratchView<value_type, exec_space, tile_layout_t>;
  using interm_type =
      NodeHandle<IntermTag, scratch_view_t, std::integral_constant<int, Rank>,
                 exec_space, NoHook>;
  using result_type   = interm_type;
  using team_member_t = typename Kokkos::TeamPolicy<exec_space>::member_type;

  // Staging keeps its scratch/result canonical (tiling_type is the canonical
  // tile). For a permuted operand it tiles the PermutedView's *native* inner
  // view so subview_tile stays on the compile-time-ordered path; the taken
  // subview is reordered into canonical order via perm_seq. All of this
  // collapses to the previous behavior for non-permuted operands (perm_seq =
  // identity).
  using stg      = Impl::staging_operand<TensorHandle<T, ModesSeq>, T, Rank>;
  using perm_seq = typename stg::perm_seq;
  using inv_perm_seq  = Impl::inverse_perm_seq_t<perm_seq>;
  using native_view_t = typename stg::view_t;
  using native_tile_t =
      decltype(reorder_tile_value(std::declval<tiling_type>(), inv_perm_seq{}));
  using tiled_input_t = TiledView<native_view_t, native_tile_t>;

  static_assert(node_type::Rank == Rank,
                "input staging tile must carry one extent per input mode");

  node_type      node;
  tiling_type    tiling;
  tiled_input_t  tiled_input_;
  scratch_view_t scratch_;
  result_type    result_;

  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type t,
                            const team_member_t& team)
      : node(n),
        tiling(t),
        tiled_input_(tile_view(stg::get(n.handle),
                               reorder_tile_value(t, inv_perm_seq{}))),
        scratch_(alloc_scratch(team)) {
    result_.storage_ = scratch_;
  }

  KOKKOS_FUNCTION result_type operator()(
      const team_member_t& team, Kokkos::Array<int, Rank> tile_idx) const {
    TIMING_SCOPE_ENTER(g_timing_stats.scratch_input_load_time,
                       g_timing_stats.scratch_input_load_count);
    // Canonical tile index -> native (operand-order) tile index; identity
    // scatter for non-permuted operands.
    fill_team(team, Impl::scatter_index(tile_idx, perm_seq{}), tile_idx,
              scratch_);
    TIMING_SCOPE_EXIT(g_timing_stats.scratch_input_load_time,
                      g_timing_stats.scratch_input_load_count);
    return result_;
  }

  // Outer tile count along canonical dimension d, read through the permutation
  // (tiled_input_ is stored in the operand's native axis order).
  KOKKOS_FUNCTION int outer_extent_canon(int d) const noexcept {
    const auto p = Impl::seq_to_karray(perm_seq{});
    return tiled_input_.extent(p[d]);
  }

  static std::size_t scratch_size_per_team(const tiling_type& t) {
    const auto layout = make_tile_layout(t, LayoutRight{});
    return backing_t::shmem_size(static_cast<std::size_t>(layout.size()));
  }

 private:
  KOKKOS_FUNCTION scratch_view_t
  alloc_scratch(const team_member_t& team) const {
    const auto layout = make_tile_layout(tiling, LayoutRight{});
    backing_t  backing(team.team_scratch(0),
                       static_cast<std::size_t>(layout.size()));
    return {backing, layout};
  }

  KOKKOS_FUNCTION void fill_team(const team_member_t&            team,
                                 const Kokkos::Array<int, Rank>& native_idx,
                                 const Kokkos::Array<int, Rank>& tile_idx,
                                 const scratch_view_t&           result) const {
    const auto sv0       = subview_tile(tiled_input_, native_idx);  // ordered
    const auto sv        = reorder_view(sv0, perm_seq{});           // canonical
    const auto sv_layout = sv.layout();
    const auto total     = sv.size();
    const auto hook = node.hook_op;  // local copy: lambda captures no `this`
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, total), [=](int i) {
      const auto               coord = sv_layout[i];
      Kokkos::Array<int, Rank> gidx{};
      for (int d = 0; d < Rank; ++d)
        gidx[d] = tile_idx[d] * sv.extent(d) + coord[d];
      auto v = sv[coord];
      Impl::apply_hook(hook, gidx, v);
      result[coord] = v;
    });
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
  using scratch_space       = typename ES::scratch_memory_space;
  using c_backing_t = Kokkos::View<value_type*, scratch_space,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

 private:
  static constexpr int RankC = Rank;
  static constexpr int NumK  = node_type::NumContracted;
  static constexpr int RankA = NA::Rank;
  static constexpr int RankB = NB::Rank;
  static constexpr int FreeA = RankA - NumK;
  static constexpr int FreeB = RankB - NumK;
  static_assert(FreeA + FreeB == RankC,
                "free-mode counts must sum to the output rank");

  // Check Tile ranks match node ranks
  static_assert(TileA::rank == RankA, "TileA rank must match node A");
  static_assert(TileB::rank == RankB, "TileB rank must match node B");
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

  using na_canon_t =
      decltype(Impl::canonicalize_input(std::declval<NA>(), permA_seq{}));
  using nb_canon_t =
      decltype(Impl::canonicalize_input(std::declval<NB>(), permB_seq{}));
  using tile_a_canon_t =
      decltype(reorder_tile_value(std::declval<TileA>(), permA_seq{}));
  using tile_b_canon_t =
      decltype(reorder_tile_value(std::declval<TileB>(), permB_seq{}));
  using tile_c_canon_t =
      decltype(reorder_tile_value(std::declval<TileC>(), permC_seq{}));

  using c_layout_t =
      decltype(make_tile_layout(tile_c_canon_t{}, LayoutRight{}));
  using stage_a_type = Evaluator<TeamPolicyTag<ES>, na_canon_t, tile_a_canon_t>;
  using stage_b_type = Evaluator<TeamPolicyTag<ES>, nb_canon_t, tile_b_canon_t>;

 public:
  using scratch_view_t = ScratchView<value_type, exec_space, c_layout_t>;
  using interm_type =
      NodeHandle<IntermTag, scratch_view_t, std::integral_constant<int, Rank>,
                 exec_space, HookOp>;
  using result_type   = interm_type;
  using team_member_t = typename Kokkos::TeamPolicy<exec_space>::member_type;

  node_type      node;
  tile_a_canon_t a_tile_;
  tile_b_canon_t b_tile_;
  tile_c_canon_t c_tile_;
  stage_a_type   stage_a_;
  stage_b_type   stage_b_;
  scratch_view_t c_scratch_;
  result_type    result_;

  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type t,
                            const team_member_t& team)
      : node(n),
        a_tile_(reorder_tile_value(t.a, permA_seq{})),
        b_tile_(reorder_tile_value(t.b, permB_seq{})),
        c_tile_(reorder_tile_value(t.c, permC_seq{})),
        stage_a_(make_evaluator<TeamPolicyTag<ES>>(
            Impl::canonicalize_input(n.node_a, permA_seq{}), a_tile_, team)),
        stage_b_(make_evaluator<TeamPolicyTag<ES>>(
            Impl::canonicalize_input(n.node_b, permB_seq{}), b_tile_, team)),
        c_scratch_(alloc_c_scratch(team)) {
    auto c = c_scratch_;
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, c.size()),
                         [c](int i) { c.data()[i] = S{0}; });
    team.team_barrier();
    result_.hook_op  = node.hook_op;
    result_.storage_ = c_scratch_;
  }

  KOKKOS_FUNCTION result_type operator()(
      const team_member_t& team, Kokkos::Array<int, Rank> tile_idx) const {
    TIMING_SCOPE_ENTER(g_timing_stats.contraction_accum_time,
                       g_timing_stats.contraction_accum_count);
    // Per-mode k-tile counts (from A's already-tiled outer extents) and their
    // product. Contracted tiles share single-buffered scratch, so the walk over
    // the linearized contracted-tile space is serialized by team barriers.
    Kokkos::Array<int, NumK> n_k_tiles{};
    int                      total_k_tiles = 1;
    for (int i = 0; i < NumK; ++i) {
      n_k_tiles[i] = stage_a_.outer_extent_canon(FreeA + i);
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

  static std::size_t scratch_size_per_team(const tiling_type& t) {
    const auto        c_canon = reorder_tile_value(t.c, permC_seq{});
    const std::size_t c_sz = c_backing_t::shmem_size(static_cast<std::size_t>(
        make_tile_layout(c_canon, LayoutRight{}).size()));
    return c_sz +
           stage_a_type::scratch_size_per_team(
               reorder_tile_value(t.a, permA_seq{})) +
           stage_b_type::scratch_size_per_team(
               reorder_tile_value(t.b, permB_seq{}));
  }

 private:
  KOKKOS_FUNCTION scratch_view_t
  alloc_c_scratch(const team_member_t& team) const {
    const auto  layout = make_tile_layout(c_tile_, LayoutRight{});
    c_backing_t backing(team.team_scratch(0),
                        static_cast<std::size_t>(layout.size()));
    return {backing, layout};
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

    stage_a_(team, a_tile_idx);
    stage_b_(team, b_tile_idx);
    team.team_barrier();

    // View each operand's scratch as a 2D GEMM matrix: A[SA,SK], B[SK,SB],
    // C[SA,SB] (all row-major). For static tiles these carry compile-time
    // extents, and the register-tiled views below carry compile-time strides.
    auto a = reshape(stage_a_.scratch_,
                     prefix_product(a_tile_, rank_c<FreeA>));  // [SA,SK]
    auto b = reshape(stage_b_.scratch_,
                     prefix_product(b_tile_, rank_c<NumK>));  // [SK,SB]
    auto c =
        reshape(c_scratch_, prefix_product(c_tile_, rank_c<FreeA>));  // [SA,SB]

    int SA = [&]() {
      if constexpr (TileA::is_static)
        return decltype(a)::layout_t::extent(0);
      else
        return a.extent(0);
    }();
    int SK = [&]() {
      if constexpr (TileA::is_static)
        return decltype(a)::layout_t::extent(1);
      else
        return a.extent(1);
    }();
    int SB = [&]() {
      if constexpr (TileB::is_static)
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
    if constexpr (TileA::is_static) {
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
    if constexpr (TileB::is_static) {
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
    const auto scratch = node.storage_;
    const auto hook    = node.hook_op;  // local copy: lambda captures no `this`

    const auto sv_layout = sv.layout();
    const auto total     = sv.size();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, total), [=](int i) {
      const auto               coord = sv_layout[i];
      Kokkos::Array<int, Rank> gidx{};
      for (int d = 0; d < Rank; ++d)
        gidx[d] = tile_idx[d] * sv.extent(d) + coord[d];
      auto v = scratch[coord];
      Impl::apply_hook(hook, gidx, v);
      sv[coord] = v;
    });
    TIMING_SCOPE_EXIT(g_timing_stats.store_write_time,
                      g_timing_stats.store_write_count);
  }
};
