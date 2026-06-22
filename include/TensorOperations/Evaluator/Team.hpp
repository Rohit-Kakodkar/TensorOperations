#pragma once
// Included from within TensorOperations namespace by Evaluator.hpp

// ---------------------------------------------------------------------------
// Specialization 2: TeamPolicyTag + InputTag + Tile_  (scratch tier)
// ---------------------------------------------------------------------------
template <typename ES, TensorLike T, typename HookOp, typename Tile_>
struct Evaluator<TeamPolicyTag<ES>, NodeHandle<InputTag, T, HookOp>, Tile_> {
  using node_type           = NodeHandle<InputTag, T, HookOp>;
  using tiling_type         = Tile_;
  using policy_tag          = TeamPolicyTag<ES>;
  static constexpr int Rank = tiling_type::rank;
  using value_type          = typename node_type::value_type;
  using exec_space          = ES;
  using scratch_space       = typename exec_space::scratch_memory_space;
  using tile_layout_t = decltype(make_tile_layout(std::declval<tiling_type>()));
  using backing_t     = Kokkos::View<value_type*, scratch_space,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using scratch_view_t = ScratchView<value_type, exec_space, tile_layout_t>;
  using interm_type =
      NodeHandle<IntermTag, scratch_view_t, std::integral_constant<int, Rank>,
                 exec_space, NoHook>;
  using result_type   = interm_type;
  using team_member_t = typename Kokkos::TeamPolicy<exec_space>::member_type;
  using tiled_input_t = TiledView<TensorHandle<T>, tiling_type>;

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
        tiled_input_(tile_view(n.handle, t)),
        scratch_(alloc_scratch(team)) {
    for (int d = 0; d < Rank; ++d) result_.shape_[d] = tiling.extent(d);
    result_.modes_   = node.modes();
    result_.storage_ = scratch_;
  }

  KOKKOS_FUNCTION result_type operator()(
      const team_member_t& team, Kokkos::Array<int, Rank> tile_idx) const {
    TIMING_SCOPE_ENTER(g_timing_stats.scratch_input_load_time,
                       g_timing_stats.scratch_input_load_count);
    fill_team(team, tile_idx, scratch_);
    TIMING_SCOPE_EXIT(g_timing_stats.scratch_input_load_time,
                      g_timing_stats.scratch_input_load_count);
    return result_;
  }

  static std::size_t scratch_size_per_team(const tiling_type& t) {
    const auto layout = make_tile_layout(t);
    return backing_t::shmem_size(static_cast<std::size_t>(layout.size()));
  }

 private:
  KOKKOS_FUNCTION scratch_view_t
  alloc_scratch(const team_member_t& team) const {
    const auto layout = make_tile_layout(tiling);
    backing_t  backing(team.team_scratch(0),
                       static_cast<std::size_t>(layout.size()));
    return {backing, layout};
  }

  KOKKOS_FUNCTION void fill_team(const team_member_t&            team,
                                 const Kokkos::Array<int, Rank>& tile_idx,
                                 const scratch_view_t&           result) const {
    const auto sv        = subview_tile(tiled_input_, tile_idx);
    const auto sv_layout = sv.layout();
    const auto total     = sv.size();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, total), [=](int i) {
      const auto coord = sv_layout[i];
      result[coord]    = sv[coord];
    });
  }
};

// ---------------------------------------------------------------------------
// Specialization 4: TeamPolicyTag + ContractionTag + Tile_  (scratch tier)
// ---------------------------------------------------------------------------
template <typename NA, typename NB, typename IntCRank, typename S, typename ES,
          typename HookOp, typename TileA, typename TileB, typename TileC>
struct Evaluator<TeamPolicyTag<ES>,
                 NodeHandle<ContractionTag, NA, NB, IntCRank, S, ES, HookOp>,
                 TensorOperations::Tile<TileA, TileB, TileC>> {
  using node_type = NodeHandle<ContractionTag, NA, NB, IntCRank, S, ES, HookOp>;
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

  using c_layout_t   = decltype(make_tile_layout(TileC{}));
  using stage_a_type = Evaluator<TeamPolicyTag<ES>, NA, TileA>;
  using stage_b_type = Evaluator<TeamPolicyTag<ES>, NB, TileB>;

 public:
  using scratch_view_t = ScratchView<value_type, exec_space, c_layout_t>;
  using interm_type =
      NodeHandle<IntermTag, scratch_view_t, std::integral_constant<int, Rank>,
                 exec_space, HookOp>;
  using result_type   = interm_type;
  using team_member_t = typename Kokkos::TeamPolicy<exec_space>::member_type;

  node_type      node;
  TileA          a_tile_;
  TileB          b_tile_;
  TileC          c_tile_;
  stage_a_type   stage_a_;
  stage_b_type   stage_b_;
  scratch_view_t c_scratch_;
  result_type    result_;

  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type t,
                            const team_member_t& team)
      : node(n),
        a_tile_(t.a),
        b_tile_(t.b),
        c_tile_(t.c),
        stage_a_(make_evaluator<TeamPolicyTag<ES>>(n.node_a, a_tile_, team)),
        stage_b_(make_evaluator<TeamPolicyTag<ES>>(n.node_b, b_tile_, team)),
        c_scratch_(alloc_c_scratch(team)) {
    auto c = c_scratch_;
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, c.size()),
                         [c](int i) { c.data()[i] = S{0}; });
    team.team_barrier();
    for (int d = 0; d < Rank; ++d) result_.shape_[d] = c_tile_.extent(d);
    result_.modes_   = node.modes();
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
      n_k_tiles[i] = stage_a_.tiled_input_.extent(FreeA + i);
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
    const std::size_t c_sz = c_backing_t::shmem_size(
        static_cast<std::size_t>(make_tile_layout(t.c).size()));
    return c_sz + stage_a_type::scratch_size_per_team(t.a) +
           stage_b_type::scratch_size_per_team(t.b);
  }

 private:
  KOKKOS_FUNCTION scratch_view_t
  alloc_c_scratch(const team_member_t& team) const {
    const auto  layout = make_tile_layout(c_tile_);
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
    // C[SA,SB] (all row-major). Static tiles fold these extents/strides at
    // compile time. Index raw pointers with hoisted/strided offsets rather than
    // View::operator() so the inner loop carries no per-access coordinate-array
    // build or flat_offset_ stride loop — that addressing overhead, not the
    // memory traffic, is what doubled the generic kernel's instruction count.
    auto a = reshape(stage_a_.scratch_, prefix_product(a_tile_, rank_c<FreeA>));
    auto b = reshape(stage_b_.scratch_, prefix_product(b_tile_, rank_c<NumK>));
    auto c = reshape(c_scratch_, prefix_product(c_tile_, rank_c<FreeA>));
    const value_type* KOKKOS_RESTRICT ap = a.data();
    const value_type* KOKKOS_RESTRICT bp = b.data();
    value_type* KOKKOS_RESTRICT       cp = c.data();
    // c is row-major [SA,SB], so the linear work index idx == fa*SB + fb. For
    // static tiles SK/SB are pulled from the layout *type* as constexpr, which
    // lets nvcc fully unroll the contraction loop and strength-reduce the
    // address math (k*SB) — a `const int` from extent() does not trip the
    // unroller, leaving the loop rolled with per-iter ISETP/BRA/IMAD overhead.
    if constexpr (TileA::is_static && TileB::is_static) {
      constexpr int SA = decltype(a)::layout_t::extent(0);
      constexpr int SK = decltype(a)::layout_t::extent(1);
      constexpr int SB = decltype(b)::layout_t::extent(1);

      // Register-blocked GEMM micro-kernel. Each work-item owns an MT x NT tile
      // of C held in registers; per k it loads MT A-values + NT B-values once
      // and issues MT*NT FMAs, cutting the shared-load:FMA ratio from 2 to
      // (MT+NT)/(MT*NT) — the unblocked 1:2 ratio left the L1/LSU pipe
      // saturated
      // (~85%) while the FP pipe starved (~30%). Performance-portable via
      // Kokkos simd: its ABI is chosen by execution space — scalar (width 1) on
      // GPU, so this is 2D scalar register-blocking there; AVX/NEON on CPU, so
      // the column dimension vectorizes. TeamThreadRange (the policy already
      // runs vector_length=1) keeps all parallelism in the thread dim on GPU
      // and avoids Kokkos vectorizing across work-items on top of our explicit
      // simd.
      namespace KE    = Kokkos::Experimental;
      using simd_t    = KE::simd<value_type>;
      constexpr int W = static_cast<int>(simd_t::size());
      // GPU (scalar simd) block factors are the register-tile dims; override
      // for tuning. 4x2 (8 accumulators) was the sweep winner on A100: a larger
      // 4x4 tile cut the load:FMA ratio further but pushed registers to 62 and
      // occupancy to 48%, leaving the kernel occupancy-bound; 4x2 drops to 48
      // registers / ~60% occupancy and is faster. CPU uses one simd-wide column
      // block (NT==W) by default — these macros only affect the scalar/GPU
      // path.
#ifndef TENSOR_OPS_GPU_MT
#define TENSOR_OPS_GPU_MT 4
#endif
#ifndef TENSOR_OPS_GPU_NT
#define TENSOR_OPS_GPU_NT 2
#endif
      constexpr int MT =
          (W == 1) ? TENSOR_OPS_GPU_MT : 8;  // output rows / item
      constexpr int NT =
          (W == 1) ? TENSOR_OPS_GPU_NT : W;  // output cols / item
      constexpr int NR = NT / W;  // simd vectors spanning the columns

      if constexpr (SA % MT == 0 && SB % NT == 0) {
        constexpr int CB = SB / NT;  // column-blocks across the C tile
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, (SA / MT) * CB), [=](int t) {
              const int i0 = (t / CB) * MT;
              const int j0 = (t % CB) * NT;
              simd_t    acc[MT][NR];
#pragma unroll
              for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int n = 0; n < NR; ++n)
                  acc[i][n] = simd_t(cp + (i0 + i) * SB + j0 + n * W,
                                     KE::simd_flag_default);
              for (int k = 0; k < SK; ++k) {
                simd_t br[NR];
#pragma unroll
                for (int n = 0; n < NR; ++n)
                  br[n] =
                      simd_t(bp + k * SB + j0 + n * W, KE::simd_flag_default);
#pragma unroll
                for (int i = 0; i < MT; ++i) {
                  const simd_t a_i(ap[(i0 + i) * SK + k]);  // broadcast A
#pragma unroll
                  for (int n = 0; n < NR; ++n) acc[i][n] += a_i * br[n];
                }
              }
#pragma unroll
              for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int n = 0; n < NR; ++n)
                  KE::simd_unchecked_store(acc[i][n],
                                           cp + (i0 + i) * SB + j0 + n * W,
                                           KE::simd_flag_default);
            });
      } else {
        // Tile extents not divisible by the block: one output per work-item.
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team, c.size()), [=](int idx) {
              const int         fb   = idx % SB;
              const value_type* arow = ap + (idx / SB) * SK;  // fa*SK, hoisted
              value_type        acc  = cp[idx];
#pragma unroll
              for (int k = 0; k < SK; ++k) acc += arow[k] * bp[k * SB + fb];
              cp[idx] = acc;
            });
      }
    } else {
      const int SK = a.extent(1);
      const int SB = b.extent(1);
      Kokkos::parallel_for(Kokkos::TeamVectorRange(team, c.size()),
                           [=](int idx) {
                             const int         fb   = idx % SB;
                             const value_type* arow = ap + (idx / SB) * SK;
                             value_type        acc  = cp[idx];
                             for (int k = 0; k < SK; ++k)
                               acc += arow[k] * bp[k * SB + fb];
                             cp[idx] = acc;
                           });
    }
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

  template <typename ViewT>
  KOKKOS_FUNCTION void operator()(const team_member_t&     team,
                                  Kokkos::Array<int, Rank> tile_idx,
                                  const ViewT&             view) const {
    TIMING_SCOPE_ENTER(g_timing_stats.store_write_time,
                       g_timing_stats.store_write_count);
    team.team_barrier();  // ensure the producer's scratch is fully visible
    const auto tv      = tile_view(view, tiling);
    const auto sv      = subview_tile(tv, tile_idx);
    const auto scratch = node.storage_;
    const auto hook    = node.hook_op;  // local copy: lambda captures no `this`

    const auto sv_layout = sv.layout();
    const auto total     = sv.size();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, total), [=](int i) {
      const auto coord = sv_layout[i];
      sv[coord]        = Impl::apply_hook(hook, scratch[coord]);
    });
    TIMING_SCOPE_EXIT(g_timing_stats.store_write_time,
                      g_timing_stats.store_write_count);
  }
};
