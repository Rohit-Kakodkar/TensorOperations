#pragma once
// Included from within TensorOperations namespace by Evaluator.hpp

// ---------------------------------------------------------------------------
// Specialization 1: RangePolicyTag + InputTag + StaticTile<E...>  (register
// tier)
// ---------------------------------------------------------------------------
template <TensorLike T, typename HookOp, int... E>
struct Evaluator<RangePolicyTag, NodeHandle<InputTag, T, HookOp>,
                 StaticTile<E...>> {
  using node_type           = NodeHandle<InputTag, T, HookOp>;
  using tiling_type         = StaticTile<E...>;
  using policy_tag          = RangePolicyTag;
  static constexpr int Rank = tiling_type::rank;
  using value_type          = typename node_type::value_type;
  using exec_space          = typename Impl::exec_space_of<T>::type;
  using register_array_t    = RegisterArray<value_type, E...>;

  static_assert(node_type::Rank == Rank,
                "input staging tile must carry one extent per input mode");

  using tiled_input_t = TiledView<TensorHandle<T>, tiling_type>;

  using interm_type =
      NodeHandle<IntermTag, register_array_t, std::integral_constant<int, Rank>,
                 exec_space, NoHook>;
  using result_type = interm_type;

  node_type      node;
  tiling_type    tiling;
  tiled_input_t  tiled_input_;

  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type t)
      : node(n), tiling(t), tiled_input_(tile_view(n.handle, t)) {}

  KOKKOS_FUNCTION result_type
  operator()(Kokkos::Array<int, Rank> tile_idx) const {
    TIMING_SCOPE_ENTER(g_timing_stats.input_stage_load_time, g_timing_stats.input_stage_load_count);
    result_type out{};
    for (int d = 0; d < Rank; ++d) out.shape_[d] = tiling_type::extent(d);
    out.modes_ = node.modes();
    fill(tile_idx, out.storage_);
    TIMING_SCOPE_EXIT(g_timing_stats.input_stage_load_time, g_timing_stats.input_stage_load_count);
    return out;
  }

 private:
  KOKKOS_FORCEINLINE_FUNCTION void fill(const Kokkos::Array<int, Rank>& tile_idx,
                                        register_array_t& result) const {
    auto sv = subview_tile(tiled_input_, tile_idx);
    if constexpr (Impl::is_layout_stride_v<T>) {
      const int f = sv.unit_stride_dim();
      if (f >= 0)
        fill_dispatch(f, sv, result, std::make_index_sequence<Rank>{});
      else
        fill_fallback(sv, result);
    } else if constexpr (Impl::is_layout_left_v<T>) {
      if (sv.stride(0) == 1)
        fill_along<0>(sv, result);
      else
        fill_fallback(sv, result);
    } else {
      if (sv.stride(Rank - 1) == 1)
        fill_along<Rank - 1>(sv, result);
      else
        fill_fallback(sv, result);
    }
  }

  template <std::size_t F, typename SV>
  KOKKOS_FORCEINLINE_FUNCTION void fill_along(const SV&         sv,
                                              register_array_t& result) const {
    constexpr int     Fi    = static_cast<int>(F);
    constexpr int     EF    = register_array_t::extent(Fi);
    constexpr int     RSF   = static_cast<int>(register_array_t::strides_[F]);
    constexpr int     Outer = static_cast<int>(register_array_t::size) / EF;
    const value_type* src   = sv.data();
    const int         sf    = sv.stride(Fi);
    for (int o = 0; o < Outer; ++o) {
      int rem       = o;
      int base_slot = 0;
      int src_base  = 0;
      for (int d = Rank - 1; d >= 0; --d) {
        if (d == Fi) continue;
        const int e     = register_array_t::extent(d);
        const int local = rem % e;
        rem /= e;
        base_slot += local * static_cast<int>(register_array_t::strides_[d]);
        src_base  += local * sv.stride(d);
      }
      TENSOR_PRAGMA_UNROLL
      for (int c = 0; c < EF; ++c)
        result.data_[base_slot + c * RSF] =
            Impl::apply_hook(node.hook_op, src[src_base + c * sf]);
    }
  }

  template <typename SV, std::size_t... K>
  KOKKOS_FORCEINLINE_FUNCTION void fill_dispatch(
      int f, const SV& sv, register_array_t& result,
      std::index_sequence<K...>) const {
    ((f == static_cast<int>(K) ? fill_along<K>(sv, result) : void()), ...);
  }

  template <typename SV>
  KOKKOS_FORCEINLINE_FUNCTION void fill_fallback(const SV&         sv,
                                                 register_array_t& result) const {
    TENSOR_PRAGMA_UNROLL
    for (int i = 0; i < static_cast<int>(register_array_t::size); ++i) {
      auto local    = sv.layout()[i];
      result[local] = Impl::apply_hook(node.hook_op, sv[local]);
    }
  }
};

// ---------------------------------------------------------------------------
// Specialization 3: RangePolicyTag + ContractionTag + StaticTile<E...>
// (register tier)
// ---------------------------------------------------------------------------
template <typename NA, typename NB, typename IntCRank, typename S, typename ES,
          typename HookOp, int... E>
struct Evaluator<RangePolicyTag,
                 NodeHandle<ContractionTag, NA, NB, IntCRank, S, ES, HookOp>,
                 StaticTile<E...>> {
  using node_type = NodeHandle<ContractionTag, NA, NB, IntCRank, S, ES, HookOp>;
  using tiling_type         = StaticTile<E...>;
  using policy_tag          = RangePolicyTag;
  static constexpr int Rank = node_type::Rank;
  using value_type          = S;
  using exec_space          = ES;
  using register_array_t    = RegisterArray<value_type, E...>;

 private:
  static constexpr int RankC = node_type::Rank;
  static constexpr int NumK  = node_type::NumContracted;
  static constexpr int RankA = NA::Rank;
  static constexpr int RankB = NB::Rank;
  static constexpr int FreeA = RankA - NumK;
  static constexpr int FreeB = RankB - NumK;

  static_assert(tiling_type::rank == RankC + NumK,
                "contraction tile must carry one extent per participating mode");
  static_assert(FreeA + FreeB == RankC,
                "free-mode counts must sum to the output rank");

  static constexpr std::array<int, RankA> a_pos = [] {
    std::array<int, RankA> p{};
    for (int i = 0; i < FreeA; ++i) p[i] = i;
    for (int i = 0; i < NumK; ++i) p[FreeA + i] = RankC + i;
    return p;
  }();
  static constexpr std::array<int, RankB> b_pos = [] {
    std::array<int, RankB> p{};
    for (int i = 0; i < NumK; ++i) p[i] = RankC + i;
    for (int i = 0; i < FreeB; ++i) p[NumK + i] = FreeA + i;
    return p;
  }();
  static constexpr std::array<int, RankC> c_pos = [] {
    std::array<int, RankC> p{};
    for (int i = 0; i < RankC; ++i) p[i] = i;
    return p;
  }();

  using a_tile_type = Impl::project_tile_t<tiling_type, a_pos>;
  using b_tile_type = Impl::project_tile_t<tiling_type, b_pos>;
  using a_array_t   = Impl::project_regs_t<value_type, tiling_type, a_pos>;
  using b_array_t   = Impl::project_regs_t<value_type, tiling_type, b_pos>;
  using stage_a_type = Evaluator<RangePolicyTag, NA, a_tile_type>;
  using stage_b_type = Evaluator<RangePolicyTag, NB, b_tile_type>;

  static constexpr int prod_range(int lo, int count) {
    int p = 1;
    for (int i = 0; i < count; ++i) p *= tiling_type::extent(lo + i);
    return p;
  }
  static constexpr int SA = prod_range(0, FreeA);
  static constexpr int SB = prod_range(FreeA, FreeB);
  static constexpr int SK = prod_range(RankC, NumK);
  static constexpr int P  = RankC + NumK;

 public:
  using accumulator_t = Impl::project_regs_t<value_type, tiling_type, c_pos>;
  using interm_type =
      NodeHandle<IntermTag, accumulator_t, std::integral_constant<int, RankC>,
                 exec_space, HookOp>;
  using result_type = interm_type;

  node_type      node;
  tiling_type    tiling;

  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type t)
      : node(n), tiling(t),
        stage_a_(make_stage_a(n.node_a)),
        stage_b_(make_stage_b(n.node_b)) {}

  KOKKOS_FUNCTION result_type
  operator()(Kokkos::Array<int, RankC> c_tile_idx) const {
    TIMING_SCOPE_ENTER(g_timing_stats.contraction_accum_time, g_timing_stats.contraction_accum_count);
    accumulator_t acc{};
    acc.fill(value_type{0});

    Kokkos::Array<int, NumK> kext{};
    if constexpr (NumK > 0) {
      const auto a_shape = node.node_a.shape();
      for (int i = 0; i < NumK; ++i) kext[i] = a_shape[FreeA + i];
    }
    reduce_contracted(c_tile_idx, kext, acc);

    result_type out{};
    out.storage_ = acc;
    for (int d = 0; d < RankC; ++d) out.shape_[d] = tiling_type::extent(d);
    out.modes_  = node.modes();
    out.hook_op = node.hook_op;
    TIMING_SCOPE_EXIT(g_timing_stats.contraction_accum_time, g_timing_stats.contraction_accum_count);
    return out;
  }

  stage_a_type stage_a_;
  stage_b_type stage_b_;

 private:
  static KOKKOS_FUNCTION stage_a_type make_stage_a(const NA& na) {
    return make_evaluator<RangePolicyTag>(na, a_tile_type{});
  }
  static KOKKOS_FUNCTION stage_b_type make_stage_b(const NB& nb) {
    return make_evaluator<RangePolicyTag>(nb, b_tile_type{});
  }

  KOKKOS_FUNCTION void reduce_contracted(
      Kokkos::Array<int, RankC>       c_tile_idx,
      const Kokkos::Array<int, NumK>& kext, accumulator_t& acc) const {
    Kokkos::Array<int, NumK> k_tile_idx{};
    reduce_contracted_impl(c_tile_idx, k_tile_idx, kext, acc,
                           std::make_index_sequence<NumK>{});
  }

  KOKKOS_FUNCTION void reduce_contracted_impl(
      const Kokkos::Array<int, RankC>& c_tile_idx,
      Kokkos::Array<int, NumK>&        k_tile_idx,
      const Kokkos::Array<int, NumK>& /*kext*/, accumulator_t& acc,
      std::index_sequence<>) const {
    Kokkos::Array<int, P> part_tile_idx{};
    for (int d = 0; d < RankC; ++d) part_tile_idx[d]         = c_tile_idx[d];
    for (int i = 0; i < NumK; ++i)  part_tile_idx[RankC + i] = k_tile_idx[i];
    Kokkos::Array<int, RankA> a_tile_idx{};
    for (int j = 0; j < RankA; ++j) a_tile_idx[j] = part_tile_idx[a_pos[j]];
    Kokkos::Array<int, RankB> b_tile_idx{};
    for (int j = 0; j < RankB; ++j) b_tile_idx[j] = part_tile_idx[b_pos[j]];
    accumulate_block(a_tile_idx, b_tile_idx, acc);
  }

  template <std::size_t I, std::size_t... Rest>
  KOKKOS_FUNCTION void reduce_contracted_impl(
      const Kokkos::Array<int, RankC>& c_tile_idx,
      Kokkos::Array<int, NumK>&        k_tile_idx,
      const Kokkos::Array<int, NumK>&  kext, accumulator_t& acc,
      std::index_sequence<I, Rest...>) const {
    constexpr int TileK       = tiling_type::extent(RankC + I);
    const int     num_k_tiles = (kext[I] + TileK - 1) / TileK;
    for (int kt = 0; kt < num_k_tiles; ++kt) {
      k_tile_idx[I] = kt;
      reduce_contracted_impl(c_tile_idx, k_tile_idx, kext, acc,
                             std::index_sequence<Rest...>{});
    }
  }

  KOKKOS_FUNCTION void accumulate_block(Kokkos::Array<int, RankA> a_tile_idx,
                                        Kokkos::Array<int, RankB> b_tile_idx,
                                        accumulator_t& result) const {
    TIMING_SCOPE_ENTER(g_timing_stats.contraction_block_load_time, g_timing_stats.contraction_block_load_count);
    const auto& a_regs = stage_a_(a_tile_idx).storage_;
    const auto& b_regs = stage_b_(b_tile_idx).storage_;
    TIMING_SCOPE_EXIT(g_timing_stats.contraction_block_load_time, g_timing_stats.contraction_block_load_count);

    {
      TIMING_SCOPE_ENTER(g_timing_stats.contraction_compute_time, g_timing_stats.contraction_compute_count);
      namespace KE    = Kokkos::Experimental;
      using simd_t    = KE::simd<value_type>;
      using mask_t    = typename simd_t::mask_type;
      constexpr int W = static_cast<int>(simd_t::size());
      for (int fa = 0; fa < SA; ++fa) {
        int fb0 = 0;
        for (; fb0 + W <= SB; fb0 += W) {
          simd_t acc(&result.data_[fa * SB + fb0], KE::simd_flag_default);
          for (int kk = 0; kk < SK; ++kk) {
            const simd_t bvec(&b_regs.data_[kk * SB + fb0], KE::simd_flag_default);
            acc = simd_t(a_regs.data_[fa * SK + kk]) * bvec + acc;
          }
          KE::simd_unchecked_store(acc, &result.data_[fa * SB + fb0],
                                   KE::simd_flag_default);
        }
        if (fb0 < SB) {
          const mask_t mask([&](auto lane) { return fb0 + int(lane) < SB; });
          simd_t       acc = KE::simd_partial_load(&result.data_[fa * SB + fb0],
                                                   mask, KE::simd_flag_default);
          for (int kk = 0; kk < SK; ++kk) {
            const simd_t bvec = KE::simd_partial_load(
                &b_regs.data_[kk * SB + fb0], mask, KE::simd_flag_default);
            acc = simd_t(a_regs.data_[fa * SK + kk]) * bvec + acc;
          }
          KE::simd_partial_store(acc, &result.data_[fa * SB + fb0], mask,
                                 KE::simd_flag_default);
        }
      }
      TIMING_SCOPE_EXIT(g_timing_stats.contraction_compute_time, g_timing_stats.contraction_compute_count);
    }
  }
};

// ---------------------------------------------------------------------------
// Specialization 5: RangePolicyTag + IntermTag(RegisterArray) — store-evaluator
// ---------------------------------------------------------------------------
template <typename V, int... E, typename IntRank, typename ES, typename HookOp,
          typename Tiling>
struct Evaluator<
    RangePolicyTag,
    NodeHandle<IntermTag, RegisterArray<V, E...>, IntRank, ES, HookOp>,
    Tiling> {
  using node_type =
      NodeHandle<IntermTag, RegisterArray<V, E...>, IntRank, ES, HookOp>;
  using storage_type        = RegisterArray<V, E...>;
  using tiling_type         = Tiling;
  using policy_tag          = RangePolicyTag;
  static constexpr int Rank = node_type::Rank;
  using value_type          = V;
  using exec_space          = ES;
  using c_tile_type         = StaticTile<E...>;

  static_assert(storage_type::rank == Rank,
                "interm storage rank must equal node rank");

  node_type   node;
  tiling_type tiling;

  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type t) : node(n), tiling(t) {}

  template <typename ViewT>
  KOKKOS_FUNCTION void operator()(Kokkos::Array<int, Rank> tile_idx,
                                  const ViewT&             view) const {
    TIMING_SCOPE_ENTER(g_timing_stats.store_write_time, g_timing_stats.store_write_count);
    auto tv = tile_view(view, c_tile_type{});
    store(tile_idx, tv, view);
    TIMING_SCOPE_EXIT(g_timing_stats.store_write_time, g_timing_stats.store_write_count);
  }

 private:
  template <typename ViewT, typename TV>
  KOKKOS_FORCEINLINE_FUNCTION void store(const Kokkos::Array<int, Rank>& tile_idx,
                                         const TV& tv, const ViewT& view) const {
    auto sv = subview_tile(tv, tile_idx);

    bool interior = true;
    for (int d = 0; d < Rank; ++d)
      if ((tile_idx[d] + 1) * storage_type::extent(d) >
          static_cast<int>(view.extent(d)))
        interior = false;

    if (interior) {
      if constexpr (Impl::is_layout_stride_v<ViewT>) {
        const int f = sv.unit_stride_dim();
        if (f >= 0)
          store_dispatch(f, sv, std::make_index_sequence<Rank>{});
        else
          store_fallback(sv);
      } else if constexpr (Impl::is_layout_left_v<ViewT>) {
        if (sv.stride(0) == 1)
          store_along<0>(sv);
        else
          store_fallback(sv);
      } else {
        if (sv.stride(Rank - 1) == 1)
          store_along<Rank - 1>(sv);
        else
          store_fallback(sv);
      }
    } else {
      Kokkos::Array<int, Rank> avail{};
      for (int d = 0; d < Rank; ++d) {
        int rem    = static_cast<int>(view.extent(d)) -
                     tile_idx[d] * storage_type::extent(d);
        avail[d] = rem < storage_type::extent(d) ? rem : storage_type::extent(d);
      }
      Kokkos::Array<int, Rank> local{};
      store_boundary<0>(sv, local, avail);
    }
  }

  template <std::size_t F, typename SV>
  KOKKOS_FORCEINLINE_FUNCTION void store_along(const SV& sv) const {
    constexpr int Fi    = static_cast<int>(F);
    constexpr int EF    = storage_type::extent(Fi);
    constexpr int RSF   = static_cast<int>(storage_type::strides_[F]);
    constexpr int Outer = static_cast<int>(storage_type::size) / EF;
    value_type*   dst   = sv.data();
    const int     sf    = sv.stride(Fi);
    for (int o = 0; o < Outer; ++o) {
      int rem       = o;
      int base_slot = 0;
      int dst_base  = 0;
      for (int d = Rank - 1; d >= 0; --d) {
        if (d == Fi) continue;
        const int e     = storage_type::extent(d);
        const int local = rem % e;
        rem /= e;
        base_slot += local * static_cast<int>(storage_type::strides_[d]);
        dst_base  += local * sv.stride(d);
      }
      TENSOR_PRAGMA_UNROLL
      for (int c = 0; c < EF; ++c)
        dst[dst_base + c * sf] =
            Impl::apply_hook(node.hook_op, node.storage_.data_[base_slot + c * RSF]);
    }
  }

  template <typename SV, std::size_t... K>
  KOKKOS_FORCEINLINE_FUNCTION void store_dispatch(
      int f, const SV& sv, std::index_sequence<K...>) const {
    ((f == static_cast<int>(K) ? store_along<K>(sv) : void()), ...);
  }

  template <typename SV>
  KOKKOS_FORCEINLINE_FUNCTION void store_fallback(const SV& sv) const {
    TENSOR_PRAGMA_UNROLL
    for (int i = 0; i < static_cast<int>(storage_type::size); ++i) {
      auto local = sv.layout()[i];
      sv[local]  = Impl::apply_hook(node.hook_op, node.storage_[local]);
    }
  }

  template <int D, typename SVType>
  KOKKOS_FORCEINLINE_FUNCTION void store_boundary(
      const SVType& sv, Kokkos::Array<int, Rank>& local,
      const Kokkos::Array<int, Rank>& avail) const {
    for (local[D] = 0; local[D] < storage_type::extent(D); ++local[D]) {
      if (local[D] >= avail[D]) break;
      if constexpr (D + 1 < Rank)
        store_boundary<D + 1>(sv, local, avail);
      else
        sv[local] = Impl::apply_hook(node.hook_op, node.storage_[local]);
    }
  }
};
