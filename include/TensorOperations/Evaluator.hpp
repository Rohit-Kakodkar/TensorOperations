#pragma once
#include <TensorOperations/Layout.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/RegisterArray.hpp>
#include <TensorOperations/Tiling.hpp>
#include <TensorOperations/Macros.hpp>
#include <TensorOperations/TimingInstrumentation.hpp>
#include <array>
#include <utility>

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
namespace TensorOperations {

// ---------------------------------------------------------------------------
// Policy tags
// ---------------------------------------------------------------------------
struct RangePolicyTag {};
struct TeamPolicyTag {};

// Tiling specs (StaticTile / DynamicTile) live in Tiling.hpp. There is no
// NoTiling: the un-staged baseline is the degenerate unit tile.

// ---------------------------------------------------------------------------
// Impl helpers
// ---------------------------------------------------------------------------
namespace Impl {

// Apply an input node's hook at load time. NoHook is the identity.
template <typename Op, typename V>
KOKKOS_FORCEINLINE_FUNCTION V apply_hook(const Op& op, V v) {
  return op(v);
}
template <typename V>
KOKKOS_FORCEINLINE_FUNCTION V apply_hook(const NoHook&, V v) {
  return v;
}

// Project a tiling type onto a set of participating-mode positions, yielding a
// StaticTile / RegisterArray whose extents are Tile::extent(Pos[i])... . `Pos`
// is a constexpr std::array of positions (Option A: pure rank arithmetic, no
// mode-label lookup). A single std::make_index_sequence drives the unpack.
template <typename Tile, auto Pos, std::size_t... I>
StaticTile<Tile::extent(static_cast<int>(Pos[I]))...> project_tile_fn(
    std::index_sequence<I...>);
template <typename Tile, auto Pos>
using project_tile_t = decltype(project_tile_fn<Tile, Pos>(
    std::make_index_sequence<Pos.size()>{}));

template <typename V, typename Tile, auto Pos, std::size_t... I>
RegisterArray<V, Tile::extent(static_cast<int>(Pos[I]))...> project_regs_fn(
    std::index_sequence<I...>);
template <typename V, typename Tile, auto Pos>
using project_regs_t = decltype(project_regs_fn<V, Tile, Pos>(
    std::make_index_sequence<Pos.size()>{}));

// Outer slot decomposition shared by the gather (Spec 1) and store (Spec 5)
// fast paths. ArrayType is RegisterArray<V,E...>; F is the compile-time
// fast-varying dim (chosen to match the source/destination view's contiguous
// dim). OuterFn computes gidx[d] for the outer (non-F) modes from (d, local);
// InnerFn receives (slot, gidx) and performs the actual load or store.
// NOTE: lambdas are fine on the Serial CPU backend. If a GPU backend is added,
// replace the lambda arguments with KOKKOS_LAMBDA or explicit function objects.
template <typename ArrayType, std::size_t F, int Rank,
          typename OuterFn, typename InnerFn>
KOKKOS_FORCEINLINE_FUNCTION void traverse_along(
    const Kokkos::Array<int, Rank>& off, OuterFn outer_fn, InnerFn inner_fn) {
  constexpr int Fi    = static_cast<int>(F);
  constexpr int EF    = ArrayType::extent(Fi);
  constexpr int RSF   = static_cast<int>(ArrayType::strides_[F]);
  constexpr int Outer = static_cast<int>(ArrayType::size) / EF;
  for (int o = 0; o < Outer; ++o) {
    Kokkos::Array<int, Rank> gidx{};
    int                      base_slot = 0;
    int                      rem       = o;
    for (int d = Rank - 1; d >= 0; --d) {
      if (d == Fi) continue;
      const int e     = ArrayType::extent(d);
      const int local = rem % e;
      rem /= e;
      gidx[d]    = outer_fn(d, local);
      base_slot += local * static_cast<int>(ArrayType::strides_[d]);
    }
    TENSOR_PRAGMA_UNROLL
    for (int c = 0; c < EF; ++c) {
      gidx[Fi] = off[Fi] + c;
      inner_fn(base_slot + c * RSF, gidx);
    }
  }
}

}  // namespace Impl

// ---------------------------------------------------------------------------
// Primary template — undefined; must use a specialization
// ---------------------------------------------------------------------------
template <typename PolicyTag, typename NodeType, typename Tiling>
struct Evaluator;

template <typename PolicyTag, typename NodeType, typename Tile>
KOKKOS_FUNCTION auto make_evaluator(NodeType node, Tile tile)
    -> Evaluator<PolicyTag, NodeType, Tile> {
  return {node, tile};
}

// ---------------------------------------------------------------------------
// Specialization 1: RangePolicyTag + InputTag + StaticTile<E...>  (register
// tier)
// ---------------------------------------------------------------------------
// Each thread stages one tile of the input into a thread-private RegisterArray.
// The caller owns the register block and passes it as an out-param, addressing
// the read by the tile's global origin `tile_offset`; the slot -> global rule
// (contiguous vs staggered/coalesced) lives in `layout`. The un-staged baseline
// is the unit tile StaticTile<1,...,1>. Range tier is static-only: a
// DynamicTile here matches no specialization.
//
// Assumes the tensor extents are divisible by the tile extents (no boundary
// guard); the hook is applied at load time. Body is inline below.
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
  using layout_type         = StridedLayout<Rank>;
  // Result is an IntermTag tile node backed by the staged RegisterArray. The
  // input hook is applied at load, so the node carries NoHook.
  using interm_type =
      NodeHandle<IntermTag, register_array_t, std::integral_constant<int, Rank>,
                 exec_space, NoHook>;
  using result_type = interm_type;

  static_assert(node_type::Rank == Rank,
                "input staging tile must carry one extent per input mode");

  node_type   node;
  tiling_type tiling;
  layout_type layout;  // default-constructed = contiguous (stride 1)

  // Stage the tile whose global origin is `tile_offset` and return it as an
  // IntermTag tile node.
  KOKKOS_FUNCTION result_type
  operator()(Kokkos::Array<int, Rank> tile_offset) const {
    TIMING_SCOPE_ENTER(g_timing_stats.input_stage_load_time, g_timing_stats.input_stage_load_count);
    result_type out{};
    for (int d = 0; d < Rank; ++d) out.shape_[d] = tiling_type::extent(d);
    out.modes_ = node.modes();
    fill(tile_offset, out.storage_);
    TIMING_SCOPE_EXIT(g_timing_stats.input_stage_load_time, g_timing_stats.input_stage_load_count);
    return out;
  }

 private:
  // ---- layout-aware staging -------------------------------------------------
  // Pick the input view's contiguous mode F (compile-time for LayoutRight/Left,
  // runtime argmin over the strides for LayoutStride) and gather slots in runs
  // along F so the per-run reads are sequential in memory. The RegisterArray
  // storage stays row-major: this only permutes the *visit order* of slots, so
  // results are bit-identical to the per-element `fill_slots` fallback. The
  // fast path assumes a unit slot->global step along F (`layout[F] == 1`, the
  // default); a non-unit / staggered layout falls back to `fill_slots`.
  KOKKOS_FORCEINLINE_FUNCTION void fill(const Kokkos::Array<int, Rank>& off,
                                        register_array_t& result) const {
    if constexpr (Impl::is_layout_stride_v<T>) {
      // Runtime: make the smallest-stride dimension vary fastest.
      int  f    = 0;
      auto best = node.handle.stride(0);
      for (int d = 1; d < Rank; ++d) {
        const auto s = node.handle.stride(d);
        if (s < best) { best = s; f = d; }
      }
      if (layout[f] == 1)
        fill_dispatch(f, off, result, std::make_index_sequence<Rank>{});
      else
        fill_slots(off, result,
                   std::make_index_sequence<register_array_t::size>{});
    } else if constexpr (Impl::is_layout_left_v<T>) {
      if (layout[0] == 1)
        fill_along<0>(off, result);
      else
        fill_slots(off, result,
                   std::make_index_sequence<register_array_t::size>{});
    } else {  // LayoutRight or a plain grid type (defaults to row-major)
      if (layout[Rank - 1] == 1)
        fill_along<Rank - 1>(off, result);
      else
        fill_slots(off, result,
                   std::make_index_sequence<register_array_t::size>{});
    }
  }

  // Gather every slot with mode F varying in the (contiguous) inner run. The
  // inner trip count and the register stride along F are compile-time, so with
  // a unit memory step the compiler emits packed loads.
  template <std::size_t F>
  KOKKOS_FORCEINLINE_FUNCTION void fill_along(const Kokkos::Array<int, Rank>& off,
                                              register_array_t& result) const {
    Impl::traverse_along<register_array_t, F, Rank>(
        off,
        [&](int d, int local) { return off[d] + local * layout[d]; },
        [&](int slot, const Kokkos::Array<int, Rank>& gidx) {
          result.data_[slot] = Impl::apply_hook(
              node.hook_op, read_handle(gidx, std::make_index_sequence<Rank>{}));
        });
  }

  // Runtime F dispatch (LayoutStride): exactly one compile-time `fill_along`
  // runs for the chosen fastest-varying dimension.
  template <std::size_t... K>
  KOKKOS_FORCEINLINE_FUNCTION void fill_dispatch(
      int f, const Kokkos::Array<int, Rank>& off, register_array_t& result,
      std::index_sequence<K...>) const {
    ((f == static_cast<int>(K) ? fill_along<K>(off, result) : void()), ...);
  }

  template <std::size_t... D>
  KOKKOS_FORCEINLINE_FUNCTION value_type
  read_handle(const Kokkos::Array<int, Rank>& gidx,
              std::index_sequence<D...>) const {
    return node.handle(gidx[D]...);
  }

  // ---- per-element fallback (any layout) ------------------------------------
  // Compile-time local index of register slot S along mode D (row-major
  // decode).
  template <std::size_t S, std::size_t D>
  static constexpr int local_index() {
    return static_cast<int>((S / register_array_t::strides_[D]) %
                            register_array_t::extents_[D]);
  }

  // Load one register slot S: gather its global multi-index and read it.
  template <std::size_t S, std::size_t... D>
  KOKKOS_FORCEINLINE_FUNCTION value_type load_slot(
      const Kokkos::Array<int, Rank>& off, std::index_sequence<D...>) const {
    return Impl::apply_hook(
        node.hook_op,
        node.handle(
            (off[D] + local_index<S, D>() * layout[static_cast<int>(D)])...));
  }

  // Unroll over every register slot; the write index S stays compile-time so
  // the RegisterArray remains register-resident.
  template <std::size_t... S>
  KOKKOS_FORCEINLINE_FUNCTION void fill_slots(
      const Kokkos::Array<int, Rank>& off, register_array_t& result,
      std::index_sequence<S...>) const {
    ((result.data_[S] = load_slot<S>(off, std::make_index_sequence<Rank>{})),
     ...);
  }
};

// ---------------------------------------------------------------------------
// Specialization 2: TeamPolicyTag + InputTag + Tile_  (scratch tier)
// ---------------------------------------------------------------------------
// Single specialization covering both StaticTile and DynamicTile. The team
// copies the tile from global memory into scratch and returns an IntermTag
// node backed by that scratch view. Body defined in Eval.hpp.
template <TensorLike T, typename HookOp, typename Tile_>
struct Evaluator<TeamPolicyTag, NodeHandle<InputTag, T, HookOp>, Tile_> {
  using node_type           = NodeHandle<InputTag, T, HookOp>;
  using tiling_type         = Tile_;
  using policy_tag          = TeamPolicyTag;
  static constexpr int Rank = tiling_type::rank;
  using value_type          = typename node_type::value_type;
  using exec_space          = typename Impl::exec_space_of<T>::type;
  using scratch_space       = typename exec_space::scratch_memory_space;
  using scratch_view_t =
      typename Impl::KokkosViewN<value_type, Rank, scratch_space>::type;
  using interm_type =
      NodeHandle<IntermTag, scratch_view_t, std::integral_constant<int, Rank>,
                 exec_space, NoHook>;
  using result_type = interm_type;

  node_type   node;
  tiling_type tiling;

  KOKKOS_FUNCTION result_type
  operator()(const typename Kokkos::TeamPolicy<exec_space>::member_type& team,
             Kokkos::Array<int, Rank> tile_offset) const;
};

// ---------------------------------------------------------------------------
// Specialization 3: RangePolicyTag + ContractionTag + StaticTile<E...>
// (register tier)
// ---------------------------------------------------------------------------
// Register-blocked, GEMM-structural convention (Option A): the participating
// tile StaticTile<E...> carries one extent per participating mode
// (Rank_C + NumContracted) ordered [freeA.., freeB.., contracted..], with each
// operand stored [freeA.., contracted..] (A) and [contracted.., freeB..] (B).
// Mirroring the input stager, operator() takes the A and B tile offsets and an
// out-param C accumulator: it stages both operand tiles into registers (reusing
// Specialization 1) and accumulates (+=) their contraction into `result`. The
// caller zeroes the accumulator, computes the offsets, loops the contracted
// blocks, and writes the finished accumulator into the interm node view.
//
// General (label-matched) projection for arbitrary operand mode orders is
// tracked in issues/0001-general-mode-projection.md.
template <typename NA, typename NB, typename IntCRank, typename S, typename ES,
          typename HookOp, int... E>
struct Evaluator<RangePolicyTag,
                 NodeHandle<ContractionTag, NA, NB, IntCRank, S, ES, HookOp>,
                 StaticTile<E...>> {
  using node_type = NodeHandle<ContractionTag, NA, NB, IntCRank, S, ES, HookOp>;
  using tiling_type         = StaticTile<E...>;
  using policy_tag          = RangePolicyTag;
  static constexpr int Rank = node_type::Rank;  // output (C) rank
  using value_type          = S;
  using exec_space          = ES;
  // Kept for the existing EvaluatorSelection static_asserts (full set of E...).
  using register_array_t = RegisterArray<value_type, E...>;

 private:
  // Structural (GEMM) convention counts — all compile-time from ranks.
  static constexpr int RankC = node_type::Rank;
  static constexpr int NumK  = node_type::NumContracted;
  static constexpr int RankA = NA::Rank;
  static constexpr int RankB = NB::Rank;
  static constexpr int FreeA = RankA - NumK;
  static constexpr int FreeB = RankB - NumK;

  static_assert(tiling_type::rank == RankC + NumK,
                "contraction tile must carry one extent per participating mode "
                "(Rank_C + NumContracted)");
  static_assert(FreeA + FreeB == RankC,
                "free-mode counts must sum to the output rank");

  // Participating-tile positions for each operand under the GEMM convention:
  //   [0,FreeA)=freeA   [FreeA,RankC)=freeB   [RankC,RankC+NumK)=contracted
  static constexpr std::array<int, RankA> a_pos = [] {
    std::array<int, RankA> p{};
    for (int i = 0; i < FreeA; ++i) p[i] = i;                 // freeA
    for (int i = 0; i < NumK; ++i) p[FreeA + i] = RankC + i;  // contracted
    return p;
  }();
  static constexpr std::array<int, RankB> b_pos = [] {
    std::array<int, RankB> p{};
    for (int i = 0; i < NumK; ++i) p[i] = RankC + i;          // contracted
    for (int i = 0; i < FreeB; ++i) p[NumK + i] = FreeA + i;  // freeB
    return p;
  }();
  static constexpr std::array<int, RankC> c_pos = [] {
    std::array<int, RankC> p{};
    for (int i = 0; i < RankC; ++i) p[i] = i;  // freeA ++ freeB
    return p;
  }();

  // Compile-time projected tile/register types for the A and B operand tiles.
  using a_tile_type = Impl::project_tile_t<tiling_type, a_pos>;
  using b_tile_type = Impl::project_tile_t<tiling_type, b_pos>;
  using a_array_t   = Impl::project_regs_t<value_type, tiling_type, a_pos>;
  using b_array_t   = Impl::project_regs_t<value_type, tiling_type, b_pos>;

  // Flat sizes of each mode group. Under the row-major register layout the
  // contraction is a plain GEMM on flat storage: C[SA x SB] += A[SA x SK] *
  // B[SK x SB].
  static constexpr int prod_range(int lo, int count) {
    int p = 1;
    for (int i = 0; i < count; ++i) p *= tiling_type::extent(lo + i);
    return p;
  }
  static constexpr int SA = prod_range(0, FreeA);      // free-A extent product
  static constexpr int SB = prod_range(FreeA, FreeB);  // free-B extent product
  static constexpr int SK = prod_range(RankC, NumK);   // contracted product

  static constexpr int P = RankC + NumK;  // participating-mode count

 public:
  // The C accumulator over the free (output) modes.
  using accumulator_t = Impl::project_regs_t<value_type, tiling_type, c_pos>;
  // Result is the finished output tile as an IntermTag node, carrying the
  // contraction's hook (applied later at store time).
  using interm_type =
      NodeHandle<IntermTag, accumulator_t, std::integral_constant<int, RankC>,
                 exec_space, HookOp>;
  using result_type = interm_type;

  node_type   node;
  tiling_type tiling;

  // Compute the complete output tile whose free-mode origin is
  // `c_tile_offset`: sum over the entire contracted extent K, then return the
  // finished tile node (hook carried, applied at store).
  KOKKOS_FUNCTION result_type
  operator()(Kokkos::Array<int, RankC> c_tile_offset) const {
    TIMING_SCOPE_ENTER(g_timing_stats.contraction_accum_time, g_timing_stats.contraction_accum_count);
    accumulator_t acc{};
    acc.fill(value_type{0});

    Kokkos::Array<int, NumK> kext{};
    if constexpr (NumK > 0) {
      const auto a_shape = node.node_a.shape();  // contracted extents (GEMM)
      for (int i = 0; i < NumK; ++i) kext[i] = a_shape[FreeA + i];
    }
    reduce_contracted(c_tile_offset, kext, acc);

    result_type out{};
    out.storage_ = acc;
    for (int d = 0; d < RankC; ++d) out.shape_[d] = tiling_type::extent(d);
    out.modes_  = node.modes();
    out.hook_op = node.hook_op;  // carried; applied by the overwrite store
    TIMING_SCOPE_EXIT(g_timing_stats.contraction_accum_time, g_timing_stats.contraction_accum_count);
    return out;
  }

 private:
  // Nested loop over the contracted modes (compile-time depth NumK, runtime
  // extents, compile-time tile step — no modulo/division). At the base case the
  // assembled A/B offsets feed one local GEMM block.
  KOKKOS_FUNCTION void reduce_contracted(Kokkos::Array<int, RankC>       c_off,
                                         const Kokkos::Array<int, NumK>& kext,
                                         accumulator_t& acc) const {
    Kokkos::Array<int, NumK> k_off{};
    reduce_contracted_impl(c_off, k_off, kext, acc,
                           std::make_index_sequence<NumK>{});
  }

  // Base case: all contracted modes stepped; assemble offsets and accumulate.
  KOKKOS_FUNCTION void reduce_contracted_impl(
      const Kokkos::Array<int, RankC>& c_off, Kokkos::Array<int, NumK>& k_off,
      const Kokkos::Array<int, NumK>& /*kext*/, accumulator_t&          acc,
      std::index_sequence<>) const {
    Kokkos::Array<int, P> part_off{};  // [free.., contracted..]
    for (int d = 0; d < RankC; ++d) part_off[d] = c_off[d];
    for (int i = 0; i < NumK; ++i) part_off[RankC + i] = k_off[i];
    Kokkos::Array<int, RankA> a_off{};
    for (int j = 0; j < RankA; ++j) a_off[j] = part_off[a_pos[j]];
    Kokkos::Array<int, RankB> b_off{};
    for (int j = 0; j < RankB; ++j) b_off[j] = part_off[b_pos[j]];
    accumulate_block(a_off, b_off, acc);
  }

  // Recursive case: peel the head index I, loop over that contracted mode.
  template <std::size_t I, std::size_t... Rest>
  KOKKOS_FUNCTION void reduce_contracted_impl(
      const Kokkos::Array<int, RankC>& c_off, Kokkos::Array<int, NumK>& k_off,
      const Kokkos::Array<int, NumK>& kext, accumulator_t& acc,
      std::index_sequence<I, Rest...>) const {
    constexpr int TileK = tiling_type::extent(RankC + I);
    for (int off = 0; off < kext[I]; off += TileK) {
      k_off[I] = off;
      reduce_contracted_impl(c_off, k_off, kext, acc,
                             std::index_sequence<Rest...>{});
    }
  }

  // Stage the A-tile at `a_off` and B-tile at `b_off` into registers, then
  // accumulate their contraction into `result` (one contracted block).
  KOKKOS_FUNCTION void accumulate_block(Kokkos::Array<int, RankA> a_off,
                                        Kokkos::Array<int, RankB> b_off,
                                        accumulator_t& result) const {

    TIMING_SCOPE_ENTER(g_timing_stats.contraction_block_load_time, g_timing_stats.contraction_block_load_count);
    auto stage_a = make_evaluator<RangePolicyTag>(node.node_a, a_tile_type{});
    auto stage_b = make_evaluator<RangePolicyTag>(node.node_b, b_tile_type{});
    const auto &a_regs = stage_a(a_off).storage_;
    const auto &b_regs = stage_b(b_off).storage_;
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
// Specialization 4: TeamPolicyTag + ContractionTag + Tile_  (scratch tier)
// ---------------------------------------------------------------------------
// Single specialization covering both StaticTile and DynamicTile. The team
// computes one output tile into scratch memory and returns the scratch-backed
// IntermTag node. Body defined in Eval.hpp.
template <typename NA, typename NB, typename IntCRank, typename S, typename ES,
          typename HookOp, typename Tile_>
struct Evaluator<TeamPolicyTag,
                 NodeHandle<ContractionTag, NA, NB, IntCRank, S, ES, HookOp>,
                 Tile_> {
  using node_type = NodeHandle<ContractionTag, NA, NB, IntCRank, S, ES, HookOp>;
  using tiling_type         = Tile_;
  using policy_tag          = TeamPolicyTag;
  static constexpr int Rank = node_type::Rank;  // output rank
  using value_type          = S;
  using exec_space          = ES;
  using scratch_space       = typename ES::scratch_memory_space;
  using scratch_view_t =
      typename Impl::KokkosViewN<value_type, Rank, scratch_space>::type;
  using interm_type =
      NodeHandle<IntermTag, scratch_view_t, std::integral_constant<int, Rank>,
                 exec_space, NoHook>;
  using result_type = interm_type;

  node_type   node;
  tiling_type tiling;

  KOKKOS_FUNCTION result_type
  operator()(const typename Kokkos::TeamPolicy<exec_space>::member_type& team,
             Kokkos::Array<int, Rank> tile_offset) const;
};

// ---------------------------------------------------------------------------
// Specialization 5: RangePolicyTag + IntermTag(RegisterArray) —
// store-evaluator
// ---------------------------------------------------------------------------
// Overwrite-drains a finished register-tier output tile into a destination
// view at a given global offset, applying the node's hook at store time. The
// Tiling parameter is unused (the tile shape lives in the RegisterArray
// storage). Assumes the destination extents are divisible by the tile
// extents; a boundary guard skips slots that fall outside the view.
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

  static_assert(storage_type::rank == Rank,
                "interm storage rank must equal node rank");

  node_type   node;
  tiling_type tiling;

  // Write the tile into `view` at global origin `offset` (hook applied).
  template <typename ViewT>
  KOKKOS_FUNCTION void operator()(Kokkos::Array<int, Rank> offset,
                                  const ViewT&             view) const {
    TIMING_SCOPE_ENTER(g_timing_stats.store_write_time, g_timing_stats.store_write_count);
    store(offset, view);
    TIMING_SCOPE_EXIT(g_timing_stats.store_write_time, g_timing_stats.store_write_count);
  }

 private:
  // ---- layout-aware draining ------------------------------------------------
  // A fully-interior tile (no mode hangs off the view edge) needs no bounds
  // guard, so drain it in contiguous runs along the view's fastest dimension F.
  // Storage stays row-major (only the visit order is permuted), so results are
  // bit-identical to the guarded `store_slots` fallback. Boundary tiles take
  // the guarded path.
  template <typename ViewT>
  KOKKOS_FORCEINLINE_FUNCTION void store(const Kokkos::Array<int, Rank>& off,
                                         const ViewT& view) const {
    bool interior = true;
    for (int d = 0; d < Rank; ++d)
      if (off[d] + storage_type::extent(d) > static_cast<int>(view.extent(d)))
        interior = false;
    if (!interior) {
      store_slots(off, view, std::make_index_sequence<storage_type::size>{});
      return;
    }
    if constexpr (Impl::is_layout_stride_v<ViewT>) {
      int  f    = 0;
      auto best = view.stride(0);
      for (int d = 1; d < Rank; ++d) {
        const auto s = view.stride(d);
        if (s < best) { best = s; f = d; }
      }
      store_dispatch(f, off, view, std::make_index_sequence<Rank>{});
    } else if constexpr (Impl::is_layout_left_v<ViewT>) {
      store_along<0>(off, view);
    } else {  // LayoutRight or default
      store_along<Rank - 1>(off, view);
    }
  }

  // Drain every slot with mode F varying in the (contiguous) inner run.
  template <std::size_t F, typename ViewT>
  KOKKOS_FORCEINLINE_FUNCTION void store_along(const Kokkos::Array<int, Rank>& off,
                                               const ViewT& view) const {
    Impl::traverse_along<storage_type, F, Rank>(
        off,
        [&](int d, int local) { return off[d] + local; },
        [&](int slot, const Kokkos::Array<int, Rank>& gidx) {
          write_handle(view, gidx,
                       Impl::apply_hook(node.hook_op, node.storage_.data_[slot]),
                       std::make_index_sequence<Rank>{});
        });
  }

  template <std::size_t... K, typename ViewT>
  KOKKOS_FORCEINLINE_FUNCTION void store_dispatch(
      int f, const Kokkos::Array<int, Rank>& off, const ViewT& view,
      std::index_sequence<K...>) const {
    ((f == static_cast<int>(K) ? store_along<K>(off, view) : void()), ...);
  }

  template <typename ViewT, std::size_t... D>
  KOKKOS_FORCEINLINE_FUNCTION void write_handle(
      const ViewT& view, const Kokkos::Array<int, Rank>& gidx, value_type v,
      std::index_sequence<D...>) const {
    view(gidx[D]...) = v;
  }

  // ---- per-element guarded fallback (boundary tiles) ------------------------
  // Compile-time local index of register slot S along mode D (row-major
  // decode).
  template <std::size_t S, std::size_t D>
  static constexpr int local_index() {
    return static_cast<int>((S / storage_type::strides_[D]) %
                            storage_type::extents_[D]);
  }

  // Write one register slot S into the view (hook applied), with a per-mode
  // bounds guard so partial boundary tiles do not write out of range.
  template <typename ViewT, std::size_t S, std::size_t... D>
  KOKKOS_FORCEINLINE_FUNCTION void store_slot(
      const Kokkos::Array<int, Rank>& off, const ViewT& view,
      std::index_sequence<D...>) const {
    const bool in =
        ((off[D] + local_index<S, D>() < static_cast<int>(view.extent(D))) &&
         ...);
    if (in)
      view((off[D] + local_index<S, D>())...) =
          Impl::apply_hook(node.hook_op, node.storage_.data_[S]);
  }

  // Unroll over every register slot; the read index S stays compile-time.
  template <typename ViewT, std::size_t... S>
  KOKKOS_FORCEINLINE_FUNCTION void store_slots(
      const Kokkos::Array<int, Rank>& off, const ViewT& view,
      std::index_sequence<S...>) const {
    (store_slot<ViewT, S>(off, view, std::make_index_sequence<Rank>{}), ...);
  }
};

}  // namespace TensorOperations
