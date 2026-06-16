#pragma once
#include <TensorOperations/Layout.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/RegisterArray.hpp>
#include <TensorOperations/Tiling.hpp>
#include <TensorOperations/Macros.hpp>
#include <array>
#include <utility>

#include <Kokkos_Core.hpp>
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
    result_type out{};
    for (int d = 0; d < Rank; ++d) out.shape_[d] = tiling_type::extent(d);
    out.modes_ = node.modes();
    fill_slots(tile_offset, out.storage_,
               std::make_index_sequence<register_array_t::size>{});
    return out;
  }

 private:
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

  // Register block (micro-kernel) dims, decoupled from the tile (SA x SB). The
  // hot accumulator working set is one MR x NR block, kept register-resident
  // even when the full SA x SB tile accumulator would spill. Tunable here.
  static constexpr int MR0 = 4;  // preferred register-block rows
  static constexpr int NR0 =
      8;  // preferred register-block lanes (contiguous SB)
  static constexpr int MR = SA < MR0 ? SA : MR0;
  static constexpr int NR = SB < NR0 ? SB : NR0;
  static_assert(SA % MR == 0 && SB % NR == 0,
                "tile free extents must be divisible by the register block");

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
  // accumulate their contraction into `result` (one contracted block). The
  // SA x SB accumulator is walked as a grid of MR x NR register blocks; each
  // block is computed by a register-resident, vectorized micro-kernel.
  KOKKOS_FUNCTION void accumulate_block(Kokkos::Array<int, RankA> a_off,
                                        Kokkos::Array<int, RankB> b_off,
                                        accumulator_t& result) const {
    auto stage_a = make_evaluator<RangePolicyTag>(node.node_a, a_tile_type{});
    auto stage_b = make_evaluator<RangePolicyTag>(node.node_b, b_tile_type{});
    const auto a_regs = stage_a(a_off).storage_;
    const auto b_regs = stage_b(b_off).storage_;

    accumulate_blocks(a_regs, b_regs, result,
                      std::make_index_sequence<(SA / MR) * (SB / NR)>{});
  }

  // Register block type: row-major MR x NR, flat data_[mr * NR + nr].
  using block_t = RegisterArray<value_type, MR, NR>;

  // Copy the MR x NR sub-block at (IA, IB) between `result` and a local block.
  // All indices are compile-time constants, so the block stays
  // register-resident.
  template <int IA, int IB, std::size_t... Pos>
  KOKKOS_FORCEINLINE_FUNCTION void mk_load(const accumulator_t& result,
                                           block_t&             c,
                                           std::index_sequence<Pos...>) const {
    ((c.data_[Pos] =
          result.data_[(IA + int(Pos) / NR) * SB + (IB + int(Pos) % NR)]),
     ...);
  }
  template <int IA, int IB, std::size_t... Pos>
  KOKKOS_FORCEINLINE_FUNCTION void mk_store(const block_t& c,
                                            accumulator_t& result,
                                            std::index_sequence<Pos...>) const {
    ((result.data_[(IA + int(Pos) / NR) * SB + (IB + int(Pos) % NR)] =
          c.data_[Pos]),
     ...);
  }

  // One contracted-step rank-1 update of the whole MR x NR block: for every
  // (mi, ni) in the block, c[mi,ni] += A[(IA+mi), kk] * B[kk, (IB+ni)]. mi/ni
  // are compile-time (the fold over Pos); only kk is runtime. The ni lanes are
  // contiguous in both b_regs and c, so this becomes a broadcast-A vector FMA.
  template <int IA, int IB, std::size_t... Pos>
  KOKKOS_FORCEINLINE_FUNCTION void mk_kstep(const a_array_t& a,
                                            const b_array_t& b, int kk,
                                            block_t& c,
                                            std::index_sequence<Pos...>) const {
    ((c.data_[Pos] += a.data_[(IA + int(Pos) / NR) * SK + kk] *
                      b.data_[kk * SB + (IB + int(Pos) % NR)]),
     ...);
  }

  // Compute the MR x NR register block whose origin in the tile is (IA, IB).
  template <int IA, int IB>
  KOKKOS_FORCEINLINE_FUNCTION void micro_kernel(const a_array_t& a,
                                                const b_array_t& b,
                                                accumulator_t&   result) const {
    block_t c{};
    mk_load<IA, IB>(result, c, std::make_index_sequence<MR * NR>{});
    TENSOR_PRAGMA_UNROLL
    for (int kk = 0; kk < SK; ++kk)
      mk_kstep<IA, IB>(a, b, kk, c, std::make_index_sequence<MR * NR>{});
    mk_store<IA, IB>(c, result, std::make_index_sequence<MR * NR>{});
  }

  // Unroll the MR x NR block grid over the SA x SB accumulator (compile-time
  // origins). Q decodes to block-row (IA) and block-col (IB).
  template <std::size_t... Q>
  KOKKOS_FORCEINLINE_FUNCTION void accumulate_blocks(
      const a_array_t& a, const b_array_t& b, accumulator_t& result,
      std::index_sequence<Q...>) const {
    constexpr int NBcol = SB / NR;
    (micro_kernel<(int(Q) / NBcol) * MR, (int(Q) % NBcol) * NR>(a, b, result),
     ...);
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
    store_slots(offset, view, std::make_index_sequence<storage_type::size>{});
  }

 private:
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
