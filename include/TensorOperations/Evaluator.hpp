#pragma once
#include <TensorOperations/Layout.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/RegisterArray.hpp>
#include <TensorOperations/Tiling.hpp>
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

// Extracts ExecSpace from a Kokkos View-like T (has ::execution_space).
// Falls back to DefaultExecutionSpace for plain TensorLike types.
template <typename T, typename = void>
struct exec_space_of {
  using type = Kokkos::DefaultExecutionSpace;
};
template <typename T>
struct exec_space_of<T, std::void_t<typename T::execution_space>> {
  using type = typename T::execution_space;
};

// Apply an input node's hook at load time. NoHook is the identity.
template <typename Op, typename V>
KOKKOS_FORCEINLINE_FUNCTION V apply_hook(const Op& op, V v) {
  return op(v);
}
template <typename V>
KOKKOS_FORCEINLINE_FUNCTION V apply_hook(const NoHook&, V v) {
  return v;
}

}  // namespace Impl

// ---------------------------------------------------------------------------
// Primary template — undefined; must use a specialization
// ---------------------------------------------------------------------------
template <typename PolicyTag, typename NodeType, typename Tiling>
struct Evaluator;

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
  using result_type         = register_array_t;

  static_assert(node_type::Rank == Rank,
                "input staging tile must carry one extent per input mode");

  node_type   node;
  tiling_type tiling;
  layout_type layout;  // default-constructed = contiguous (stride 1)

  // Stage the tile whose global origin is `tile_offset` into `result`.
  KOKKOS_FUNCTION void operator()(std::array<int, Rank> tile_offset,
                                  register_array_t&     result) const {
    fill_slots(tile_offset, result,
               std::make_index_sequence<register_array_t::size>{});
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
  KOKKOS_FORCEINLINE_FUNCTION value_type
  load_slot(const std::array<int, Rank>& off, std::index_sequence<D...>) const {
    return Impl::apply_hook(
        node.hook_op,
        node.handle(
            (off[D] + local_index<S, D>() * layout[static_cast<int>(D)])...));
  }

  // Unroll over every register slot; the write index S stays compile-time so
  // the RegisterArray remains register-resident.
  template <std::size_t... S>
  KOKKOS_FORCEINLINE_FUNCTION void fill_slots(const std::array<int, Rank>& off,
                                              register_array_t& result,
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
      NodeHandle<IntermTag, value_type, std::integral_constant<int, Rank>,
                 exec_space, NoHook>;
  using result_type = interm_type;

  node_type   node;
  tiling_type tiling;

  KOKKOS_FUNCTION result_type
  operator()(const typename Kokkos::TeamPolicy<exec_space>::member_type& team,
             std::array<int, Rank> tile_offset) const;
};

// ---------------------------------------------------------------------------
// Specialization 3: RangePolicyTag + ContractionTag + StaticTile<E...>
// (register tier)
// ---------------------------------------------------------------------------
// Register-blocked: each thread holds a register working set (StaticTile
// carries one extent per participating mode = Rank_C + NumContracted), runs an
// outer- product update over the contracted modes, and writes its output
// sub-tile back to the pre-allocated global result. The kernel projects the
// accumulator sub-tile out of the full participating-mode tile. Body defined in
// Eval.hpp.
template <typename NA, typename NB, typename IntCRank, typename S, typename ES,
          typename HookOp, int... E>
struct Evaluator<RangePolicyTag,
                 NodeHandle<ContractionTag, NA, NB, IntCRank, S, ES, HookOp>,
                 StaticTile<E...>> {
  using node_type = NodeHandle<ContractionTag, NA, NB, IntCRank, S, ES, HookOp>;
  using tiling_type         = StaticTile<E...>;
  using policy_tag          = RangePolicyTag;
  static constexpr int Rank = node_type::Rank;  // output rank
  using value_type          = S;
  using exec_space          = ES;
  using register_array_t =
      RegisterArray<value_type, E...>;  // staging working set
  using interm_type =
      NodeHandle<IntermTag, value_type, std::integral_constant<int, Rank>,
                 exec_space, NoHook>;
  using result_type = interm_type;

  node_type   node;
  tiling_type tiling;
  result_type result;  // pre-allocated output node; view_ written by operator()

  KOKKOS_FUNCTION void operator()(int flat_idx) const;
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
      NodeHandle<IntermTag, value_type, std::integral_constant<int, Rank>,
                 exec_space, NoHook>;
  using result_type = interm_type;

  node_type   node;
  tiling_type tiling;

  KOKKOS_FUNCTION result_type
  operator()(const typename Kokkos::TeamPolicy<exec_space>::member_type& team,
             std::array<int, Rank> tile_offset) const;
};

}  // namespace TensorOperations
