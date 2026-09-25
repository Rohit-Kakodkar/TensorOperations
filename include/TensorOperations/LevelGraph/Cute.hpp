#pragma once
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph/Team.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

namespace TensorOperations {
namespace Impl {

inline constexpr std::size_t lg_cute_default_smem_bytes = 48 * 1024;

struct LinearTileScheduler {
  static dim3 grid(int league) { return dim3(static_cast<unsigned>(league)); }

  template <std::size_t RootR, typename GridTile>
  __device__ static Kokkos::Array<int, RootR> tile_coord(
      const Kokkos::Array<int, RootR>& shape) {
    return decode_tile_index<static_cast<int>(RootR)>(
        static_cast<int>(blockIdx.x), shape, GridTile{});
  }
};

template <typename LevelsT, std::size_t... Ls>
constexpr bool lg_cute_all_staged(std::index_sequence<Ls...>) {
  return (lg_all_staged_v<tuple_element_t<Ls, LevelsT>> && ...);
}

template <typename Operand>
struct lg_cute_array_layout {
  static_assert(
      has_node_tag_v<InputTag, Operand>,
      "CuTe level graph: a staged operand must be a View-backed input node");
  using type = typename std::decay_t<
      decltype(std::declval<Operand>().handle)>::array_layout;
};

template <typename Node, int NumThreads>
struct lg_cute_stage {
  using tile_shape = cute_shape_of_t<member_out_tile_t<Node>>;
  using order      = view_contiguity_t<
      Node::Rank,
      typename lg_cute_array_layout<typename Node::operand_type>::type>;
  using thr_layout = cute_thr_layout_t<tile_shape, order, NumThreads>;
  using part       = CuteThreadPartitioner<thr_layout>;
};

template <typename ES, int NumThreads, typename LevelsT, typename GridModes,
          std::size_t RootR, std::size_t L, std::size_t M>
__device__ auto lg_cute_stage_member(
    const LevelsT& levels, const Kokkos::Array<int, RootR>& grid_idx) {
  using Node     = tuple_element_t<M, tuple_element_t<L, LevelsT>>;
  using S        = lg_cute_stage<Node, NumThreads>;
  using Gather   = gather_seq_t<typename Node::modes_seq, GridModes>;
  const auto idx = node_index<Node::Rank, RootR>(grid_idx, Gather{});
  auto       ev  = make_evaluator<CutePolicyTag<ES>>(
      levels.template get<L>().template get<M>(),
      CuteStagedTag<typename S::tile_shape, typename S::thr_layout>{
          {typename S::thr_layout{}, static_cast<int>(threadIdx.x)}});
  return ev(idx);
}

template <typename ES, int NumThreads, typename LevelsT, typename GridModes,
          std::size_t RootR, std::size_t L, std::size_t... Ms>
__device__ auto lg_cute_run_level(const LevelsT&                   levels,
                                  const Kokkos::Array<int, RootR>& grid_idx,
                                  std::index_sequence<Ms...>) {
  using R = DeviceTuple<decltype(lg_cute_stage_member<ES, NumThreads, LevelsT,
                                                      GridModes, RootR, L, Ms>(
      levels, grid_idx))...>;
  return R{
      lg_cute_stage_member<ES, NumThreads, LevelsT, GridModes, RootR, L, Ms>(
          levels, grid_idx)...};
}

template <typename Acc, typename Level, std::size_t... As, std::size_t... Bs>
__device__ auto lg_cute_concat(const Acc& acc, const Level& lv,
                               std::index_sequence<As...>,
                               std::index_sequence<Bs...>) {
  using R = DeviceTuple<std::decay_t<decltype(acc.template get<As>())>...,
                        std::decay_t<decltype(lv.template get<Bs>())>...>;
  return R{acc.template get<As>()..., lv.template get<Bs>()...};
}

template <typename ES, int NumThreads, typename LevelsT, typename GridModes,
          std::size_t RootR, std::size_t L, typename Acc>
__device__ auto lg_cute_run_levels(const LevelsT&                   levels,
                                   const Kokkos::Array<int, RootR>& grid_idx,
                                   const Acc&                       acc) {
  if constexpr (L == tuple_size_v<LevelsT>) {
    return acc;
  } else {
    constexpr std::size_t NM = tuple_size_v<tuple_element_t<L, LevelsT>>;
    static_assert(lg_member_base_v<LevelsT, L, 0> == tuple_size_v<Acc>,
                  "CuTe level graph: slots must be numbered in level order");
    const auto lv =
        lg_cute_run_level<ES, NumThreads, LevelsT, GridModes, RootR, L>(
            levels, grid_idx, std::make_index_sequence<NM>{});
    return lg_cute_run_levels<ES, NumThreads, LevelsT, GridModes, RootR, L + 1>(
        levels, grid_idx,
        lg_cute_concat(acc, lv, std::make_index_sequence<tuple_size_v<Acc>>{},
                       std::make_index_sequence<NM>{}));
  }
}

template <typename ES, int NumThreads, typename LevelsT, typename GridModes,
          std::size_t RootR, std::size_t Rt, typename Slots, typename ViewT>
__device__ void lg_cute_store_root(const Slots&                     slots,
                                   const Kokkos::Array<int, RootR>& grid_idx,
                                   const ViewT&                     view) {
  constexpr std::size_t L = lg_slot_level_v<LevelsT, Rt>;
  constexpr std::size_t M = lg_slot_member_v<LevelsT, Rt>;
  using Node              = tuple_element_t<M, tuple_element_t<L, LevelsT>>;
  using S                 = lg_cute_stage<Node, NumThreads>;
  using Gather            = gather_seq_t<typename Node::modes_seq, GridModes>;

  const auto& f   = slots.template get<Rt>().node();
  const auto  idx = node_index<Node::Rank, RootR>(grid_idx, Gather{});
  const auto  node =
      make_cute_fragment_node<ES, Node::Rank, typename S::tile_shape>(
          f.frag_, f.coords_);
  const auto out =
      make_handle_seq(view, typename lg_member_decl_modes<Node>::type{});
  make_evaluator<CutePolicyTag<ES>>(
      node, CuteFragmentStoreTag<typename S::part>{typename S::part{
                typename S::thr_layout{}, static_cast<int>(threadIdx.x)}})(
      idx, out, output_perm_seq<Node>());
}

template <typename ES, int NumThreads, typename LevelsT, typename GridModes,
          std::size_t RootR, typename Slots, typename ViewArr,
          std::size_t... Rts>
__device__ void lg_cute_store_roots(const Slots&                     slots,
                                    const Kokkos::Array<int, RootR>& grid_idx,
                                    const ViewArr&                   views,
                                    std::index_sequence<Rts...>) {
  int i = 0;
  (lg_cute_store_root<ES, NumThreads, LevelsT, GridModes, RootR, Rts>(
       slots, grid_idx, views[i++]),
   ...);
}

template <typename ES, int NumThreads, typename LevelsT, typename GridModes,
          typename GridTile, typename Scheduler, typename RootsSeq,
          std::size_t RootR, typename ViewArr>
__global__ void lg_cute_kernel(LevelsT levels, Kokkos::Array<int, RootR> shape,
                               ViewArr views) {
  const auto grid_idx = Scheduler::template tile_coord<RootR, GridTile>(shape);
  const auto slots =
      lg_cute_run_levels<ES, NumThreads, LevelsT, GridModes, RootR, 0>(
          levels, grid_idx, DeviceTuple<>{});
  lg_cute_store_roots<ES, NumThreads, LevelsT, GridModes, RootR>(
      slots, grid_idx, views, RootsSeq{});
}

template <typename LT, typename GridModes, std::size_t N>
void lg_cute_check_divisible(const Kokkos::Array<int, N>& shape) {
  constexpr auto grid = seq_to_array(GridModes{});
  for (std::size_t d = 0; d < N; ++d)
    if (shape[d] % label_tile_of<LT>(grid[d]) != 0)
      Kokkos::abort(
          "level graph (CuTe): a gridded label's extent is not a multiple of "
          "its tile; the CuTe backend requires tiles to divide the extents");
}

template <typename LT, typename Node>
void lg_cute_check_whole(const Node& n) {
  constexpr auto modes = seq_to_array(typename Node::modes_seq{});
  const auto     shape = n.shape();
  for (std::size_t d = 0; d < modes.size(); ++d)
    if (!label_gridded_of<LT>(modes[d]) &&
        shape[d] != label_tile_of<LT>(modes[d]))
      Kokkos::abort(
          "level graph (CuTe): a LabelWhole label's extent differs from its "
          "tile; the CuTe backend requires tiles to divide the extents");
}

template <typename LT, typename LevelsT, std::size_t... Ls>
void lg_cute_check_wholes(const LevelsT& levels, std::index_sequence<Ls...>) {
  (
      [&]<std::size_t... Ms>(const auto& lv, std::index_sequence<Ms...>) {
        (lg_cute_check_whole<LT>(lv.template get<Ms>()), ...);
      }(levels.template get<Ls>(),
        std::make_index_sequence<tuple_size_v<tuple_element_t<Ls, LevelsT>>>{}),
      ...);
}

template <typename V, typename ES, typename LT, typename LevelsT,
          int NumThreads, typename RootsSeq, typename... ViewTs>
int lg_execute_cute(const LevelsT& levels, RootsSeq, const ViewTs&... views) {
  static_assert(lg_cute_all_staged<LevelsT>(
                    std::make_index_sequence<tuple_size_v<LevelsT>>{}),
                "level graph (CuTe): contraction and combine levels are not "
                "supported by the CuTe backend yet");
  using GridModes      = lg_grid_modes_t<LT, LevelsT>;
  using GridTile       = lg_grid_tile_t<LT, LevelsT>;
  using Scheduler      = LinearTileScheduler;
  constexpr auto RootR = GridModes::size();

  const auto grid_shape = lg_grid_shape<LT>(
      levels, std::make_index_sequence<tuple_size_v<LevelsT>>{});
  lg_cute_check_divisible<LT, GridModes>(grid_shape);
  lg_cute_check_wholes<LT>(levels,
                           std::make_index_sequence<tuple_size_v<LevelsT>>{});
  const int         wk    = lg_league_size<GridTile>(grid_shape);
  const auto        varr  = lg_view_array(views...);
  const std::size_t bytes = 0;

  const auto kernel =
      lg_cute_kernel<ES, NumThreads, LevelsT, GridModes, GridTile, Scheduler,
                     RootsSeq, RootR, std::decay_t<decltype(varr)>>;
  if (bytes > lg_cute_default_smem_bytes &&
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(bytes)) != cudaSuccess)
    Kokkos::abort(
        "level graph (CuTe): the graph's scratch exceeds the device's "
        "shared memory per block");

  kernel<<<Scheduler::grid(wk), NumThreads, bytes, ES{}.cuda_stream()>>>(
      levels, grid_shape, varr);
  if (cudaGetLastError() != cudaSuccess)
    Kokkos::abort("level graph (CuTe): kernel launch failed");
  return wk;
}

}  // namespace Impl
}  // namespace TensorOperations
