#pragma once
#include <TensorOperations/LevelGraph/Team.hpp>

#include <cstddef>
#include <utility>

#include <Kokkos_Core.hpp>

namespace TensorOperations {
namespace Impl {

inline constexpr int         lg_cute_default_threads    = 128;
inline constexpr std::size_t lg_cute_default_smem_bytes = 48 * 1024;

template <typename LevelsT, std::size_t RootR, typename ViewArr>
__global__ void lg_cute_kernel(LevelsT, Kokkos::Array<int, RootR>, ViewArr) {}

template <typename V, typename ES, typename LT, typename LevelsT,
          typename RootsSeq, typename... ViewTs>
int lg_execute_cute(const LevelsT& levels, std::size_t bytes, int team_size,
                    RootsSeq, const ViewTs&... views) {
  using GridModes      = lg_grid_modes_t<LT, LevelsT>;
  using GridTile       = lg_grid_tile_t<LT, LevelsT>;
  constexpr auto RootR = GridModes::size();

  const auto grid_shape = lg_grid_shape<LT>(
      levels, std::make_index_sequence<tuple_size_v<LevelsT>>{});
  const int  wk      = lg_league_size<GridTile>(grid_shape);
  const int  threads = team_size > 0 ? team_size : lg_cute_default_threads;
  const auto varr    = lg_view_array(views...);

  const auto kernel = lg_cute_kernel<LevelsT, RootR, decltype(varr)>;
  if (bytes > lg_cute_default_smem_bytes &&
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(bytes)) != cudaSuccess)
    Kokkos::abort(
        "level graph (CuTe): the graph's scratch exceeds the device's "
        "shared memory per block");

  kernel<<<wk, threads, bytes, ES{}.cuda_stream()>>>(levels, grid_shape, varr);
  if (cudaGetLastError() != cudaSuccess)
    Kokkos::abort("level graph (CuTe): kernel launch failed");
  return wk;
}

}  // namespace Impl
}  // namespace TensorOperations
