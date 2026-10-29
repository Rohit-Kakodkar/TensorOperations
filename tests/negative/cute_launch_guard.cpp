// lg_cute_kernel ASSUMES it runs as a (NumThreads, 1, 1) block
// (LevelGraph/Cute.hpp), and with TENSOR_OPS_CHECK_CUTE_LAUNCH on it traps on
// any other shape. This launches the real kernel with NumThreads + 32 threads
// -- the mistake the check exists for -- and reports whether it trapped. It
// runs as its own process: a trap leaves the CUDA context unusable.
//
// The normal execute() path runs first and must succeed, so a trap can only
// come from the wrong block shape.
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/NodeHandle.hpp>

#include <Kokkos_Core.hpp>

#include <cstdio>
#include <utility>

#if !TENSOR_OPS_CHECK_CUTE_LAUNCH
#error "build with TENSOR_OPS_CHECK_CUTE_LAUNCH=1"
#endif

using namespace TensorOperations;

namespace {

using ES   = Kokkos::Cuda;
using V4   = Kokkos::View<float****, Kokkos::LayoutRight, ES>;
using Map4 = LabelTiles<LabelTile<'e', 2>, LabelWhole<'a', 5>,
                        LabelWhole<'b', 5>, LabelWhole<'c', 5>>;

int run() {
  // Never destroyed: after the trap the context is gone, and a View's
  // destructor would abort in cudaFree.
  V4& u   = *new V4("u", 8, 5, 5, 5);
  V4& out = *new V4("out", 8, 5, 5, 5);
  Kokkos::deep_copy(u, 1.0f);

  auto g0      = make_level_graph<float, ES>(Map4{});
  auto [g1, s] = g0.add(
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(u))));
  const auto lo = g1.outputs(s);

  // Control: the real launch path passes the check.
  lo.execute(CutePolicyTag<>{}, out);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::printf("control launch failed\n");
    return 1;
  }

  // The same kernel lg_execute_cute launches, with a block that is too big.
  using namespace Impl;
  using LevelsT        = typename decltype(lo.graph)::levels_type;
  using RootsSeq       = typename decltype(lo)::roots_seq;
  constexpr int NT     = lg_cute_num_threads_v<LevelsT>;
  using GridModes      = lg_grid_modes_t<Map4, LevelsT>;
  using GridTile       = lg_grid_tile_t<Map4, LevelsT>;
  constexpr auto RootR = GridModes::size();
  const LevelsT& lv    = lo.graph.levels;
  const auto     shape = lg_grid_shape<Map4>(
      lv, std::make_index_sequence<tuple_size_v<LevelsT>>{});
  const auto varr   = lg_view_array(out);
  const auto bytes  = lg_cute_smem_bytes<float, ES, LevelsT>();
  const auto kernel = lg_cute_kernel<float, ES, NT, LevelsT, GridModes,
                                     GridTile, LinearTileScheduler, RootsSeq,
                                     RootR, std::decay_t<decltype(varr)>>;
  kernel<<<lg_league_size<GridTile>(shape), NT + 32, bytes>>>(lv, shape, varr);

  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess)
    std::printf("trap fired: %s\n", cudaGetErrorString(e));
  else
    std::printf("no trap: the oversized block ran\n");
  return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  const int rc = run();
  // No Kokkos::finalize: after a trap the context is gone and finalize would
  // abort on the deallocations. The process exits either way.
  std::fflush(stdout);
  std::_Exit(rc);
}
