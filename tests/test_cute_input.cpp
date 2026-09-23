#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/TensorHandle.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

using namespace TensorOperations;

namespace {

using Tiler     = cute::Shape<cute::_4, cute::_4>;
constexpr int I = 8, J = 12, TI = 4, TJ = 4;

template <typename Node, typename View>
__global__ void copy_tiles(Node node, View out) {
  const int  ti = blockIdx.x, tj = blockIdx.y;
  const int  a = threadIdx.x / TJ, b = threadIdx.x % TJ;
  const auto tile =
      make_evaluator<CutePolicyTag<>>(node, Tiler{})(cute::make_coord(ti, tj))
          .node()
          .storage_;
  out(TI * ti + a, TJ * tj + b) = tile(a, b);
}

template <typename Layout>
int count_tile_mismatches() {
  using View = Kokkos::View<float**, Layout, Kokkos::Cuda>;

  View v("v", I, J), out("out", I, J);
  auto hv = Kokkos::create_mirror_view(v);
  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j) hv(i, j) = 100.0f * i + j;
  Kokkos::deep_copy(v, hv);

  auto node = make_input_node(make_handle<'i', 'j'>(v));
  copy_tiles<<<dim3(I / TI, J / TJ), TI * TJ>>>(node, out);
  if (cudaGetLastError() != cudaSuccess ||
      cudaDeviceSynchronize() != cudaSuccess)
    return -1;

  auto ho  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  int  bad = 0;
  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j)
      if (ho(i, j) != hv(i, j)) ++bad;
  return bad;
}

}  // namespace

TEST(CuteInput, TilesMatchView) {
  EXPECT_EQ(count_tile_mismatches<Kokkos::LayoutRight>(), 0);
  EXPECT_EQ(count_tile_mismatches<Kokkos::LayoutLeft>(), 0);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
