#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/TensorHandle.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

using namespace TensorOperations;

namespace {

using View      = Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::Cuda>;
using Tiler     = cute::Shape<cute::_4, cute::_8>;
using ThrLayout = cute::Layout<cute::Shape<cute::_2, cute::_8>>;
constexpr int I = 8, J = 16, TI = 4, TJ = 8, NT = 16;

template <typename Node>
__global__ void stage_tiles(Node node, View out) {
  __shared__ float buf[TI * TJ];
  const int        ti = blockIdx.x, tj = blockIdx.y;
  auto             stile =
      cute::make_tensor(cute::make_smem_ptr(buf),
                        cute::make_layout(Tiler{}, cute::LayoutRight{}));

  auto src =
      make_evaluator<CutePolicyTag<>>(node, Tiler{})(cute::make_coord(ti, tj));
  auto stager = make_evaluator<CutePolicyTag<>>(
      make_cute_interm_node<Kokkos::Cuda>(stile),
      CuteStageTag<ThrLayout>{ThrLayout{}, static_cast<int>(threadIdx.x)});
  auto staged = (stager = src);
  __syncthreads();

  const auto s = staged.node().storage_;
  for (int e = threadIdx.x; e < TI * TJ; e += NT)
    out(TI * ti + e / TJ, TJ * tj + e % TJ) = s(e / TJ, e % TJ);
}

int count_staged_mismatches() {
  View v("v", I, J), out("out", I, J);
  auto hv = Kokkos::create_mirror_view(v);
  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j) hv(i, j) = 100.0f * i + j;
  Kokkos::deep_copy(v, hv);

  auto node = make_input_node(make_handle<'i', 'j'>(v));
  stage_tiles<<<dim3(I / TI, J / TJ), NT>>>(node, out);
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

TEST(CuteStage, StagedTileMatchesView) {
  EXPECT_EQ(count_staged_mismatches(), 0);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
