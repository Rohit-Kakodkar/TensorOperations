#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/TensorHandle.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

using namespace TensorOperations;

namespace {

using View      = Kokkos::View<float***, Kokkos::LayoutLeft, Kokkos::Cuda>;
using Tiler     = cute::Shape<cute::_3, cute::_4, cute::_4>;
using ThrLayout = cute::Layout<cute::Shape<cute::_3, cute::_2, cute::_4>>;
constexpr int P = 6, Q = 8, R = 12, TP = 3, TQ = 4, TR = 4, NT = 24;
constexpr int NQ = Q / TQ, NR = R / TR, TS = TP * TQ * TR;

template <typename StageNode>
__global__ void stage_node_tiles(StageNode sn, View out, View out_storage) {
  __shared__ float buf[TS];
  const int        t  = blockIdx.x;
  const int        tp = t / (NQ * NR), tq = (t / NR) % NQ, tr = t % NR;
  auto             stile =
      cute::make_tensor(cute::make_smem_ptr(buf),
                        cute::make_layout(Tiler{}, cute::LayoutRight{}));

  auto ev = make_evaluator<CutePolicyTag<>>(
      sn, CuteStagedTag<decltype(stile), ThrLayout>{
              {ThrLayout{}, static_cast<int>(threadIdx.x)}, stile});
  auto staged = ev(cute::make_coord(tp, tq, tr));
  __syncthreads();

  const auto s = staged.node().storage_;
  const auto e = ev.storage();
  for (int n = threadIdx.x; n < TS; n += NT) {
    const int a = n / (TQ * TR), b = (n / TR) % TQ, c = n % TR;
    out(TP * tp + a, TQ * tq + b, TR * tr + c)         = s(a, b, c);
    out_storage(TP * tp + a, TQ * tq + b, TR * tr + c) = e(a, b, c);
  }
}

int count_staged_node_mismatches() {
  View v("v", P, Q, R), out("out", P, Q, R),
      out_storage("out_storage", P, Q, R);
  auto hv = Kokkos::create_mirror_view(v);
  for (int p = 0; p < P; ++p)
    for (int q = 0; q < Q; ++q)
      for (int r = 0; r < R; ++r)
        hv(p, q, r) = 100.0f * p + 10.0f * q + r + 0.5f * p * r;
  Kokkos::deep_copy(v, hv);

  auto sn = make_stage_node(make_input_node(make_handle<'p', 'q', 'r'>(v)));
  stage_node_tiles<<<(P / TP) * NQ * NR, NT>>>(sn, out, out_storage);
  if (cudaGetLastError() != cudaSuccess ||
      cudaDeviceSynchronize() != cudaSuccess)
    return -1;

  auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  auto hs =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out_storage);
  int bad = 0;
  for (int p = 0; p < P; ++p)
    for (int q = 0; q < Q; ++q)
      for (int r = 0; r < R; ++r)
        if (ho(p, q, r) != hv(p, q, r) || hs(p, q, r) != hv(p, q, r)) ++bad;
  return bad;
}

}  // namespace

TEST(CuteStaged, StagedNodeTileMatchesView) {
  EXPECT_EQ(count_staged_node_mismatches(), 0);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
