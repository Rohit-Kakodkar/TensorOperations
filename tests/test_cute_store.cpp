#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/TensorHandle.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <utility>

using namespace TensorOperations;

namespace {

using InView    = Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::Cuda>;
using OutView   = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::Cuda>;
using Tiler     = cute::Shape<cute::_4, cute::_8>;
using ThrLayout = cute::Layout<cute::Shape<cute::_2, cute::_8>>;
constexpr int I = 8, J = 16, TI = 4, TJ = 8, NT = 16;

struct ShiftHook {
  KOKKOS_FUNCTION void operator()(int i, int j, float& v) const {
    v += 0.5f * i - j;
  }
};

InView make_input() {
  InView v("in", I, J);
  auto   h = Kokkos::create_mirror_view(v);
  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j) h(i, j) = 100.0f * i + j;
  Kokkos::deep_copy(v, h);
  return v;
}

template <typename Node, typename Hook, typename OutHandle, typename Perm>
__global__ void round_trip_tiles(Node node, OutHandle out, Hook hook,
                                 Perm perm) {
  __shared__ float buf[TI * TJ];
  const int        ti = blockIdx.x, tj = blockIdx.y;
  const int        thr = static_cast<int>(threadIdx.x);
  auto             stile =
      cute::make_tensor(cute::make_smem_ptr(buf),
                        cute::make_layout(Tiler{}, cute::LayoutRight{}));

  const auto coord  = cute::make_coord(ti, tj);
  auto       src    = make_evaluator<CutePolicyTag<>>(node, Tiler{})(coord);
  auto       stager = make_evaluator<CutePolicyTag<>>(
      make_cute_interm_node<Kokkos::Cuda>(stile),
      CuteStageTag<ThrLayout>{ThrLayout{}, thr});
  stager = src;
  __syncthreads();

  auto store = make_evaluator<CutePolicyTag<>>(
      make_cute_interm_node<Kokkos::Cuda>(stile, hook),
      CuteStoreTag<ThrLayout>{ThrLayout{}, thr});
  store(coord, out, perm);
}

template <typename Hook, typename OutHandle, int... Perm>
bool round_trip(const InView& in, const OutHandle& out, Hook hook,
                std::integer_sequence<int, Perm...> perm) {
  auto node = make_input_node(make_handle<'i', 'j'>(in));
  round_trip_tiles<<<dim3(I / TI, J / TJ), NT>>>(node, out, hook, perm);
  return cudaGetLastError() == cudaSuccess &&
         cudaDeviceSynchronize() == cudaSuccess;
}

int count_hooked_mismatches() {
  const auto in = make_input();
  OutView    out("out", I, J);
  if (!round_trip(in, make_handle<'i', 'j'>(out), ShiftHook{},
                  std::integer_sequence<int, 0, 1>{}))
    return -1;

  auto h_in  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, in);
  auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  int  bad   = 0;
  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j)
      if (h_out(i, j) != h_in(i, j) + 0.5f * i - j) ++bad;
  return bad;
}

int count_permuted_mismatches() {
  const auto in = make_input();
  OutView    out("out", J, I);
  if (!round_trip(in, make_handle<'j', 'i'>(out), NoHook{},
                  std::integer_sequence<int, 1, 0>{}))
    return -1;

  auto h_in  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, in);
  auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  int  bad   = 0;
  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j)
      if (h_out(j, i) != h_in(i, j)) ++bad;
  return bad;
}

}  // namespace

TEST(CuteStore, HookedRoundTripMatchesInput) {
  EXPECT_EQ(count_hooked_mismatches(), 0);
}

TEST(CuteStore, PermutedOutputIsTransposed) {
  EXPECT_EQ(count_permuted_mismatches(), 0);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
