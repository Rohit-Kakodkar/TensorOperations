#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/TensorHandle.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

using namespace TensorOperations;

namespace {

using View3     = Kokkos::View<float***, Kokkos::LayoutLeft, Kokkos::Cuda>;
using Tiler     = cute::Shape<cute::_3, cute::_4, cute::_4>;
using ThrLayout = cute::Layout<cute::Shape<cute::_3, cute::_2, cute::_4>>;
constexpr int P = 6, Q = 8, R = 12, TP = 3, TQ = 4, TR = 4;
constexpr int NP = P / TP, NQ = Q / TQ, NR = R / TR;

struct ReadView {
  View3                 v;
  KOKKOS_FUNCTION float operator()(int p, int q, int r) const {
    return v(p, q, r);
  }
};

struct Probe {
  KOKKOS_FUNCTION float operator()(int p, int q, int r) const {
    return 10000.0f * p + 100.0f * q + r;
  }
};

float probe_ref(int p, int q, int r) { return 10000.0f * p + 100.0f * q + r; }

__device__ cute::tuple<int, int, int> tile_of_block() {
  const int t = blockIdx.x;
  return {t / (NQ * NR), (t / NR) % NQ, t % NR};
}

template <typename Node, typename Out>
__global__ void copy_tiles(Node node, Out out) {
  const auto [tp, tq, tr] = tile_of_block();
  const int  a = threadIdx.x / (TQ * TR), b = (threadIdx.x / TR) % TQ,
             c    = threadIdx.x % TR;
  const auto tile = make_evaluator<CutePolicyTag<>>(
                        node, Tiler{})(cute::make_coord(tp, tq, tr))
                        .node()
                        .storage_;
  out(TP * tp + a, TQ * tq + b, TR * tr + c) = tile(a, b, c);
}

template <typename Node>
View3 tiles_of(Node node) {
  View3 out("out", P, Q, R);
  Kokkos::deep_copy(out, -999.0f);
  copy_tiles<<<NP * NQ * NR, TP * TQ * TR>>>(node, out);
  if (cudaGetLastError() != cudaSuccess ||
      cudaDeviceSynchronize() != cudaSuccess)
    ADD_FAILURE() << "kernel launch failed";
  return out;
}

template <typename StageNode, typename OutHandle>
__global__ void stage_node_tiles(StageNode sn, OutHandle out) {
  const auto [tp, tq, tr] = tile_of_block();
  const int thr           = static_cast<int>(threadIdx.x);

  auto ev = make_evaluator<CutePolicyTag<>>(
      sn, CuteStagedTag<Tiler, ThrLayout>{{ThrLayout{}, thr}});
  const auto coord = cute::make_coord(tp, tq, tr);
  auto       frag  = ev(coord);
  static_assert(
      std::is_same_v<typename decltype(frag)::node_type::node_tag, FragmentTag>,
      "a CuTe stage of a functional input must produce a register fragment");

  using Part = CuteThreadPartitioner<ThrLayout>;
  make_evaluator<CutePolicyTag<>>(
      frag.node(), CuteFragmentStoreTag<Part>{Part{ThrLayout{}, thr}})(
      coord, out, std::integer_sequence<int, 0, 1, 2>{});
}

int count_staged_mismatches(int threads) {
  View3 out("out", P, Q, R);
  Kokkos::deep_copy(out, -999.0f);
  auto sn =
      make_stage_node(make_functional_input_node<Kokkos::Cuda, 'p', 'q', 'r'>(
          Kokkos::Array<int, 3>{P, Q, R}, Probe{}));
  stage_node_tiles<<<NP * NQ * NR, threads>>>(sn,
                                              make_handle<'p', 'q', 'r'>(out));
  if (cudaGetLastError() != cudaSuccess ||
      cudaDeviceSynchronize() != cudaSuccess)
    return -1;

  auto ho  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  int  bad = 0;
  for (int p = 0; p < P; ++p)
    for (int q = 0; q < Q; ++q)
      for (int r = 0; r < R; ++r)
        if (ho(p, q, r) != probe_ref(p, q, r)) ++bad;
  return bad;
}

}  // namespace

TEST(CuteFunctionalInput, ReReadingAViewIsBitwiseTheInputTile) {
  View3 v("v", P, Q, R);
  auto  hv = Kokkos::create_mirror_view(v);
  for (int p = 0; p < P; ++p)
    for (int q = 0; q < Q; ++q)
      for (int r = 0; r < R; ++r)
        hv(p, q, r) = 0.1f * p + 1.7f * q - 0.3f * r + 0.01f * p * r;
  Kokkos::deep_copy(v, hv);

  const auto via_input = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{},
      tiles_of(make_input_node(make_handle<'p', 'q', 'r'>(v))));
  const auto via_fn = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{},
      tiles_of(make_functional_input_node<Kokkos::Cuda, 'p', 'q', 'r'>(
          Kokkos::Array<int, 3>{P, Q, R}, ReadView{v})));

  int bad = 0;
  for (int p = 0; p < P; ++p)
    for (int q = 0; q < Q; ++q)
      for (int r = 0; r < R; ++r)
        if (via_fn(p, q, r) != via_input(p, q, r) ||
            via_fn(p, q, r) != hv(p, q, r))
          ++bad;
  EXPECT_EQ(bad, 0);
}

TEST(CuteFunctionalInput, FunctorSeesTheGlobalCoordinate) {
  const auto out = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{},
      tiles_of(make_functional_input_node<Kokkos::Cuda, 'p', 'q', 'r'>(
          Kokkos::Array<int, 3>{P, Q, R}, Probe{})));
  int bad = 0;
  for (int p = 0; p < P; ++p)
    for (int q = 0; q < Q; ++q)
      for (int r = 0; r < R; ++r)
        if (out(p, q, r) != probe_ref(p, q, r)) ++bad;
  EXPECT_EQ(bad, 0);
}

TEST(CuteFunctionalInput, StagedTileRoundTripsThroughRegisters) {
  EXPECT_EQ(count_staged_mismatches(24), 0);
}

TEST(CuteFunctionalInput, ExtraThreadsSitOut) {
  EXPECT_EQ(count_staged_mismatches(40), 0);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
