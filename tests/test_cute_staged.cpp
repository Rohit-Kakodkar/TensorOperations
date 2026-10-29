#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/TensorHandle.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

using namespace TensorOperations;

struct AddOne {
  KOKKOS_FUNCTION void operator()(int, int, int, float& v) const { v += 1.0f; }
};

namespace {

using View      = Kokkos::View<float***, Kokkos::LayoutLeft, Kokkos::Cuda>;
using Tiler     = cute::Shape<cute::_3, cute::_4, cute::_4>;
using ThrLayout = cute::Layout<cute::Shape<cute::_3, cute::_2, cute::_4>>;
constexpr int P = 6, Q = 8, R = 12, TP = 3, TQ = 4, TR = 4;
constexpr int NQ = Q / TQ, NR = R / TR;

template <typename StageNode, typename OutHandle>
__global__ void stage_node_tiles(StageNode sn, OutHandle out) {
  const int t  = blockIdx.x;
  const int tp = t / (NQ * NR), tq = (t / NR) % NQ, tr = t % NR;
  const int thr = static_cast<int>(threadIdx.x);

  auto ev = make_evaluator<CutePolicyTag<>>(
      sn, CuteStagedTag<Tiler, CuteThreadPartitioner<ThrLayout>>{
              {ThrLayout{}, thr}});
  const auto coord = cute::make_coord(tp, tq, tr);
  auto       frag  = ev(coord);
  static_assert(
      std::is_same_v<typename decltype(frag)::node_type::node_tag, FragmentTag>,
      "a CuTe stage must produce a register fragment");
  static_assert(std::is_same_v<std::decay_t<decltype(frag.node().hook_op)>,
                               std::decay_t<decltype(sn.operand_.hook_op)>>,
                "the fragment must carry the input's hook");

  using Part = CuteThreadPartitioner<ThrLayout>;
  make_evaluator<CutePolicyTag<>>(
      frag.node(), CuteFragmentStoreTag<Part>{Part{ThrLayout{}, thr}})(
      coord, out, std::integer_sequence<int, 0, 1, 2>{});
}

template <typename Hook>
int count_staged_mismatches(int threads, Hook hook, float shift) {
  View v("v", P, Q, R), out("out", P, Q, R);
  auto hv = Kokkos::create_mirror_view(v);
  for (int p = 0; p < P; ++p)
    for (int q = 0; q < Q; ++q)
      for (int r = 0; r < R; ++r)
        hv(p, q, r) = 100.0f * p + 10.0f * q + r + 0.5f * p * r;
  Kokkos::deep_copy(v, hv);
  Kokkos::deep_copy(out, -999.0f);

  auto sn =
      make_stage_node(make_input_node(make_handle<'p', 'q', 'r'>(v), hook));
  stage_node_tiles<<<(P / TP) * NQ * NR, threads>>>(
      sn, make_handle<'p', 'q', 'r'>(out));
  if (cudaGetLastError() != cudaSuccess ||
      cudaDeviceSynchronize() != cudaSuccess)
    return -1;

  auto ho  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  int  bad = 0;
  for (int p = 0; p < P; ++p)
    for (int q = 0; q < Q; ++q)
      for (int r = 0; r < R; ++r)
        if (ho(p, q, r) != hv(p, q, r) + shift) ++bad;
  return bad;
}

using TransposedTV =
    cute::Layout<cute::Shape<cute::Shape<cute::_4, cute::_4>, cute::_3>,
                 cute::Stride<cute::Stride<cute::_12, cute::_3>, cute::_1>>;

template <typename StageNode>
__global__ void stage_tv_tiles(StageNode sn, View out) {
  const int t  = blockIdx.x;
  const int tp = t / (NQ * NR), tq = (t / NR) % NQ, tr = t % NR;
  const int thr = static_cast<int>(threadIdx.x);

  using Part = CuteTVPartitioner<TransposedTV>;
  auto ev    = make_evaluator<CutePolicyTag<>>(
      sn, CuteStagedTag<Tiler, Part>{Part{TransposedTV{}, thr}});
  auto frag = ev(cute::make_coord(tp, tq, tr));
  if (thr >= 16) return;
  for (int v = 0; v < 3; ++v) {
    const int idx                  = static_cast<int>(TransposedTV{}(thr, v));
    out(tp * TP + idx % TP, tq * TQ + (idx / TP) % TQ,
        tr * TR + idx / (TP * TQ)) = frag.node().frag_(v);
  }
}

int count_tv_mismatches(int threads) {
  View v("v", P, Q, R), out("out", P, Q, R);
  auto hv = Kokkos::create_mirror_view(v);
  for (int p = 0; p < P; ++p)
    for (int q = 0; q < Q; ++q)
      for (int r = 0; r < R; ++r)
        hv(p, q, r) = 100.0f * p + 10.0f * q + r + 0.5f * p * r;
  Kokkos::deep_copy(v, hv);
  Kokkos::deep_copy(out, -999.0f);

  auto sn = make_stage_node(make_input_node(make_handle<'p', 'q', 'r'>(v)),
                            TransposedTV{});
  stage_tv_tiles<<<(P / TP) * NQ * NR, threads>>>(sn, out);
  if (cudaGetLastError() != cudaSuccess ||
      cudaDeviceSynchronize() != cudaSuccess)
    return -1;

  auto ho  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  int  bad = 0;
  for (int p = 0; p < P; ++p)
    for (int q = 0; q < Q; ++q)
      for (int r = 0; r < R; ++r)
        if (ho(p, q, r) != hv(p, q, r)) ++bad;
  return bad;
}

}  // namespace

TEST(CuteStaged, ThreadValueLayoutPicksEachThreadsCoordinates) {
  EXPECT_EQ(count_tv_mismatches(16), 0);
}

TEST(CuteStaged, ThreadValueLayoutLeavesExtraThreadsIdle) {
  EXPECT_EQ(count_tv_mismatches(32), 0);
}

TEST(CuteStaged, StagedTileRoundTripsThroughRegisters) {
  EXPECT_EQ(count_staged_mismatches(24, NoHook{}, 0.0f), 0);
}

TEST(CuteStaged, ExtraThreadsSitOut) {
  EXPECT_EQ(count_staged_mismatches(40, NoHook{}, 0.0f), 0);
}

TEST(CuteStaged, InputHookIsCarriedToTheStore) {
  EXPECT_EQ(count_staged_mismatches(24, AddOne{}, 1.0f), 0);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
