#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/TensorHandle.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cmath>

using namespace TensorOperations;

struct DoubleHook {
  KOKKOS_FUNCTION void operator()(int, int, int, float& v) const { v *= 2.0f; }
};

namespace {

using V2    = Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::Cuda>;
using V3    = Kokkos::View<float***, Kokkos::LayoutLeft, Kokkos::Cuda>;
using V4    = Kokkos::View<float****, Kokkos::LayoutLeft, Kokkos::Cuda>;
using Count = Kokkos::View<int, Kokkos::Cuda>;
using MmaF  = cute::UniversalFMA<float, float, float>;

float a_val(int n) { return 0.25f * n - 3.0f + static_cast<float>(n % 7); }
float b_val(int n) { return 0.5f - 0.125f * n + static_cast<float>(n % 5); }

template <typename Shape, typename Thr, typename StageNode>
__device__ auto stage(StageNode sn, float* ptr, int thr) {
  auto dst = cute::make_tensor(cute::make_smem_ptr(ptr),
                               cute::make_layout(Shape{}, cute::LayoutRight{}));
  auto src = make_evaluator<CutePolicyTag<>>(
      sn.operand_, Shape{})(cute::repeat<cute::rank_v<Shape>>(0));
  auto ld =
      make_evaluator<CutePolicyTag<>>(make_cute_interm_node<Kokkos::Cuda>(dst),
                                      CuteSmemLoadTag<Thr>{Thr{}, thr});
  return (ld = src);
}

template <typename X>
__device__ void store_at(const V4& out, const X& x, float v) {
  out(cute::get<0, 0>(x), cute::get<0, 1>(x), cute::get<1, 0>(x),
      cute::get<1, 1>(x)) = v;
}

template <typename X>
__device__ void store_at(const V3& out, const X& x, float v) {
  out(cute::get<0, 0>(x), cute::get<1, 0>(x), cute::get<1, 1>(x)) = v;
}

template <typename AShape, typename BShape, typename ThrA, typename ThrB,
          typename SA, typename SB, typename CN, typename Mma, typename Out>
__global__ void contract_kernel(SA sa, SB sb, CN cn, Mma mma, Out out,
                                int* writes) {
  __shared__ float bufA[decltype(cute::size(AShape{}))::value];
  __shared__ float bufB[decltype(cute::size(BShape{}))::value];
  const int        thr = static_cast<int>(threadIdx.x);

  auto ea = stage<AShape, ThrA>(sa, bufA, thr);
  auto eb = stage<BShape, ThrB>(sb, bufB, thr);
  __syncthreads();

  auto res = make_evaluator<CutePolicyTag<>>(
      cn, CuteContractTag<decltype(ea), decltype(eb), Mma>{ea, eb, mma, thr})();
  static_assert(std::is_same_v<typename decltype(res)::node_type::hook_type,
                               typename CN::hook_type>,
                "the fragment must carry the contraction's hook");

  const auto& f = res.node().frag_;
  const auto& c = res.node().coords_;
  for (int v = 0; v < static_cast<int>(cute::size(f)); ++v) {
    store_at(out, c(v), f(v));
    atomicAdd(writes, 1);
  }
}

template <typename V>
void fill(V v, float (*val)(int)) {
  auto h = Kokkos::create_mirror_view(v);
  for (std::size_t n = 0; n < h.size(); ++n)
    h.data()[n] = val(static_cast<int>(n));
  Kokkos::deep_copy(v, h);
}

bool close(float got, float want) {
  return std::abs(got - want) <= 1e-4f * (1.0f + std::abs(want));
}

bool launched() {
  return cudaGetLastError() == cudaSuccess &&
         cudaDeviceSynchronize() == cudaSuccess;
}

}  // namespace

TEST(CuteContract, PermutedMultiModeOperands) {
  using AShape = cute::Shape<cute::_4, cute::_2, cute::_5, cute::_3>;
  using BShape = cute::Shape<cute::_4, cute::_5, cute::_5, cute::_5>;
  using ThrA =
      cute::Layout<cute::Shape<cute::_1, cute::_2, cute::_5, cute::_1>>;
  using ThrB =
      cute::Layout<cute::Shape<cute::_2, cute::_5, cute::_1, cute::_1>>;

  V4    a("a", 4, 2, 5, 3), b("b", 4, 5, 5, 5), out("out", 2, 3, 5, 5);
  Count writes("writes");
  fill(a, a_val);
  fill(b, b_val);
  Kokkos::deep_copy(out, -999.0f);

  auto an  = make_input_node(make_handle<'p', 'i', 'q', 'j'>(a));
  auto bn  = make_input_node(make_handle<'p', 'q', 'x', 'y'>(b));
  auto cn  = make_contraction_node<'i', 'j', 'x', 'y'>(an, bn);
  auto mma = cute::make_tiled_mma(
      MmaF{}, cute::Layout<cute::Shape<cute::_2, cute::_5, cute::_1>>{});
  contract_kernel<AShape, BShape, ThrA, ThrB><<<1, 10>>>(
      make_stage_node(an), make_stage_node(bn), cn, mma, out, writes.data());
  ASSERT_TRUE(launched());

  auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a);
  auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b);
  auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  auto hw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, writes);
  EXPECT_EQ(hw(), 2 * 3 * 5 * 5);

  int bad = 0;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      for (int x = 0; x < 5; ++x)
        for (int y = 0; y < 5; ++y) {
          float acc = 0.0f;
          for (int p = 0; p < 4; ++p)
            for (int q = 0; q < 5; ++q) acc += ha(p, i, q, j) * hb(p, q, x, y);
          if (!close(ho(i, j, x, y), acc)) ++bad;
        }
  EXPECT_EQ(bad, 0);
}

TEST(CuteContract, SemShapedWithPermutedBAndRawHook) {
  using AShape = cute::Shape<cute::_5, cute::_5>;
  using BShape = cute::Shape<cute::_5, cute::_5, cute::_5>;
  using ThrA   = cute::Layout<cute::Shape<cute::_5, cute::_5>>;
  using ThrB   = cute::Layout<cute::Shape<cute::_1, cute::_5, cute::_5>>;

  V2    a("a", 5, 5);
  V3    b("b", 5, 5, 5), out("out", 5, 5, 5);
  Count writes("writes");
  fill(a, a_val);
  fill(b, b_val);
  Kokkos::deep_copy(out, -999.0f);

  auto an  = make_input_node(make_handle<'i', 'a'>(a));
  auto bn  = make_input_node(make_handle<'b', 'a', 'c'>(b));
  auto cn  = make_contraction_node<'i', 'b', 'c'>(an, bn, DoubleHook{});
  auto mma = cute::make_tiled_mma(
      MmaF{}, cute::Layout<cute::Shape<cute::_5, cute::_5, cute::_1>>{});
  contract_kernel<AShape, BShape, ThrA, ThrB><<<1, 25>>>(
      make_stage_node(an), make_stage_node(bn), cn, mma, out, writes.data());
  ASSERT_TRUE(launched());

  auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a);
  auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b);
  auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  auto hw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, writes);
  EXPECT_EQ(hw(), 5 * 5 * 5);

  int bad = 0;
  for (int i = 0; i < 5; ++i)
    for (int bb = 0; bb < 5; ++bb)
      for (int cc = 0; cc < 5; ++cc) {
        float acc = 0.0f;
        for (int aa = 0; aa < 5; ++aa) acc += ha(i, aa) * hb(bb, aa, cc);
        if (!close(ho(i, bb, cc), acc)) ++bad;
      }
  EXPECT_EQ(bad, 0);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
