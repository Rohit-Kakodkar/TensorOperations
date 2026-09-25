#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/TensorHandle.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cmath>

using namespace TensorOperations;

struct EpilogueFn {
  KOKKOS_FUNCTION Kokkos::Array<float, 2> operator()(int i, int, int c, float x,
                                                     float w) const {
    return {x * w + static_cast<float>(i), x - 2.0f * static_cast<float>(c)};
  }
};

struct PointwiseFn {
  KOKKOS_FUNCTION float operator()(int p, int q, int r, float x,
                                   float y) const {
    return x - 3.0f * y + static_cast<float>(p * q - r);
  }
};

namespace {

using V2    = Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::Cuda>;
using V3    = Kokkos::View<float***, Kokkos::LayoutLeft, Kokkos::Cuda>;
using Count = Kokkos::View<int, Kokkos::Cuda>;

float h_val(int n) { return 0.25f * n - 3.0f + static_cast<float>(n % 7); }
float u_val(int n) { return 0.5f - 0.125f * n + static_cast<float>(n % 5); }
float w_val(int n) { return 1.0f + 0.0625f * n - static_cast<float>(n % 3); }

template <typename Shape, typename Thr, typename StageNode, typename Coord>
__device__ auto stage(StageNode sn, float* ptr, int thr, Coord coord) {
  auto dst = cute::make_tensor(cute::make_smem_ptr(ptr),
                               cute::make_layout(Shape{}, cute::LayoutRight{}));
  auto src = make_evaluator<CutePolicyTag<>>(sn.operand_, Shape{})(coord);
  auto ld =
      make_evaluator<CutePolicyTag<>>(make_cute_interm_node<Kokkos::Cuda>(dst),
                                      CuteSmemLoadTag<Thr>{Thr{}, thr});
  return (ld = src);
}

template <typename Shape>
__device__ auto zeros() {
  return cute::repeat<cute::rank_v<Shape>>(0);
}

template <typename X>
__device__ void store_at(const V3& out, const X& x, float v) {
  out(cute::get<0>(x), cute::get<1>(x), cute::get<2>(x)) = v;
}

using S5  = cute::_5;
using HSh = cute::Shape<S5, S5>;
using USh = cute::Shape<S5, S5, S5>;

template <typename SH, typename SU, typename SW, typename CN, typename GN,
          typename Mma>
__global__ void epilogue_kernel(SH sh, SU su, SW sw, CN cn, GN gn, Mma mma,
                                Kokkos::Array<int, 3> origin, V3 out0, V3 out1,
                                int* writes) {
  using ThrH = cute::Layout<cute::Shape<S5, S5>>;
  using ThrU = cute::Layout<cute::Shape<cute::_1, S5, S5>>;
  __shared__ float bh[25];
  __shared__ float bu[125];
  __shared__ float bw[125];
  const int        thr = static_cast<int>(threadIdx.x);

  auto eh = stage<HSh, ThrH>(sh, bh, thr, zeros<HSh>());
  auto eu = stage<USh, ThrU>(su, bu, thr, zeros<USh>());
  auto ew = stage<USh, ThrU>(sw, bw, thr, zeros<USh>());
  __syncthreads();

  auto x = make_evaluator<CutePolicyTag<>>(
      cn, CuteContractTag<decltype(eh), decltype(eu), Mma>{eh, eu, mma, thr})();
  auto outs = make_evaluator<CutePolicyTag<>>(
      gn, CuteCombineTag<decltype(x), decltype(ew)>{
              DeviceTuple<decltype(x), decltype(ew)>(x, ew), origin})();

  const auto& c = outs[0].node().coords_;
  for (int v = 0; v < static_cast<int>(cute::size(c)); ++v) {
    const auto oc = cute::flatten(c(v));
    store_at(out0, oc, outs[0].node().frag_(v));
    store_at(out1, oc, outs[1].node().frag_(v));
    atomicAdd(writes, 1);
  }
}

constexpr int P = 4, Q = 6, R = 8, TP = 2, TQ = 3, TR = 4;

template <typename SA, typename SB, typename GN>
__global__ void pointwise_kernel(SA sa, SB sb, GN gn, V3 out, int* writes) {
  using ATile = cute::Shape<cute::_2, cute::_3, cute::_4>;
  using BTile = cute::Shape<cute::_4, cute::_2, cute::_3>;
  using ThrA  = cute::Layout<cute::Shape<cute::_2, cute::_3, cute::_2>>;
  using ThrB  = cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_3>>;
  using ThrC  = cute::Layout<cute::Shape<cute::_2, cute::_3, cute::_2>>;
  __shared__ float ba[TP * TQ * TR];
  __shared__ float bb[TP * TQ * TR];
  const int        thr = static_cast<int>(threadIdx.x);
  const int        tp = blockIdx.x, tq = blockIdx.y, tr = blockIdx.z;

  auto ea = stage<ATile, ThrA>(sa, ba, thr, cute::make_coord(tp, tq, tr));
  auto eb = stage<BTile, ThrB>(sb, bb, thr, cute::make_coord(tr, tp, tq));
  __syncthreads();

  const Kokkos::Array<int, 3> origin{TP * tp, TQ * tq, TR * tr};
  auto                        outs = make_evaluator<CutePolicyTag<>>(
      gn, CuteCombineThreadTag<ThrC, decltype(ea), decltype(eb)>{
              {ThrC{}, thr},
              DeviceTuple<decltype(ea), decltype(eb)>(ea, eb),
              origin})();

  const auto& c = outs[0].node().coords_;
  for (int v = 0; v < static_cast<int>(cute::size(c)); ++v) {
    const auto oc                     = cute::flatten(c(v));
    out(origin[0] + cute::get<0>(oc), origin[1] + cute::get<1>(oc),
        origin[2] + cute::get<2>(oc)) = outs[0].node().frag_(v);
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

TEST(CuteCombine, FragmentDrivenEpilogueWithPermutedSmemOperand) {
  V2    h("h", 5, 5);
  V3    u("u", 5, 5, 5), w("w", 5, 5, 5);
  V3    out0("out0", 5, 5, 5), out1("out1", 5, 5, 5);
  Count writes("writes");
  fill(h, h_val);
  fill(u, u_val);
  fill(w, w_val);

  auto hn = make_input_node(make_handle<'i', 'a'>(h));
  auto un = make_input_node(make_handle<'b', 'a', 'c'>(u));
  auto wn = make_input_node(make_handle<'c', 'i', 'b'>(w));
  auto cn = make_contraction_node<'i', 'b', 'c'>(hn, un);
  auto gn = make_combine_node<'i', 'b', 'c'>(cn, wn, EpilogueFn{});
  auto mma =
      cute::make_tiled_mma(cute::UniversalFMA<float, float, float>{},
                           cute::Layout<cute::Shape<S5, S5, cute::_1>>{});
  const Kokkos::Array<int, 3> origin{10, 20, 30};
  epilogue_kernel<<<1, 25>>>(make_stage_node(hn), make_stage_node(un),
                             make_stage_node(wn), cn, gn, mma, origin, out0,
                             out1, writes.data());
  ASSERT_TRUE(launched());

  auto hh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, h);
  auto hu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u);
  auto hw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, w);
  auto o0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out0);
  auto o1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out1);
  auto nw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, writes);
  EXPECT_EQ(nw(), 5 * 5 * 5);

  int bad = 0;
  for (int i = 0; i < 5; ++i)
    for (int b = 0; b < 5; ++b)
      for (int c = 0; c < 5; ++c) {
        float x = 0.0f;
        for (int a = 0; a < 5; ++a) x += hh(i, a) * hu(b, a, c);
        const float want0 = x * hw(c, i, b) + static_cast<float>(i + 10);
        const float want1 = x - 2.0f * static_cast<float>(c + 30);
        if (!close(o0(i, b, c), want0) || !close(o1(i, b, c), want1)) ++bad;
      }
  EXPECT_EQ(bad, 0);
}

TEST(CuteCombine, SmemOnlyTilesWithPermutedOperand) {
  V3    a("a", P, Q, R), b("b", R, P, Q), out("out", P, Q, R);
  Count writes("writes");
  fill(a, u_val);
  fill(b, w_val);

  auto an = make_input_node(make_handle<'p', 'q', 'r'>(a));
  auto bn = make_input_node(make_handle<'r', 'p', 'q'>(b));
  auto gn = make_combine_node<'p', 'q', 'r'>(an, bn, PointwiseFn{});
  pointwise_kernel<<<dim3(P / TP, Q / TQ, R / TR), 12>>>(
      make_stage_node(an), make_stage_node(bn), gn, out, writes.data());
  ASSERT_TRUE(launched());

  auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a);
  auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b);
  auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  auto nw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, writes);
  EXPECT_EQ(nw(), P * Q * R);

  int bad = 0;
  for (int p = 0; p < P; ++p)
    for (int q = 0; q < Q; ++q)
      for (int r = 0; r < R; ++r) {
        const float want =
            ha(p, q, r) - 3.0f * hb(r, p, q) + static_cast<float>(p * q - r);
        if (!close(ho(p, q, r), want)) ++bad;
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
