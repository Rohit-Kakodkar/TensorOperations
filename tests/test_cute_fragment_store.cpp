#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/TensorHandle.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <type_traits>
#include <utility>

using namespace TensorOperations;

struct DoubleHook {
  KOKKOS_FUNCTION void operator()(int, int, int, float& v) const { v *= 2.0f; }
};

struct CoordHook {
  KOKKOS_FUNCTION void operator()(int i, int b, int c, float& v) const {
    v += static_cast<float>(100 * i + 10 * b + c);
  }
};

struct EpilogueFn {
  KOKKOS_FUNCTION Kokkos::Array<float, 2> operator()(int i, int, int c, float x,
                                                     float w) const {
    return {x * w + static_cast<float>(i), x - 2.0f * static_cast<float>(c)};
  }
};

struct SplitFn {
  KOKKOS_FUNCTION Kokkos::Array<float, 2> operator()(int p, int q, int r,
                                                     float x, float y) const {
    return {x - 3.0f * y + static_cast<float>(p * q - r),
            x * y + static_cast<float>(p)};
  }
};

namespace {

enum class Path { ViaSmem, Direct };

using V2   = Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::Cuda>;
using V3   = Kokkos::View<float***, Kokkos::LayoutLeft, Kokkos::Cuda>;
using V4   = Kokkos::View<float****, Kokkos::LayoutLeft, Kokkos::Cuda>;
using MmaF = cute::UniversalFMA<float, float, float>;
using S5   = cute::_5;
using Id3  = std::integer_sequence<int, 0, 1, 2>;
using Id4  = std::integer_sequence<int, 0, 1, 2, 3>;

float a_val(int n) { return 0.25f * n - 3.0f + static_cast<float>(n % 7); }
float b_val(int n) { return 0.5f - 0.125f * n + static_cast<float>(n % 5); }
float w_val(int n) { return 1.0f + 0.0625f * n - static_cast<float>(n % 3); }

template <typename Shape>
__device__ auto smem_tile(float* ptr) {
  return cute::make_tensor(cute::make_smem_ptr(ptr),
                           cute::make_layout(Shape{}, cute::LayoutRight{}));
}

template <typename Shape, typename Thr, typename StageNode, typename Coord>
__device__ auto stage(StageNode sn, float* ptr, int thr, Coord coord) {
  auto dst = smem_tile<Shape>(ptr);
  auto ev  = make_evaluator<CutePolicyTag<>>(
      sn, CuteStagedTag<decltype(dst), Thr>{{Thr{}, thr}, dst});
  return ev(coord);
}

template <typename Shape>
__device__ auto zeros() {
  return cute::repeat<cute::rank_v<Shape>>(0);
}

template <Path How, typename Shape, typename ThrOut, typename Part,
          typename FragEval, typename Coord, typename Out, typename Perm>
__device__ void store_frag(float* buf, Part part, const FragEval& frag, int thr,
                           Coord coord, const Out& out, Perm perm) {
  if constexpr (How == Path::Direct) {
    make_evaluator<CutePolicyTag<>>(
        frag.node(), CuteFragmentStoreTag<Part>{part})(coord, out, perm);
  } else {
    auto sink = make_evaluator<CutePolicyTag<>>(
        make_cute_interm_node<Kokkos::Cuda>(smem_tile<Shape>(buf)),
        CuteFragmentStoreTag<Part>{part});
    auto ev = (sink = frag);
    static_assert(std::is_same_v<std::decay_t<decltype(ev.node().hook_op)>,
                                 std::decay_t<decltype(frag.node().hook_op)>>,
                  "the smem tile must carry the fragment's hook");
    __syncthreads();
    make_evaluator<CutePolicyTag<>>(
        ev.node(), CuteStoreTag<ThrOut>{ThrOut{}, thr})(coord, out, perm);
  }
}

template <Path How, typename AShape, typename BShape, typename CShape,
          typename ThrA, typename ThrB, typename ThrOut, int FreeA, typename SA,
          typename SB, typename CN, typename Mma, typename Out, typename Perm>
__global__ void contract_kernel(SA sa, SB sb, CN cn, Mma mma, Out out,
                                Perm perm) {
  __shared__ float bufA[decltype(cute::size(AShape{}))::value];
  __shared__ float bufB[decltype(cute::size(BShape{}))::value];
  __shared__ float bufC[decltype(cute::size(CShape{}))::value];
  const int        thr = static_cast<int>(threadIdx.x);

  auto ea = stage<AShape, ThrA>(sa, bufA, thr, zeros<AShape>());
  auto eb = stage<BShape, ThrB>(sb, bufB, thr, zeros<BShape>());
  __syncthreads();

  auto res = make_evaluator<CutePolicyTag<>>(
      cn, CuteContractTag<decltype(ea), decltype(eb), Mma>{ea, eb, mma, thr})();
  store_frag<How, CShape, ThrOut>(bufC,
                                  CuteMmaPartitioner<Mma, FreeA>{mma, thr}, res,
                                  thr, zeros<CShape>(), out, perm);
}

using HSh = cute::Shape<S5, S5>;
using USh = cute::Shape<S5, S5, S5>;

template <Path How, typename SH, typename SU, typename SW, typename CN,
          typename GN, typename Mma, typename Out>
__global__ void epilogue_kernel(SH sh, SU su, SW sw, CN cn, GN gn, Mma mma,
                                Kokkos::Array<int, 3> origin, Out out0,
                                Out out1) {
  using ThrH   = cute::Layout<cute::Shape<S5, S5>>;
  using ThrU   = cute::Layout<cute::Shape<cute::_1, S5, S5>>;
  using ThrOut = cute::Layout<cute::Shape<S5, cute::_1, S5>>;
  __shared__ float bh[25];
  __shared__ float bu[125];
  __shared__ float bw[125];
  __shared__ float b0[125];
  __shared__ float b1[125];
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

  const CuteMmaPartitioner<Mma, 1> part{mma, thr};
  store_frag<How, USh, ThrOut>(b0, part, outs[0], thr, zeros<USh>(), out0,
                               Id3{});
  store_frag<How, USh, ThrOut>(b1, part, outs[1], thr, zeros<USh>(), out1,
                               Id3{});
}

constexpr int P = 4, Q = 6, R = 8, TP = 2, TQ = 3, TR = 4;

template <Path How, typename SA, typename SB, typename GN, typename Out>
__global__ void pointwise_kernel(SA sa, SB sb, GN gn, Out out0, Out out1) {
  using ATile  = cute::Shape<cute::_2, cute::_3, cute::_4>;
  using BTile  = cute::Shape<cute::_4, cute::_2, cute::_3>;
  using ThrA   = cute::Layout<cute::Shape<cute::_2, cute::_3, cute::_2>>;
  using ThrB   = cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_3>>;
  using ThrC   = cute::Layout<cute::Shape<cute::_2, cute::_3, cute::_2>>;
  using ThrOut = cute::Layout<cute::Shape<cute::_1, cute::_3, cute::_4>>;
  __shared__ float ba[TP * TQ * TR];
  __shared__ float bb[TP * TQ * TR];
  __shared__ float b0[TP * TQ * TR];
  __shared__ float b1[TP * TQ * TR];
  const int        thr = static_cast<int>(threadIdx.x);
  const int        tp = blockIdx.x, tq = blockIdx.y, tr = blockIdx.z;
  const auto       coord = cute::make_coord(tp, tq, tr);

  auto ea = stage<ATile, ThrA>(sa, ba, thr, coord);
  auto eb = stage<BTile, ThrB>(sb, bb, thr, cute::make_coord(tr, tp, tq));
  __syncthreads();

  const Kokkos::Array<int, 3> origin{TP * tp, TQ * tq, TR * tr};
  auto                        outs = make_evaluator<CutePolicyTag<>>(
      gn, CuteCombineThreadTag<ThrC, decltype(ea), decltype(eb)>{
              {ThrC{}, thr},
              DeviceTuple<decltype(ea), decltype(eb)>(ea, eb),
              origin})();

  const CuteThreadPartitioner<ThrC> part{ThrC{}, thr};
  store_frag<How, ATile, ThrOut>(b0, part, outs[0], thr, coord, out0, Id3{});
  store_frag<How, ATile, ThrOut>(b1, part, outs[1], thr, coord, out1, Id3{});
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

template <Path How>
int nested_multi_mode_contraction() {
  using AShape = cute::Shape<cute::_4, cute::_2, cute::_5, cute::_3>;
  using BShape = cute::Shape<cute::_4, cute::_5, cute::_5, cute::_5>;
  using CShape = cute::Shape<cute::_2, cute::_3, cute::_5, cute::_5>;
  using ThrA =
      cute::Layout<cute::Shape<cute::_1, cute::_2, cute::_5, cute::_1>>;
  using ThrB =
      cute::Layout<cute::Shape<cute::_2, cute::_5, cute::_1, cute::_1>>;
  using ThrOut =
      cute::Layout<cute::Shape<cute::_2, cute::_1, cute::_1, cute::_5>>;

  V4 a("a", 4, 2, 5, 3), b("b", 4, 5, 5, 5), out("out", 2, 3, 5, 5);
  fill(a, a_val);
  fill(b, b_val);
  Kokkos::deep_copy(out, -999.0f);

  auto an  = make_input_node(make_handle<'p', 'i', 'q', 'j'>(a));
  auto bn  = make_input_node(make_handle<'p', 'q', 'x', 'y'>(b));
  auto cn  = make_contraction_node<'i', 'j', 'x', 'y'>(an, bn);
  auto mma = cute::make_tiled_mma(
      MmaF{}, cute::Layout<cute::Shape<cute::_2, cute::_5, cute::_1>>{});
  contract_kernel<How, AShape, BShape, CShape, ThrA, ThrB, ThrOut, 2>
      <<<1, 10>>>(make_stage_node(an), make_stage_node(bn), cn, mma,
                  make_handle<'i', 'j', 'x', 'y'>(out), Id4{});
  if (!launched()) return -1;

  auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a);
  auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b);
  auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);

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
  return bad;
}

template <Path How>
int sem_shaped_contraction_carries_hook() {
  using ThrA   = cute::Layout<cute::Shape<S5, S5>>;
  using ThrB   = cute::Layout<cute::Shape<cute::_1, S5, S5>>;
  using ThrOut = cute::Layout<cute::Shape<S5, S5, cute::_1>>;

  V2 a("a", 5, 5);
  V3 b("b", 5, 5, 5), out("out", 5, 5, 5);
  fill(a, a_val);
  fill(b, b_val);
  Kokkos::deep_copy(out, -999.0f);

  auto an  = make_input_node(make_handle<'i', 'a'>(a));
  auto bn  = make_input_node(make_handle<'b', 'a', 'c'>(b));
  auto cn  = make_contraction_node<'i', 'b', 'c'>(an, bn, DoubleHook{});
  auto mma = cute::make_tiled_mma(
      MmaF{}, cute::Layout<cute::Shape<S5, S5, cute::_1>>{});
  contract_kernel<How, HSh, USh, USh, ThrA, ThrB, ThrOut, 1>
      <<<1, 25>>>(make_stage_node(an), make_stage_node(bn), cn, mma,
                  make_handle<'i', 'b', 'c'>(out), Id3{});
  if (!launched()) return -1;

  auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a);
  auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b);
  auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);

  int bad = 0;
  for (int i = 0; i < 5; ++i)
    for (int bb = 0; bb < 5; ++bb)
      for (int cc = 0; cc < 5; ++cc) {
        float acc = 0.0f;
        for (int aa = 0; aa < 5; ++aa) acc += ha(i, aa) * hb(bb, aa, cc);
        if (!close(ho(i, bb, cc), 2.0f * acc)) ++bad;
      }
  return bad;
}

template <Path How>
int permuted_output_with_coordinate_hook() {
  using ThrA   = cute::Layout<cute::Shape<S5, S5>>;
  using ThrB   = cute::Layout<cute::Shape<cute::_1, S5, S5>>;
  using ThrOut = cute::Layout<cute::Shape<S5, S5, cute::_1>>;

  V2 a("a", 5, 5);
  V3 b("b", 5, 5, 5), out("out", 5, 5, 5);
  fill(a, a_val);
  fill(b, b_val);
  Kokkos::deep_copy(out, -999.0f);

  auto an  = make_input_node(make_handle<'i', 'a'>(a));
  auto bn  = make_input_node(make_handle<'b', 'a', 'c'>(b));
  auto cn  = make_contraction_node<'i', 'b', 'c'>(an, bn, CoordHook{});
  auto mma = cute::make_tiled_mma(
      MmaF{}, cute::Layout<cute::Shape<S5, S5, cute::_1>>{});
  contract_kernel<How, HSh, USh, USh, ThrA, ThrB, ThrOut, 1><<<1, 25>>>(
      make_stage_node(an), make_stage_node(bn), cn, mma,
      make_handle<'c', 'i', 'b'>(out), std::integer_sequence<int, 1, 2, 0>{});
  if (!launched()) return -1;

  auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a);
  auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b);
  auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);

  int bad = 0;
  for (int i = 0; i < 5; ++i)
    for (int bb = 0; bb < 5; ++bb)
      for (int cc = 0; cc < 5; ++cc) {
        float acc = 0.0f;
        for (int aa = 0; aa < 5; ++aa) acc += ha(i, aa) * hb(bb, aa, cc);
        const float want = acc + static_cast<float>(100 * i + 10 * bb + cc);
        if (!close(ho(cc, i, bb), want)) ++bad;
      }
  return bad;
}

template <Path How>
int fragment_driven_combine_outputs() {
  V2 h("h", 5, 5);
  V3 u("u", 5, 5, 5), w("w", 5, 5, 5);
  V3 out0("out0", 5, 5, 5), out1("out1", 5, 5, 5);
  fill(h, a_val);
  fill(u, b_val);
  fill(w, w_val);

  auto hn  = make_input_node(make_handle<'i', 'a'>(h));
  auto un  = make_input_node(make_handle<'b', 'a', 'c'>(u));
  auto wn  = make_input_node(make_handle<'c', 'i', 'b'>(w));
  auto cn  = make_contraction_node<'i', 'b', 'c'>(hn, un);
  auto gn  = make_combine_node<'i', 'b', 'c'>(cn, wn, EpilogueFn{});
  auto mma = cute::make_tiled_mma(
      MmaF{}, cute::Layout<cute::Shape<S5, S5, cute::_1>>{});
  const Kokkos::Array<int, 3> origin{10, 20, 30};
  epilogue_kernel<How><<<1, 25>>>(make_stage_node(hn), make_stage_node(un),
                                  make_stage_node(wn), cn, gn, mma, origin,
                                  make_handle<'i', 'b', 'c'>(out0),
                                  make_handle<'i', 'b', 'c'>(out1));
  if (!launched()) return -1;

  auto hh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, h);
  auto hu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u);
  auto hw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, w);
  auto o0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out0);
  auto o1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out1);

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
  return bad;
}

template <Path How>
int thread_layout_combine_outputs() {
  V3 a("a", P, Q, R), b("b", R, P, Q), out0("out0", P, Q, R),
      out1("out1", P, Q, R);
  fill(a, b_val);
  fill(b, w_val);

  auto an = make_input_node(make_handle<'p', 'q', 'r'>(a));
  auto bn = make_input_node(make_handle<'r', 'p', 'q'>(b));
  auto gn = make_combine_node<'p', 'q', 'r'>(an, bn, SplitFn{});
  pointwise_kernel<How><<<dim3(P / TP, Q / TQ, R / TR), 12>>>(
      make_stage_node(an), make_stage_node(bn), gn,
      make_handle<'p', 'q', 'r'>(out0), make_handle<'p', 'q', 'r'>(out1));
  if (!launched()) return -1;

  auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a);
  auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b);
  auto o0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out0);
  auto o1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out1);

  int bad = 0;
  for (int p = 0; p < P; ++p)
    for (int q = 0; q < Q; ++q)
      for (int r = 0; r < R; ++r) {
        const float x = ha(p, q, r), y = hb(r, p, q);
        const float want0 = x - 3.0f * y + static_cast<float>(p * q - r);
        const float want1 = x * y + static_cast<float>(p);
        if (!close(o0(p, q, r), want0) || !close(o1(p, q, r), want1)) ++bad;
      }
  return bad;
}

}  // namespace

TEST(CuteFragmentStore, NestedMultiModeContractionViaSmem) {
  EXPECT_EQ(nested_multi_mode_contraction<Path::ViaSmem>(), 0);
}

TEST(CuteFragmentStore, NestedMultiModeContractionDirect) {
  EXPECT_EQ(nested_multi_mode_contraction<Path::Direct>(), 0);
}

TEST(CuteFragmentStore, SemShapedContractionCarriesHookViaSmem) {
  EXPECT_EQ(sem_shaped_contraction_carries_hook<Path::ViaSmem>(), 0);
}

TEST(CuteFragmentStore, SemShapedContractionAppliesHookDirect) {
  EXPECT_EQ(sem_shaped_contraction_carries_hook<Path::Direct>(), 0);
}

TEST(CuteFragmentStore, PermutedOutputWithCoordinateHookViaSmem) {
  EXPECT_EQ(permuted_output_with_coordinate_hook<Path::ViaSmem>(), 0);
}

TEST(CuteFragmentStore, PermutedOutputWithCoordinateHookDirect) {
  EXPECT_EQ(permuted_output_with_coordinate_hook<Path::Direct>(), 0);
}

TEST(CuteFragmentStore, FragmentDrivenCombineOutputsViaSmem) {
  EXPECT_EQ(fragment_driven_combine_outputs<Path::ViaSmem>(), 0);
}

TEST(CuteFragmentStore, FragmentDrivenCombineOutputsDirect) {
  EXPECT_EQ(fragment_driven_combine_outputs<Path::Direct>(), 0);
}

TEST(CuteFragmentStore, ThreadLayoutCombineOutputsViaSmem) {
  EXPECT_EQ(thread_layout_combine_outputs<Path::ViaSmem>(), 0);
}

TEST(CuteFragmentStore, ThreadLayoutCombineOutputsDirect) {
  EXPECT_EQ(thread_layout_combine_outputs<Path::Direct>(), 0);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
