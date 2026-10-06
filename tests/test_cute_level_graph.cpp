#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/NodeHandle.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>

using namespace TensorOperations;

namespace {

using ES = Kokkos::Cuda;

static_assert(std::is_same_v<
              Impl::cute_thr_layout_t<cute::Shape<cute::_5, cute::_5, cute::_5>,
                                      std::integer_sequence<int, 0, 1, 2>, 128>,
              cute::Layout<cute::Shape<cute::_5, cute::_5, cute::_5>,
                           cute::Stride<cute::_1, cute::_5, cute::Int<25>>>>);
static_assert(std::is_same_v<
              Impl::cute_thr_layout_t<
                  cute::Shape<cute::_2, cute::_5, cute::_5, cute::_5>,
                  std::integer_sequence<int, 3, 2, 1, 0>, 128>,
              cute::Layout<cute::Shape<cute::_1, cute::_5, cute::_5, cute::_5>,
                           cute::Stride<cute::Int<125>, cute::Int<25>, cute::_5,
                                        cute::_1>>>);
static_assert(std::is_same_v<
              Impl::cute_thr_layout_t<cute::Shape<cute::_3, cute::_16>,
                                      std::integer_sequence<int, 0, 1>, 32>,
              cute::Layout<cute::Shape<cute::_3, cute::_8>,
                           cute::Stride<cute::_1, cute::_3>>>);

constexpr int N = 5;

template <typename Layout>
using View4 = Kokkos::View<float****, Layout, ES>;
using ViewR = View4<Kokkos::LayoutRight>;
using View2 = Kokkos::View<float**, Kokkos::LayoutRight, ES>;

template <int TE>
using Map4 = LabelTiles<LabelTile<'e', TE>, LabelWhole<'a', N>,
                        LabelWhole<'b', N>, LabelWhole<'c', N>>;

template <typename V>
void fill(const V& v, float seed) {
  auto h = Kokkos::create_mirror_view(v);
  for (std::size_t n = 0; n < h.size(); ++n)
    h.data()[n] =
        seed + 0.25f * static_cast<float>(n) - static_cast<float>((n * 7) % 11);
  Kokkos::deep_copy(v, h);
}

template <typename A, typename B>
int mismatches(const A& got, const B& want) {
  auto hg  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, got);
  auto hw  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, want);
  int  bad = 0;
  for (int e = 0; e < static_cast<int>(hw.extent(0)); ++e)
    for (int a = 0; a < static_cast<int>(hw.extent(1)); ++a)
      for (int b = 0; b < static_cast<int>(hw.extent(2)); ++b)
        for (int c = 0; c < static_cast<int>(hw.extent(3)); ++c)
          if (hg(e, a, b, c) != hw(e, a, b, c)) ++bad;
  return bad;
}

template <typename A, typename B>
int mismatches2(const A& got, const B& want) {
  auto hg  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, got);
  auto hw  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, want);
  int  bad = 0;
  for (int i = 0; i < static_cast<int>(hw.extent(0)); ++i)
    for (int j = 0; j < static_cast<int>(hw.extent(1)); ++j)
      if (hg(i, j) != hw(i, j)) ++bad;
  return bad;
}

template <typename A, typename B>
float max_rel_err(const A& got, const B& want) {
  auto  hg = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, got);
  auto  hw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, want);
  float m  = 0.0f;
  for (std::size_t n = 0; n < hw.size(); ++n)
    m = std::max(m, std::abs(hg.data()[n] - hw.data()[n]) /
                        (1.0f + std::abs(hw.data()[n])));
  return m;
}

template <typename HV, typename UV>
auto gradient_ref(const HV& h, const UV& u, int axis) {
  auto      hh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, h);
  auto      hu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u);
  const int E  = static_cast<int>(hu.extent(0));
  const int Q  = static_cast<int>(hh.extent(0));
  const int K  = static_cast<int>(hh.extent(1));
  const int M  = static_cast<int>(hu.extent(2));
  Kokkos::View<float****, Kokkos::LayoutRight, Kokkos::HostSpace> r("ref", Q, E,
                                                                    M, M);
  for (int q = 0; q < Q; ++q)
    for (int e = 0; e < E; ++e)
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < M; ++n) {
          float acc = 0.0f;
          for (int k = 0; k < K; ++k)
            acc += hh(q, k) * (axis == 0   ? hu(e, k, m, n)
                               : axis == 1 ? hu(e, m, k, n)
                                           : hu(e, m, n, k));
          r(q, e, m, n) = acc;
        }
  return r;
}

bool synced() {
  return cudaGetLastError() == cudaSuccess &&
         cudaDeviceSynchronize() == cudaSuccess;
}

template <int TE, typename Policy, typename InView>
void single_stage_root(int E, Policy policy, int expected_league) {
  InView u("u", E, N, N, N);
  ViewR  cute_out("cute_out", E, N, N, N), team_out("team_out", E, N, N, N);
  fill(u, 1.0f);
  Kokkos::deep_copy(cute_out, -999.0f);

  auto g0      = make_level_graph<float, ES>(Map4<TE>{});
  auto [g1, s] = g0.add(
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(u))));
  const auto out = g1.outputs(s);

  EXPECT_EQ(out.execute(policy, cute_out), expected_league);
  ASSERT_TRUE(synced());
  out.execute(TeamPolicyTag<ES>{}, team_out);
  ASSERT_TRUE(synced());

  EXPECT_EQ(mismatches(cute_out, u), 0);
  EXPECT_EQ(mismatches(cute_out, team_out), 0);
}

}  // namespace

TEST(CuteLevelGraph, SingleStageRootCopiesTheInput) {
  single_stage_root<2, CutePolicyTag<>, ViewR>(8, CutePolicyTag<>{}, 4);
}

TEST(CuteLevelGraph, LayoutLeftInputToLayoutRightOutput) {
  single_stage_root<2, CutePolicyTag<>, View4<Kokkos::LayoutLeft>>(
      8, CutePolicyTag<>{}, 4);
}

TEST(CuteLevelGraph, IdleThreadsWithDefaultBlock) {
  single_stage_root<1, CutePolicyTag<>, ViewR>(6, CutePolicyTag<>{}, 6);
}

TEST(CuteLevelGraph, SmallerBlockStillCoversTheTile) {
  single_stage_root<1, CutePolicyTag<ES, 32>, ViewR>(6, CutePolicyTag<ES, 32>{},
                                                     6);
}

TEST(CuteLevelGraph, TwoMemberStageLevelBothRoots) {
  constexpr int E = 8;
  ViewR         u("u", E, N, N, N), w("w", E, N, N, N);
  ViewR         cu("cu", E, N, N, N), cw("cw", E, N, N, N);
  ViewR         tu("tu", E, N, N, N), tw("tw", E, N, N, N);
  fill(u, 1.0f);
  fill(w, -3.0f);

  auto g0           = make_level_graph<float, ES>(Map4<2>{});
  auto [g1, su, sw] = g0.add(
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(u))),
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(w))));
  const auto out = g1.outputs(su, sw);

  EXPECT_EQ(out.execute(CutePolicyTag<>{}, cu, cw), E / 2);
  ASSERT_TRUE(synced());
  out.execute(TeamPolicyTag<ES>{}, tu, tw);
  ASSERT_TRUE(synced());

  EXPECT_EQ(mismatches(cu, u), 0);
  EXPECT_EQ(mismatches(cw, w), 0);
  EXPECT_EQ(mismatches(cu, tu), 0);
  EXPECT_EQ(mismatches(cw, tw), 0);
}

TEST(CuteLevelGraph, TwoStageLevelsWithDifferentTiles) {
  constexpr int E = 12, B = 5, C = 3;
  using Map =
      LabelTiles<LabelTile<'e', 4>, LabelWhole<'b', B>, LabelWhole<'c', C>>;
  View2 a("a", E, B), b("b", E, C);
  View2 ca("ca", E, B), cb("cb", E, C), ta("ta", E, B), tb("tb", E, C);
  fill(a, 2.0f);
  fill(b, -1.0f);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, sa] =
      g0.add(make_stage_node(make_input_node(make_handle<'e', 'b'>(a))));
  auto [g2, sb] =
      g1.add(make_stage_node(make_input_node(make_handle<'e', 'c'>(b))));
  const auto out = g2.outputs(sa, sb);

  EXPECT_EQ(out.execute(CutePolicyTag<ES, 16>{}, ca, cb), E / 4);
  ASSERT_TRUE(synced());
  out.execute(TeamPolicyTag<ES>{}, ta, tb);
  ASSERT_TRUE(synced());

  EXPECT_EQ(mismatches2(ca, a), 0);
  EXPECT_EQ(mismatches2(cb, b), 0);
  EXPECT_EQ(mismatches2(ca, ta), 0);
  EXPECT_EQ(mismatches2(cb, tb), 0);
}

namespace {

template <int NG, int TE>
using MapQ =
    LabelTiles<LabelTile<'e', TE>, LabelWhole<'q', NG>, LabelWhole<'p', NG>,
               LabelWhole<'a', NG>, LabelWhole<'b', NG>, LabelWhole<'c', NG>>;

template <int NG>
using OpView = Kokkos::View<float**, Kokkos::LayoutRight, ES>;

template <int NG, int TE, typename Policy>
void one_member_gradient(int E, Policy policy) {
  OpView<NG> h("h", NG, NG);
  ViewR      u("u", E, NG, NG, NG);
  ViewR      cute_out("cute_out", NG, E, NG, NG),
      team_out("team_out", NG, E, NG, NG);
  fill(h, 0.5f);
  fill(u, -2.0f);
  Kokkos::deep_copy(cute_out, -999.0f);

  auto g0 = make_level_graph<float, ES>(MapQ<NG, TE>{});
  auto [g1, sh] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'a'>(h))));
  auto [g2, su] = g1.add(
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(u))));
  auto [g3, c]   = g2.add(make_contraction_node<'q', 'e', 'b', 'c'>(sh, su));
  const auto out = g3.outputs(c);

  EXPECT_EQ(out.execute(policy, cute_out), E / TE);
  ASSERT_TRUE(synced());
  out.execute(TeamPolicyTag<ES>{}, team_out);
  ASSERT_TRUE(synced());

  EXPECT_LT(max_rel_err(cute_out, gradient_ref(h, u, 0)), 1e-5f);
  EXPECT_LT(max_rel_err(cute_out, team_out), 1e-5f);
}

}  // namespace

TEST(CuteLevelGraph, ContractionDefaultMma) {
  one_member_gradient<8, 2>(8, CutePolicyTag<>{});
}

TEST(CuteLevelGraph, ContractionDefaultMmaOddExtents) {
  one_member_gradient<5, 2>(8, CutePolicyTag<>{});
}

TEST(CuteLevelGraph, ContractionSmallBlock) {
  one_member_gradient<5, 1>(6, CutePolicyTag<ES, 32>{});
}

TEST(CuteLevelGraph, ThreeMembersShareTheOperator) {
  constexpr int NG = 5, TE = 2, E = 6;
  OpView<NG>    h("h", NG, NG);
  ViewR         u("u", E, NG, NG, NG);
  ViewR         ca("ca", NG, E, NG, NG), cb("cb", NG, E, NG, NG),
      cc("cc", NG, E, NG, NG);
  ViewR ta("ta", NG, E, NG, NG), tb("tb", NG, E, NG, NG),
      tc("tc", NG, E, NG, NG);
  fill(h, 1.5f);
  fill(u, -0.5f);

  auto g0 = make_level_graph<float, ES>(MapQ<NG, TE>{});
  auto [g1, sh] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'a'>(h))));
  auto [g2, su] = g1.add(
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(u))));
  auto [g3, xa, xb, xc] = g2.add(
      make_contraction_node<'q', 'e', 'b', 'c'>(sh, su),
      make_contraction_node<'q', 'e', 'a', 'c'>(sh.template as<'q', 'b'>(), su),
      make_contraction_node<'q', 'e', 'a', 'b'>(sh.template as<'q', 'c'>(),
                                                su));
  const auto out = g3.outputs(xa, xb, xc);

  out.execute(CutePolicyTag<>{}, ca, cb, cc);
  ASSERT_TRUE(synced());
  out.execute(TeamPolicyTag<ES>{}, ta, tb, tc);
  ASSERT_TRUE(synced());

  EXPECT_LT(max_rel_err(ca, gradient_ref(h, u, 0)), 1e-5f);
  EXPECT_LT(max_rel_err(cb, gradient_ref(h, u, 1)), 1e-5f);
  EXPECT_LT(max_rel_err(cc, gradient_ref(h, u, 2)), 1e-5f);
  EXPECT_LT(max_rel_err(ca, ta), 1e-5f);
  EXPECT_LT(max_rel_err(cb, tb), 1e-5f);
  EXPECT_LT(max_rel_err(cc, tc), 1e-5f);
}

TEST(CuteLevelGraph, UserMmaWithIdleThreadsBesideDefault) {
  constexpr int NG = 8, TE = 2, E = 8;
  OpView<NG>    h("h", NG, NG);
  ViewR         u("u", E, NG, NG, NG);
  ViewR         ca("ca", NG, E, NG, NG), cb("cb", NG, E, NG, NG);
  ViewR         ta("ta", NG, E, NG, NG), tb("tb", NG, E, NG, NG);
  fill(h, 0.75f);
  fill(u, 3.0f);

  const auto mma = cute::make_tiled_mma(
      cute::UniversalFMA<float, float, float>{},
      cute::Layout<cute::Shape<cute::_4, cute::_8, cute::_1>>{});
  static_assert(decltype(cute::size(mma))::value == 32);

  auto g0 = make_level_graph<float, ES>(MapQ<NG, TE>{});
  auto [g1, sh] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'a'>(h))));
  auto [g2, su] = g1.add(
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(u))));
  auto [g3, xa, xb] =
      g2.add(make_contraction_node<'q', 'e', 'b', 'c'>(sh, su, NoHook{}, mma),
             make_contraction_node<'q', 'e', 'a', 'c'>(
                 sh.template as<'q', 'b'>(), su));
  const auto out = g3.outputs(xa, xb);

  out.execute(CutePolicyTag<>{}, ca, cb);
  ASSERT_TRUE(synced());
  out.execute(TeamPolicyTag<ES>{}, ta, tb);
  ASSERT_TRUE(synced());

  EXPECT_LT(max_rel_err(ca, gradient_ref(h, u, 0)), 1e-5f);
  EXPECT_LT(max_rel_err(cb, gradient_ref(h, u, 1)), 1e-5f);
  EXPECT_LT(max_rel_err(ca, ta), 1e-5f);
  EXPECT_LT(max_rel_err(cb, tb), 1e-5f);
}

TEST(CuteLevelGraph, ChainedContractionReadsPermutedRoot) {
  constexpr int NG = 5, TE = 2, E = 6;
  OpView<NG>    h("h", NG, NG);
  ViewR         u("u", E, NG, NG, NG);
  ViewR         cx("cx", E, NG, NG, NG), tx("tx", E, NG, NG, NG);
  ViewR         cd("cd", NG, E, NG, NG), td("td", NG, E, NG, NG);
  fill(h, 1.25f);
  fill(u, 0.5f);
  Kokkos::deep_copy(cx, -999.0f);
  Kokkos::deep_copy(cd, -999.0f);

  auto g0 = make_level_graph<float, ES>(MapQ<NG, TE>{});
  auto [g1, sh] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'a'>(h))));
  auto [g2, su] = g1.add(
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(u))));
  auto [g3, x] = g2.add(make_contraction_node<'e', 'b', 'c', 'q'>(sh, su));
  auto [g4, d] = g3.add(
      make_contraction_node<'p', 'e', 'b', 'c'>(sh.template as<'p', 'q'>(), x));
  const auto out = g4.outputs(x, d);

  out.execute(CutePolicyTag<>{}, cx, cd);
  ASSERT_TRUE(synced());
  out.execute(TeamPolicyTag<ES>{}, tx, td);
  ASSERT_TRUE(synced());

  const auto g  = gradient_ref(h, u, 0);
  auto       hh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, h);
  Kokkos::View<float****, Kokkos::LayoutRight, Kokkos::HostSpace> rx(
      "rx", E, NG, NG, NG),
      rd("rd", NG, E, NG, NG);
  for (int q = 0; q < NG; ++q)
    for (int e = 0; e < E; ++e)
      for (int b = 0; b < NG; ++b)
        for (int c = 0; c < NG; ++c) rx(e, b, c, q) = g(q, e, b, c);
  for (int p = 0; p < NG; ++p)
    for (int e = 0; e < E; ++e)
      for (int b = 0; b < NG; ++b)
        for (int c = 0; c < NG; ++c) {
          float acc = 0.0f;
          for (int q = 0; q < NG; ++q) acc += hh(p, q) * g(q, e, b, c);
          rd(p, e, b, c) = acc;
        }

  EXPECT_LT(max_rel_err(cx, rx), 1e-5f);
  EXPECT_LT(max_rel_err(cd, rd), 1e-5f);
  EXPECT_LT(max_rel_err(cx, tx), 1e-5f);
  EXPECT_LT(max_rel_err(cd, td), 1e-5f);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
