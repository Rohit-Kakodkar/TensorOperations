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
  EXPECT_EQ(out.cute_smem_bytes(), out.cute_unpooled_smem_bytes());

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

namespace {

constexpr int kPN = 5, kPTE = 2, kPE = 6;

using MapP =
    LabelTiles<LabelTile<'e', kPTE>, LabelWhole<'q', kPN>, LabelWhole<'p', kPN>,
               LabelWhole<'r', kPN>, LabelWhole<'a', kPN>, LabelWhole<'b', kPN>,
               LabelWhole<'c', kPN>>;

auto pooled_chain(OpView<kPN> h, ViewR u) {
  auto g0 = make_level_graph<float, ES>(MapP{});
  auto [g1, sh] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'a'>(h))));
  auto [g2, su] = g1.add(
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(u))));
  auto [g3, x] = g2.add(make_contraction_node<'q', 'e', 'b', 'c'>(sh, su));
  auto [g4, y] = g3.add(
      make_contraction_node<'p', 'e', 'b', 'c'>(sh.template as<'p', 'q'>(), x));
  auto [g5, z] = g4.add(
      make_contraction_node<'r', 'e', 'b', 'c'>(sh.template as<'r', 'p'>(), y));
  return std::make_tuple(g5, z);
}

using PooledLevels = std::decay_t<decltype(std::get<0>(pooled_chain(
    std::declval<OpView<kPN>>(), std::declval<ViewR>())))>::levels_type;

static_assert(Impl::lg_cute_first_reader_v<PooledLevels, 128, 0> == 2 &&
              Impl::lg_cute_last_reader_v<PooledLevels, 128, 0> == 4);
static_assert(Impl::lg_cute_first_reader_v<PooledLevels, 128, 1> == 2 &&
              Impl::lg_cute_last_reader_v<PooledLevels, 128, 1> == 2);
static_assert(Impl::lg_cute_first_reader_v<PooledLevels, 128, 2> == 3 &&
              Impl::lg_cute_last_reader_v<PooledLevels, 128, 2> == 3);
static_assert(Impl::lg_cute_first_reader_v<PooledLevels, 128, 3> == 4 &&
              Impl::lg_cute_last_reader_v<PooledLevels, 128, 3> == 4);
static_assert(!Impl::lg_cute_smem_slot_v<PooledLevels, 128, 4>,
              "the root is only ever in registers");
static_assert(Impl::lg_cute_slot_pool_v<PooledLevels, 128, 1> !=
                  Impl::lg_cute_slot_pool_v<PooledLevels, 128, 0>,
              "h and u are copied at the same level");
static_assert(Impl::lg_cute_slot_pool_v<PooledLevels, 128, 2> ==
                  Impl::lg_cute_slot_pool_v<PooledLevels, 128, 1>,
              "x reclaims u's buffer");
static_assert(Impl::lg_cute_slot_pool_v<PooledLevels, 128, 3> ==
                  Impl::lg_cute_slot_pool_v<PooledLevels, 128, 1>,
              "y reclaims x's buffer");
static_assert(!Impl::lg_cute_reuses_at_v<PooledLevels, 128, 2,
                                         std::make_index_sequence<2>>);
static_assert(Impl::lg_cute_reuses_at_v<PooledLevels, 128, 3,
                                        std::make_index_sequence<3>>);
static_assert(Impl::lg_cute_reuses_at_v<PooledLevels, 128, 4,
                                        std::make_index_sequence<4>>);

}  // namespace

TEST(CuteLevelGraph, PooledChainReusesBuffers) {
  OpView<kPN> h("h", kPN, kPN);
  ViewR       u("u", kPE, kPN, kPN, kPN);
  ViewR       cz("cz", kPN, kPE, kPN, kPN), tz("tz", kPN, kPE, kPN, kPN);
  fill(h, 0.25f);
  fill(u, -1.5f);
  Kokkos::deep_copy(cz, -999.0f);

  auto [g, z]    = pooled_chain(h, u);
  const auto out = g.outputs(z);

  constexpr std::size_t tile =
      Impl::slot_arena_step<float, ES>(kPN * kPTE * kPN * kPN);
  constexpr std::size_t hstep = Impl::slot_arena_step<float, ES>(kPN * kPN);
  EXPECT_EQ(out.cute_unpooled_smem_bytes(), (hstep + 3 * tile) * sizeof(float));
  EXPECT_EQ(out.cute_smem_bytes(), (hstep + tile) * sizeof(float));

  EXPECT_EQ(out.execute(CutePolicyTag<>{}, cz), kPE / kPTE);
  ASSERT_TRUE(synced());
  out.execute(TeamPolicyTag<ES>{}, tz);
  ASSERT_TRUE(synced());

  const auto g1 = gradient_ref(h, u, 0);
  auto       hh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, h);
  Kokkos::View<float****, Kokkos::LayoutRight, Kokkos::HostSpace> y(
      "y", kPN, kPE, kPN, kPN),
      rz("rz", kPN, kPE, kPN, kPN);
  for (int p = 0; p < kPN; ++p)
    for (int e = 0; e < kPE; ++e)
      for (int b = 0; b < kPN; ++b)
        for (int c = 0; c < kPN; ++c) {
          float acc = 0.0f;
          for (int q = 0; q < kPN; ++q) acc += hh(p, q) * g1(q, e, b, c);
          y(p, e, b, c) = acc;
        }
  for (int r = 0; r < kPN; ++r)
    for (int e = 0; e < kPE; ++e)
      for (int b = 0; b < kPN; ++b)
        for (int c = 0; c < kPN; ++c) {
          float acc = 0.0f;
          for (int p = 0; p < kPN; ++p) acc += hh(r, p) * y(p, e, b, c);
          rz(r, e, b, c) = acc;
        }

  EXPECT_LT(max_rel_err(cz, rz), 1e-5f);
  EXPECT_LT(max_rel_err(cz, tz), 1e-5f);
}

struct ScaleAt {
  KOKKOS_FUNCTION float operator()(int i, int j, int k, int l, float g) const {
    return 2.0f * g + 0.5f + 0.01f * static_cast<float>(i) -
           0.002f * static_cast<float>(l) + 0.003f * static_cast<float>(j * k);
  }
};

struct DupAt {
  KOKKOS_FUNCTION Kokkos::Array<float, 2> operator()(int i, int j, int k, int l,
                                                     float g) const {
    return {ScaleAt{}(i, j, k, l, g), -3.0f * g + 0.1f * static_cast<float>(j)};
  }
};

struct MixAt {
  KOKKOS_FUNCTION float operator()(int i, int j, int k, int l, float x,
                                   float y) const {
    return x - 0.5f * y + 0.01f * static_cast<float>(i + 3 * j) -
           0.002f * static_cast<float>(k * l);
  }
};

namespace {

using HostV4 = Kokkos::View<float****, Kokkos::LayoutRight, Kokkos::HostSpace>;

template <typename F>
HostV4 host_map(const HostV4& g, F f) {
  HostV4 r("r", g.extent(0), g.extent(1), g.extent(2), g.extent(3));
  for (int i = 0; i < static_cast<int>(g.extent(0)); ++i)
    for (int j = 0; j < static_cast<int>(g.extent(1)); ++j)
      for (int k = 0; k < static_cast<int>(g.extent(2)); ++k)
        for (int l = 0; l < static_cast<int>(g.extent(3)); ++l)
          r(i, j, k, l) = f(i, j, k, l, g(i, j, k, l));
  return r;
}

template <typename G>
using LevelsOf = typename std::decay_t<G>::levels_type;

template <int NG, int TE>
struct GradFixture {
  static constexpr int E = 6;
  OpView<NG>           h{"h", NG, NG};
  ViewR                u{"u", E, NG, NG, NG};

  GradFixture() {
    fill(h, 0.5f);
    fill(u, -2.0f);
  }

  auto graph() const {
    auto g0 = make_level_graph<float, ES>(MapQ<NG, TE>{});
    auto [g1, sh] =
        g0.add(make_stage_node(make_input_node(make_handle<'q', 'a'>(h))));
    auto [g2, su] = g1.add(
        make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(u))));
    auto [g3, ca] = g2.add(make_contraction_node<'q', 'e', 'b', 'c'>(sh, su));
    return std::make_tuple(g3, sh, ca);
  }

  HostV4 grad() const { return gradient_ref(h, u, 0); }
};

}  // namespace

TEST(CuteLevelGraph, CombineRegisterDrivenAfterContraction) {
  GradFixture<5, 2> fx;
  auto [g3, sh, ca] = fx.graph();
  auto [g4, pv] = g3.add(make_combine_node<'q', 'e', 'b', 'c'>(ca, ScaleAt{}));
  using Plan    = Impl::lg_cute_combine_plan<LevelsOf<decltype(g4)>, 3, 0, 128>;
  static_assert(Plan::register_driven && Plan::D == 0);

  ViewR cp("cp", 5, fx.E, 5, 5), tp("tp", 5, fx.E, 5, 5);
  Kokkos::deep_copy(cp, -999.0f);
  const auto out = g4.outputs(pv);
  EXPECT_EQ(out.cute_smem_bytes(), g3.outputs(ca).cute_smem_bytes());

  out.execute(CutePolicyTag<>{}, cp);
  ASSERT_TRUE(synced());
  out.execute(TeamPolicyTag<ES>{}, tp);
  ASSERT_TRUE(synced());

  EXPECT_LT(max_rel_err(cp, host_map(fx.grad(), ScaleAt{})), 1e-5f);
  EXPECT_LT(max_rel_err(cp, tp), 1e-5f);
}

TEST(CuteLevelGraph, CombineMultiOutputBothRoots) {
  GradFixture<8, 2> fx;
  auto [g3, sh, ca] = fx.graph();
  auto [g4, p0, p1] =
      g3.add(make_combine_node<'q', 'e', 'b', 'c'>(ca, DupAt{}));
  ViewR      c0("c0", 8, fx.E, 8, 8), c1("c1", 8, fx.E, 8, 8);
  ViewR      t0("t0", 8, fx.E, 8, 8), t1("t1", 8, fx.E, 8, 8);
  const auto out = g4.outputs(p0, p1);

  out.execute(CutePolicyTag<>{}, c0, c1);
  ASSERT_TRUE(synced());
  out.execute(TeamPolicyTag<ES>{}, t0, t1);
  ASSERT_TRUE(synced());

  const auto g = fx.grad();
  EXPECT_LT(max_rel_err(c0, host_map(g, ScaleAt{})), 1e-5f);
  EXPECT_LT(max_rel_err(c1, host_map(g,
                                     [](int, int j, int, int, float v) {
                                       return -3.0f * v +
                                              0.1f * static_cast<float>(j);
                                     })),
            1e-5f);
  EXPECT_LT(max_rel_err(c0, t0), 1e-5f);
  EXPECT_LT(max_rel_err(c1, t1), 1e-5f);
}

TEST(CuteLevelGraph, CombinePermutedReadIsThreadDriven) {
  GradFixture<5, 2> fx;
  auto [g3, sh, ca] = fx.graph();
  auto [g4, pv] = g3.add(make_combine_node<'q', 'c', 'e', 'b'>(ca, ScaleAt{}));
  using Plan    = Impl::lg_cute_combine_plan<LevelsOf<decltype(g4)>, 3, 0, 128>;
  static_assert(!Plan::register_driven);
  static_assert(decltype(cute::size(typename Plan::thr_layout{}))::value < 128,
                "exercises idle threads on the thread-driven path");

  ViewR cp("cp", 5, 5, fx.E, 5), tp("tp", 5, 5, fx.E, 5);
  Kokkos::deep_copy(cp, -999.0f);
  const auto out = g4.outputs(pv);
  EXPECT_GT(out.cute_smem_bytes(), g3.outputs(ca).cute_smem_bytes());

  out.execute(CutePolicyTag<>{}, cp);
  ASSERT_TRUE(synced());
  out.execute(TeamPolicyTag<ES>{}, tp);
  ASSERT_TRUE(synced());

  const auto g = fx.grad();
  HostV4     r("r", 5, 5, fx.E, 5);
  for (int q = 0; q < 5; ++q)
    for (int c = 0; c < 5; ++c)
      for (int e = 0; e < fx.E; ++e)
        for (int b = 0; b < 5; ++b)
          r(q, c, e, b) = ScaleAt{}(q, c, e, b, g(q, e, b, c));
  EXPECT_LT(max_rel_err(cp, r), 1e-5f);
  EXPECT_LT(max_rel_err(cp, tp), 1e-5f);
}

TEST(CuteLevelGraph, CombineMixedPartitionsGoThroughSmem) {
  GradFixture<5, 2> fx;
  ViewR             w("w", 5, fx.E, 5, 5);
  fill(w, 4.0f);
  auto [g3, sh, ca] = fx.graph();
  auto [g4, sw]     = g3.add(
      make_stage_node(make_input_node(make_handle<'q', 'e', 'b', 'c'>(w))));
  auto [g5, pv] =
      g4.add(make_combine_node<'q', 'e', 'b', 'c'>(ca, sw, MixAt{}));
  using Plan = Impl::lg_cute_combine_plan<LevelsOf<decltype(g5)>, 4, 0, 128>;
  static_assert(Plan::register_driven && Plan::D == 0);
  static_assert(Plan::template in_register<0>() &&
                !Plan::template in_register<1>());

  ViewR      cp("cp", 5, fx.E, 5, 5), tp("tp", 5, fx.E, 5, 5);
  const auto out = g5.outputs(pv);
  out.execute(CutePolicyTag<>{}, cp);
  ASSERT_TRUE(synced());
  out.execute(TeamPolicyTag<ES>{}, tp);
  ASSERT_TRUE(synced());

  const auto g  = fx.grad();
  auto       hw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, w);
  HostV4     r("r", 5, fx.E, 5, 5);
  for (int q = 0; q < 5; ++q)
    for (int e = 0; e < fx.E; ++e)
      for (int b = 0; b < 5; ++b)
        for (int c = 0; c < 5; ++c)
          r(q, e, b, c) = MixAt{}(q, e, b, c, g(q, e, b, c), hw(q, e, b, c));
  EXPECT_LT(max_rel_err(cp, r), 1e-5f);
  EXPECT_LT(max_rel_err(cp, tp), 1e-5f);
}

TEST(CuteLevelGraph, CombineOfStagesStaysInRegisters) {
  constexpr int NG = 5, TE = 2, E = 6;
  ViewR         a("a", E, NG, NG, NG), b("b", E, NG, NG, NG);
  ViewR         cp("cp", E, NG, NG, NG), tp("tp", E, NG, NG, NG);
  fill(a, 1.0f);
  fill(b, -6.0f);

  auto g0           = make_level_graph<float, ES>(MapQ<NG, TE>{});
  auto [g1, sa, sb] = g0.add(
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(a))),
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(b))));
  auto [g2, pv] =
      g1.add(make_combine_node<'e', 'a', 'b', 'c'>(sa, sb, MixAt{}));
  using Plan = Impl::lg_cute_combine_plan<LevelsOf<decltype(g2)>, 1, 0, 128>;
  static_assert(Plan::register_driven && Plan::template in_register<0>() &&
                Plan::template in_register<1>());

  const auto out = g2.outputs(pv);
  EXPECT_EQ(out.cute_smem_bytes(), 0u);
  out.execute(CutePolicyTag<>{}, cp);
  ASSERT_TRUE(synced());
  out.execute(TeamPolicyTag<ES>{}, tp);
  ASSERT_TRUE(synced());

  auto   ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a);
  auto   hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b);
  HostV4 r("r", E, NG, NG, NG);
  for (int e = 0; e < E; ++e)
    for (int i = 0; i < NG; ++i)
      for (int j = 0; j < NG; ++j)
        for (int k = 0; k < NG; ++k)
          r(e, i, j, k) = MixAt{}(e, i, j, k, ha(e, i, j, k), hb(e, i, j, k));
  EXPECT_LT(max_rel_err(cp, r), 1e-5f);
  EXPECT_LT(max_rel_err(cp, tp), 1e-5f);
}

TEST(CuteLevelGraph, CombineChainsIntoCombineAndContraction) {
  GradFixture<5, 2> fx;
  auto [g3, sh, ca] = fx.graph();
  auto [g4, p] = g3.add(make_combine_node<'q', 'e', 'b', 'c'>(ca, ScaleAt{}));
  auto [g5, r] = g4.add(make_combine_node<'q', 'e', 'b', 'c'>(p, ScaleAt{}));
  auto [g6, d] = g5.add(
      make_contraction_node<'p', 'e', 'b', 'c'>(sh.template as<'p', 'q'>(), r));
  using Levels = LevelsOf<decltype(g6)>;
  using PlanR  = Impl::lg_cute_combine_plan<Levels, 4, 0, 128>;
  static_assert(PlanR::register_driven && PlanR::template in_register<0>(),
                "combine -> combine stays in registers");
  static_assert(!Impl::lg_cute_smem_slot_v<Levels, 128, 3> &&
                    Impl::lg_cute_smem_slot_v<Levels, 128, 4>,
                "only the combine read by the contraction goes to smem");

  ViewR      cr("cr", 5, fx.E, 5, 5), tr("tr", 5, fx.E, 5, 5);
  ViewR      cd("cd", 5, fx.E, 5, 5), td("td", 5, fx.E, 5, 5);
  const auto out = g6.outputs(r, d);
  out.execute(CutePolicyTag<>{}, cr, cd);
  ASSERT_TRUE(synced());
  out.execute(TeamPolicyTag<ES>{}, tr, td);
  ASSERT_TRUE(synced());

  const auto rr = host_map(host_map(fx.grad(), ScaleAt{}), ScaleAt{});
  auto   hh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fx.h);
  HostV4 rd("rd", 5, fx.E, 5, 5);
  for (int p = 0; p < 5; ++p)
    for (int e = 0; e < fx.E; ++e)
      for (int b = 0; b < 5; ++b)
        for (int c = 0; c < 5; ++c) {
          float acc = 0.0f;
          for (int q = 0; q < 5; ++q) acc += hh(p, q) * rr(q, e, b, c);
          rd(p, e, b, c) = acc;
        }
  EXPECT_LT(max_rel_err(cr, rr), 1e-5f);
  EXPECT_LT(max_rel_err(cd, rd), 1e-4f);
  EXPECT_LT(max_rel_err(td, rd), 1e-4f);
  EXPECT_LT(max_rel_err(cr, tr), 1e-5f);
  EXPECT_LT(max_rel_err(cd, td), 1e-5f);
}

TEST(CuteLevelGraph, CombineOfPermutedContractionOutput) {
  constexpr int NG = 5, TE = 5, E = 10;
  OpView<NG>    h("h", NG, NG);
  ViewR         u("u", E, NG, NG, NG);
  ViewR         cp("cp", E, NG, NG, NG), tp("tp", E, NG, NG, NG);
  fill(h, 0.5f);
  fill(u, -2.0f);

  auto g0 = make_level_graph<float, ES>(MapQ<NG, TE>{});
  auto [g1, sh] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'a'>(h))));
  auto [g2, su] = g1.add(
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(u))));
  auto [g3, x]  = g2.add(make_contraction_node<'e', 'b', 'c', 'q'>(sh, su));
  auto [g4, pv] = g3.add(make_combine_node<'e', 'b', 'c', 'q'>(x, ScaleAt{}));
  using Plan    = Impl::lg_cute_combine_plan<LevelsOf<decltype(g4)>, 3, 0, 128>;
  static_assert(
      !Plan::register_driven,
      "x is stored in canonical (q,e,b,c) order: a fragment cannot be "
      "read as (e,b,c,q) even though the tile extents agree");

  const auto out = g4.outputs(pv);
  out.execute(CutePolicyTag<>{}, cp);
  ASSERT_TRUE(synced());
  out.execute(TeamPolicyTag<ES>{}, tp);
  ASSERT_TRUE(synced());

  const auto g = gradient_ref(h, u, 0);
  HostV4     r("r", E, NG, NG, NG);
  for (int e = 0; e < E; ++e)
    for (int b = 0; b < NG; ++b)
      for (int c = 0; c < NG; ++c)
        for (int q = 0; q < NG; ++q)
          r(e, b, c, q) = ScaleAt{}(e, b, c, q, g(q, e, b, c));
  EXPECT_LT(max_rel_err(cp, r), 1e-5f);
  EXPECT_LT(max_rel_err(cp, tp), 1e-5f);
}

namespace {

constexpr int kFE = 8, kFN = 5, kFTE = 2, kNglob = 97;

using ViewGlob = Kokkos::View<float**, Kokkos::LayoutLeft, ES>;
using ViewMap  = Kokkos::View<int****, Kokkos::LayoutRight, ES>;

struct ReadU {
  ViewR                 u;
  KOKKOS_FUNCTION float operator()(int e, int a, int b, int c) const {
    return u(e, a, b, c);
  }
};

struct GatherU {
  ViewGlob              glob;
  ViewMap               iglob;
  int                   comp;
  KOKKOS_FUNCTION float operator()(int e, int a, int b, int c) const {
    return glob(iglob(e, a, b, c), comp);
  }
};

int gid(int e, int a, int b, int c) {
  return (e * 37 + a * 11 + b * 5 + c * 3) % kNglob;
}

const Kokkos::Array<int, 4> kFExt{kFE, kFN, kFN, kFN};

template <typename MakeU>
auto functional_gradient(OpView<kFN> h, MakeU make_u, ViewR cute_out,
                         ViewR team_out) {
  Kokkos::deep_copy(cute_out, -999.0f);
  auto g0 = make_level_graph<float, ES>(MapQ<kFN, kFTE>{});
  auto [g1, sh] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'a'>(h))));
  auto [g2, su]  = g1.add(make_stage_node(make_u()));
  auto [g3, c]   = g2.add(make_contraction_node<'q', 'e', 'b', 'c'>(sh, su));
  const auto out = g3.outputs(c);
  EXPECT_EQ(out.execute(CutePolicyTag<>{}, cute_out), kFE / kFTE);
  EXPECT_TRUE(synced());
  out.execute(TeamPolicyTag<ES>{}, team_out);
  EXPECT_TRUE(synced());
}

ViewR gathered_u(ViewGlob glob, int comp) {
  auto  hg = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, glob);
  ViewR u("u_ref", kFE, kFN, kFN, kFN);
  auto  hu = Kokkos::create_mirror_view(u);
  for (int e = 0; e < kFE; ++e)
    for (int a = 0; a < kFN; ++a)
      for (int b = 0; b < kFN; ++b)
        for (int c = 0; c < kFN; ++c)
          hu(e, a, b, c) = hg(gid(e, a, b, c), comp);
  Kokkos::deep_copy(u, hu);
  return u;
}

struct GatherFixture {
  OpView<kFN> h{"h", kFN, kFN};
  ViewGlob    glob{"glob", kNglob, 3};
  ViewMap     iglob{"iglob", kFE, kFN, kFN, kFN};

  GatherFixture() {
    fill(h, 0.5f);
    fill(glob, -3.0f);
    auto hm = Kokkos::create_mirror_view(iglob);
    for (int e = 0; e < kFE; ++e)
      for (int a = 0; a < kFN; ++a)
        for (int b = 0; b < kFN; ++b)
          for (int c = 0; c < kFN; ++c) hm(e, a, b, c) = gid(e, a, b, c);
    Kokkos::deep_copy(iglob, hm);
  }

  GatherU fn() const { return {glob, iglob, 1}; }
};

template <typename Operand>
using stage_order_t = typename Impl::lg_cute_stage_order<Operand, 4>::type;

template <typename Layout>
using functional_stage_t =
    decltype(make_functional_input_node<ES, 'e', 'a', 'b', 'c'>(
        std::declval<Layout>(), std::declval<GatherU>()));

static_assert(
    std::is_same_v<stage_order_t<functional_stage_t<DynamicTileLayoutRight<4>>>,
                   std::integer_sequence<int, 3, 2, 1, 0>>);
static_assert(
    std::is_same_v<stage_order_t<functional_stage_t<DynamicTileLayoutLeft<4>>>,
                   std::integer_sequence<int, 0, 1, 2, 3>>);
static_assert(
    std::is_same_v<stage_order_t<functional_stage_t<StaticTileLayoutStride<
                       StaticTile<kFE, kFN, kFN, kFN>, 2, 3, 1, 0>>>,
                   std::integer_sequence<int, 2, 3, 1, 0>>);

}  // namespace

TEST(CuteLevelGraph, FunctionalStageIsBitwiseTheInputStage) {
  OpView<kFN> h("h", kFN, kFN);
  ViewR       u("u", kFE, kFN, kFN, kFN);
  fill(h, 0.5f);
  fill(u, -2.0f);
  ViewR plain("plain", kFN, kFE, kFN, kFN), fn_out("fn", kFN, kFE, kFN, kFN),
      team("team", kFN, kFE, kFN, kFN);

  functional_gradient(
      h, [&] { return make_input_node(make_handle<'e', 'a', 'b', 'c'>(u)); },
      plain, team);
  functional_gradient(
      h,
      [&] {
        return make_functional_input_node<ES, 'e', 'a', 'b', 'c'>(kFExt,
                                                                  ReadU{u});
      },
      fn_out, team);

  EXPECT_EQ(mismatches(fn_out, plain), 0);
  EXPECT_LT(max_rel_err(fn_out, team), 1e-5f);
}

TEST(CuteLevelGraph, FunctionalStageGathersThroughAnIndexMap) {
  const GatherFixture f;
  ViewR cute_out("cute", kFN, kFE, kFN, kFN), team("team", kFN, kFE, kFN, kFN);

  functional_gradient(
      f.h,
      [&] {
        return make_functional_input_node<ES, 'e', 'a', 'b', 'c'>(kFExt,
                                                                  f.fn());
      },
      cute_out, team);

  EXPECT_LT(max_rel_err(cute_out, gradient_ref(f.h, gathered_u(f.glob, 1), 0)),
            1e-5f);
  EXPECT_LT(max_rel_err(cute_out, team), 1e-5f);
}

TEST(CuteLevelGraph, FunctionalStageOrderDoesNotChangeTheResult) {
  const GatherFixture f;
  ViewR right("right", kFN, kFE, kFN, kFN), left("left", kFN, kFE, kFN, kFN),
      perm("perm", kFN, kFE, kFN, kFN), team("team", kFN, kFE, kFN, kFN);

  functional_gradient(
      f.h,
      [&] {
        return make_functional_input_node<ES, 'e', 'a', 'b', 'c'>(
            DynamicTileLayoutRight<4>{kFExt}, f.fn());
      },
      right, team);
  functional_gradient(
      f.h,
      [&] {
        return make_functional_input_node<ES, 'e', 'a', 'b', 'c'>(
            DynamicTileLayoutLeft<4>{kFExt}, f.fn());
      },
      left, team);
  functional_gradient(
      f.h,
      [&] {
        return make_functional_input_node<ES, 'e', 'a', 'b', 'c'>(
            StaticTileLayoutStride<StaticTile<kFE, kFN, kFN, kFN>, 2, 3, 1,
                                   0>{},
            f.fn());
      },
      perm, team);

  EXPECT_LT(max_rel_err(right, gradient_ref(f.h, gathered_u(f.glob, 1), 0)),
            1e-5f);
  EXPECT_EQ(mismatches(left, right), 0);
  EXPECT_EQ(mismatches(perm, right), 0);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
