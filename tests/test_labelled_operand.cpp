#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Permute.hpp>
#include <TensorOperations/TensorHandle.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

using namespace TensorOperations;

namespace {

template <int32_t... L>
using M = std::integer_sequence<int32_t, L...>;
template <int... Ps>
using Pm = std::integer_sequence<int, Ps...>;

using AU = M<'e', 'a', 'b', 'c'>;

using H0 = M<'i', 'a'>;
using H1 = M<'j', 'b'>;
using H2 = M<'k', 'c'>;

using PermU0 = Impl::permB_seq_t<H0, AU>;
using PermU1 = Impl::permB_seq_t<H1, AU>;
using PermU2 = Impl::permB_seq_t<H2, AU>;

static_assert(std::is_same_v<PermU0, Pm<1, 0, 2, 3>>);
static_assert(std::is_same_v<PermU1, Pm<2, 0, 1, 3>>);
static_assert(std::is_same_v<PermU2, Pm<3, 0, 1, 2>>);

static_assert(std::is_same_v<Impl::permA_seq_t<H0, AU>, Pm<0, 1>>);
static_assert(Impl::is_identity_v<Impl::permA_seq_t<H0, AU>>);
static_assert(!Impl::is_identity_v<PermU0>);

static_assert(
    std::is_same_v<Impl::canonC_modes_seq_t<4, H0, AU>, M<'i', 'e', 'b', 'c'>>);
static_assert(
    std::is_same_v<Impl::canonC_modes_seq_t<4, H1, AU>, M<'j', 'e', 'a', 'c'>>);
static_assert(
    std::is_same_v<Impl::canonC_modes_seq_t<4, H2, AU>, M<'k', 'e', 'a', 'b'>>);

static_assert(Impl::valid_contraction_v<4, H0, AU, M<'i', 'e', 'b', 'c'>>);
static_assert(Impl::same_label_set_v<M<'e', 'i', 'b', 'c'>,
                                     Impl::canonC_modes_seq_t<4, H0, AU>>);

static_assert(Impl::labels_distinct_v<AU>);
static_assert(!Impl::labels_distinct_v<M<'e', 'a', 'a', 'c'>>);

static_assert(!Impl::valid_contraction_v<4, H0, AU, M<'i', 'e', 'b', 'z'>>);
static_assert(!Impl::valid_contraction_v<4, H0, AU, M<'i', 'e', 'b', 'a'>>);

}  // namespace

TEST(LabelledOperand, DerivedPermsMatchHandWritten) { SUCCEED(); }

namespace stagenode {

using ES     = Kokkos::DefaultExecutionSpace;
using team_t = typename Kokkos::TeamPolicy<ES>::member_type;
using V3     = Kokkos::View<float***, Kokkos::LayoutRight, ES>;
using Buf1D  = Kokkos::View<float*, ES>;

constexpr int sP = 2, sQ = 3, sR = 4;
using TS = StaticTile<sP, sQ, sR>;

KOKKOS_INLINE_FUNCTION float s_val(int p, int q, int r) {
  return 0.5f + 0.25f * p - 0.125f * q + 0.375f * r + 0.0625f * p * r;
}

void run_stage_node(V3 v, Buf1D out, Kokkos::View<int, ES> rank_out) {
  auto sn = make_stage_node(make_input_node(make_handle<'p', 'q', 'r'>(v)));
  const std::size_t bytes = Impl::scratch_tile_bytes<float, ES>(TS{});
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const team_t& team) {
        auto S = make_evaluator<TeamPolicyTag2<ES>>(
            sn, TS{}, team)(Kokkos::Array<int, 3>{0, 0, 0});
        team.team_barrier();
        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          const auto sv = S.node().storage_;
          for (int p = 0; p < sP; ++p)
            for (int q = 0; q < sQ; ++q)
              for (int r = 0; r < sR; ++r)
                out((p * sQ + q) * sR + r) = sv(p, q, r);
          rank_out() = decltype(S)::Rank;
        });
      });
  Kokkos::fence();
}

}  // namespace stagenode

TEST(StageNode, StagesNativelyAndRepublishesLabels) {
  using namespace stagenode;
  V3   v("v", sP, sQ, sR);
  auto vh = Kokkos::create_mirror_view(v);
  for (int p = 0; p < sP; ++p)
    for (int q = 0; q < sQ; ++q)
      for (int r = 0; r < sR; ++r) vh(p, q, r) = s_val(p, q, r);
  Kokkos::deep_copy(v, vh);

  Buf1D                 out("out", sP * sQ * sR);
  Kokkos::View<int, ES> rk("rk");
  run_stage_node(v, out, rk);

  auto oh = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(oh, out);
  int rkh = 0;
  Kokkos::deep_copy(rkh, rk);

  EXPECT_EQ(rkh, 3);
  for (int p = 0; p < sP; ++p)
    for (int q = 0; q < sQ; ++q)
      for (int r = 0; r < sR; ++r)
        EXPECT_FLOAT_EQ(oh((p * sQ + q) * sR + r), s_val(p, q, r))
            << "p=" << p << " q=" << q << " r=" << r;
}

namespace lvl {

using ES     = Kokkos::DefaultExecutionSpace;
using team_t = typename Kokkos::TeamPolicy<ES>::member_type;
using V2     = Kokkos::View<float**, Kokkos::LayoutRight, ES>;
using V4     = Kokkos::View<float****, Kokkos::LayoutRight, ES>;
using Buf1D  = Kokkos::View<float*, ES>;

constexpr int uE = 2, uA = 3, uB = 4, uC = 5;

using TU  = StaticTile<uE, uA, uB, uC>;
using TH0 = StaticTile<uE, uA>;
using TH1 = StaticTile<uA, uB>;
using TH2 = StaticTile<uB, uC>;

KOKKOS_INLINE_FUNCTION float u_val(int e, int a, int b, int c) {
  return 0.5f + 0.25f * a - 0.125f * b + 0.375f * c + 0.0625f * e +
         0.03125f * a * c;
}
KOKKOS_INLINE_FUNCTION float h_val(int d, int m, int k) {
  return 1.25f - 0.5f * d + 0.75f * m - 0.25f * k + 0.125f * m * k * (d + 1);
}

template <typename... Tiles>
std::size_t scratch_for() {
  return (Impl::scratch_tile_bytes<float, ES>(Tiles{}) + ... + std::size_t{0});
}

void run_level(V4 u, V2 h0, V2 h1, V2 h2, Buf1D o0, Buf1D o1, Buf1D o2) {
  auto un =
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(u)));
  auto h0n = make_stage_node(make_input_node(make_handle<'i', 'a'>(h0)));
  auto h1n = make_stage_node(make_input_node(make_handle<'j', 'b'>(h1)));
  auto h2n = make_stage_node(make_input_node(make_handle<'k', 'c'>(h2)));
  auto n0  = make_contraction_node<'i', 'e', 'b', 'c'>(h0n, un);
  auto n1  = make_contraction_node<'j', 'e', 'a', 'c'>(h1n, un);
  auto n2  = make_contraction_node<'k', 'e', 'a', 'b'>(h2n, un);

  const std::size_t bytes =
      scratch_for<TU, TH0, TH1, TH2>() +
      Impl::scratch_tile_bytes<float, ES>(StaticTile<uE, uE, uB, uC>{}) +
      Impl::scratch_tile_bytes<float, ES>(StaticTile<uA, uE, uA, uC>{}) +
      Impl::scratch_tile_bytes<float, ES>(StaticTile<uB, uE, uA, uB>{});

  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const team_t& team) {
        auto U = make_evaluator<TeamPolicyTag2<ES>>(
            un, TU{}, team)(Kokkos::Array<int, 4>{0, 0, 0, 0});
        auto A0 = make_evaluator<TeamPolicyTag2<ES>>(
            h0n, TH0{}, team)(Kokkos::Array<int, 2>{0, 0});
        auto A1 = make_evaluator<TeamPolicyTag2<ES>>(
            h1n, TH1{}, team)(Kokkos::Array<int, 2>{0, 0});
        auto A2 = make_evaluator<TeamPolicyTag2<ES>>(
            h2n, TH2{}, team)(Kokkos::Array<int, 2>{0, 0});
        team.team_barrier();

        using T0 = contract_c_tile_t<decltype(n0), decltype(A0), decltype(U)>;
        using T1 = contract_c_tile_t<decltype(n1), decltype(A1), decltype(U)>;
        using T2 = contract_c_tile_t<decltype(n2), decltype(A2), decltype(U)>;
        static_assert(std::is_same_v<T0, StaticTile<uE, uE, uB, uC>>);
        static_assert(std::is_same_v<T1, StaticTile<uA, uE, uA, uC>>);
        static_assert(std::is_same_v<T2, StaticTile<uB, uE, uA, uB>>);

        auto C0 =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, T0{}));
        auto C1 =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, T1{}));
        auto C2 =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, T2{}));

        contract_into(n0, A0, U, C0, team)();
        contract_into(n1, A1, U, C1, team)();
        contract_into(n2, A2, U, C2, team)();
        team.team_barrier();

        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          const auto v0 = C0.storage_;
          const auto v1 = C1.storage_;
          const auto v2 = C2.storage_;
          for (int m = 0; m < uE; ++m)
            for (int e = 0; e < uE; ++e)
              for (int b = 0; b < uB; ++b)
                for (int c = 0; c < uC; ++c)
                  o0(((m * uE + e) * uB + b) * uC + c) = v0(m, e, b, c);
          for (int m = 0; m < uA; ++m)
            for (int e = 0; e < uE; ++e)
              for (int a = 0; a < uA; ++a)
                for (int c = 0; c < uC; ++c)
                  o1(((m * uE + e) * uA + a) * uC + c) = v1(m, e, a, c);
          for (int m = 0; m < uB; ++m)
            for (int e = 0; e < uE; ++e)
              for (int a = 0; a < uA; ++a)
                for (int b = 0; b < uB; ++b)
                  o2(((m * uE + e) * uA + a) * uB + b) = v2(m, e, a, b);
        });
      });
  Kokkos::fence();
}

}  // namespace lvl

TEST(LevelContract, ThreeDirectionsFromOneStagedTensor) {
  using namespace lvl;
  V4   u("u", uE, uA, uB, uC);
  V2   h0("h0", uE, uA), h1("h1", uA, uB), h2("h2", uB, uC);
  auto uh  = Kokkos::create_mirror_view(u);
  auto h0h = Kokkos::create_mirror_view(h0);
  auto h1h = Kokkos::create_mirror_view(h1);
  auto h2h = Kokkos::create_mirror_view(h2);
  for (int e = 0; e < uE; ++e)
    for (int a = 0; a < uA; ++a)
      for (int b = 0; b < uB; ++b)
        for (int c = 0; c < uC; ++c) uh(e, a, b, c) = u_val(e, a, b, c);
  for (int m = 0; m < uE; ++m)
    for (int a = 0; a < uA; ++a) h0h(m, a) = h_val(0, m, a);
  for (int m = 0; m < uA; ++m)
    for (int b = 0; b < uB; ++b) h1h(m, b) = h_val(1, m, b);
  for (int m = 0; m < uB; ++m)
    for (int c = 0; c < uC; ++c) h2h(m, c) = h_val(2, m, c);
  Kokkos::deep_copy(u, uh);
  Kokkos::deep_copy(h0, h0h);
  Kokkos::deep_copy(h1, h1h);
  Kokkos::deep_copy(h2, h2h);

  Buf1D o0("o0", uE * uE * uB * uC), o1("o1", uA * uE * uA * uC),
      o2("o2", uB * uE * uA * uB);
  run_level(u, h0, h1, h2, o0, o1, o2);

  auto g0 = Kokkos::create_mirror_view(o0);
  auto g1 = Kokkos::create_mirror_view(o1);
  auto g2 = Kokkos::create_mirror_view(o2);
  Kokkos::deep_copy(g0, o0);
  Kokkos::deep_copy(g1, o1);
  Kokkos::deep_copy(g2, o2);

  for (int m = 0; m < uE; ++m)
    for (int e = 0; e < uE; ++e)
      for (int b = 0; b < uB; ++b)
        for (int c = 0; c < uC; ++c) {
          double acc = 0.0;
          for (int a = 0; a < uA; ++a)
            acc += static_cast<double>(h0h(m, a)) * uh(e, a, b, c);
          EXPECT_NEAR(g0(((m * uE + e) * uB + b) * uC + c),
                      static_cast<float>(acc), 1e-4f)
              << "d0 m=" << m << " e=" << e << " b=" << b << " c=" << c;
        }
  for (int m = 0; m < uA; ++m)
    for (int e = 0; e < uE; ++e)
      for (int a = 0; a < uA; ++a)
        for (int c = 0; c < uC; ++c) {
          double acc = 0.0;
          for (int b = 0; b < uB; ++b)
            acc += static_cast<double>(h1h(m, b)) * uh(e, a, b, c);
          EXPECT_NEAR(g1(((m * uE + e) * uA + a) * uC + c),
                      static_cast<float>(acc), 1e-4f)
              << "d1 m=" << m << " e=" << e << " a=" << a << " c=" << c;
        }
  for (int m = 0; m < uB; ++m)
    for (int e = 0; e < uE; ++e)
      for (int a = 0; a < uA; ++a)
        for (int b = 0; b < uB; ++b) {
          double acc = 0.0;
          for (int c = 0; c < uC; ++c)
            acc += static_cast<double>(h2h(m, c)) * uh(e, a, b, c);
          EXPECT_NEAR(g2(((m * uE + e) * uA + a) * uB + b),
                      static_cast<float>(acc), 1e-4f)
              << "d2 m=" << m << " e=" << e << " a=" << a << " b=" << b;
        }
}

namespace sem {

using ES     = Kokkos::DefaultExecutionSpace;
using team_t = typename Kokkos::TeamPolicy<ES>::member_type;
using V2     = Kokkos::View<float**, Kokkos::LayoutRight, ES>;
using V4     = Kokkos::View<float****, Kokkos::LayoutRight, ES>;
using Buf1D  = Kokkos::View<float*, ES>;

constexpr int nT = 2, nN = 3;

using TU = StaticTile<nT, nN, nN, nN>;
using TH = StaticTile<nN, nN>;
using TC = StaticTile<nN, nT, nN, nN>;

KOKKOS_INLINE_FUNCTION float su(int e, int a, int b, int c) {
  return 0.5f + 0.25f * a - 0.125f * b + 0.375f * c + 0.0625f * e +
         0.03125f * a * c - 0.25f * b * c;
}
KOKKOS_INLINE_FUNCTION float sh(int i, int k) {
  return 1.25f + 0.75f * i - 0.5f * k + 0.125f * i * k;
}

void run_sem(V4 u, V2 h, Buf1D o0, Buf1D o1, Buf1D o2) {
  auto un =
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(u)));
  auto hn = make_stage_node(make_input_node(make_handle<'i', 'a'>(h)));
  auto q0 = make_contraction_node<'i', 'e', 'b', 'c'>(hn, un);
  auto q1 = make_contraction_node<'i', 'e', 'a', 'c'>(hn.as<'i', 'b'>(), un);
  auto q2 = make_contraction_node<'i', 'e', 'a', 'b'>(hn.as<'i', 'c'>(), un);

  const std::size_t bytes = lvl::scratch_for<TU, TH, TC, TC, TC>();
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const team_t& team) {
        auto U = make_evaluator<TeamPolicyTag2<ES>>(
            un, TU{}, team)(Kokkos::Array<int, 4>{0, 0, 0, 0});
        auto H = make_evaluator<TeamPolicyTag2<ES>>(
            hn, TH{}, team)(Kokkos::Array<int, 2>{0, 0});
        team.team_barrier();

        static_assert(
            std::is_same_v<
                contract_c_tile_t<decltype(q0), decltype(H), decltype(U)>, TC>);

        auto C0 =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, TC{}));
        auto C1 =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, TC{}));
        auto C2 =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, TC{}));

        contract_into(q0, H, U, C0, team)();
        contract_into(q1, H, U, C1, team)();
        contract_into(q2, H, U, C2, team)();
        team.team_barrier();

        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          const auto v0 = C0.storage_;
          const auto v1 = C1.storage_;
          const auto v2 = C2.storage_;
          for (int i = 0; i < nN; ++i)
            for (int e = 0; e < nT; ++e)
              for (int x = 0; x < nN; ++x)
                for (int y = 0; y < nN; ++y) {
                  const int f = ((i * nT + e) * nN + x) * nN + y;
                  o0(f)       = v0(i, e, x, y);
                  o1(f)       = v1(i, e, x, y);
                  o2(f)       = v2(i, e, x, y);
                }
        });
      });
  Kokkos::fence();
}

}  // namespace sem

TEST(LevelContract, SemGradientLevelViaOneStageAndThreeAs) {
  using namespace sem;
  V4   u("u", nT, nN, nN, nN);
  V2   h("h", nN, nN);
  auto uh = Kokkos::create_mirror_view(u);
  auto hh = Kokkos::create_mirror_view(h);
  for (int e = 0; e < nT; ++e)
    for (int a = 0; a < nN; ++a)
      for (int b = 0; b < nN; ++b)
        for (int c = 0; c < nN; ++c) uh(e, a, b, c) = su(e, a, b, c);
  for (int i = 0; i < nN; ++i)
    for (int k = 0; k < nN; ++k) hh(i, k) = sh(i, k);
  Kokkos::deep_copy(u, uh);
  Kokkos::deep_copy(h, hh);

  const int n = nN * nT * nN * nN;
  Buf1D     o0("o0", n), o1("o1", n), o2("o2", n);
  run_sem(u, h, o0, o1, o2);

  auto g0 = Kokkos::create_mirror_view(o0);
  auto g1 = Kokkos::create_mirror_view(o1);
  auto g2 = Kokkos::create_mirror_view(o2);
  Kokkos::deep_copy(g0, o0);
  Kokkos::deep_copy(g1, o1);
  Kokkos::deep_copy(g2, o2);

  for (int i = 0; i < nN; ++i)
    for (int e = 0; e < nT; ++e)
      for (int x = 0; x < nN; ++x)
        for (int y = 0; y < nN; ++y) {
          const int f  = ((i * nT + e) * nN + x) * nN + y;
          double    a0 = 0.0, a1 = 0.0, a2 = 0.0;
          for (int k = 0; k < nN; ++k) {
            a0 += static_cast<double>(hh(i, k)) * uh(e, k, x, y);
            a1 += static_cast<double>(hh(i, k)) * uh(e, x, k, y);
            a2 += static_cast<double>(hh(i, k)) * uh(e, x, y, k);
          }
          EXPECT_NEAR(g0(f), static_cast<float>(a0), 1e-4f)
              << "d0 i=" << i << " e=" << e << " x=" << x << " y=" << y;
          EXPECT_NEAR(g1(f), static_cast<float>(a1), 1e-4f)
              << "d1 i=" << i << " e=" << e << " x=" << x << " y=" << y;
          EXPECT_NEAR(g2(f), static_cast<float>(a2), 1e-4f)
              << "d2 i=" << i << " e=" << e << " x=" << x << " y=" << y;
        }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
