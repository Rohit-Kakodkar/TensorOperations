// ===========================================================================
// test_reduce_node.cpp -- make_reduce_node, a parallel_reduce-shaped node.
//
//   make_reduce_node<Out...>(over<Red...>{}, [outputs<M>{},] ops..., fn
//                            [, reducer])
//   fn(o..., rho..., accessor_0, ..., accessor_{N-1}, acc&)
//
// Operands arrive as accessors: axes whose labels are output or reduction
// labels are bound; the remaining (free) axes are passed by fn. Each test
// holds a graph to a hand-written host loop. The gridded label 'e' has three
// tiles, so a dropped tile origin shows up as a duplicated answer, and no
// two labels differ only by case.
//
//   1. over<>, fully bound operands == make_combine_node, bitwise.
//   2. Broadcast: operands carrying strict subsets of the output labels.
//   3. A declared reduction label, bound on the operands.
//   4. Free axes indexed by fn itself, with an early-return predicate.
//   5. outputs<2>.
//   6. Max reducer.
//   7. A reduce reading a level output, LevelPlan guards instantiated.
//   8. Rank-9 output through StridedAlias, output labels no operand carries,
//      whole tile above the host level-0 scratch cap (scratch level 1).
// ===========================================================================
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/LevelPlan.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/StridedAlias.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

using namespace TensorOperations;
using ES = Kokkos::DefaultExecutionSpace;

namespace reduce_test {

constexpr int kE = 6, kTE = 2;  // 3 teams along 'e'
constexpr int kQ = 4, kA = 3, kN = 5;

using View2 = Kokkos::View<float**, Kokkos::LayoutRight, ES>;
using View3 = Kokkos::View<float***, Kokkos::LayoutRight, ES>;

// 'i' and 'l' are the two point labels an output compares; 'f' is the free
// function axis of h.
using Map = LabelTiles<LabelTile<'e', kTE>, LabelWhole<'q', kQ>,
                       LabelWhole<'a', kA>, LabelWhole<'i', kN>,
                       LabelWhole<'l', kN>, LabelWhole<'f', kN>>;

float hval(int q, int i) { return 0.1f + 0.3f * q - 0.2f * i + 0.05f * q * i; }
float uval(int e, int a) { return 0.2f + 0.11f * e - 0.07f * a; }
float mval(int e, int q) { return 0.5f + 0.13f * e - 0.09f * q + 0.02f * e * q; }

View2 make_h() {  // (q, f)
  View2 v("h", kQ, kN);
  auto  m = Kokkos::create_mirror_view(v);
  for (int q = 0; q < kQ; ++q)
    for (int i = 0; i < kN; ++i) m(q, i) = hval(q, i);
  Kokkos::deep_copy(v, m);
  return v;
}
View2 make_u() {  // (e, a)
  View2 v("u", kE, kA);
  auto  m = Kokkos::create_mirror_view(v);
  for (int e = 0; e < kE; ++e)
    for (int a = 0; a < kA; ++a) m(e, a) = uval(e, a);
  Kokkos::deep_copy(v, m);
  return v;
}
View2 make_m() {  // (e, q)
  View2 v("m", kE, kQ);
  auto  m = Kokkos::create_mirror_view(v);
  for (int e = 0; e < kE; ++e)
    for (int q = 0; q < kQ; ++q) m(e, q) = mval(e, q);
  Kokkos::deep_copy(v, m);
  return v;
}

double ref_diag(int e, int i, int l, double mscale = 1.0, double mshift = 0.0) {
  if ((i + l) % 2 != 0) return 0.0;
  double r = 0.0;
  for (int q = 0; q < kQ; ++q)
    r += static_cast<double>(hval(q, i)) * hval(q, l) *
         (mscale * mval(e, q) + mshift);
  return r;
}

// --- functors (named structs: a KOKKOS_LAMBDA in a private TestBody cannot
// be a device functor) ------------------------------------------------------

struct AffineCombine {
  KOKKOS_FUNCTION float operator()(int e, int a, float x, float y) const {
    return 2.0f * x - 0.5f * y + 100.0f * e + a;
  }
};
struct AffineReduce {
  template <typename U>
  KOKKOS_FUNCTION void operator()(int e, int a, const U& x, const U& y,
                                  float& acc) const {
    acc += 2.0f * x() - 0.5f * y() + 100.0f * e + a;
  }
};

struct Broadcast {
  template <typename H, typename U>
  KOKKOS_FUNCTION void operator()(int e, int a, int i, const H& h, const U& u,
                                  float& acc) const {
    acc += h() * u() + 0.001f * (e * 100 + a * 10 + i);
  }
};

// sum_q [ (i + l) even ] h(q, i) h(q, l) m(e, q), with q a declared label.
struct BoundDiagonal {
  template <typename H1, typename H2, typename M>
  KOKKOS_FUNCTION void operator()(int, int i, int l, int, const H1& hqi,
                                  const H2& hql, const M& m,
                                  float& acc) const {
    if ((i + l) % 2 != 0) return;
    acc += hqi() * hql() * m();
  }
};

// The same sum with q looped by fn over FREE axes: h(q, f), m(q).
struct FreeDiagonal {
  template <typename H, typename M>
  KOKKOS_FUNCTION void operator()(int, int i, int l, const H& h, const M& m,
                                  float& acc) const {
    if ((i + l) % 2 != 0) return;
    for (int q = 0; q < kQ; ++q) acc += h(q, i) * h(q, l) * m(q);
  }
};

struct TwoOut {
  template <typename H1, typename H2, typename M>
  KOKKOS_FUNCTION void operator()(int, int i, int l, int, const H1& hqi,
                                  const H2& hql, const M& m,
                                  Kokkos::Array<float, 2>& acc) const {
    acc[0] += hqi() * hql() * m();
    acc[1] += (i == l ? 1.0f : 0.0f) * hql() * m();
  }
};

struct MaxTerm {
  template <typename H1, typename H2, typename M>
  KOKKOS_FUNCTION void operator()(int, int, int, int, const H1& hqi,
                                  const H2& hql, const M& m,
                                  float& acc) const {
    acc = Kokkos::max(acc, hqi() * hql() * m());
  }
};

struct ScaleShift {
  template <typename M>
  KOKKOS_FUNCTION void operator()(int, int, const M& m, float& acc) const {
    acc += 3.0f * m() + 1.0f;
  }
};

template <typename View>
double max_diff_3(const View& d, double (*ref)(int, int, int)) {
  auto   h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d);
  double err = 0.0;
  for (int e = 0; e < kE; ++e)
    for (int i = 0; i < kN; ++i)
      for (int l = 0; l < kN; ++l)
        err = std::max(err, std::abs(ref(e, i, l) - h(e, i, l)));
  return err;
}
double ref_diag_plain(int e, int i, int l) { return ref_diag(e, i, l); }
double ref_diag_scaled(int e, int i, int l) {
  return ref_diag(e, i, l, 3.0, 1.0);
}

}  // namespace reduce_test

using namespace reduce_test;

// ---------------------------------------------------------------------------
TEST(ReduceNode, NoReductionLabelsIsACombine) {
  auto  Ud = make_u();
  View2 Pr("Pr", kE, kA), Pc("Pc", kE, kA);
  {
    auto g0 = make_level_graph<float, ES>(Map{});
    auto [g1, u] =
        g0.add(make_stage_node(make_input_node(make_handle<'e', 'a'>(Ud))));
    auto [g2, p] =
        g1.add(make_reduce_node<'e', 'a'>(over<>{}, u, u, AffineReduce{}));
    g2.outputs(p).execute(TeamPolicyTag2<ES>{}, Pr);
  }
  {
    auto g0 = make_level_graph<float, ES>(Map{});
    auto [g1, u] =
        g0.add(make_stage_node(make_input_node(make_handle<'e', 'a'>(Ud))));
    auto [g2, p] = g1.add(make_combine_node<'e', 'a'>(u, u, AffineCombine{}));
    g2.outputs(p).execute(TeamPolicyTag2<ES>{}, Pc);
  }
  Kokkos::fence();
  auto Hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Pr);
  auto Hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Pc);
  for (int e = 0; e < kE; ++e)
    for (int a = 0; a < kA; ++a) {
      EXPECT_EQ(Hr(e, a), Hc(e, a)) << "e=" << e << " a=" << a;
      EXPECT_NEAR(Hr(e, a), 1.5f * uval(e, a) + 100.0f * e + a, 1e-4f);
    }
}

// ---------------------------------------------------------------------------
TEST(ReduceNode, BroadcastOperandsMatchHostLoop) {
  View2 Hd("h_ai", kA, kN);
  {
    auto m = Kokkos::create_mirror_view(Hd);
    for (int a = 0; a < kA; ++a)
      for (int i = 0; i < kN; ++i) m(a, i) = hval(a, i);
    Kokkos::deep_copy(Hd, m);
  }
  auto  Ud = make_u();
  View3 Bd("B", kE, kA, kN);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, h] =
      g0.add(make_stage_node(make_input_node(make_handle<'a', 'i'>(Hd))));
  auto [g2, u] =
      g1.add(make_stage_node(make_input_node(make_handle<'e', 'a'>(Ud))));
  auto [g3, b] =
      g2.add(make_reduce_node<'e', 'a', 'i'>(over<>{}, h, u, Broadcast{}));
  g3.outputs(b).execute(TeamPolicyTag2<ES>{}, Bd);
  Kokkos::fence();

  auto   Bh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Bd);
  double err = 0.0;
  for (int e = 0; e < kE; ++e)
    for (int a = 0; a < kA; ++a)
      for (int i = 0; i < kN; ++i) {
        const double ref = static_cast<double>(hval(a, i)) * uval(e, a) +
                           0.001 * (e * 100 + a * 10 + i);
        err = std::max(err, std::abs(ref - Bh(e, a, i)));
      }
  EXPECT_LT(err, 1e-4);
}

// ---------------------------------------------------------------------------
TEST(ReduceNode, DeclaredReductionLabelBoundOnOperands) {
  auto  Hd = make_h();
  auto  Md = make_m();
  View3 Kd("K", kE, kN, kN);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, h] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'f'>(Hd))));
  auto [g2, m] =
      g1.add(make_stage_node(make_input_node(make_handle<'e', 'q'>(Md))));
  auto node = make_reduce_node<'e', 'i', 'l'>(
      over<'q'>{}, h.template as<'q', 'i'>(), h.template as<'q', 'l'>(), m,
      BoundDiagonal{});
  static_assert(decltype(node)::RedRank == 1);
  auto [g3, k] = g2.add(node);
  g3.outputs(k).execute(TeamPolicyTag2<ES>{}, Kd);
  Kokkos::fence();
  EXPECT_LT(max_diff_3(Kd, ref_diag_plain), 1e-4);
}

// ---------------------------------------------------------------------------
TEST(ReduceNode, FreeAxesIndexedByTheFunctor) {
  auto  Hd = make_h();
  auto  Md = make_m();
  View3 Kd("K", kE, kN, kN);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, h] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'f'>(Hd))));
  auto [g2, m] =
      g1.add(make_stage_node(make_input_node(make_handle<'e', 'q'>(Md))));
  // h fully free, m bound on 'e' and free on 'q'.
  auto [g3, k] = g2.add(
      make_reduce_node<'e', 'i', 'l'>(over<>{}, h, m, FreeDiagonal{}));
  g3.outputs(k).execute(TeamPolicyTag2<ES>{}, Kd);
  Kokkos::fence();
  EXPECT_LT(max_diff_3(Kd, ref_diag_plain), 1e-4);
}

// ---------------------------------------------------------------------------
TEST(ReduceNode, MultiOutputReduce) {
  auto  Hd = make_h();
  auto  Md = make_m();
  View3 K0("K0", kE, kN, kN), K1("K1", kE, kN, kN);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, h] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'f'>(Hd))));
  auto [g2, m] =
      g1.add(make_stage_node(make_input_node(make_handle<'e', 'q'>(Md))));
  auto [g3, k0, k1] = g2.add(make_reduce_node<'e', 'i', 'l'>(
      over<'q'>{}, outputs<2>{}, h.template as<'q', 'i'>(),
      h.template as<'q', 'l'>(), m, TwoOut{}));
  g3.outputs(k0, k1).execute(TeamPolicyTag2<ES>{}, K0, K1);
  Kokkos::fence();

  auto   H0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, K0);
  auto   H1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, K1);
  double e0 = 0.0, e1 = 0.0;
  for (int e = 0; e < kE; ++e)
    for (int i = 0; i < kN; ++i)
      for (int l = 0; l < kN; ++l) {
        double r0 = 0.0, r1 = 0.0;
        for (int q = 0; q < kQ; ++q) {
          r0 += static_cast<double>(hval(q, i)) * hval(q, l) * mval(e, q);
          if (i == l) r1 += static_cast<double>(hval(q, l)) * mval(e, q);
        }
        e0 = std::max(e0, std::abs(r0 - H0(e, i, l)));
        e1 = std::max(e1, std::abs(r1 - H1(e, i, l)));
      }
  EXPECT_LT(e0, 1e-4);
  EXPECT_LT(e1, 1e-4);
}

// ---------------------------------------------------------------------------
TEST(ReduceNode, MaxReducer) {
  auto  Hd = make_h();
  auto  Md = make_m();
  View3 Kd("K", kE, kN, kN);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, h] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'f'>(Hd))));
  auto [g2, m] =
      g1.add(make_stage_node(make_input_node(make_handle<'e', 'q'>(Md))));
  auto [g3, k] = g2.add(make_reduce_node<'e', 'i', 'l'>(
      over<'q'>{}, h.template as<'q', 'i'>(), h.template as<'q', 'l'>(), m,
      MaxTerm{}, Max<float>{}));
  g3.outputs(k).execute(TeamPolicyTag2<ES>{}, Kd);
  Kokkos::fence();

  auto   Kh  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Kd);
  double err = 0.0;
  for (int e = 0; e < kE; ++e)
    for (int i = 0; i < kN; ++i)
      for (int l = 0; l < kN; ++l) {
        float r = -std::numeric_limits<float>::max();
        for (int q = 0; q < kQ; ++q)
          r = std::max(r, hval(q, i) * hval(q, l) * mval(e, q));
        err = std::max(err, static_cast<double>(std::abs(r - Kh(e, i, l))));
      }
  EXPECT_LT(err, 1e-5);
}

// ---------------------------------------------------------------------------
TEST(ReduceNode, ReadsALevelOutputAndPassesThePlanGuards) {
  auto  Hd = make_h();
  auto  Md = make_m();
  View3 Kd("K", kE, kN, kN);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, h] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'f'>(Hd))));
  auto [g2, m] =
      g1.add(make_stage_node(make_input_node(make_handle<'e', 'q'>(Md))));
  auto [g3, m2] =
      g2.add(make_reduce_node<'e', 'q'>(over<>{}, m, ScaleShift{}));
  auto [g4, k] = g3.add(make_reduce_node<'e', 'i', 'l'>(
      over<'q'>{}, h.template as<'q', 'i'>(), h.template as<'q', 'l'>(), m2,
      BoundDiagonal{}));
  using Plan = LevelPlan<std::decay_t<decltype(g4.levels)>>;
  static_assert(Plan::num_levels == 4);
  static_assert(Plan::num_slots == 4);
  g4.outputs(k).execute(TeamPolicyTag2<ES>{}, Kd);
  Kokkos::fence();
  EXPECT_LT(max_diff_3(Kd, ref_diag_scaled), 1e-4);
}

// ---------------------------------------------------------------------------
// 8. Rank 9: (e, a, k, j, i, b, n, m, l) -- the element-stiffness frame.
//    k, j, n, m are carried by no operand (their extents come from the map);
//    the whole tile (2 x 4^3 x 2 x 4^3 floats = 64 KB) is above the host
//    level-0 scratch cap, so the host run carves from level 1.
// ---------------------------------------------------------------------------
namespace r9 {
constexpr int N = 4, NC = 2, NE = 3;
using Map9 = LabelTiles<LabelTile<'e', 1>, LabelWhole<'a', NC>,
                        LabelWhole<'b', NC>, LabelWhole<'k', N>,
                        LabelWhole<'j', N>, LabelWhole<'i', N>,
                        LabelWhole<'n', N>, LabelWhole<'m', N>,
                        LabelWhole<'l', N>, LabelWhole<'q', N>,
                        LabelWhole<'f', N>>;
using ViewH = Kokkos::View<float**, Kokkos::LayoutRight, ES>;
using ViewM = Kokkos::View<float****, Kokkos::LayoutRight, ES>;
using View1 = Kokkos::View<float*, Kokkos::LayoutRight, ES>;
float h9(int q, int i) { return 0.2f + 0.7f * q - 0.3f * i; }
float m9(int e, int a, int b, int q) {
  return 0.4f + 0.15f * e - 0.25f * a + 0.35f * b + 0.1f * q;
}
struct Term {
  template <typename H1, typename H2, typename M>
  KOKKOS_FUNCTION void operator()(int, int, int k, int j, int, int, int n,
                                  int m, int, int, const H1& hqi,
                                  const H2& hql, const M& mm,
                                  float& acc) const {
    if (k == n && j == m) acc += hqi() * hql() * mm();
  }
};
}  // namespace r9

TEST(ReduceNode, RankNineOutputThroughAStridedAlias) {
  using namespace r9;
  ViewH Hd("h9", N, N);
  ViewM Md("m9", NE, NC, NC, N);
  {
    auto hm = Kokkos::create_mirror_view(Hd);
    for (int q = 0; q < N; ++q)
      for (int i = 0; i < N; ++i) hm(q, i) = h9(q, i);
    Kokkos::deep_copy(Hd, hm);
    auto mm = Kokkos::create_mirror_view(Md);
    for (int e = 0; e < NE; ++e)
      for (int a = 0; a < NC; ++a)
        for (int b = 0; b < NC; ++b)
          for (int q = 0; q < N; ++q) mm(e, a, b, q) = m9(e, a, b, q);
    Kokkos::deep_copy(Md, mm);
  }
  constexpr int block = NC * N * N * N * NC * N * N * N;
  View1         flat("K9", NE * block);
  const auto    alias =
      make_strided_alias<ES, NC, N, N, N, NC, N, N, N>(flat.data(), NE);

  auto g0 = make_level_graph<float, ES>(Map9{});
  auto [g1, h] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'f'>(Hd))));
  auto [g2, m] = g1.add(
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'q'>(Md))));
  auto [g3, k] = g2.add(
      make_reduce_node<'e', 'a', 'k', 'j', 'i', 'b', 'n', 'm', 'l'>(
          over<'q'>{}, h.template as<'q', 'i'>(), h.template as<'q', 'l'>(),
          m, Term{}));
  const auto out = g3.outputs(k);
  EXPECT_GT(out.scratch_bytes(), std::size_t{32} * 1024);
  constexpr bool on_host =
      Kokkos::SpaceAccessibility<ES, Kokkos::HostSpace>::accessible;
  out.scratch_level(on_host ? 1 : 0).execute(TeamPolicyTag2<ES>{}, alias);
  Kokkos::fence();

  auto   Kh  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, flat);
  double err = 0.0;
  int    lin = 0;
  for (int e = 0; e < NE; ++e)
    for (int a = 0; a < NC; ++a)
      for (int kk = 0; kk < N; ++kk)
        for (int j = 0; j < N; ++j)
          for (int i = 0; i < N; ++i)
            for (int b = 0; b < NC; ++b)
              for (int n = 0; n < N; ++n)
                for (int mm = 0; mm < N; ++mm)
                  for (int l = 0; l < N; ++l, ++lin) {
                    double ref = 0.0;
                    if (kk == n && j == mm)
                      for (int q = 0; q < N; ++q)
                        ref += static_cast<double>(h9(q, i)) * h9(q, l) *
                               m9(e, a, b, q);
                    err = std::max(err, std::abs(ref - Kh(lin)));
                  }
  EXPECT_LT(err, 1e-5);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
