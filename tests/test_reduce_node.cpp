// ===========================================================================
// test_reduce_node.cpp -- a combine generalized to BROADCAST operands and
// REDUCTION labels, in a level graph.
//
//   out[o] = sum_{rho in R} fn(o..., rho..., op_k[proj_k(o, rho)]...)
//
// Each test holds the graph to a hand-written host loop. Extents are pairwise
// distinct wherever the math allows (a transposed argument cannot hide behind
// two equal axes), and the gridded label has more than one tile so a dropped
// origin duplicates tile 0's answer instead of passing by luck.
//
// What each test pins:
//   1. R empty, full-rank operands: bitwise identical to a combine.
//   2. Broadcast: operands carrying strict subsets of the output labels.
//   3. One reduction label with a predicate on the OUTPUT coordinate and one
//      staged operand relabeled twice (the stiffness kernel's h(q,i) h(q,I)).
//   4. Two reduction labels and PINNED axes (fixed<I>) on one staged slot.
//   5. Multi-output reduce.
//   6. A reduce reading a level output (not only staged leaves) and the
//      LevelPlan guards firing on a reduce level.
//   7. A rank-8 output over a rank-8 alias view.
// ===========================================================================
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/LevelPlan.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cmath>

using namespace TensorOperations;
using ES = Kokkos::DefaultExecutionSpace;

namespace reduce_test {

constexpr int kE = 6, kTE = 2;  // 3 teams
constexpr int kQ = 4, kA = 3, kI = 5, kS = 3, kR = 3;

using View2 = Kokkos::View<float**, Kokkos::LayoutRight, ES>;
using View3 = Kokkos::View<float***, Kokkos::LayoutRight, ES>;
using View4 = Kokkos::View<float****, Kokkos::LayoutRight, ES>;

// Map: 'e' gridded; everything else whole. 'i' and 'I' share an extent
// because the predicate tests compare them.
using Map = LabelTiles<LabelTile<'e', kTE>, LabelWhole<'q', kQ>,
                       LabelWhole<'a', kA>, LabelWhole<'i', kI>,
                       LabelWhole<'I', kI>, LabelWhole<'s', kS>,
                       LabelWhole<'r', kR>>;

float hval(int q, int i) { return 0.1f + 0.3f * q - 0.2f * i + 0.05f * q * i; }
float uval(int e, int a) { return 0.2f + 0.11f * e - 0.07f * a; }
float mval(int e, int q) { return 0.5f + 0.13f * e - 0.09f * q + 0.02f * e * q; }
float m4val(int e, int r, int s, int q) {
  return 0.3f + 0.07f * e + 0.11f * r - 0.05f * s + 0.03f * q +
         0.01f * (r + 1) * (s + 2) * q;
}

View2 make_h() {
  View2 v("h", kQ, kI);
  auto  m = Kokkos::create_mirror_view(v);
  for (int q = 0; q < kQ; ++q)
    for (int i = 0; i < kI; ++i) m(q, i) = hval(q, i);
  Kokkos::deep_copy(v, m);
  return v;
}
View2 make_u() {
  View2 v("u", kE, kA);
  auto  m = Kokkos::create_mirror_view(v);
  for (int e = 0; e < kE; ++e)
    for (int a = 0; a < kA; ++a) m(e, a) = uval(e, a);
  Kokkos::deep_copy(v, m);
  return v;
}
View2 make_m() {
  View2 v("m", kE, kQ);
  auto  m = Kokkos::create_mirror_view(v);
  for (int e = 0; e < kE; ++e)
    for (int q = 0; q < kQ; ++q) m(e, q) = mval(e, q);
  Kokkos::deep_copy(v, m);
  return v;
}
View4 make_m4() {
  View4 v("m4", kE, kR, kS, kQ);
  auto  m = Kokkos::create_mirror_view(v);
  for (int e = 0; e < kE; ++e)
    for (int r = 0; r < kR; ++r)
      for (int s = 0; s < kS; ++s)
        for (int q = 0; q < kQ; ++q) m(e, r, s, q) = m4val(e, r, s, q);
  Kokkos::deep_copy(v, m);
  return v;
}

// --- functors (named structs: a KOKKOS_LAMBDA in a private TestBody cannot
// be a device functor) ------------------------------------------------------

// 1. P(e,a) = fn(e, a, u(e,a), u(e,a)) with no reduction.
struct AffineTwo {
  KOKKOS_FUNCTION float operator()(int e, int a, float x, float y) const {
    return 2.0f * x - 0.5f * y + 100.0f * e + a;
  }
};

// 2. B(e,a,i) = h(a? no: uses h(q->a slot) ...) -- broadcast: h carries (a,i)
// as a (kA x kI) tile, u carries (e,a).
struct Broadcast {
  KOKKOS_FUNCTION float operator()(int e, int a, int i, float h,
                                   float u) const {
    return h * u + 0.001f * (e * 100 + a * 10 + i);
  }
};

// 3. K(e,i,I) = sum_q [ (i + I) even ] h(q,i) h(q,I) M(e,q)
struct DiagonalTerm {
  KOKKOS_FUNCTION float operator()(int, int i, int I, int, float hqi,
                                   float hqI, float m) const {
    return ((i + I) % 2 == 0) ? hqi * hqI * m : 0.0f;
  }
};

// 4. K(e,i,I) = sum_{s,q} h(q,i) h(q,I) (M(e,0,s,q) + (s+1) M(e,2,s,q))
//    with a predicate on the reduction coordinate: only q != I. The reduction
//    coordinates arrive in FIRST-APPEARANCE order over the operand list -- q
//    (from h) before s (from M) -- which the static_assert in the test pins.
struct TwoRedPinned {
  KOKKOS_FUNCTION float operator()(int, int, int I, int q, int s, float hqi,
                                   float hqI, float m0, float m2) const {
    if (q == I) return 0.0f;
    return hqi * hqI * (m0 + static_cast<float>(s + 1) * m2);
  }
};

// 5. Two outputs from one pass.
struct TwoOut {
  KOKKOS_FUNCTION Kokkos::Array<float, 2> operator()(int, int i, int I, int q,
                                                     float hqi, float hqI,
                                                     float m) const {
    return {hqi * hqI * m, (i == I ? 1.0f : 0.0f) * hqI * m};
  }
};

// 6. A level-1 reduce feeding a level-2 reduce.
struct Scale {
  KOKKOS_FUNCTION float operator()(int, int, float m) const {
    return 3.0f * m + 1.0f;
  }
};

}  // namespace reduce_test

using namespace reduce_test;

// ---------------------------------------------------------------------------
// 1. R empty, full-rank operands == combine, bitwise.
// ---------------------------------------------------------------------------
TEST(ReduceNode, NoReductionLabelsIsACombine) {
  auto  Ud = make_u();
  View2 Pr("Pr", kE, kA), Pc("Pc", kE, kA);

  {
    auto g0 = make_level_graph<float, ES>(Map{});
    auto [g1, u] =
        g0.add(make_stage_node(make_input_node(make_handle<'e', 'a'>(Ud))));
    auto [g2, p] = g1.add(make_reduce_node<'e', 'a'>(u, u, AffineTwo{}));
    static_assert(decltype(g2)::num_levels == 2);
    using Plan = LevelPlan<std::decay_t<decltype(g2.levels)>>;
    static_assert(Plan::num_slots == 2);
    g2.outputs(p).execute(TeamPolicyTag2<ES>{}, Pr);
  }
  {
    auto g0 = make_level_graph<float, ES>(Map{});
    auto [g1, u] =
        g0.add(make_stage_node(make_input_node(make_handle<'e', 'a'>(Ud))));
    auto [g2, p] = g1.add(make_combine_node<'e', 'a'>(u, u, AffineTwo{}));
    g2.outputs(p).execute(TeamPolicyTag2<ES>{}, Pc);
  }
  Kokkos::fence();

  auto Hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Pr);
  auto Hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Pc);
  for (int e = 0; e < kE; ++e)
    for (int a = 0; a < kA; ++a) {
      EXPECT_EQ(Hr(e, a), Hc(e, a)) << "e=" << e << " a=" << a;
      const float ref = 2.0f * uval(e, a) - 0.5f * uval(e, a) + 100.0f * e + a;
      EXPECT_NEAR(Hr(e, a), ref, 1e-4f);
    }
}

// ---------------------------------------------------------------------------
// 2. Broadcast operands: h carries (a,i) but not e; u carries (e,a) but not i.
// ---------------------------------------------------------------------------
TEST(ReduceNode, BroadcastOperandsMatchHostLoop) {
  // h staged over (a, i): reuse hval but with the 'a' extent.
  View2 Hd("h_ai", kA, kI);
  {
    auto m = Kokkos::create_mirror_view(Hd);
    for (int a = 0; a < kA; ++a)
      for (int i = 0; i < kI; ++i) m(a, i) = hval(a, i);
    Kokkos::deep_copy(Hd, m);
  }
  auto  Ud = make_u();
  View3 Bd("B", kE, kA, kI);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, h] =
      g0.add(make_stage_node(make_input_node(make_handle<'a', 'i'>(Hd))));
  auto [g2, u] =
      g1.add(make_stage_node(make_input_node(make_handle<'e', 'a'>(Ud))));
  auto [g3, b] = g2.add(make_reduce_node<'e', 'a', 'i'>(h, u, Broadcast{}));
  static_assert(std::is_same_v<member_out_tile_t<std::decay_t<decltype(
                                   g3.levels.template get<2>().template get<0>())>>,
                               StaticTile<kTE, kA, kI>>);
  g3.outputs(b).execute(TeamPolicyTag2<ES>{}, Bd);
  Kokkos::fence();

  auto Bh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Bd);
  double max_err = 0.0;
  for (int e = 0; e < kE; ++e)
    for (int a = 0; a < kA; ++a)
      for (int i = 0; i < kI; ++i) {
        const double ref = static_cast<double>(hval(a, i)) * uval(e, a) +
                           0.001 * (e * 100 + a * 10 + i);
        max_err = std::max(max_err, std::abs(ref - Bh(e, a, i)));
      }
  EXPECT_LT(max_err, 1e-4) << "broadcast reduce != reference";
}

// ---------------------------------------------------------------------------
// 3. One reduction label, a predicate on the output coordinate, one staged
//    operand relabeled twice.
// ---------------------------------------------------------------------------
TEST(ReduceNode, ReductionWithPredicateMatchesHostLoop) {
  auto  Hd = make_h();
  auto  Md = make_m();
  View3 Kd("K", kE, kI, kI);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, h] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'i'>(Hd))));
  auto [g2, m] =
      g1.add(make_stage_node(make_input_node(make_handle<'e', 'q'>(Md))));
  auto node = make_reduce_node<'e', 'i', 'I'>(
      h.template as<'q', 'i'>(), h.template as<'q', 'I'>(), m, DiagonalTerm{});
  static_assert(decltype(node)::RedRank == 1);
  static_assert(std::is_same_v<typename decltype(node)::reduce_seq,
                               std::integer_sequence<int32_t, 'q'>>);
  auto [g3, k] = g2.add(node);
  g3.outputs(k).execute(TeamPolicyTag2<ES>{}, Kd);
  Kokkos::fence();

  auto Kh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Kd);
  double max_err = 0.0, scale = 0.0;
  for (int e = 0; e < kE; ++e)
    for (int i = 0; i < kI; ++i)
      for (int I = 0; I < kI; ++I) {
        double ref = 0.0;
        if ((i + I) % 2 == 0)
          for (int q = 0; q < kQ; ++q)
            ref += static_cast<double>(hval(q, i)) * hval(q, I) * mval(e, q);
        scale   = std::max(scale, std::abs(ref));
        max_err = std::max(max_err, std::abs(ref - Kh(e, i, I)));
      }
  ASSERT_GT(scale, 0.0);
  EXPECT_LT(max_err, 1e-4 * scale) << "reduction != reference";
}

// ---------------------------------------------------------------------------
// 4. Two reduction labels and pinned axes on one staged slot.
// ---------------------------------------------------------------------------
TEST(ReduceNode, TwoReductionLabelsAndPinnedAxes) {
  auto  Hd = make_h();
  auto  M4 = make_m4();
  View3 Kd("K", kE, kI, kI);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, h] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'i'>(Hd))));
  auto [g2, m] = g1.add(
      make_stage_node(make_input_node(make_handle<'e', 'r', 's', 'q'>(M4))));
  auto node = make_reduce_node<'e', 'i', 'I'>(
      h.template as<'q', 'i'>(), h.template as<'q', 'I'>(),
      m.template as<'e', fixed<0>, 's', 'q'>(),
      m.template as<'e', fixed<2>, 's', 'q'>(), TwoRedPinned{});
  static_assert(decltype(node)::RedRank == 2);
  // First appearance order over the operand list: q (from h), then s.
  static_assert(std::is_same_v<typename decltype(node)::reduce_seq,
                               std::integer_sequence<int32_t, 'q', 's'>>);
  auto [g3, k] = g2.add(node);
  g3.outputs(k).execute(TeamPolicyTag2<ES>{}, Kd);
  Kokkos::fence();

  auto Kh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Kd);
  double max_err = 0.0, scale = 0.0;
  for (int e = 0; e < kE; ++e)
    for (int i = 0; i < kI; ++i)
      for (int I = 0; I < kI; ++I) {
        double ref = 0.0;
        for (int s = 0; s < kS; ++s)
          for (int q = 0; q < kQ; ++q) {
            if (q == I) continue;
            ref += static_cast<double>(hval(q, i)) * hval(q, I) *
                   (m4val(e, 0, s, q) + (s + 1) * m4val(e, 2, s, q));
          }
        scale   = std::max(scale, std::abs(ref));
        max_err = std::max(max_err, std::abs(ref - Kh(e, i, I)));
      }
  ASSERT_GT(scale, 0.0);
  EXPECT_LT(max_err, 1e-4 * scale) << "two-label reduction != reference";
}

// ---------------------------------------------------------------------------
// 5. Multi-output reduce.
// ---------------------------------------------------------------------------
TEST(ReduceNode, MultiOutputReduce) {
  auto  Hd = make_h();
  auto  Md = make_m();
  View3 K0("K0", kE, kI, kI), K1("K1", kE, kI, kI);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, h] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'i'>(Hd))));
  auto [g2, m] =
      g1.add(make_stage_node(make_input_node(make_handle<'e', 'q'>(Md))));
  auto [g3, k0, k1] = g2.add(make_reduce_node<'e', 'i', 'I'>(
      h.template as<'q', 'i'>(), h.template as<'q', 'I'>(), m, TwoOut{}));
  g3.outputs(k0, k1).execute(TeamPolicyTag2<ES>{}, K0, K1);
  Kokkos::fence();

  auto H0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, K0);
  auto H1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, K1);
  double e0 = 0.0, e1 = 0.0;
  for (int e = 0; e < kE; ++e)
    for (int i = 0; i < kI; ++i)
      for (int I = 0; I < kI; ++I) {
        double r0 = 0.0, r1 = 0.0;
        for (int q = 0; q < kQ; ++q) {
          r0 += static_cast<double>(hval(q, i)) * hval(q, I) * mval(e, q);
          if (i == I) r1 += static_cast<double>(hval(q, I)) * mval(e, q);
        }
        e0 = std::max(e0, std::abs(r0 - H0(e, i, I)));
        e1 = std::max(e1, std::abs(r1 - H1(e, i, I)));
      }
  EXPECT_LT(e0, 1e-4) << "multi-output reduce o0";
  EXPECT_LT(e1, 1e-4) << "multi-output reduce o1";
}

// ---------------------------------------------------------------------------
// 6. A reduce reading a LEVEL OUTPUT, with the plan guards instantiated.
// ---------------------------------------------------------------------------
TEST(ReduceNode, ReadsALevelOutputAndPassesThePlanGuards) {
  auto  Hd = make_h();
  auto  Md = make_m();
  View3 Kd("K", kE, kI, kI);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, h] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'i'>(Hd))));
  auto [g2, m] =
      g1.add(make_stage_node(make_input_node(make_handle<'e', 'q'>(Md))));
  auto [g3, m2] = g2.add(make_reduce_node<'e', 'q'>(m, Scale{}));
  auto [g4, k]  = g3.add(make_reduce_node<'e', 'i', 'I'>(
      h.template as<'q', 'i'>(), h.template as<'q', 'I'>(), m2,
      DiagonalTerm{}));
  using Plan = LevelPlan<std::decay_t<decltype(g4.levels)>>;
  static_assert(Plan::num_levels == 4);
  static_assert(Plan::num_slots == 4);
  g4.outputs(k).execute(TeamPolicyTag2<ES>{}, Kd);
  Kokkos::fence();

  auto Kh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Kd);
  double max_err = 0.0, scale = 0.0;
  for (int e = 0; e < kE; ++e)
    for (int i = 0; i < kI; ++i)
      for (int I = 0; I < kI; ++I) {
        double ref = 0.0;
        if ((i + I) % 2 == 0)
          for (int q = 0; q < kQ; ++q)
            ref += static_cast<double>(hval(q, i)) * hval(q, I) *
                   (3.0 * mval(e, q) + 1.0);
        scale   = std::max(scale, std::abs(ref));
        max_err = std::max(max_err, std::abs(ref - Kh(e, i, I)));
      }
  ASSERT_GT(scale, 0.0);
  EXPECT_LT(max_err, 1e-4 * scale) << "reduce over a level output";
}

// ---------------------------------------------------------------------------
// 7. Rank-8 output: the shape the element-stiffness block needs
//    (E, k, j, i, b, K, J, I). Tiny extents; the point is that the frame,
//    the broadcast operands and the rank-8 root write all go through.
// ---------------------------------------------------------------------------
namespace r8 {
constexpr int N = 2, NE = 4, TE = 2;
using Map8 = LabelTiles<LabelTile<'E', TE>, LabelWhole<'k', N>,
                        LabelWhole<'j', N>, LabelWhole<'i', N>,
                        LabelWhole<'b', N>, LabelWhole<'K', N>,
                        LabelWhole<'J', N>, LabelWhole<'I', N>,
                        LabelWhole<'q', N>>;
using View8 =
    Kokkos::View<float*[N][N][N][N][N][N][N], Kokkos::LayoutRight, ES>;
using ViewH = Kokkos::View<float**, Kokkos::LayoutRight, ES>;
using ViewM = Kokkos::View<float***, Kokkos::LayoutRight, ES>;

float h8(int q, int i) { return 0.2f + 0.7f * q - 0.3f * i; }
float m8(int E, int b, int q) { return 0.4f + 0.15f * E - 0.25f * b + 0.1f * q; }

struct Term {
  KOKKOS_FUNCTION float operator()(int, int k, int j, int, int, int K, int J,
                                   int, int, float hqi, float hqI,
                                   float m) const {
    return (k == K && j == J) ? hqi * hqI * m : 0.0f;
  }
};
}  // namespace r8

TEST(ReduceNode, RankEightOutput) {
  using namespace r8;
  ViewH Hd("h8", N, N);
  ViewM Md("m8", NE, N, N);
  {
    auto hm = Kokkos::create_mirror_view(Hd);
    for (int q = 0; q < N; ++q)
      for (int i = 0; i < N; ++i) hm(q, i) = h8(q, i);
    Kokkos::deep_copy(Hd, hm);
    auto mm = Kokkos::create_mirror_view(Md);
    for (int E = 0; E < NE; ++E)
      for (int b = 0; b < N; ++b)
        for (int q = 0; q < N; ++q) mm(E, b, q) = m8(E, b, q);
    Kokkos::deep_copy(Md, mm);
  }
  View8 Kd("K8", NE);

  auto g0 = make_level_graph<float, ES>(Map8{});
  auto [g1, h] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'i'>(Hd))));
  auto [g2, m] =
      g1.add(make_stage_node(make_input_node(make_handle<'E', 'b', 'q'>(Md))));
  // k, j, K, J are carried by NO operand: their extents come from the map.
  auto [g3, k] = g2.add(make_reduce_node<'E', 'k', 'j', 'i', 'b', 'K', 'J', 'I'>(
      h.template as<'q', 'i'>(), h.template as<'q', 'I'>(), m, Term{}));
  using Plan8 = LevelPlan<std::decay_t<decltype(g3.levels)>>;
  static_assert(Plan8::num_levels == 3);
  g3.outputs(k).execute(TeamPolicyTag2<ES>{}, Kd);
  Kokkos::fence();

  auto Kh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Kd);
  double max_err = 0.0;
  for (int E = 0; E < NE; ++E)
    for (int k_ = 0; k_ < N; ++k_)
      for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
          for (int b = 0; b < N; ++b)
            for (int K = 0; K < N; ++K)
              for (int J = 0; J < N; ++J)
                for (int I = 0; I < N; ++I) {
                  double ref = 0.0;
                  if (k_ == K && j == J)
                    for (int q = 0; q < N; ++q)
                      ref += static_cast<double>(h8(q, i)) * h8(q, I) *
                             m8(E, b, q);
                  max_err = std::max(
                      max_err, std::abs(ref - Kh(E, k_, j, i, b, K, J, I)));
                }
  EXPECT_LT(max_err, 1e-5) << "rank-8 reduce != reference";
}

// ---------------------------------------------------------------------------
// 8. Scratch level 1. The host backends cap level-0 team scratch at 32 KB;
//    a whole-tile output of 5^6 floats is 62.5 KB. Carving from level 1 is
//    what lets the same tile map run on host and GPU.
// ---------------------------------------------------------------------------
namespace big {
constexpr int N = 5, NE = 3, TE = 1;
using MapB = LabelTiles<LabelTile<'E', TE>, LabelWhole<'k', N>,
                        LabelWhole<'j', N>, LabelWhole<'i', N>,
                        LabelWhole<'K', N>, LabelWhole<'J', N>,
                        LabelWhole<'I', N>, LabelWhole<'q', N>>;
using View7 = Kokkos::View<float*[N][N][N][N][N][N], Kokkos::LayoutRight, ES>;
using ViewH = Kokkos::View<float**, Kokkos::LayoutRight, ES>;
using ViewM = Kokkos::View<float**, Kokkos::LayoutRight, ES>;
float hb(int q, int i) { return 0.2f + 0.7f * q - 0.3f * i; }
float mb(int E, int q) { return 0.4f + 0.15f * E + 0.1f * q; }
struct Term {
  KOKKOS_FUNCTION float operator()(int, int k, int j, int, int K, int J, int,
                                   int, float hqi, float hqI, float m) const {
    return (k == K && j == J) ? hqi * hqI * m : 0.0f;
  }
};
}  // namespace big

TEST(ReduceNode, ScratchLevelOneCarriesAWholeTileAboveTheHostLevelZeroCap) {
  using namespace big;
  ViewH Hd("hb", N, N);
  ViewM Md("mb", NE, N);
  {
    auto hm = Kokkos::create_mirror_view(Hd);
    for (int q = 0; q < N; ++q)
      for (int i = 0; i < N; ++i) hm(q, i) = hb(q, i);
    Kokkos::deep_copy(Hd, hm);
    auto mm = Kokkos::create_mirror_view(Md);
    for (int E = 0; E < NE; ++E)
      for (int q = 0; q < N; ++q) mm(E, q) = mb(E, q);
    Kokkos::deep_copy(Md, mm);
  }
  View7 Kd("K7", NE);

  auto g0 = make_level_graph<float, ES>(MapB{});
  auto [g1, h] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'i'>(Hd))));
  auto [g2, m] =
      g1.add(make_stage_node(make_input_node(make_handle<'E', 'q'>(Md))));
  auto [g3, k] = g2.add(make_reduce_node<'E', 'k', 'j', 'i', 'K', 'J', 'I'>(
      h.template as<'q', 'i'>(), h.template as<'q', 'I'>(), m, Term{}));
  const auto out = g3.outputs(k);
  EXPECT_GT(out.scratch_bytes(), std::size_t{32} * 1024)
      << "the case must exceed the host level-0 cap to prove anything";
  constexpr bool on_host =
      Kokkos::SpaceAccessibility<ES, Kokkos::HostSpace>::accessible;
  out.scratch_level(on_host ? 1 : 0).execute(TeamPolicyTag2<ES>{}, Kd);
  Kokkos::fence();

  auto Kh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Kd);
  double max_err = 0.0;
  for (int E = 0; E < NE; ++E)
    for (int k_ = 0; k_ < N; ++k_)
      for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
          for (int K = 0; K < N; ++K)
            for (int J = 0; J < N; ++J)
              for (int I = 0; I < N; ++I) {
                double ref = 0.0;
                if (k_ == K && j == J)
                  for (int q = 0; q < N; ++q)
                    ref += static_cast<double>(hb(q, i)) * hb(q, I) * mb(E, q);
                max_err = std::max(max_err,
                                   std::abs(ref - Kh(E, k_, j, i, K, J, I)));
              }
  EXPECT_LT(max_err, 1e-4) << "level-1 scratch reduce != reference";
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
