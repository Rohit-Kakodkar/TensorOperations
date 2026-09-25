// ===========================================================================
// test_general_contraction.cpp -- make_contraction_node beyond the binary
// GEMM: the labels decide what is summed.
//
// Every case is held to a host loop over the same formula. Extents are
// pairwise distinct wherever two labels could be confused (a transposed
// operand cannot hide behind two equal axes), and the gridded element label
// has more than one tile so a dropped tile origin shows up.
//
//   1. matmul with a batch label carried by one operand -- a valid binary
//      contraction, so it dispatches to the GEMM; the same product through
//      outer(a) takes the general path, and both must agree with the host;
//   2. a batch (Hadamard) label shared by operands and output;
//   3. a four-operand contraction whose operands are ALL functional inputs,
//      so the grid's extent comes from a general-contraction leaf, not a
//      stage;
//   4. structured operands: the 3D stiffness pattern K = B^T M B with the
//      gradient B = stack<'r'>(outer(h, delta, delta), ...), i.e.
//        K = sum_{r,s,z,y,x} B_r(zyx, kji) M(e,a,b,r,s,zyx) B_s(zyx, nml)
//      against the same expression with a DENSE B and against host loops,
//      into a rank-9 StridedAlias on team scratch level 1, with the term
//      structure the delta elimination must produce pinned at compile time;
//   5. VALUE LABELS: with a and b LabelWhole the stiffness node's value labels
//      are exactly {a, b} and the level runs in-register value blocks; the
//      result must be BITWISE the gridded-(a, b) graph's, which runs entry by
//      entry. The same for a functional value leaf between two other leaves,
//      and for a value leaf that differs between the terms of a stack;
//   6. STREAMED ROOTS: a general-contraction root nobody reads is written
//      straight to its output view -- bitwise the same values as when a later
//      member reads it (so it lives in scratch and is copied out), with the
//      scratch request smaller by exactly its tile; and a level of several
//      streamed members, blocked only when their value labels agree.
// ===========================================================================
#include <TensorOperations/GeneralContraction.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/LevelPlan.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/StridedAlias.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace TensorOperations;
using ES = Kokkos::DefaultExecutionSpace;

namespace {

float aval(int e, int i, int j) {
  return 0.3f + 0.07f * e - 0.11f * i + 0.05f * j + 0.01f * e * j;
}
float bval(int j, int k) {
  return -0.2f + 0.13f * j + 0.09f * k - 0.02f * j * k;
}
float wval(int e, int j) { return 1.0f + 0.1f * e - 0.03f * j; }

// --- 1. matmul with a batch label on one operand ---------------------------

TEST(GeneralContraction, MatmulWithBatchLabel) {
  constexpr int fE = 6, fTE = 2, fI = 4, fJ = 5, fK = 3;
  using Map = LabelTiles<LabelTile<'e', fTE>, LabelWhole<'i', fI>,
                         LabelWhole<'j', fJ>, LabelWhole<'k', fK>>;

  Kokkos::View<float***, Kokkos::LayoutRight, ES> A("A", fE, fI, fJ);
  Kokkos::View<float**, Kokkos::LayoutRight, ES>  B("B", fJ, fK);
  Kokkos::View<float***, Kokkos::LayoutRight, ES> C("C", fE, fI, fK);
  auto Ah = Kokkos::create_mirror_view(A);
  auto Bh = Kokkos::create_mirror_view(B);
  for (int e = 0; e < fE; ++e)
    for (int i = 0; i < fI; ++i)
      for (int j = 0; j < fJ; ++j) Ah(e, i, j) = aval(e, i, j);
  for (int j = 0; j < fJ; ++j)
    for (int k = 0; k < fK; ++k) Bh(j, k) = bval(j, k);
  Kokkos::deep_copy(A, Ah);
  Kokkos::deep_copy(B, Bh);

  auto g0 = make_level_graph<float, ES>(Map{});
  // One stage level per tile shape: a stage level is one copy range.
  auto [g1, a] =
      g0.add(make_stage_node(make_input_node(make_handle<'e', 'i', 'j'>(A))));
  auto [g2, b] =
      g1.add(make_stage_node(make_input_node(make_handle<'j', 'k'>(B))));
  // Two node operands, each output label on exactly one: the binary GEMM.
  // A one-operand outer(a) is the same tensor as a, but structured, so the
  // same product through it is a general contraction.
  const auto gemm = make_contraction_node<'e', 'i', 'k'>(a, b);
  const auto gen  = make_contraction_node<'e', 'i', 'k'>(outer(a), b);
  static_assert(Impl::has_node_tag_v<ContractionTag, decltype(gemm)>);
  static_assert(Impl::has_node_tag_v<GeneralContractionTag, decltype(gen)>);

  auto [g3, c] = g2.add(gemm);
  using Plan   = LevelPlan<std::decay_t<decltype(g3.levels)>>;
  static_assert(Plan::num_levels == 3);
  auto [g3g, cg] = g2.add(gen);
  using PlanG    = LevelPlan<std::decay_t<decltype(g3g.levels)>>;
  static_assert(PlanG::num_levels == 3);

  Kokkos::View<float***, Kokkos::LayoutRight, ES> CG("CG", fE, fI, fK);
  g3.outputs(c).execute(TeamPolicyTag<ES>{}, C);
  g3g.outputs(cg).execute(TeamPolicyTag<ES>{}, CG);
  Kokkos::fence();

  auto Ch  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  auto CGh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, CG);
  for (int e = 0; e < fE; ++e)
    for (int i = 0; i < fI; ++i)
      for (int k = 0; k < fK; ++k) {
        float ref = 0;
        for (int j = 0; j < fJ; ++j) ref += aval(e, i, j) * bval(j, k);
        EXPECT_NEAR(Ch(e, i, k), ref, 1e-5f) << e << " " << i << " " << k;
        EXPECT_NEAR(CGh(e, i, k), ref, 1e-5f) << e << " " << i << " " << k;
      }
}

// --- 2. a batch label shared by operands and output -------------------------

TEST(GeneralContraction, SharedBatchLabel) {
  constexpr int fE = 6, fTE = 3, fI = 4, fJ = 5;
  using Map =
      LabelTiles<LabelTile<'e', fTE>, LabelWhole<'i', fI>, LabelWhole<'j', fJ>>;

  Kokkos::View<float***, Kokkos::LayoutRight, ES> A("A", fE, fI, fJ);
  Kokkos::View<float**, Kokkos::LayoutRight, ES>  W("W", fE, fJ);
  Kokkos::View<float**, Kokkos::LayoutRight, ES>  C("C", fE, fI);
  auto Ah = Kokkos::create_mirror_view(A);
  auto Wh = Kokkos::create_mirror_view(W);
  for (int e = 0; e < fE; ++e)
    for (int j = 0; j < fJ; ++j) {
      Wh(e, j) = wval(e, j);
      for (int i = 0; i < fI; ++i) Ah(e, i, j) = aval(e, i, j);
    }
  Kokkos::deep_copy(A, Ah);
  Kokkos::deep_copy(W, Wh);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, a] =
      g0.add(make_stage_node(make_input_node(make_handle<'e', 'i', 'j'>(A))));
  auto [g2, w] =
      g1.add(make_stage_node(make_input_node(make_handle<'e', 'j'>(W))));
  // 'e' is on both operands AND in the output: not a GEMM.
  const auto node = make_contraction_node<'e', 'i'>(a, w);
  static_assert(Impl::has_node_tag_v<GeneralContractionTag, decltype(node)>);
  auto [g3, c] = g2.add(node);
  g3.outputs(c).execute(TeamPolicyTag<ES>{}, C);
  Kokkos::fence();

  auto Ch = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  for (int e = 0; e < fE; ++e)
    for (int i = 0; i < fI; ++i) {
      float ref = 0;
      for (int j = 0; j < fJ; ++j) ref += aval(e, i, j) * wval(e, j);
      EXPECT_NEAR(Ch(e, i), ref, 1e-5f) << e << " " << i;
    }
}

// --- 3. four functional operands, the grid carried only by them ------------

KOKKOS_INLINE_FUNCTION float xfun(int e, int c, int a) {
  return 0.5f + 0.1f * e + 0.2f * c - 0.15f * a + 0.03f * c * a;
}
KOKKOS_INLINE_FUNCTION float cfun(int e, int a, int c, int b, int d) {
  return 1.0f + 0.01f * e + 0.1f * (a == c) + 0.2f * (b == d) + 0.05f * a * b -
         0.02f * c * d;
}
KOKKOS_INLINE_FUNCTION float wfun(int e) { return 0.25f + 0.125f * e; }

struct XFn {
  KOKKOS_INLINE_FUNCTION float operator()(int e, int c, int a) const {
    return xfun(e, c, a);
  }
};
struct CFn {
  KOKKOS_INLINE_FUNCTION float operator()(int e, int a, int c, int b,
                                          int d) const {
    return cfun(e, a, c, b, d);
  }
};
struct WFn {
  KOKKOS_INLINE_FUNCTION float operator()(int e) const { return wfun(e); }
};

TEST(GeneralContraction, FunctionalOperandsCarryTheGrid) {
  constexpr int fE = 6, fTE = 2, fA = 3, fB = 2, fC = 4, fD = 5;
  // a and b differ from c and d so a transposed functor argument shows up.
  using Map =
      LabelTiles<LabelTile<'e', fTE>, LabelWhole<'a', fA>, LabelWhole<'b', fB>,
                 LabelWhole<'c', fC>, LabelWhole<'d', fD>>;

  auto x1 = make_functional_input_node<'e', 'c', 'a'>(
      Kokkos::Array<int, 3>{fE, fC, fA}, XFn{});
  auto x2 = make_functional_input_node<'e', 'd', 'b'>(
      Kokkos::Array<int, 3>{fE, fD, fB}, XFn{});
  auto cc = make_functional_input_node<'e', 'a', 'c', 'b', 'd'>(
      Kokkos::Array<int, 5>{fE, fA, fC, fB, fD}, CFn{});
  auto ww = make_functional_input_node<'e'>(Kokkos::Array<int, 1>{fE}, WFn{});

  Kokkos::View<float***, Kokkos::LayoutRight, ES> M("M", fE, fA, fB);
  auto g0      = make_level_graph<float, ES>(Map{});
  auto [g1, m] = g0.add(make_contraction_node<'e', 'a', 'b'>(x1, cc, x2, ww));
  g1.outputs(m).execute(TeamPolicyTag<ES>{}, M);
  Kokkos::fence();

  auto Mh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, M);
  for (int e = 0; e < fE; ++e)
    for (int a = 0; a < fA; ++a)
      for (int b = 0; b < fB; ++b) {
        float ref = 0;
        for (int c = 0; c < fC; ++c)
          for (int d = 0; d < fD; ++d)
            ref +=
                xfun(e, c, a) * cfun(e, a, c, b, d) * xfun(e, d, b) * wfun(e);
        EXPECT_NEAR(Mh(e, a, b), ref, 1e-4f * std::fabs(ref))
            << e << " " << a << " " << b;
      }
}

// --- 4. structured operands: the 3D stiffness pattern ----------------------

// Distinct extents per direction: x (i, l) 3, y (j, m) 4, z (k, n) 2.
constexpr int kNX = 3, kNY = 4, kNZ = 2, kNE = 3, kNC = 2;

float hx_val(int q, int f) {
  return 0.2f + 0.3f * q - 0.25f * f + 0.07f * q * f;
}
float hy_val(int q, int f) {
  return -0.1f + 0.2f * q + 0.15f * f - 0.05f * q * f;
}
float hz_val(int q, int f) {
  return 0.4f - 0.3f * q + 0.35f * f + 0.1f * q * f;
}
KOKKOS_INLINE_FUNCTION float mval(int e, int a, int b, int r, int s, int z,
                                  int y, int x) {
  return 1.0f + 0.1f * e + 0.2f * a - 0.3f * b + 0.05f * r - 0.07f * s +
         0.011f * z + 0.013f * y + 0.017f * x + 0.01f * r * s + 0.02f * a * b;
}
struct MFn {
  KOKKOS_INLINE_FUNCTION float operator()(int e, int a, int b, int r, int s,
                                          int z, int y, int x) const {
    return mval(e, a, b, r, s, z, y, x);
  }
};

template <typename H>
struct DenseD {
  H hx, hy, hz;
  // B(r, z, y, x, k, j, i) = h_r(q_r, i_r) prod_{t != r} delta(q_t, i_t).
  KOKKOS_INLINE_FUNCTION float operator()(int r, int z, int y, int x, int k,
                                          int j, int i) const {
    if (r == 0) return (y == j && z == k) ? hx(x, i) : 0.0f;
    if (r == 1) return (x == i && z == k) ? hy(y, j) : 0.0f;
    return (x == i && y == j) ? hz(z, k) : 0.0f;
  }
};

float dref(int r, int z, int y, int x, int k, int j, int i) {
  if (r == 0) return (y == j && z == k) ? hx_val(x, i) : 0.0f;
  if (r == 1) return (x == i && z == k) ? hy_val(y, j) : 0.0f;
  return (x == i && y == j) ? hz_val(z, k) : 0.0f;
}

using StiffnessMap =
    LabelTiles<LabelTile<'e', 1>, LabelTile<'a', 1>, LabelTile<'b', 1>,
               LabelWhole<'k', kNZ>, LabelWhole<'j', kNY>, LabelWhole<'i', kNX>,
               LabelWhole<'n', kNZ>, LabelWhole<'m', kNY>, LabelWhole<'l', kNX>,
               LabelWhole<'r', 3>, LabelWhole<'s', 3>, LabelWhole<'z', kNZ>,
               LabelWhole<'y', kNY>, LabelWhole<'x', kNX>>;

using H2 = Kokkos::View<float**, Kokkos::LayoutRight, ES>;

struct StiffnessInputs {
  H2 hx{"hx", kNX, kNX}, hy{"hy", kNY, kNY}, hz{"hz", kNZ, kNZ};
  StiffnessInputs() {
    auto fill = [](H2 v, float (*f)(int, int)) {
      auto h = Kokkos::create_mirror_view(v);
      for (int q = 0; q < static_cast<int>(v.extent(0)); ++q)
        for (int p = 0; p < static_cast<int>(v.extent(1)); ++p)
          h(q, p) = f(q, p);
      Kokkos::deep_copy(v, h);
    };
    fill(hx, hx_val);
    fill(hy, hy_val);
    fill(hz, hz_val);
  }
};

auto stiffness_m_node() {
  return make_stage_node(
      make_functional_input_node<'e', 'a', 'b', 'r', 's', 'z', 'y', 'x'>(
          Kokkos::Array<int, 8>{kNE, kNC, kNC, 3, 3, kNZ, kNY, kNX}, MFn{}));
}

using KAlias = StridedAlias<float, ES, kNC, kNZ, kNY, kNX, kNC, kNZ, kNY, kNX>;
constexpr int kBlock = kNC * kNZ * kNY * kNX * kNC * kNZ * kNY * kNX;

float kref(int e, int a, int k, int j, int i, int b, int n, int m, int l) {
  float acc = 0;
  for (int r = 0; r < 3; ++r)
    for (int s = 0; s < 3; ++s)
      for (int z = 0; z < kNZ; ++z)
        for (int y = 0; y < kNY; ++y)
          for (int x = 0; x < kNX; ++x)
            acc += dref(r, z, y, x, k, j, i) * mval(e, a, b, r, s, z, y, x) *
                   dref(s, z, y, x, n, m, l);
  return acc;
}

void check_stiffness(const Kokkos::View<float*, ES>& out, const char* what) {
  auto h   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  int  bad = 0;
  for (int e = 0; e < kNE; ++e)
    for (int a = 0; a < kNC; ++a)
      for (int k = 0; k < kNZ; ++k)
        for (int j = 0; j < kNY; ++j)
          for (int i = 0; i < kNX; ++i)
            for (int b = 0; b < kNC; ++b)
              for (int n = 0; n < kNZ; ++n)
                for (int m = 0; m < kNY; ++m)
                  for (int l = 0; l < kNX; ++l) {
                    const int idx =
                        ((((((((e * kNC + a) * kNZ + k) * kNY + j) * kNX + i) *
                                kNC +
                            b) *
                               kNZ +
                           n) *
                              kNY +
                          m) *
                             kNX +
                         l);
                    const float ref = kref(e, a, k, j, i, b, n, m, l);
                    if (std::fabs(h(idx) - ref) >
                            1e-4f * (1 + std::fabs(ref)) &&
                        bad++ < 5)
                      ADD_FAILURE()
                          << what << " at (" << e << a << k << j << i << b << n
                          << m << l << "): " << h(idx) << " vs " << ref;
                  }
  EXPECT_EQ(bad, 0) << what;
}

TEST(GeneralContraction, StackedDeltasSumFactorTheStiffness) {
  StiffnessInputs in;
  auto            g0 = make_level_graph<float, ES>(StiffnessMap{});
  auto [g1a, hx] =
      g0.add(make_stage_node(make_input_node(make_handle<'x', 'i'>(in.hx))));
  auto [g1b, hy] =
      g1a.add(make_stage_node(make_input_node(make_handle<'y', 'j'>(in.hy))));
  auto [g1, hz] =
      g1b.add(make_stage_node(make_input_node(make_handle<'z', 'k'>(in.hz))));
  auto [g2, M] = g1.add(stiffness_m_node());

  // B_r(z, y, x; k, j, i): the derivative of the tensor-product basis
  // function (k, j, i) along reference direction r, at point (z, y, x) --
  // branch r is h along r times the identity along the other two.
  const auto B = stack<'r'>(outer(hx, delta<'y', 'j'>(), delta<'z', 'k'>()),
                            outer(delta<'x', 'i'>(), hy, delta<'z', 'k'>()),
                            outer(delta<'x', 'i'>(), delta<'y', 'j'>(), hz));
  // B's labels are 'r' then branch 0's: (r, x, i, y, j, z, k). The second
  // factor is B relabelled positionally r->s, i->l, j->m, k->n.
  using BModes = std::decay_t<decltype(B)>::modes_seq;
  static_assert(
      std::is_same_v<BModes, std::integer_sequence<int32_t, 'r', 'x', 'i', 'y',
                                                   'j', 'z', 'k'>>);
  const auto Bt = B.as<'s', 'x', 'l', 'y', 'm', 'z', 'n'>();

  auto [g3, K] =
      g2.add(make_contraction_node<'e', 'a', 'k', 'j', 'i', 'b', 'n', 'm', 'l'>(
          B, M, Bt));

  // The delta elimination, pinned: 9 terms, (r, s) with r outermost; the
  // three with r == s sum one quadrature coordinate and test two directions,
  // the six with r != s sum nothing and test one.
  using KNode =
      std::decay_t<decltype(g3.levels.template get<4>().template get<0>())>;
  static_assert(KNode::NumTerms == 9);
  static_assert(KNode::template term_info<0>::num_loop == 1);  // (x, x)
  static_assert(KNode::template term_info<1>::num_loop == 0);  // (x, y)
  static_assert(KNode::template term_info<4>::num_loop == 1);  // (y, y)
  static_assert(KNode::template term_info<8>::num_loop == 1);  // (z, z)
  static_assert(KNode::template term_info<0>::chk_a_seq::size() == 2);
  static_assert(KNode::template term_info<1>::chk_a_seq::size() == 1);
  static_assert(
      std::is_same_v<typename KNode::template term_info<0>::loop_ext_seq,
                     std::integer_sequence<int, kNX>>);
  static_assert(
      std::is_same_v<typename KNode::template term_info<4>::loop_ext_seq,
                     std::integer_sequence<int, kNY>>);
  using Plan = LevelPlan<std::decay_t<decltype(g3.levels)>>;
  static_assert(Plan::num_levels == 5);

  Kokkos::View<float*, ES> out("K", kNE * kBlock);
  Kokkos::deep_copy(out, std::nanf(""));
  g3.outputs(K).scratch_level(1).execute(TeamPolicyTag<ES>{},
                                         KAlias{out.data(), kNE});
  Kokkos::fence();
  check_stiffness(out, "stacked B");
}

TEST(GeneralContraction, DenseOperandIsTheSameExpression) {
  StiffnessInputs in;
  auto            g0 = make_level_graph<float, ES>(StiffnessMap{});
  auto [g1, M]       = g0.add(stiffness_m_node());

  const DenseD<H2> dfn{in.hx, in.hy, in.hz};
  auto Dr = make_functional_input_node<'r', 'z', 'y', 'x', 'k', 'j', 'i'>(
      Kokkos::Array<int, 7>{3, kNZ, kNY, kNX, kNZ, kNY, kNX}, dfn);
  auto Ds = make_functional_input_node<'s', 'z', 'y', 'x', 'n', 'm', 'l'>(
      Kokkos::Array<int, 7>{3, kNZ, kNY, kNX, kNZ, kNY, kNX}, dfn);

  auto [g2, K] =
      g1.add(make_contraction_node<'e', 'a', 'k', 'j', 'i', 'b', 'n', 'm', 'l'>(
          Dr, M, Ds));
  using KNode =
      std::decay_t<decltype(g2.levels.template get<1>().template get<0>())>;
  static_assert(KNode::NumTerms == 1);
  static_assert(KNode::template term_info<0>::num_loop == 5);  // r s z y x

  Kokkos::View<float*, ES> out("K", kNE * kBlock);
  Kokkos::deep_copy(out, std::nanf(""));
  g2.outputs(K).scratch_level(1).execute(TeamPolicyTag<ES>{},
                                         KAlias{out.data(), kNE});
  Kokkos::fence();
  check_stiffness(out, "dense B");
}

// --- 5. value labels: the in-register block ---------------------------------

// The stiffness map with a and b whole: every team holds one element's full
// 2 x 2 block of (a, b) sub-blocks.
using StiffnessWholeABMap =
    LabelTiles<LabelTile<'e', 1>, LabelWhole<'a', kNC>, LabelWhole<'b', kNC>,
               LabelWhole<'k', kNZ>, LabelWhole<'j', kNY>, LabelWhole<'i', kNX>,
               LabelWhole<'n', kNZ>, LabelWhole<'m', kNY>, LabelWhole<'l', kNX>,
               LabelWhole<'r', 3>, LabelWhole<'s', 3>, LabelWhole<'z', kNZ>,
               LabelWhole<'y', kNY>, LabelWhole<'x', kNX>>;

// The stacked-delta stiffness graph of case 4 over map `Map`: (graph, K).
template <typename Map>
auto stiffness_graph(const StiffnessInputs& in) {
  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1a, hx] =
      g0.add(make_stage_node(make_input_node(make_handle<'x', 'i'>(in.hx))));
  auto [g1b, hy] =
      g1a.add(make_stage_node(make_input_node(make_handle<'y', 'j'>(in.hy))));
  auto [g1, hz] =
      g1b.add(make_stage_node(make_input_node(make_handle<'z', 'k'>(in.hz))));
  auto [g2, M]  = g1.add(stiffness_m_node());
  const auto B  = stack<'r'>(outer(hx, delta<'y', 'j'>(), delta<'z', 'k'>()),
                             outer(delta<'x', 'i'>(), hy, delta<'z', 'k'>()),
                             outer(delta<'x', 'i'>(), delta<'y', 'j'>(), hz));
  const auto Bt = B.template as<'s', 'x', 'l', 'y', 'm', 'z', 'n'>();
  return g2.add(
      make_contraction_node<'e', 'a', 'k', 'j', 'i', 'b', 'n', 'm', 'l'>(B, M,
                                                                         Bt));
}
template <typename Graph>
using stiffness_level_t = std::decay_t<
    decltype(std::declval<const Graph&>().levels.template get<4>())>;
template <typename Graph>
using stiffness_node_t = std::decay_t<decltype(std::declval<const Graph&>()
                                                   .levels.template get<4>()
                                                   .template get<0>())>;

template <typename Graph, typename Handle>
Kokkos::View<float*, ES> run_stiffness(const Graph& g, const Handle& K) {
  Kokkos::View<float*, ES> out("K", kNE * kBlock);
  Kokkos::deep_copy(out, std::nanf(""));
  g.outputs(K).scratch_level(1).execute(TeamPolicyTag<ES>{},
                                        KAlias{out.data(), kNE});
  Kokkos::fence();
  return out;
}

// Entries that differ bit for bit (a NaN -- an entry never written -- counts).
template <typename ViewA, typename ViewB>
int bitwise_mismatches(const ViewA& a, const ViewB& b) {
  auto         ah = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a);
  auto         bh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b);
  const float* pa = ah.data();
  const float* pb = bh.data();
  int          n  = 0;
  for (std::size_t k = 0; k < ah.size(); ++k)
    if (!(pa[k] == pb[k])) ++n;
  return n;
}

TEST(GeneralContraction, ValueLabelsBlockTheStiffnessBitwise) {
  StiffnessInputs in;
  auto [gg, Kg] = stiffness_graph<StiffnessMap>(in);
  auto [gw, Kw] = stiffness_graph<StiffnessWholeABMap>(in);
  using GridG   = std::decay_t<decltype(gg)>;
  using GridW   = std::decay_t<decltype(gw)>;

  // Gridded a and b are not value labels: that graph runs entry by entry.
  static_assert(
      std::is_same_v<typename stiffness_node_t<GridG>::value_labels_seq,
                     std::integer_sequence<int32_t>>);
  static_assert(!Impl::lg_gc_level_blocked_v<stiffness_level_t<GridG>>);
  // Whole, they are: only M carries them, and no delta touches them. The
  // gridded e and the delta-bound k j i n m l are not.
  static_assert(
      std::is_same_v<typename stiffness_node_t<GridW>::value_labels_seq,
                     std::integer_sequence<int32_t, 'a', 'b'>>);
  static_assert(std::is_same_v<typename stiffness_node_t<GridW>::value_dims_seq,
                               std::integer_sequence<int, 1, 5>>);
  static_assert(Impl::lg_gc_level_blocked_v<stiffness_level_t<GridW>>);
  // The value leaf is M, the middle leaf of every term: (h_r, M, h_s).
  static_assert(Impl::gc_value_leaf_pos_v<
                    typename stiffness_node_t<GridW>::structure_type,
                    StiffnessWholeABMap, 0> == 1);

  const auto og = run_stiffness(gg, Kg);
  const auto ow = run_stiffness(gw, Kw);
  check_stiffness(ow, "value-blocked");
  EXPECT_EQ(bitwise_mismatches(og, ow), 0)
      << "the value-blocked stiffness must equal the entry-by-entry one bit "
         "for bit";
}

// C(e,i,k) = sum_j W(e,j) F(e,i,j) G(j,k), F functional. Only F carries i and
// only G carries k: i is the value label (accepted first, on F) and k is not
// (it would need G). So the value leaf is functional and sits BETWEEN two
// other leaves -- a prefix W and a suffix G.
constexpr int vE = 6, vTE = 2, vI = 5, vJ = 4, vK = 3;

KOKKOS_INLINE_FUNCTION float ffun(int e, int i, int j) {
  return 0.4f + 0.05f * e - 0.13f * i + 0.07f * j + 0.011f * i * j;
}
struct FFn {
  KOKKOS_INLINE_FUNCTION float operator()(int e, int i, int j) const {
    return ffun(e, i, j);
  }
};

template <typename Map, typename ExpectedValueLabels>
Kokkos::View<float***, Kokkos::LayoutRight, ES> functional_value_case(
    const Kokkos::View<float**, Kokkos::LayoutRight, ES>& W,
    const Kokkos::View<float**, Kokkos::LayoutRight, ES>& G) {
  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, w] =
      g0.add(make_stage_node(make_input_node(make_handle<'e', 'j'>(W))));
  auto [g2, gk] =
      g1.add(make_stage_node(make_input_node(make_handle<'j', 'k'>(G))));
  auto f = make_functional_input_node<'e', 'i', 'j'>(
      Kokkos::Array<int, 3>{vE, vI, vJ}, FFn{});
  auto [g3, c] = g2.add(make_contraction_node<'e', 'i', 'k'>(w, f, gk));
  using Node =
      std::decay_t<decltype(g3.levels.template get<2>().template get<0>())>;
  static_assert(
      std::is_same_v<typename Node::value_labels_seq, ExpectedValueLabels>);

  Kokkos::View<float***, Kokkos::LayoutRight, ES> C("C", vE, vI, vK);
  Kokkos::deep_copy(C, std::nanf(""));
  g3.outputs(c).execute(TeamPolicyTag<ES>{}, C);
  Kokkos::fence();
  return C;
}

TEST(GeneralContraction, FunctionalValueLeafBetweenTwoLeaves) {
  Kokkos::View<float**, Kokkos::LayoutRight, ES> W("W", vE, vJ);
  Kokkos::View<float**, Kokkos::LayoutRight, ES> G("G", vJ, vK);
  auto Wh = Kokkos::create_mirror_view(W);
  auto Gh = Kokkos::create_mirror_view(G);
  for (int e = 0; e < vE; ++e)
    for (int j = 0; j < vJ; ++j) Wh(e, j) = wval(e, j);
  for (int j = 0; j < vJ; ++j)
    for (int k = 0; k < vK; ++k) Gh(j, k) = bval(j, k);
  Kokkos::deep_copy(W, Wh);
  Kokkos::deep_copy(G, Gh);

  using Whole = LabelTiles<LabelTile<'e', vTE>, LabelWhole<'i', vI>,
                           LabelWhole<'j', vJ>, LabelWhole<'k', vK>>;
  // i and k gridded with a single tile each: the same problem, no value
  // labels. (Gridding i alone would make k the value label, on G.)
  using Gridded = LabelTiles<LabelTile<'e', vTE>, LabelTile<'i', vI>,
                             LabelWhole<'j', vJ>, LabelTile<'k', vK>>;
  const auto Cb =
      functional_value_case<Whole, std::integer_sequence<int32_t, 'i'>>(W, G);
  const auto Cu =
      functional_value_case<Gridded, std::integer_sequence<int32_t>>(W, G);

  EXPECT_EQ(bitwise_mismatches(Cb, Cu), 0);
  auto Ch = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Cb);
  for (int e = 0; e < vE; ++e)
    for (int i = 0; i < vI; ++i)
      for (int k = 0; k < vK; ++k) {
        float ref = 0;
        for (int j = 0; j < vJ; ++j)
          ref += wval(e, j) * ffun(e, i, j) * bval(j, k);
        EXPECT_NEAR(Ch(e, i, k), ref, 1e-5f) << e << " " << i << " " << k;
      }
}

// C(e,i) = sum_r w(e,r) stack<'r'>(a, b)(r,e,i) = w(e,0) a(e,i) + w(e,1)
// b(e,i): two terms whose value leaf is a in one and b in the other.
template <typename Map, typename ExpectedValueLabels>
Kokkos::View<float**, Kokkos::LayoutRight, ES> stacked_value_case(
    const Kokkos::View<float**, Kokkos::LayoutRight, ES>& A,
    const Kokkos::View<float**, Kokkos::LayoutRight, ES>& B,
    const Kokkos::View<float**, Kokkos::LayoutRight, ES>& Wr) {
  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, a, b] =
      g0.add(make_stage_node(make_input_node(make_handle<'e', 'i'>(A))),
             make_stage_node(make_input_node(make_handle<'e', 'i'>(B))));
  auto [g2, w] =
      g1.add(make_stage_node(make_input_node(make_handle<'e', 'r'>(Wr))));
  auto [g3, c] = g2.add(make_contraction_node<'e', 'i'>(stack<'r'>(a, b), w));
  using Node =
      std::decay_t<decltype(g3.levels.template get<2>().template get<0>())>;
  static_assert(Node::NumTerms == 2);
  static_assert(
      std::is_same_v<typename Node::value_labels_seq, ExpectedValueLabels>);

  Kokkos::View<float**, Kokkos::LayoutRight, ES> C("C", vE, vI);
  Kokkos::deep_copy(C, std::nanf(""));
  g3.outputs(c).execute(TeamPolicyTag<ES>{}, C);
  Kokkos::fence();
  return C;
}

TEST(GeneralContraction, ValueLeafMayDifferBetweenTerms) {
  Kokkos::View<float**, Kokkos::LayoutRight, ES> A("A", vE, vI);
  Kokkos::View<float**, Kokkos::LayoutRight, ES> B("B", vE, vI);
  Kokkos::View<float**, Kokkos::LayoutRight, ES> Wr("Wr", vE, 2);
  auto Ah = Kokkos::create_mirror_view(A);
  auto Bh = Kokkos::create_mirror_view(B);
  auto Rh = Kokkos::create_mirror_view(Wr);
  for (int e = 0; e < vE; ++e) {
    for (int i = 0; i < vI; ++i) {
      Ah(e, i) = aval(e, i, 1);
      Bh(e, i) = aval(e, i, 3);
    }
    for (int r = 0; r < 2; ++r) Rh(e, r) = wval(e, r);
  }
  Kokkos::deep_copy(A, Ah);
  Kokkos::deep_copy(B, Bh);
  Kokkos::deep_copy(Wr, Rh);

  using Whole =
      LabelTiles<LabelTile<'e', vTE>, LabelWhole<'i', vI>, LabelWhole<'r', 2>>;
  using Gridded =
      LabelTiles<LabelTile<'e', vTE>, LabelTile<'i', vI>, LabelWhole<'r', 2>>;
  const auto Cb =
      stacked_value_case<Whole, std::integer_sequence<int32_t, 'i'>>(A, B, Wr);
  const auto Cu =
      stacked_value_case<Gridded, std::integer_sequence<int32_t>>(A, B, Wr);

  EXPECT_EQ(bitwise_mismatches(Cb, Cu), 0);
  auto Ch = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Cb);
  for (int e = 0; e < vE; ++e)
    for (int i = 0; i < vI; ++i) {
      const float ref = wval(e, 0) * aval(e, i, 1) + wval(e, 1) * aval(e, i, 3);
      EXPECT_NEAR(Ch(e, i), ref, 1e-5f) << e << " " << i;
    }
}

// --- 6. streamed roots -------------------------------------------------------

// A sink that copies K into a second rank-9 buffer: a later member READING K,
// which is all it takes to keep K in scratch.
struct CopyK {
  KAlias                      out;
  KOKKOS_INLINE_FUNCTION void operator()(int e, int a, int k, int j, int i,
                                         int b, int n, int m, int l,
                                         float v) const {
    out(e, a, k, j, i, b, n, m, l) = v;
  }
};

template <typename Map>
void streamed_root_case(const char* what) {
  StiffnessInputs in;
  // A: K is a root nobody reads -- streamed.
  auto [ga, Ka] = stiffness_graph<Map>(in);
  // B: the same K, also read by a later sink -- computed into scratch, copied
  // out by the root store.
  auto [gb0, Kb] = stiffness_graph<Map>(in);
  Kokkos::View<float*, ES> copy("Kcopy", kNE * kBlock);
  Kokkos::deep_copy(copy, std::nanf(""));
  auto gb =
      gb0.add(make_combine_node<'e', 'a', 'k', 'j', 'i', 'b', 'n', 'm', 'l'>(
          Kb, CopyK{KAlias{copy.data(), kNE}}));

  constexpr std::size_t S = std::decay_t<decltype(Ka)>::SlotIdx;
  static_assert(S == std::decay_t<decltype(Kb)>::SlotIdx);
  using Roots   = std::index_sequence<S>;
  using LevelsA = std::decay_t<decltype(ga.levels)>;
  using LevelsB = std::decay_t<decltype(gb.levels)>;
  static_assert(Impl::lg_slot_streamed_v<LevelsA, Roots, S>);
  static_assert(Impl::lg_slot_pool_v<LevelsA, Roots, S> ==
                Impl::slot_pool_none);
  static_assert(!Impl::lg_slot_streamed_v<LevelsB, Roots, S>);
  // Not a root at all: never streamed, whoever reads it.
  static_assert(!Impl::lg_slot_streamed_v<LevelsA, std::index_sequence<>, S>);

  // The scratch request drops by exactly K's tile (its own pool: K is live
  // with every stage slot at its level, so it shares with nothing).
  using KTile = member_out_tile_t<stiffness_node_t<std::decay_t<decltype(ga)>>>;
  constexpr std::size_t tile_bytes =
      Impl::slot_arena_step<float, ES>(Impl::slot_tile_elems<KTile>()) *
      sizeof(float);
  const std::size_t bytes_a = ga.outputs(Ka).scratch_bytes();
  const std::size_t bytes_b = gb.outputs(Kb).scratch_bytes();
  EXPECT_LT(bytes_a, bytes_b) << what;
  EXPECT_EQ(bytes_b - bytes_a, tile_bytes) << what;
  EXPECT_EQ(decltype(gb.outputs(Kb))::num_pools,
            decltype(ga.outputs(Ka))::num_pools + 1)
      << what;

  const auto oa = run_stiffness(ga, Ka);
  const auto ob = run_stiffness(gb, Kb);
  check_stiffness(oa, what);
  EXPECT_EQ(bitwise_mismatches(oa, ob), 0) << what;
  EXPECT_EQ(bitwise_mismatches(ob, copy), 0) << what;
}

TEST(GeneralContraction, StreamedRootNeedsNoScratchTile) {
  streamed_root_case<StiffnessMap>("streamed, entry by entry");
  streamed_root_case<StiffnessWholeABMap>("streamed, value-blocked");
}

// Two general-contraction members in ONE level, both roots nobody reads (so
// both streamed, each to its own view). Blocking is per level: when the
// members' value dims agree the level runs value blocks for both; when one
// has none the whole level falls back to entry by entry. Either way each
// member must match its host loop.
//   c1(e,i) = sum_j A(e,i,j) W(e,j)      value label i, on A
//   c2(e,i) = sum_j A(e,i,j) V(e,j)      value label i, on A   (agrees)
//   c3(e,i) = sum_j A(e,i,j) A(e,i,j)    i on both leaves: none (disagrees)
TEST(GeneralContraction, LevelBlocksOnlyWhenItsMembersAgree) {
  constexpr int fE = 6, fTE = 2, fI = 4, fJ = 5;
  using Map =
      LabelTiles<LabelTile<'e', fTE>, LabelWhole<'i', fI>, LabelWhole<'j', fJ>>;
  using V2 = Kokkos::View<float**, Kokkos::LayoutRight, ES>;
  Kokkos::View<float***, Kokkos::LayoutRight, ES> A("A", fE, fI, fJ);
  V2   W("W", fE, fJ), Vv("V", fE, fJ);
  auto Ah = Kokkos::create_mirror_view(A);
  auto Wh = Kokkos::create_mirror_view(W);
  auto Vh = Kokkos::create_mirror_view(Vv);
  for (int e = 0; e < fE; ++e)
    for (int j = 0; j < fJ; ++j) {
      Wh(e, j) = wval(e, j);
      Vh(e, j) = bval(e, j);
      for (int i = 0; i < fI; ++i) Ah(e, i, j) = aval(e, i, j);
    }
  Kokkos::deep_copy(A, Ah);
  Kokkos::deep_copy(W, Wh);
  Kokkos::deep_copy(Vv, Vh);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, a] =
      g0.add(make_stage_node(make_input_node(make_handle<'e', 'i', 'j'>(A))));
  auto [g2, w, v] =
      g1.add(make_stage_node(make_input_node(make_handle<'e', 'j'>(W))),
             make_stage_node(make_input_node(make_handle<'e', 'j'>(Vv))));
  const auto n1 = make_contraction_node<'e', 'i'>(a, w);
  const auto n2 = make_contraction_node<'e', 'i'>(a, v);
  const auto n3 = make_contraction_node<'e', 'i'>(a, a);

  auto [ga, c1, c2] = g2.add(n1, n2);  // agree
  auto [gd, d1, d3] = g2.add(n1, n3);  // disagree
  using LevelA      = std::decay_t<decltype(ga.levels.template get<2>())>;
  using LevelD      = std::decay_t<decltype(gd.levels.template get<2>())>;
  static_assert(
      std::is_same_v<typename tuple_element_t<1, LevelD>::value_labels_seq,
                     std::integer_sequence<int32_t>>);
  static_assert(Impl::lg_gc_level_blocked_v<LevelA>);
  static_assert(!Impl::lg_gc_level_blocked_v<LevelD>);
  using LevelsA            = std::decay_t<decltype(ga.levels)>;
  constexpr std::size_t S1 = std::decay_t<decltype(c1)>::SlotIdx;
  constexpr std::size_t S2 = std::decay_t<decltype(c2)>::SlotIdx;
  static_assert(
      Impl::lg_slot_streamed_v<LevelsA, std::index_sequence<S1, S2>, S1> &&
      Impl::lg_slot_streamed_v<LevelsA, std::index_sequence<S1, S2>, S2>);

  V2 C1("C1", fE, fI), C2("C2", fE, fI), D1("D1", fE, fI), D3("D3", fE, fI);
  // Roots listed in the opposite order to their slots: each streamed member
  // must find ITS view, not the one at its member index.
  ga.outputs(c2, c1).execute(TeamPolicyTag<ES>{}, C2, C1);
  gd.outputs(d3, d1).execute(TeamPolicyTag<ES>{}, D3, D1);
  Kokkos::fence();

  auto C1h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C1);
  auto C2h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C2);
  auto D1h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, D1);
  auto D3h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, D3);
  for (int e = 0; e < fE; ++e)
    for (int i = 0; i < fI; ++i) {
      float r1 = 0, r2 = 0, r3 = 0;
      for (int j = 0; j < fJ; ++j) {
        r1 += aval(e, i, j) * wval(e, j);
        r2 += aval(e, i, j) * bval(e, j);
        r3 += aval(e, i, j) * aval(e, i, j);
      }
      EXPECT_NEAR(C1h(e, i), r1, 1e-5f) << e << " " << i;
      EXPECT_NEAR(C2h(e, i), r2, 1e-5f) << e << " " << i;
      EXPECT_NEAR(D1h(e, i), r1, 1e-5f) << e << " " << i;
      EXPECT_NEAR(D3h(e, i), r3, 1e-5f) << e << " " << i;
      // The same member, blocked or not: bit for bit.
      EXPECT_EQ(C1h(e, i), D1h(e, i)) << e << " " << i;
    }
}

}  // namespace

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int rc = RUN_ALL_TESTS();
  Kokkos::finalize();
  return rc;
}
