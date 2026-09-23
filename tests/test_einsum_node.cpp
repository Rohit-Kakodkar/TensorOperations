// ===========================================================================
// test_einsum_node.cpp -- make_einsum_node: the labels decide what is summed.
//
// Every case is held to a host loop over the same formula. Extents are
// pairwise distinct wherever two labels could be confused (a transposed
// operand cannot hide behind two equal axes), and the gridded element label
// has more than one tile so a dropped tile origin shows up.
//
//   1. matmul with a batch label carried by one operand;
//   2. a batch (Hadamard) label shared by operands and output;
//   3. a four-operand contraction whose operands are ALL functional inputs,
//      so the grid's extent comes from an einsum leaf, not a stage;
//   4. delta-structured operands: the 3D stiffness pattern
//        K = sum_{r,s,z,y,x} D_r(zyx, kji) M(e,a,b,r,s,zyx) D_s(zyx, nml)
//      against the same expression with a DENSE D and against host loops,
//      into a rank-9 StridedAlias on team scratch level 1, with the term
//      structure the delta elimination must produce pinned at compile time.
// ===========================================================================
#include <TensorOperations/Einsum.hpp>
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
float bval(int j, int k) { return -0.2f + 0.13f * j + 0.09f * k - 0.02f * j * k; }
float wval(int e, int j) { return 1.0f + 0.1f * e - 0.03f * j; }

// --- 1. matmul with a batch label on one operand ---------------------------

TEST(EinsumNode, MatmulWithBatchLabel) {
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
  auto [g3, c] = g2.add(make_einsum_node<'e', 'i', 'k'>(a, b));
  using Plan   = LevelPlan<std::decay_t<decltype(g3.levels)>>;
  static_assert(Plan::num_levels == 3);
  g3.outputs(c).execute(TeamPolicyTag<ES>{}, C);
  Kokkos::fence();

  auto Ch = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  for (int e = 0; e < fE; ++e)
    for (int i = 0; i < fI; ++i)
      for (int k = 0; k < fK; ++k) {
        float ref = 0;
        for (int j = 0; j < fJ; ++j) ref += aval(e, i, j) * bval(j, k);
        EXPECT_NEAR(Ch(e, i, k), ref, 1e-5f) << e << " " << i << " " << k;
      }
}

// --- 2. a batch label shared by operands and output -------------------------

TEST(EinsumNode, SharedBatchLabel) {
  constexpr int fE = 6, fTE = 3, fI = 4, fJ = 5;
  using Map = LabelTiles<LabelTile<'e', fTE>, LabelWhole<'i', fI>,
                         LabelWhole<'j', fJ>>;

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
  auto [g3, c] = g2.add(make_einsum_node<'e', 'i'>(a, w));
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
  return 1.0f + 0.01f * e + 0.1f * (a == c) + 0.2f * (b == d) +
         0.05f * a * b - 0.02f * c * d;
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

TEST(EinsumNode, FunctionalOperandsCarryTheGrid) {
  constexpr int fE = 6, fTE = 2, fA = 3, fB = 2, fC = 4, fD = 5;
  // a and b differ from c and d so a transposed functor argument shows up.
  using Map = LabelTiles<LabelTile<'e', fTE>, LabelWhole<'a', fA>,
                         LabelWhole<'b', fB>, LabelWhole<'c', fC>,
                         LabelWhole<'d', fD>>;

  auto x1 = make_functional_input_node<'e', 'c', 'a'>(
      Kokkos::Array<int, 3>{fE, fC, fA}, XFn{});
  auto x2 = make_functional_input_node<'e', 'd', 'b'>(
      Kokkos::Array<int, 3>{fE, fD, fB}, XFn{});
  auto cc = make_functional_input_node<'e', 'a', 'c', 'b', 'd'>(
      Kokkos::Array<int, 5>{fE, fA, fC, fB, fD}, CFn{});
  auto ww = make_functional_input_node<'e'>(Kokkos::Array<int, 1>{fE}, WFn{});

  Kokkos::View<float***, Kokkos::LayoutRight, ES> M("M", fE, fA, fB);
  auto g0      = make_level_graph<float, ES>(Map{});
  auto [g1, m] = g0.add(make_einsum_node<'e', 'a', 'b'>(x1, cc, x2, ww));
  g1.outputs(m).execute(TeamPolicyTag<ES>{}, M);
  Kokkos::fence();

  auto Mh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, M);
  for (int e = 0; e < fE; ++e)
    for (int a = 0; a < fA; ++a)
      for (int b = 0; b < fB; ++b) {
        float ref = 0;
        for (int c = 0; c < fC; ++c)
          for (int d = 0; d < fD; ++d)
            ref += xfun(e, c, a) * cfun(e, a, c, b, d) * xfun(e, d, b) *
                   wfun(e);
        EXPECT_NEAR(Mh(e, a, b), ref, 1e-4f * std::fabs(ref))
            << e << " " << a << " " << b;
      }
}

// --- 4. delta-structured operands: the 3D stiffness pattern ----------------

// Distinct extents per direction: x (i, l) 3, y (j, m) 4, z (k, n) 2.
constexpr int kNX = 3, kNY = 4, kNZ = 2, kNE = 3, kNC = 2;

float hx_val(int q, int f) { return 0.2f + 0.3f * q - 0.25f * f + 0.07f * q * f; }
float hy_val(int q, int f) { return -0.1f + 0.2f * q + 0.15f * f - 0.05f * q * f; }
float hz_val(int q, int f) { return 0.4f - 0.3f * q + 0.35f * f + 0.1f * q * f; }
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
  // D(r, z, y, x, k, j, i) = h_r(q_r, i_r) prod_{t != r} delta(q_t, i_t).
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

using StiffnessMap = LabelTiles<
    LabelTile<'e', 1>, LabelTile<'a', 1>, LabelTile<'b', 1>,
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
        for (int p = 0; p < static_cast<int>(v.extent(1)); ++p) h(q, p) = f(q, p);
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

using KAlias =
    StridedAlias<float, ES, kNC, kNZ, kNY, kNX, kNC, kNZ, kNY, kNX>;
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
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
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
                    if (std::fabs(h(idx) - ref) > 1e-4f * (1 + std::fabs(ref)) &&
                        bad++ < 5)
                      ADD_FAILURE() << what << " at (" << e << a << k << j << i
                                    << b << n << m << l << "): " << h(idx)
                                    << " vs " << ref;
                  }
  EXPECT_EQ(bad, 0) << what;
}

TEST(EinsumNode, DeltaOperandSumFactorsTheStiffness) {
  StiffnessInputs in;
  auto            g0 = make_level_graph<float, ES>(StiffnessMap{});
  auto [g1a, hx] =
      g0.add(make_stage_node(make_input_node(make_handle<'x', 'i'>(in.hx))));
  auto [g1b, hy] =
      g1a.add(make_stage_node(make_input_node(make_handle<'y', 'j'>(in.hy))));
  auto [g1, hz] =
      g1b.add(make_stage_node(make_input_node(make_handle<'z', 'k'>(in.hz))));
  auto [g2, M] = g1.add(stiffness_m_node());

  // D_r(z, y, x; k, j, i): the derivative of the tensor-product basis
  // function (k, j, i) along reference direction r, at point (z, y, x).
  const auto D = make_delta_operand<'r', 'z', 'y', 'x', 'k', 'j', 'i'>(
      select<'r'>(kase(hx, delta<'y', 'j'>{}, delta<'z', 'k'>{}),
                  kase(hy, delta<'x', 'i'>{}, delta<'z', 'k'>{}),
                  kase(hz, delta<'x', 'i'>{}, delta<'y', 'j'>{})));

  auto [g3, K] =
      g2.add(make_einsum_node<'e', 'a', 'k', 'j', 'i', 'b', 'n', 'm', 'l'>(
          D, M, D.as<'s', 'z', 'y', 'x', 'n', 'm', 'l'>()));

  // The delta elimination, pinned: 9 terms; the three with r == s sum one
  // quadrature coordinate and test two directions, the six with r != s sum
  // nothing and test one.
  using KNode = std::decay_t<decltype(g3.levels.template get<4>()
                                          .template get<0>())>;
  static_assert(KNode::NumTerms == 9);
  static_assert(KNode::template term_info<0>::num_loop == 1);  // (x, x)
  static_assert(KNode::template term_info<1>::num_loop == 0);  // (x, y)
  static_assert(KNode::template term_info<4>::num_loop == 1);  // (y, y)
  static_assert(KNode::template term_info<8>::num_loop == 1);  // (z, z)
  static_assert(KNode::template term_info<0>::chk_a_seq::size() == 2);
  static_assert(KNode::template term_info<1>::chk_a_seq::size() == 1);
  static_assert(std::is_same_v<typename KNode::template term_info<0>::loop_ext_seq,
                               std::integer_sequence<int, kNX>>);
  static_assert(std::is_same_v<typename KNode::template term_info<4>::loop_ext_seq,
                               std::integer_sequence<int, kNY>>);
  using Plan = LevelPlan<std::decay_t<decltype(g3.levels)>>;
  static_assert(Plan::num_levels == 5);

  Kokkos::View<float*, ES> out("K", kNE * kBlock);
  Kokkos::deep_copy(out, std::nanf(""));
  g3.outputs(K).scratch_level(1).execute(TeamPolicyTag<ES>{},
                                         KAlias{out.data(), kNE});
  Kokkos::fence();
  check_stiffness(out, "delta D");
}

TEST(EinsumNode, DenseOperandIsTheSameExpression) {
  StiffnessInputs in;
  auto            g0 = make_level_graph<float, ES>(StiffnessMap{});
  auto [g1, M]       = g0.add(stiffness_m_node());

  const DenseD<H2> dfn{in.hx, in.hy, in.hz};
  auto Dr = make_functional_input_node<'r', 'z', 'y', 'x', 'k', 'j', 'i'>(
      Kokkos::Array<int, 7>{3, kNZ, kNY, kNX, kNZ, kNY, kNX}, dfn);
  auto Ds = make_functional_input_node<'s', 'z', 'y', 'x', 'n', 'm', 'l'>(
      Kokkos::Array<int, 7>{3, kNZ, kNY, kNX, kNZ, kNY, kNX}, dfn);

  auto [g2, K] =
      g1.add(make_einsum_node<'e', 'a', 'k', 'j', 'i', 'b', 'n', 'm', 'l'>(
          Dr, M, Ds));
  using KNode = std::decay_t<decltype(g2.levels.template get<1>()
                                          .template get<0>())>;
  static_assert(KNode::NumTerms == 1);
  static_assert(KNode::template term_info<0>::num_loop == 5);  // r s z y x

  Kokkos::View<float*, ES> out("K", kNE * kBlock);
  Kokkos::deep_copy(out, std::nanf(""));
  g2.outputs(K).scratch_level(1).execute(TeamPolicyTag<ES>{},
                                         KAlias{out.data(), kNE});
  Kokkos::fence();
  check_stiffness(out, "dense D");
}

}  // namespace

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int rc = RUN_ALL_TESTS();
  Kokkos::finalize();
  return rc;
}
