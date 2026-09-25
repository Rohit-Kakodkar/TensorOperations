// ===========================================================================
// test_structured_operands.cpp -- Structured.hpp: delta, outer and stack
// operands used directly as operands of make_contraction_node<Out...>, the
// unified n-ary contraction entry point. make_contraction_node dispatches to
// ContractionTag for a plain two-operand GEMM shape (every output label in
// exactly one input, no batch label) and to GeneralContractionTag for
// everything else -- a shared batch label, or any structured operand.
//
// Every executed case is checked against a host loop over the same formula.
// Extents are pairwise distinct wherever two labels could be confused (a
// transposed index cannot hide behind two equal extents), and the gridded
// element label 'e' has more than one tile so a dropped tile origin shows up.
//
//   1. delta<> as a trace: C(e) = sum_{i,j} A(e,i,j) delta(i,j);
//   2. delta<> as an identity embedding: C(e,i,k) = sum_j A(e,i,j) delta(j,k);
//   3. delta<> as a diag(w) embedding: C(e,p,f) = w(e,p) delta(p,f);
//   4. delta<> standing in for an un-formed Kronecker factor;
//   5. delta<'r'>(idx<1>) as a slice: C(e,a) = sum_r M(e,a,r) e_1(r);
//   6. stack<> mixing a dense branch and a delta branch;
//   7. stack<> nested two deep;
//   8. outer() grouping two operands with disjoint labels, one of them
//      carrying a batch label straight into the output;
//   9. .as<>() relabelling a stack;
//  10. delta<> reading a diagonal: C(e,i) = A(e,i,i);
//  11. dispatch: node_tag is ContractionTag only for a plain two-operand GEMM
//      shape, and GeneralContractionTag for a shared batch label or any
//      delta operand (compile-time only, nothing executes).
// ===========================================================================
#include <TensorOperations/GeneralContraction.hpp>
#include <TensorOperations/Structured.hpp>

#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/LevelPlan.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

using namespace TensorOperations;
using ES = Kokkos::DefaultExecutionSpace;

namespace {

// --- 1. delta<> as a trace: C(e) = sum_{i,j} A(e,i,j) delta(i,j) -----------

float trace_aval(int e, int i, int j) {
  return 0.4f + 0.05f * e - 0.09f * i + 0.07f * j + 0.01f * e * i;
}

TEST(StructuredOperands, DeltaTrace) {
  constexpr int fE = 6, fTE = 2, fN = 4;  // i and j share the extent fN
  using Map = LabelTiles<LabelTile<'e', fTE>, LabelWhole<'i', fN>,
                         LabelWhole<'j', fN>>;

  Kokkos::View<float***, Kokkos::LayoutRight, ES> A("A", fE, fN, fN);
  Kokkos::View<float*, Kokkos::LayoutRight, ES>    C("C", fE);
  auto Ah = Kokkos::create_mirror_view(A);
  for (int e = 0; e < fE; ++e)
    for (int i = 0; i < fN; ++i)
      for (int j = 0; j < fN; ++j) Ah(e, i, j) = trace_aval(e, i, j);
  Kokkos::deep_copy(A, Ah);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, a] =
      g0.add(make_stage_node(make_input_node(make_handle<'e', 'i', 'j'>(A))));
  auto [g2, c] = g1.add(make_contraction_node<'e'>(a, delta<'i', 'j'>()));
  g2.outputs(c).execute(TeamPolicyTag<ES>{}, C);
  Kokkos::fence();

  auto Ch = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  for (int e = 0; e < fE; ++e) {
    float ref = 0;
    for (int i = 0; i < fN; ++i) ref += trace_aval(e, i, i);
    EXPECT_NEAR(Ch(e), ref, 1e-5f) << e;
  }
}

// --- 2. delta<> as an identity: C(e,i,k) = sum_j A(e,i,j) delta(j,k) -------
// equals A(e,i,k).

float ident_aval(int e, int i, int j) {
  return 0.3f + 0.07f * e - 0.11f * i + 0.05f * j + 0.01f * e * j;
}

TEST(StructuredOperands, DeltaIdentity) {
  constexpr int fE = 6, fTE = 3, fI = 4, fJ = 5;  // delta<'j','k'>: j, k = fJ
  using Map = LabelTiles<LabelTile<'e', fTE>, LabelWhole<'i', fI>,
                         LabelWhole<'j', fJ>, LabelWhole<'k', fJ>>;

  Kokkos::View<float***, Kokkos::LayoutRight, ES> A("A", fE, fI, fJ);
  Kokkos::View<float***, Kokkos::LayoutRight, ES> C("C", fE, fI, fJ);
  auto Ah = Kokkos::create_mirror_view(A);
  for (int e = 0; e < fE; ++e)
    for (int i = 0; i < fI; ++i)
      for (int j = 0; j < fJ; ++j) Ah(e, i, j) = ident_aval(e, i, j);
  Kokkos::deep_copy(A, Ah);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, a] =
      g0.add(make_stage_node(make_input_node(make_handle<'e', 'i', 'j'>(A))));
  auto [g2, c] =
      g1.add(make_contraction_node<'e', 'i', 'k'>(a, delta<'j', 'k'>()));
  g2.outputs(c).execute(TeamPolicyTag<ES>{}, C);
  Kokkos::fence();

  auto Ch = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  for (int e = 0; e < fE; ++e)
    for (int i = 0; i < fI; ++i)
      for (int k = 0; k < fJ; ++k)
        EXPECT_NEAR(Ch(e, i, k), ident_aval(e, i, k), 1e-5f)
            << e << " " << i << " " << k;
}

// --- 3. delta<> as a diag(w) embedding: C(e,p,f) = w(e,p) delta(p,f) ------

float diagw_val(int e, int p) {
  return 0.6f + 0.11f * e - 0.08f * p + 0.02f * e * p;
}

TEST(StructuredOperands, DeltaDiagEmbedding) {
  constexpr int fE = 6, fTE = 2, fP = 5;  // p and f share the extent fP
  using Map = LabelTiles<LabelTile<'e', fTE>, LabelWhole<'p', fP>,
                         LabelWhole<'f', fP>>;

  Kokkos::View<float**, Kokkos::LayoutRight, ES>  W("W", fE, fP);
  Kokkos::View<float***, Kokkos::LayoutRight, ES> C("C", fE, fP, fP);
  auto Wh = Kokkos::create_mirror_view(W);
  for (int e = 0; e < fE; ++e)
    for (int p = 0; p < fP; ++p) Wh(e, p) = diagw_val(e, p);
  Kokkos::deep_copy(W, Wh);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, w] =
      g0.add(make_stage_node(make_input_node(make_handle<'e', 'p'>(W))));
  auto [g2, c] =
      g1.add(make_contraction_node<'e', 'p', 'f'>(w, delta<'p', 'f'>()));
  g2.outputs(c).execute(TeamPolicyTag<ES>{}, C);
  Kokkos::fence();

  auto Ch = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  for (int e = 0; e < fE; ++e)
    for (int p = 0; p < fP; ++p)
      for (int f = 0; f < fP; ++f) {
        const float ref = (p == f) ? diagw_val(e, p) : 0.0f;
        EXPECT_NEAR(Ch(e, p, f), ref, 1e-5f) << e << " " << p << " " << f;
      }
}

// --- 4. delta<> standing in for an un-formed Kronecker factor -------------
// C(e,i,j) = sum_{p,q} A(i,p) delta(j,q) U(e,p,q)  ==  (A tensor I) applied
// to U, without ever forming the (fI*fJ) x (fP*fJ) Kronecker matrix.

float kron_aval(int i, int p) {
  return 0.5f + 0.09f * i - 0.13f * p + 0.02f * i * p;
}
float kron_uval(int e, int p, int q) {
  return -0.2f + 0.06f * e + 0.1f * p - 0.04f * q + 0.015f * e * q;
}

TEST(StructuredOperands, DeltaAsKroneckerFactor) {
  constexpr int fE = 6, fTE = 2, fI = 3, fP = 4, fJ = 5;  // q shares fJ, j
  using Map = LabelTiles<LabelTile<'e', fTE>, LabelWhole<'i', fI>,
                         LabelWhole<'p', fP>, LabelWhole<'j', fJ>,
                         LabelWhole<'q', fJ>>;

  Kokkos::View<float**, Kokkos::LayoutRight, ES>  A("A", fI, fP);
  Kokkos::View<float***, Kokkos::LayoutRight, ES> U("U", fE, fP, fJ);
  Kokkos::View<float***, Kokkos::LayoutRight, ES> C("C", fE, fI, fJ);
  auto Ah = Kokkos::create_mirror_view(A);
  auto Uh = Kokkos::create_mirror_view(U);
  for (int i = 0; i < fI; ++i)
    for (int p = 0; p < fP; ++p) Ah(i, p) = kron_aval(i, p);
  for (int e = 0; e < fE; ++e)
    for (int p = 0; p < fP; ++p)
      for (int q = 0; q < fJ; ++q) Uh(e, p, q) = kron_uval(e, p, q);
  Kokkos::deep_copy(A, Ah);
  Kokkos::deep_copy(U, Uh);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, a] =
      g0.add(make_stage_node(make_input_node(make_handle<'i', 'p'>(A))));
  auto [g2, u] =
      g1.add(make_stage_node(make_input_node(make_handle<'e', 'p', 'q'>(U))));
  auto [g3, c] = g2.add(
      make_contraction_node<'e', 'i', 'j'>(a, delta<'j', 'q'>(), u));
  g3.outputs(c).execute(TeamPolicyTag<ES>{}, C);
  Kokkos::fence();

  auto Ch = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  for (int e = 0; e < fE; ++e)
    for (int i = 0; i < fI; ++i)
      for (int j = 0; j < fJ; ++j) {
        float ref = 0;
        for (int p = 0; p < fP; ++p)
          ref += kron_aval(i, p) * kron_uval(e, p, j);
        EXPECT_NEAR(Ch(e, i, j), ref, 1e-5f) << e << " " << i << " " << j;
      }
}

// --- 5. delta<'r'>(idx<1>) as a slice: C(e,a) = sum_r M(e,a,r) e_1(r) -----
// equals M(e,a,1).

float slice_mval(int e, int a, int r) {
  return 0.35f + 0.08f * e - 0.06f * a + 0.04f * r + 0.01f * e * r;
}

TEST(StructuredOperands, DeltaUnitVectorSlices) {
  constexpr int fE = 6, fTE = 2, fA = 4, fR = 5;  // idx<1> picks r == 1
  using Map = LabelTiles<LabelTile<'e', fTE>, LabelWhole<'a', fA>,
                         LabelWhole<'r', fR>>;

  Kokkos::View<float***, Kokkos::LayoutRight, ES> M("M", fE, fA, fR);
  Kokkos::View<float**, Kokkos::LayoutRight, ES>  C("C", fE, fA);
  auto Mh = Kokkos::create_mirror_view(M);
  for (int e = 0; e < fE; ++e)
    for (int a = 0; a < fA; ++a)
      for (int r = 0; r < fR; ++r) Mh(e, a, r) = slice_mval(e, a, r);
  Kokkos::deep_copy(M, Mh);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, m] =
      g0.add(make_stage_node(make_input_node(make_handle<'e', 'a', 'r'>(M))));
  auto [g2, c] =
      g1.add(make_contraction_node<'e', 'a'>(m, delta<'r'>(idx<1>)));
  g2.outputs(c).execute(TeamPolicyTag<ES>{}, C);
  Kokkos::fence();

  auto Ch = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  for (int e = 0; e < fE; ++e)
    for (int a = 0; a < fA; ++a)
      EXPECT_NEAR(Ch(e, a), slice_mval(e, a, 1), 1e-5f) << e << " " << a;
}

// --- 6. stack<> mixing a dense branch and a delta branch --------------------
// S = stack<'r'>(H.as<'p','f'>(), delta<'p','f'>())     (r has extent 2)
// C(e,r,f) = sum_p U(e,p) S(r,p,f)
//   r=0: sum_p U(e,p) H(p,f)          r=1: U(e,f)

float stack_hval(int p, int f) {
  return 0.45f - 0.07f * p + 0.11f * f - 0.02f * p * f;
}
float stack_uval(int e, int p) {
  return 0.2f + 0.09f * e - 0.05f * p + 0.01f * e * p;
}

TEST(StructuredOperands, StackMixesDenseAndDelta) {
  constexpr int fE = 6, fTE = 2, fP = 4;  // p and f share the extent fP
  using Map = LabelTiles<LabelTile<'e', fTE>, LabelWhole<'r', 2>,
                         LabelWhole<'p', fP>, LabelWhole<'f', fP>>;

  Kokkos::View<float**, Kokkos::LayoutRight, ES>  H("H", fP, fP);
  Kokkos::View<float**, Kokkos::LayoutRight, ES>  U("U", fE, fP);
  Kokkos::View<float***, Kokkos::LayoutRight, ES> C("C", fE, 2, fP);
  auto Hh = Kokkos::create_mirror_view(H);
  auto Uh = Kokkos::create_mirror_view(U);
  for (int p = 0; p < fP; ++p)
    for (int f = 0; f < fP; ++f) Hh(p, f) = stack_hval(p, f);
  for (int e = 0; e < fE; ++e)
    for (int p = 0; p < fP; ++p) Uh(e, p) = stack_uval(e, p);
  Kokkos::deep_copy(H, Hh);
  Kokkos::deep_copy(U, Uh);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, h] =
      g0.add(make_stage_node(make_input_node(make_handle<'p', 'f'>(H))));
  auto [g2, u] =
      g1.add(make_stage_node(make_input_node(make_handle<'e', 'p'>(U))));
  auto s        = stack<'r'>(h.as<'p', 'f'>(), delta<'p', 'f'>());
  auto [g3, c] = g2.add(make_contraction_node<'e', 'r', 'f'>(u, s));
  g3.outputs(c).execute(TeamPolicyTag<ES>{}, C);
  Kokkos::fence();

  auto Ch = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  for (int e = 0; e < fE; ++e)
    for (int f = 0; f < fP; ++f) {
      float ref0 = 0;
      for (int p = 0; p < fP; ++p) ref0 += stack_uval(e, p) * stack_hval(p, f);
      EXPECT_NEAR(Ch(e, 0, f), ref0, 1e-5f) << "r=0 " << e << " " << f;
      EXPECT_NEAR(Ch(e, 1, f), stack_uval(e, f), 1e-5f)
          << "r=1 " << e << " " << f;
    }
}

// --- 7. stack<> nested two deep ---------------------------------------------
// Outer = stack<'s'>(stack<'r'>(X, Y), stack<'r'>(Y, delta<'p','f'>()))
// C(e,s,r,f) = sum_p U(e,p) Outer(s,r,p,f)
//   (s,r) = (0,0): X   (0,1): Y   (1,0): Y   (1,1): delta

float nest_xval(int p, int f) {
  return 0.3f + 0.04f * p - 0.09f * f + 0.015f * p * f;
}
float nest_yval(int p, int f) {
  return -0.15f + 0.08f * p + 0.05f * f - 0.02f * p * f;
}
float nest_uval(int e, int p) {
  return 0.25f - 0.03f * e + 0.07f * p - 0.01f * e * p;
}

TEST(StructuredOperands, NestedStack) {
  constexpr int fE = 6, fTE = 3, fP = 3;  // p and f share the extent fP
  using Map = LabelTiles<LabelTile<'e', fTE>, LabelWhole<'s', 2>,
                         LabelWhole<'r', 2>, LabelWhole<'p', fP>,
                         LabelWhole<'f', fP>>;

  Kokkos::View<float**, Kokkos::LayoutRight, ES>   X("X", fP, fP);
  Kokkos::View<float**, Kokkos::LayoutRight, ES>   Y("Y", fP, fP);
  Kokkos::View<float**, Kokkos::LayoutRight, ES>   U("U", fE, fP);
  Kokkos::View<float****, Kokkos::LayoutRight, ES> C("C", fE, 2, 2, fP);
  auto Xh = Kokkos::create_mirror_view(X);
  auto Yh = Kokkos::create_mirror_view(Y);
  auto Uh = Kokkos::create_mirror_view(U);
  for (int p = 0; p < fP; ++p)
    for (int f = 0; f < fP; ++f) {
      Xh(p, f) = nest_xval(p, f);
      Yh(p, f) = nest_yval(p, f);
    }
  for (int e = 0; e < fE; ++e)
    for (int p = 0; p < fP; ++p) Uh(e, p) = nest_uval(e, p);
  Kokkos::deep_copy(X, Xh);
  Kokkos::deep_copy(Y, Yh);
  Kokkos::deep_copy(U, Uh);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, x] =
      g0.add(make_stage_node(make_input_node(make_handle<'p', 'f'>(X))));
  auto [g2, y] =
      g1.add(make_stage_node(make_input_node(make_handle<'p', 'f'>(Y))));
  auto [g3, u] =
      g2.add(make_stage_node(make_input_node(make_handle<'e', 'p'>(U))));

  auto s0      = stack<'r'>(x.as<'p', 'f'>(), y.as<'p', 'f'>());
  auto s1      = stack<'r'>(y.as<'p', 'f'>(), delta<'p', 'f'>());
  auto outer_s = stack<'s'>(s0, s1);

  auto [g4, c] = g3.add(make_contraction_node<'e', 's', 'r', 'f'>(u, outer_s));
  g4.outputs(c).execute(TeamPolicyTag<ES>{}, C);
  Kokkos::fence();

  auto Ch     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  auto branch = [](int s, int r, int p, int f) -> float {
    if (s == 0 && r == 0) return nest_xval(p, f);
    if (s == 0 && r == 1) return nest_yval(p, f);
    if (s == 1 && r == 0) return nest_yval(p, f);
    return (p == f) ? 1.0f : 0.0f;
  };
  for (int e = 0; e < fE; ++e)
    for (int s = 0; s < 2; ++s)
      for (int r = 0; r < 2; ++r)
        for (int f = 0; f < fP; ++f) {
          float ref = 0;
          for (int p = 0; p < fP; ++p)
            ref += nest_uval(e, p) * branch(s, r, p, f);
          EXPECT_NEAR(Ch(e, s, r, f), ref, 1e-5f)
              << e << " " << s << " " << r << " " << f;
        }
}

// --- 8. outer() groups operands with disjoint labels into their product ---
// C(e,i,j) = outer(A.as<'e','i'>(), B.as<'j'>())     (A(e,i); B declared as
// B(m), renamed to 'j'.)  'e' rides straight through as a batch label --
// nothing here sums it.

float outer_aval(int e, int i) {
  return 0.4f + 0.06f * e - 0.1f * i + 0.02f * e * i;
}
float outer_bval(int m) { return 1.3f - 0.17f * m; }

TEST(StructuredOperands, OuterGroupsDisjointOperands) {
  constexpr int fE = 6, fTE = 2, fI = 4, fJ = 5;
  using Map = LabelTiles<LabelTile<'e', fTE>, LabelWhole<'i', fI>,
                         LabelWhole<'m', fJ>, LabelWhole<'j', fJ>>;

  Kokkos::View<float**, Kokkos::LayoutRight, ES>  A("A", fE, fI);
  Kokkos::View<float*, Kokkos::LayoutRight, ES>   B("B", fJ);
  Kokkos::View<float***, Kokkos::LayoutRight, ES> C("C", fE, fI, fJ);
  auto Ah = Kokkos::create_mirror_view(A);
  auto Bh = Kokkos::create_mirror_view(B);
  for (int e = 0; e < fE; ++e)
    for (int i = 0; i < fI; ++i) Ah(e, i) = outer_aval(e, i);
  for (int m = 0; m < fJ; ++m) Bh(m) = outer_bval(m);
  Kokkos::deep_copy(A, Ah);
  Kokkos::deep_copy(B, Bh);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, a] =
      g0.add(make_stage_node(make_input_node(make_handle<'e', 'i'>(A))));
  auto [g2, b] =
      g1.add(make_stage_node(make_input_node(make_handle<'m'>(B))));
  auto [g3, c] = g2.add(make_contraction_node<'e', 'i', 'j'>(
      outer(a.as<'e', 'i'>(), b.as<'j'>())));
  g3.outputs(c).execute(TeamPolicyTag<ES>{}, C);
  Kokkos::fence();

  auto Ch = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  for (int e = 0; e < fE; ++e)
    for (int i = 0; i < fI; ++i)
      for (int j = 0; j < fJ; ++j)
        EXPECT_NEAR(Ch(e, i, j), outer_aval(e, i) * outer_bval(j), 1e-5f)
            << e << " " << i << " " << j;
}

// --- 9. .as<>() relabels a stack --------------------------------------------
// S  = stack<'r'>(P.as<'e','x'>(), Q.as<'e','x'>())
// S2 = S.as<'k','e','y'>()
// C(e,k,y) = S2(e,k,y):   k=0 -> P(e,y)     k=1 -> Q(e,y)

float relabel_pval(int e, int x) {
  return 0.5f + 0.05f * e - 0.02f * x + 0.008f * e * x;
}
float relabel_qval(int e, int x) {
  return -0.3f + 0.09f * e + 0.04f * x - 0.006f * e * x;
}

TEST(StructuredOperands, StackRelabelledWithAs) {
  constexpr int fE = 6, fTE = 3, fX = 5;
  // 'r' and 'x' are the stack's ORIGINAL labels (before .as<>()); 'k' and 'y'
  // are what it is read as afterwards. Both are listed since it is not
  // spelled out anywhere whether the map is consulted before or after the
  // relabel -- see the report for this ambiguity.
  using Map = LabelTiles<LabelTile<'e', fTE>, LabelWhole<'r', 2>,
                         LabelWhole<'k', 2>, LabelWhole<'x', fX>,
                         LabelWhole<'y', fX>>;

  Kokkos::View<float**, Kokkos::LayoutRight, ES>  P("P", fE, fX);
  Kokkos::View<float**, Kokkos::LayoutRight, ES>  Q("Q", fE, fX);
  Kokkos::View<float***, Kokkos::LayoutRight, ES> C("C", fE, 2, fX);
  auto Ph = Kokkos::create_mirror_view(P);
  auto Qh = Kokkos::create_mirror_view(Q);
  for (int e = 0; e < fE; ++e)
    for (int x = 0; x < fX; ++x) {
      Ph(e, x) = relabel_pval(e, x);
      Qh(e, x) = relabel_qval(e, x);
    }
  Kokkos::deep_copy(P, Ph);
  Kokkos::deep_copy(Q, Qh);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, p] =
      g0.add(make_stage_node(make_input_node(make_handle<'e', 'x'>(P))));
  auto [g2, q] =
      g1.add(make_stage_node(make_input_node(make_handle<'e', 'x'>(Q))));
  auto s  = stack<'r'>(p.as<'e', 'x'>(), q.as<'e', 'x'>());
  auto s2 = s.as<'k', 'e', 'y'>();
  auto [g3, c] = g2.add(make_contraction_node<'e', 'k', 'y'>(s2));
  g3.outputs(c).execute(TeamPolicyTag<ES>{}, C);
  Kokkos::fence();

  auto Ch = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  for (int e = 0; e < fE; ++e)
    for (int y = 0; y < fX; ++y) {
      EXPECT_NEAR(Ch(e, 0, y), relabel_pval(e, y), 1e-5f)
          << "k=0 " << e << " " << y;
      EXPECT_NEAR(Ch(e, 1, y), relabel_qval(e, y), 1e-5f)
          << "k=1 " << e << " " << y;
    }
}

// --- 10. delta<> reads a diagonal: C(e,i) = A(e,i,i) -----------------------

float diagread_aval(int e, int i, int j) {
  return 0.55f + 0.04f * e - 0.07f * i + 0.09f * j - 0.015f * i * j;
}

TEST(StructuredOperands, DeltaReadsDiagonal) {
  constexpr int fE = 6, fTE = 2, fN = 5;  // i and j share the extent fN
  using Map = LabelTiles<LabelTile<'e', fTE>, LabelWhole<'i', fN>,
                         LabelWhole<'j', fN>>;

  Kokkos::View<float***, Kokkos::LayoutRight, ES> A("A", fE, fN, fN);
  Kokkos::View<float**, Kokkos::LayoutRight, ES>  C("C", fE, fN);
  auto Ah = Kokkos::create_mirror_view(A);
  for (int e = 0; e < fE; ++e)
    for (int i = 0; i < fN; ++i)
      for (int j = 0; j < fN; ++j) Ah(e, i, j) = diagread_aval(e, i, j);
  Kokkos::deep_copy(A, Ah);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, a] =
      g0.add(make_stage_node(make_input_node(make_handle<'e', 'i', 'j'>(A))));
  auto [g2, c] =
      g1.add(make_contraction_node<'e', 'i'>(a, delta<'i', 'j'>()));
  g2.outputs(c).execute(TeamPolicyTag<ES>{}, C);
  Kokkos::fence();

  auto Ch = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  for (int e = 0; e < fE; ++e)
    for (int i = 0; i < fN; ++i)
      EXPECT_NEAR(Ch(e, i), diagread_aval(e, i, i), 1e-5f) << e << " " << i;
}

// --- 11. dispatch: node_tag depends on operand shape, not just rank -------
// No level graph and nothing executes: these are compile-time-only checks of
// make_contraction_node's dispatch rule.

struct Dispatch2D {
  static constexpr int rank = 2;
  int                  extent(int) const { return 0; }
  std::ptrdiff_t        stride(int) const { return 0; }
  float*                data() { return nullptr; }
  const float*          data() const { return nullptr; }
  float                 operator()(int, int) const { return 0.f; }
};
struct Dispatch3D {
  static constexpr int rank = 3;
  int                  extent(int) const { return 0; }
  std::ptrdiff_t        stride(int) const { return 0; }
  float*                data() { return nullptr; }
  const float*          data() const { return nullptr; }
  float                 operator()(int, int, int) const { return 0.f; }
};

TEST(StructuredOperands, DispatchPlainMatmulIsContraction) {
  // Two plain inputs, every output label in exactly one input, no batch
  // label: a.as<'i','j'>() @ b.as<'j','k'>() -> ContractionTag.
  using A = decltype(make_input_node(make_handle<'i', 'j'>(Dispatch2D{})));
  using B = decltype(make_input_node(make_handle<'j', 'k'>(Dispatch2D{})));
  using Node = decltype(make_contraction_node<'i', 'k'>(std::declval<A>(),
                                                        std::declval<B>()));
  static_assert(std::is_same_v<typename Node::node_tag, ContractionTag>);
}

TEST(StructuredOperands, DispatchSharedBatchLabelIsGeneral) {
  // 'e' is carried by both inputs and the output: an output label may
  // appear in only one input for the plain-GEMM dispatch, so this is not it.
  using A = decltype(make_input_node(make_handle<'e', 'i', 'j'>(Dispatch3D{})));
  using B = decltype(make_input_node(make_handle<'e', 'j', 'k'>(Dispatch3D{})));
  using Node = decltype(make_contraction_node<'e', 'i', 'k'>(
      std::declval<A>(), std::declval<B>()));
  static_assert(std::is_same_v<typename Node::node_tag, GeneralContractionTag>);
}

TEST(StructuredOperands, DispatchDeltaOperandIsGeneral) {
  // Same shape as a matmul, but one operand is a delta: never ContractionTag.
  using A = decltype(make_input_node(make_handle<'i', 'j'>(Dispatch2D{})));
  using Node = decltype(make_contraction_node<'i', 'k'>(std::declval<A>(),
                                                        delta<'j', 'k'>()));
  static_assert(std::is_same_v<typename Node::node_tag, GeneralContractionTag>);
}

}  // namespace

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int rc = RUN_ALL_TESTS();
  Kokkos::finalize();
  return rc;
}
