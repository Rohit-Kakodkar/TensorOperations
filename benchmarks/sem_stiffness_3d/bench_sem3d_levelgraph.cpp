// ===========================================================================
// 3D SEM stiffness, NGLL=5: the DECLARATIVE level graph against a hand-written
// fused kernel of the same shape.  (Sprint 5 of plans/level-graph-sprints.md.)
//
// The sibling benchmark, bench_sem_stiffness_3d.cpp, carries a Tag2
// fused-per-level kernel called `library2` whose own header says:
//
//     "This is what Sprint 3's add() would generate; hand-writing it here is a
//      measurement, not an API."
//
// It is 353 lines of caller-driven level walking. This file is that kernel
// GENERATED from a declaration: four `.add()` calls, no barriers, no scratch
// arithmetic, no index decoding.
//
// THE PIPELINE (identical to the sibling file, same shared physics):
//
//   A2  gradient    du_c/d(xi), du_c/d(eta), du_c/d(gamma)   9 contractions
//   --  chain rule  ds[c][d] = du_c/dx_d                     pointwise
//   --  stress      T = compute_stress(ds)                   BLACK BOX
//   --  integrand   F^r_c = J * sum_d T[c][d] * dr/dx_d      3 dirs x 3 comps
//   A7  divergence  t^r_c = sum_q Hw[q,.] F^r_c              9 contractions
//   --  force       f_c = w_j w_k t^xi + w_i w_k t^eta + ... 3 combines
//
// as four LEVELS: 9 contraction members | 1 nine-output combine | 9 contraction
// members | 3 combines.
//
// THE ONE IDEA WORTH READING THE FILE FOR: every contraction declares its
// output as <'e','k','j','i'>. Declared-order output (PR #33) lets a Tag2
// contraction keep STORING C canonically (freeA ++ freeB) while the declared
// labels ride in the slot's view type, so a gradient whose canonical storage is
// (i,e,k,j) is READ as the element-major frame everyone else speaks. The
// consequence is that there is not one permuting `.as<>()` in this file --
// every `.as<>()` is a positional rename naming the summed axis. The DAG path
// in the sibling file needs 30 permuting ones.
//
// That also refutes the old "an all-9-aligned element-major frame is
// impossible" analysis in plans/specfem-kernel-graph.md, which assumed
// canonical-C pinned the declared order. It no longer does.
//
// WHY ONE 9-OUTPUT INTEGRAND NODE, AND WHAT IT COSTS: the single combine member
// runs chain_rule and compute_stress ONCE per point, where the sibling file's
// three integrand nodes each rebuild the whole stress tensor (its header
// attributes 7% of its float work to exactly that). The price is that all nine
// F share one frame, so the eta and gamma divergences read their B operand at a
// non-mergeable stride where `library2`, having three separate F frames, gets
// permB = identity. That trade is the thing this benchmark measures.
//
// SCRATCH IS THE KNOWN WEAKNESS. LevelGraph does not pool slots by live range
// (that is Sprint 4), so all 5 stage tiles and all 30 member slots are live for
// the whole kernel: 200 + 16500*TE bytes, against `library2`'s hand-aliased 18
// tiles. It does not bite at the tile that wins -- see the table this prints.
//
// Usage: bench_sem3d_levelgraph [nspec [reps [warmup [TE team]]]]
//        5 arguments selects PROFILE MODE: one launch per implementation.
// ===========================================================================

#include <Kokkos_Core.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/LevelPlan.hpp>
#include <TensorOperations/Tiling.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <vector>

using namespace TensorOperations;

using ES = Kokkos::DefaultExecutionSpace;

namespace cfg {
inline constexpr bool kIsGPU =
    !Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace,
                                Kokkos::HostSpace>::accessible;

inline constexpr int N = 5;  // GLL points per direction (production NGLL)
inline constexpr int C = 3;  // displacement components

// Unlike the sibling benchmark, TE=1 is LEGAL here and TE=16 is not.
//
// TE=1: the Tag2 contraction is a plain dot product with no register blocking,
// so SB = TE*N*N = 25 being odd is irrelevant. The sibling file cannot express
// this tile at all -- its CMake target pins NR=2, which requires SB even.
//
// TE=16: 200 + 16500*16 = 264 KB exceeds the H100's ~227 KB opt-in shared
// memory, so the launch throws. The level-fusion spike's best tile does not
// survive four unpooled levels; that is a Sprint 5 finding, not a tuning miss.
inline constexpr int TE_max = 8;

inline constexpr int E_default = 786432;

// Team scratch a level-0 request may not exceed. The H100 figure is
// sharedMemPerBlockOptin minus the reserved and team-reduce slices; Kokkos opts
// into the large limit automatically via cudaFuncSetAttribute, so there is no
// 48 KB wall -- but 227 KB is a hard ceiling and TE=16 (264 KB) sails past it.
// Serial's cap is 32 KB, which admits TE=1 only. Rows above the cap are
// SKIPPED and labelled rather than left to throw at launch.
inline constexpr std::size_t kScratchCap = kIsGPU ? 227u * 1024u : 32u * 1024u;
}  // namespace cfg

// Element-local fields are [E, k, j, i] = (element, z, y, x).
using V4  = Kokkos::View<float * [cfg::N][cfg::N][cfg::N], Kokkos::LayoutRight>;
using V2  = Kokkos::View<float**, Kokkos::LayoutRight>;
using V1  = Kokkos::View<const float*, Kokkos::LayoutRight>;
using V4H = V4::host_mirror_type;

// ===========================================================================
// SHARED PHYSICS -- verbatim from bench_sem_stiffness_3d.cpp, and verified
// against the real SPECFEM++ source this time rather than reconstructed:
// core/specfem/medium/dim3/elastic/isotropic/stress.hpp writes
//   sigma_ii = (lambda+2mu)*du(i,i) + lambda*(du(j,j)+du(k,k))
//   sigma_ij = mu*(du(i,j)+du(j,i))
// which is the same expression as the `2mu*ds[c][c] + lambda*tr` form below,
// with the same du(i,j) = du_i/dx_j index convention.
//
// Counted once in the source-size table, against nobody.
// ===========================================================================
inline constexpr int kPhysBegin = __LINE__;
struct Mat3 {
  float m[3][3];
};

KOKKOS_INLINE_FUNCTION Mat3 chain_rule(const float dxi[3], const float det[3],
                                       const float dgm[3], const Mat3& J) {
  Mat3 ds{};
  for (int c = 0; c < 3; ++c)
    for (int d = 0; d < 3; ++d)
      ds.m[c][d] = dxi[c] * J.m[0][d] + det[c] * J.m[1][d] + dgm[c] * J.m[2][d];
  return ds;
}

KOKKOS_INLINE_FUNCTION Mat3 compute_stress(const Mat3& ds, float l2m,
                                           float mu) {
  const float lambda = l2m - 2.0f * mu;
  const float tr     = ds.m[0][0] + ds.m[1][1] + ds.m[2][2];
  Mat3        s{};
  for (int c = 0; c < 3; ++c)
    for (int d = 0; d < 3; ++d) s.m[c][d] = mu * (ds.m[c][d] + ds.m[d][c]);
  for (int c = 0; c < 3; ++c) s.m[c][c] += lambda * tr;
  return s;
}

KOKKOS_INLINE_FUNCTION Kokkos::Array<float, 3> project(const Mat3& sig,
                                                       const float dref[3],
                                                       float       jac) {
  Kokkos::Array<float, 3> f{};
  for (int c = 0; c < 3; ++c)
    f[c] = jac * (sig.m[c][0] * dref[0] + sig.m[c][1] * dref[1] +
                  sig.m[c][2] * dref[2]);
  return f;
}
inline constexpr int kPhysEnd = __LINE__;

template <typename VT>
KOKKOS_INLINE_FUNCTION Mat3 metric_at(VT xix, VT xiy, VT xiz, VT etx, VT ety,
                                      VT etz, VT gmx, VT gmy, VT gmz, int e,
                                      int k, int j, int i) {
  Mat3 J{};
  J.m[0][0] = xix(e, k, j, i);
  J.m[0][1] = xiy(e, k, j, i);
  J.m[0][2] = xiz(e, k, j, i);
  J.m[1][0] = etx(e, k, j, i);
  J.m[1][1] = ety(e, k, j, i);
  J.m[1][2] = etz(e, k, j, i);
  J.m[2][0] = gmx(e, k, j, i);
  J.m[2][1] = gmy(e, k, j, i);
  J.m[2][2] = gmz(e, k, j, i);
  return J;
}

struct Fields {
  V4 u0, u1, u2;     // displacement, per component
  V4 xix, xiy, xiz;  // d(xi)/dx_d
  V4 etx, ety, etz;  // d(eta)/dx_d
  V4 gmx, gmy, gmz;  // d(gamma)/dx_d
  V4 l2m, mu, jac;   // material + Jacobian determinant
  V2 H, Hw;          // derivative operator, weighted transpose
  V1 w;              // GLL quadrature weights
  V4 f0, f1, f2;     // outputs, per component
};

// ===========================================================================
// IMPLEMENTATION 1 -- the declarative level graph
// ===========================================================================

// Only TWO tile types appear in this file. Every intermediate tile is DERIVED:
// a contraction's from contract_c_tile_t, a combine's from combine_out_tile_t.
// The sibling file spells five (H_, E_, Q_, Grad, Div, Comb, Sum) by hand.
template <int TE>
struct GTiles {
  using H_ = StaticTile<cfg::N, cfg::N>;              // H and Hw
  using E_ = StaticTile<TE, cfg::N, cfg::N, cfg::N>;  // the frame
};

// The same extents, said once per LABEL instead of once per tensor. 'e' is the
// element axis and the only one that is blocked; every spatial and summed axis
// is a whole NGLL. This is what every stage tile is looked up in, and every
// downstream tile follows from those by derivation.
template <int TE>
using GMap = LabelTiles<LabelTile<'e', TE>, LabelTile<'k', cfg::N>,
                        LabelTile<'j', cfg::N>, LabelTile<'i', cfg::N>,
                        LabelTile<'p', cfg::N>, LabelTile<'r', cfg::N>>;

inline constexpr int kFnBegin = __LINE__;
// F^r_c for ALL NINE (r,c) from one chain rule and one stress evaluation.
//
// The nine metric/material fields ride in the closure rather than being
// operands: a combine reads each operand exactly once per point, so staging
// them would add 12 * 500*TE bytes of scratch and buy no reuse -- and `w` is
// rank-1, so it could not be an operand at all (every combine operand must
// carry the output's full label set).
//
// The coordinate is GLOBAL (e,k,j,i), not tile-local: the combine member
// derives its own tile origin from the grid index (PR #34). Without that these
// metric reads would be wrong for every team but the first.
struct Integrand9 {
  Fields          d;
  KOKKOS_FUNCTION Kokkos::Array<float, 9> operator()(
      int e, int k, int j, int i, float x0, float x1, float x2, float y0,
      float y1, float y2, float z0, float z1, float z2) const {
    const float dxi[3] = {x0, x1, x2};
    const float det[3] = {y0, y1, y2};
    const float dgm[3] = {z0, z1, z2};
    const Mat3  J   = metric_at(d.xix, d.xiy, d.xiz, d.etx, d.ety, d.etz, d.gmx,
                                d.gmy, d.gmz, e, k, j, i);
    const Mat3  sig = compute_stress(chain_rule(dxi, det, dgm, J),
                                     d.l2m(e, k, j, i), d.mu(e, k, j, i));
    const float jc  = d.jac(e, k, j, i);
    const auto  fx  = project(sig, J.m[0], jc);
    const auto  fe  = project(sig, J.m[1], jc);
    const auto  fg  = project(sig, J.m[2], jc);
    return {fx[0], fx[1], fx[2], fe[0], fe[1], fe[2], fg[0], fg[1], fg[2]};
  }
};

// The quadrature weight at the SUMMED index is already inside Hw
// (Hw(q,i) = H(i,q)*w(q), the standard hprimewgll), so only the two transverse
// weights appear here. All three directions are weighted.
struct WeightedSum {
  Fields                d;
  KOKKOS_FUNCTION float operator()(int, int k, int j, int i, float t1, float t2,
                                   float t3) const {
    return d.w(j) * d.w(k) * t1 + d.w(i) * d.w(k) * t2 + d.w(i) * d.w(j) * t3;
  }
};
inline constexpr int kFnEnd = __LINE__;

inline constexpr int kGraphBegin = __LINE__;
template <int TE>
void levelgraph_sem3d(Fields d, int team) {
  using T = GTiles<TE>;
  const Integrand9  integrand{d};
  const WeightedSum wsum{d};

  // Stages. The LAST stage is the GRID node -- the league size is its tile
  // count -- so a rank-4 u must come last and the operators first.
  //
  // H is (free, summed) and Hw is (summed, free): Hw is H's weighted
  // TRANSPOSE. 'r' is renamed per use to the axis that operator reconstructs;
  // 'p' is the summed point and never appears in an output.
  auto g0       = make_level_graph<float, ES>(GMap<TE>{});
  auto [g1, h]  = g0.stage(make_input_node(make_handle<'r', 'p'>(d.H)));
  auto [g2, hw] = g1.stage(make_input_node(make_handle<'p', 'r'>(d.Hw)));
  auto [g3, u0] =
      g2.stage(make_input_node(make_handle<'e', 'k', 'j', 'i'>(d.u0)));
  auto [g4, u1] =
      g3.stage(make_input_node(make_handle<'e', 'k', 'j', 'i'>(d.u1)));
  auto [g5, u2] =
      g4.stage(make_input_node(make_handle<'e', 'k', 'j', 'i'>(d.u2)));

  // LEVEL 1 -- nine gradients, one fused range. The direction is which axis of
  // the one staged u is summed ('p' in slot 3 = d/d(xi), slot 2 = d/d(eta),
  // slot 1 = d/d(gamma)) and which row of the one staged H pairs with it.
  auto gx = [&](auto uu) {
    return make_contraction_node<'e', 'k', 'j', 'i'>(
        h.template as<'i', 'p'>(), uu.template as<'e', 'k', 'j', 'p'>());
  };
  auto ge = [&](auto uu) {
    return make_contraction_node<'e', 'k', 'j', 'i'>(
        h.template as<'j', 'p'>(), uu.template as<'e', 'k', 'p', 'i'>());
  };
  auto gg = [&](auto uu) {
    return make_contraction_node<'e', 'k', 'j', 'i'>(
        h.template as<'k', 'p'>(), uu.template as<'e', 'p', 'j', 'i'>());
  };
  auto [g6, gx0, gx1, gx2, ge0, ge1, ge2, gg0, gg1, gg2] = g5.add(
      gx(u0), gx(u1), gx(u2), ge(u0), ge(u1), ge(u2), gg(u0), gg(u1), gg(u2));

  // LEVEL 2 -- chain rule, stress and nine integrands, ONE member.
  //
  // The operands are passed BARE. Their declared labels are already the frame,
  // so an explicit .as<'e','k','j','i'>() would be the identity -- and a
  // hazard: at NGLL=5 every GLL axis is 5 and the operand check is
  // extents-only, blind to axis order, so a MISTYPED "identity" relabel
  // compiles clean and silently returns garbage. Bare cannot be mistyped.
  auto [g7, fx0, fx1, fx2, fe0, fe1, fe2, fg0, fg1, fg2] =
      g6.add(make_combine_node<'e', 'k', 'j', 'i'>(gx0, gx1, gx2, ge0, ge1, ge2,
                                                   gg0, gg1, gg2, integrand));

  // LEVEL 3 -- nine divergences. Structurally level 1 with H -> Hw, u -> F.
  auto dvx = [&](auto f) {
    return make_contraction_node<'e', 'k', 'j', 'i'>(
        hw.template as<'p', 'i'>(), f.template as<'e', 'k', 'j', 'p'>());
  };
  auto dve = [&](auto f) {
    return make_contraction_node<'e', 'k', 'j', 'i'>(
        hw.template as<'p', 'j'>(), f.template as<'e', 'k', 'p', 'i'>());
  };
  auto dvg = [&](auto f) {
    return make_contraction_node<'e', 'k', 'j', 'i'>(
        hw.template as<'p', 'k'>(), f.template as<'e', 'p', 'j', 'i'>());
  };
  auto [g8, tx0, tx1, tx2, te0, te1, te2, tg0, tg1, tg2] =
      g7.add(dvx(fx0), dvx(fx1), dvx(fx2), dve(fe0), dve(fe1), dve(fe2),
             dvg(fg0), dvg(fg1), dvg(fg2));

  // LEVEL 4 -- the three weighted sums.
  auto ws = [&](auto a, auto b, auto c) {
    return make_combine_node<'e', 'k', 'j', 'i'>(a, b, c, wsum);
  };
  auto [g9, r0, r1, r2] =
      g8.add(ws(tx0, te0, tg0), ws(tx1, te1, tg1), ws(tx2, te2, tg2));

  // LevelGraph does NOT instantiate LevelPlan, so its guards -- level
  // homogeneity, one iteration space per level, no member reading its own
  // level's output -- do not fire on their own. Naming the plan here is what
  // runs them. Without this an SA/SB disagreement between members would
  // silently index a sibling's output out of bounds.
  using Plan = LevelPlan<std::decay_t<decltype(g9.levels)>, 5>;
  static_assert(Plan::num_levels == 4, "four levels");
  static_assert(Plan::num_slots == 5 + 9 + 9 + 9 + 3, "5 stages + 30 members");

  // What liveness pooling costs those 35 slots. 19 is the maximum number
  // simultaneously live, which happens at levels 2 and 3: Hw (still needed by
  // the back-contractions) plus 9 gradients plus 9 integrands. Left-edge
  // coloring attains that bound, so a regression here is a plan that got worse,
  // not a plan that got unlucky.
  static_assert(decltype(g9.outputs(r0, r1, r2))::num_pools == 19,
                "19 pools for the SEM3D graph");

  g9.outputs(r0, r1, r2)
      .team_size(team)
      .execute(TeamPolicyTag2<ES>{}, d.f0, d.f1, d.f2);
}
inline constexpr int kGraphEnd = __LINE__;

// Host-side scratch query, for the table. Needs no launch.
template <int TE>
std::size_t levelgraph_scratch(Fields d) {
  using T       = GTiles<TE>;
  auto g0       = make_level_graph<float, ES>(GMap<TE>{});
  auto [g1, h]  = g0.stage(make_input_node(make_handle<'r', 'p'>(d.H)));
  auto [g2, hw] = g1.stage(make_input_node(make_handle<'p', 'r'>(d.Hw)));
  auto [g3, u0] =
      g2.stage(make_input_node(make_handle<'e', 'k', 'j', 'i'>(d.u0)));
  auto [g4, u1] =
      g3.stage(make_input_node(make_handle<'e', 'k', 'j', 'i'>(d.u1)));
  auto [g5, u2] =
      g4.stage(make_input_node(make_handle<'e', 'k', 'j', 'i'>(d.u2)));
  auto [g6, gx0, gx1, gx2, ge0, ge1, ge2, gg0, gg1, gg2] = g5.add(
      make_contraction_node<'e', 'k', 'j', 'i'>(
          h.template as<'i', 'p'>(), u0.template as<'e', 'k', 'j', 'p'>()),
      make_contraction_node<'e', 'k', 'j', 'i'>(
          h.template as<'i', 'p'>(), u1.template as<'e', 'k', 'j', 'p'>()),
      make_contraction_node<'e', 'k', 'j', 'i'>(
          h.template as<'i', 'p'>(), u2.template as<'e', 'k', 'j', 'p'>()),
      make_contraction_node<'e', 'k', 'j', 'i'>(
          h.template as<'j', 'p'>(), u0.template as<'e', 'k', 'p', 'i'>()),
      make_contraction_node<'e', 'k', 'j', 'i'>(
          h.template as<'j', 'p'>(), u1.template as<'e', 'k', 'p', 'i'>()),
      make_contraction_node<'e', 'k', 'j', 'i'>(
          h.template as<'j', 'p'>(), u2.template as<'e', 'k', 'p', 'i'>()),
      make_contraction_node<'e', 'k', 'j', 'i'>(
          h.template as<'k', 'p'>(), u0.template as<'e', 'p', 'j', 'i'>()),
      make_contraction_node<'e', 'k', 'j', 'i'>(
          h.template as<'k', 'p'>(), u1.template as<'e', 'p', 'j', 'i'>()),
      make_contraction_node<'e', 'k', 'j', 'i'>(
          h.template as<'k', 'p'>(), u2.template as<'e', 'p', 'j', 'i'>()));
  auto [g7, fx0, fx1, fx2, fe0, fe1, fe2, fg0, fg1, fg2] =
      g6.add(make_combine_node<'e', 'k', 'j', 'i'>(
          gx0, gx1, gx2, ge0, ge1, ge2, gg0, gg1, gg2, Integrand9{d}));
  auto [g8, tx0, tx1, tx2, te0, te1, te2, tg0, tg1, tg2] = g7.add(
      make_contraction_node<'e', 'k', 'j', 'i'>(
          hw.template as<'p', 'i'>(), fx0.template as<'e', 'k', 'j', 'p'>()),
      make_contraction_node<'e', 'k', 'j', 'i'>(
          hw.template as<'p', 'i'>(), fx1.template as<'e', 'k', 'j', 'p'>()),
      make_contraction_node<'e', 'k', 'j', 'i'>(
          hw.template as<'p', 'i'>(), fx2.template as<'e', 'k', 'j', 'p'>()),
      make_contraction_node<'e', 'k', 'j', 'i'>(
          hw.template as<'p', 'j'>(), fe0.template as<'e', 'k', 'p', 'i'>()),
      make_contraction_node<'e', 'k', 'j', 'i'>(
          hw.template as<'p', 'j'>(), fe1.template as<'e', 'k', 'p', 'i'>()),
      make_contraction_node<'e', 'k', 'j', 'i'>(
          hw.template as<'p', 'j'>(), fe2.template as<'e', 'k', 'p', 'i'>()),
      make_contraction_node<'e', 'k', 'j', 'i'>(
          hw.template as<'p', 'k'>(), fg0.template as<'e', 'p', 'j', 'i'>()),
      make_contraction_node<'e', 'k', 'j', 'i'>(
          hw.template as<'p', 'k'>(), fg1.template as<'e', 'p', 'j', 'i'>()),
      make_contraction_node<'e', 'k', 'j', 'i'>(
          hw.template as<'p', 'k'>(), fg2.template as<'e', 'p', 'j', 'i'>()));
  auto [g9, r0, r1, r2] = g8.add(
      make_combine_node<'e', 'k', 'j', 'i'>(tx0, te0, tg0, WeightedSum{d}),
      make_combine_node<'e', 'k', 'j', 'i'>(tx1, te1, tg1, WeightedSum{d}),
      make_combine_node<'e', 'k', 'j', 'i'>(tx2, te2, tg2, WeightedSum{d}));
  return g9.outputs(r0, r1, r2).scratch_bytes();
}

// ===========================================================================
// IMPLEMENTATION 2 -- the hand-written control, same four stages
//
// Deliberately NOT the sibling file's `hand`, which fuses the gradient and
// integrand phases (gradients live in registers and are never stored). That is
// a DIFFERENT pipeline; comparing against it would measure the graph's level
// structure rather than its declarative surface. This one materialises every
// level, so the only difference left is who writes the plumbing.
//
// It stores its gradients and divergences in the same new-index-first order the
// graph's contractions store -- not to imitate the graph, but because that IS
// the natural contiguous write for a loop whose outer index is the new point.
//
// It carves 30 tiles against the graph's 33: the graph needs three more for
// level 4, because a root slot is copied to global in a fifth pass rather than
// being written through. That difference is the graph's, not the pipeline's.
// ===========================================================================
inline constexpr int kHandBegin = __LINE__;
template <int TE>
void hand4_sem3d(Fields d, int E, int team) {
  using member_t   = Kokkos::TeamPolicy<>::member_type;
  using scratch    = ES::scratch_memory_space;
  constexpr int NN = cfg::N;
  constexpr int P  = NN * NN * NN;  // points per element
  constexpr int SB = TE * NN * NN;  // the contraction levels' inner extent

  // (s, new, le, b, c) -- a contraction's canonical freeA ++ freeB.
  using SQ =
      Kokkos::View<float** [TE][NN][NN], scratch, Kokkos::MemoryUnmanaged>;
  // (s, le, k, j, i) -- the element-major frame.
  using SE =
      Kokkos::View<float** [NN][NN][NN], scratch, Kokkos::MemoryUnmanaged>;
  using SO = Kokkos::View<float* [NN][NN], scratch, Kokkos::MemoryUnmanaged>;

  const std::size_t bytes = SO::shmem_size(2)         // H, Hw
                            + SE::shmem_size(3, TE)   // staged u
                            + SQ::shmem_size(9, NN)   // gradients
                            + SE::shmem_size(9, TE)   // integrands
                            + SQ::shmem_size(9, NN);  // divergences

  Kokkos::parallel_for(
      "hand4_sem3d",
      Kokkos::TeamPolicy<>(E / TE, team)
          .set_scratch_size(0, Kokkos::PerTeam(static_cast<int>(bytes))),
      KOKKOS_LAMBDA(const member_t& t) {
        const int e0 = t.league_rank() * TE;
        SO        op(t.team_scratch(0), 2);
        SE        us(t.team_scratch(0), 3, TE);
        SQ        G(t.team_scratch(0), 9, NN);
        SE        F(t.team_scratch(0), 9, TE);
        SQ        D(t.team_scratch(0), 9, NN);

        // --- PHASE 0: stage, mirroring the five stage() calls --------------
        Kokkos::parallel_for(Kokkos::TeamVectorRange(t, NN * NN), [&](int n) {
          const int a = n / NN, b = n % NN;
          op(0, a, b) = d.H(a, b);
          op(1, a, b) = d.Hw(a, b);
        });
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(t, 3 * TE * P), [&](int n) {
              const int i = n % NN, j = (n / NN) % NN, k = (n / (NN * NN)) % NN;
              const int le = (n / P) % TE, c = n / (P * TE);
              const V4& src      = (c == 0) ? d.u0 : (c == 1) ? d.u1 : d.u2;
              us(c, le, k, j, i) = src(e0 + le, k, j, i);
            });
        t.team_barrier();

        // --- PHASE 1: nine gradients --------------------------------------
        // One decode of (q, le, b, c) serves all nine members, which is the
        // property a fused level has and nine separate launches do not.
        Kokkos::parallel_for(Kokkos::TeamVectorRange(t, NN * SB), [&](int n) {
          const int q  = n / SB;
          const int r  = n % SB;
          const int c2 = r % NN, b2 = (r / NN) % NN, le = r / (NN * NN);
          for (int c = 0; c < 3; ++c) {
            float ax = 0.0f, ae = 0.0f, ag = 0.0f;
            for (int p = 0; p < NN; ++p) {
              ax += op(0, q, p) * us(c, le, b2, c2, p);  // sums i -> d/d(xi)
              ae += op(0, q, p) * us(c, le, b2, p, c2);  // sums j -> d/d(eta)
              ag += op(0, q, p) * us(c, le, p, b2, c2);  // sums k -> d/d(gamma)
            }
            G(0 + c, q, le, b2, c2) = ax;
            G(3 + c, q, le, b2, c2) = ae;
            G(6 + c, q, le, b2, c2) = ag;
          }
        });
        t.team_barrier();

        // --- PHASE 2: chain rule, stress, nine integrands -----------------
        // One chain rule and one stress per point, as the 9-output combine.
        // The three gradient families are read at their own canonical order.
        Kokkos::parallel_for(Kokkos::TeamVectorRange(t, TE * P), [&](int n) {
          const int i = n % NN, j = (n / NN) % NN, k = (n / (NN * NN)) % NN;
          const int le = n / P, e = e0 + le;
          float     dxi[3], det[3], dgm[3];
          for (int c = 0; c < 3; ++c) {
            dxi[c] = G(0 + c, i, le, k, j);
            det[c] = G(3 + c, j, le, k, i);
            dgm[c] = G(6 + c, k, le, j, i);
          }
          const Mat3  J   = metric_at(d.xix, d.xiy, d.xiz, d.etx, d.ety, d.etz,
                                      d.gmx, d.gmy, d.gmz, e, k, j, i);
          const Mat3  sig = compute_stress(chain_rule(dxi, det, dgm, J),
                                           d.l2m(e, k, j, i), d.mu(e, k, j, i));
          const float jc  = d.jac(e, k, j, i);
          for (int dir = 0; dir < 3; ++dir) {
            const auto f = project(sig, J.m[dir], jc);
            for (int c = 0; c < 3; ++c) F(3 * dir + c, le, k, j, i) = f[c];
          }
        });
        t.team_barrier();

        // --- PHASE 3: nine divergences ------------------------------------
        Kokkos::parallel_for(Kokkos::TeamVectorRange(t, NN * SB), [&](int n) {
          const int q  = n / SB;
          const int r  = n % SB;
          const int c2 = r % NN, b2 = (r / NN) % NN, le = r / (NN * NN);
          for (int c = 0; c < 3; ++c) {
            float ax = 0.0f, ae = 0.0f, ag = 0.0f;
            for (int p = 0; p < NN; ++p) {
              ax += op(1, p, q) * F(0 + c, le, b2, c2, p);
              ae += op(1, p, q) * F(3 + c, le, b2, p, c2);
              ag += op(1, p, q) * F(6 + c, le, p, b2, c2);
            }
            D(0 + c, q, le, b2, c2) = ax;
            D(3 + c, q, le, b2, c2) = ae;
            D(6 + c, q, le, b2, c2) = ag;
          }
        });
        t.team_barrier();

        // --- PHASE 4: the three weighted sums, straight to global ---------
        Kokkos::parallel_for(Kokkos::TeamVectorRange(t, TE * P), [&](int n) {
          const int   i = n % NN, j = (n / NN) % NN, k = (n / (NN * NN)) % NN;
          const int   le = n / P, e = e0 + le;
          const float wjk = d.w(j) * d.w(k), wik = d.w(i) * d.w(k),
                      wij = d.w(i) * d.w(j);
          float       f[3];
          for (int c = 0; c < 3; ++c)
            f[c] = wjk * D(0 + c, i, le, k, j) + wik * D(3 + c, j, le, k, i) +
                   wij * D(6 + c, k, le, j, i);
          d.f0(e, k, j, i) = f[0];
          d.f1(e, k, j, i) = f[1];
          d.f2(e, k, j, i) = f[2];
        });
      });
}
inline constexpr int kHandEnd = __LINE__;

template <int TE>
std::size_t hand4_scratch() {
  using scratch    = ES::scratch_memory_space;
  constexpr int NN = cfg::N;
  using SQ =
      Kokkos::View<float** [TE][NN][NN], scratch, Kokkos::MemoryUnmanaged>;
  using SE =
      Kokkos::View<float** [NN][NN][NN], scratch, Kokkos::MemoryUnmanaged>;
  using SO = Kokkos::View<float* [NN][NN], scratch, Kokkos::MemoryUnmanaged>;
  return SO::shmem_size(2) + SE::shmem_size(3, TE) + SQ::shmem_size(9, NN) +
         SE::shmem_size(9, TE) + SQ::shmem_size(9, NN);
}

// ===========================================================================
// Setup, host reference, timing -- verbatim from bench_sem_stiffness_3d.cpp so
// the two files' PASS/FAIL and timings mean the same thing.
// ===========================================================================
namespace {

void fill(V4 v, int E, int seed) {
  auto h = Kokkos::create_mirror_view(v);
  for (int e = 0; e < E; ++e)
    for (int k = 0; k < cfg::N; ++k)
      for (int j = 0; j < cfg::N; ++j)
        for (int i = 0; i < cfg::N; ++i)
          h(e, k, j, i) = 0.1f * std::sin(0.7f * (seed + 1) * (e % 11) +
                                          0.3f * k + 0.17f * j + 0.11f * i) +
                          0.6f;
  Kokkos::deep_copy(v, h);
}

void reference(Fields d, int E, V4H r0, V4H r1, V4H r2) {
  auto hu0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d.u0);
  auto hu1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d.u1);
  auto hu2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d.u2);
  auto hxx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d.xix);
  auto hxy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d.xiy);
  auto hxz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d.xiz);
  auto hex = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d.etx);
  auto hey = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d.ety);
  auto hez = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d.etz);
  auto hgx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d.gmx);
  auto hgy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d.gmy);
  auto hgz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d.gmz);
  auto hlm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d.l2m);
  auto hmu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d.mu);
  auto hja = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d.jac);
  auto hH  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d.H);
  auto hHw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d.Hw);
  auto hw  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d.w);

  static std::vector<float> F(9 * cfg::N * cfg::N * cfg::N);
  auto                      idx = [](int s, int k, int j, int i) {
    return ((s * cfg::N + k) * cfg::N + j) * cfg::N + i;
  };
  for (int e = 0; e < E; ++e) {
    for (int k = 0; k < cfg::N; ++k)
      for (int j = 0; j < cfg::N; ++j)
        for (int i = 0; i < cfg::N; ++i) {
          float dxi[3] = {0, 0, 0}, det[3] = {0, 0, 0}, dgm[3] = {0, 0, 0};
          for (int p = 0; p < cfg::N; ++p) {
            dxi[0] += hH(i, p) * hu0(e, k, j, p);
            dxi[1] += hH(i, p) * hu1(e, k, j, p);
            dxi[2] += hH(i, p) * hu2(e, k, j, p);
            det[0] += hH(j, p) * hu0(e, k, p, i);
            det[1] += hH(j, p) * hu1(e, k, p, i);
            det[2] += hH(j, p) * hu2(e, k, p, i);
            dgm[0] += hH(k, p) * hu0(e, p, j, i);
            dgm[1] += hH(k, p) * hu1(e, p, j, i);
            dgm[2] += hH(k, p) * hu2(e, p, j, i);
          }
          const Mat3 J = metric_at(hxx, hxy, hxz, hex, hey, hez, hgx, hgy, hgz,
                                   e, k, j, i);
          const Mat3 sig = compute_stress(chain_rule(dxi, det, dgm, J),
                                          hlm(e, k, j, i), hmu(e, k, j, i));
          for (int dir = 0; dir < 3; ++dir) {
            const auto f = project(sig, J.m[dir], hja(e, k, j, i));
            for (int c = 0; c < 3; ++c) F[idx(3 * dir + c, k, j, i)] = f[c];
          }
        }
    for (int k = 0; k < cfg::N; ++k)
      for (int j = 0; j < cfg::N; ++j)
        for (int i = 0; i < cfg::N; ++i) {
          float acc[3] = {0, 0, 0};
          for (int q = 0; q < cfg::N; ++q)
            for (int c = 0; c < 3; ++c)
              acc[c] += hw(j) * hw(k) * hHw(q, i) * F[idx(0 + c, k, j, q)] +
                        hw(i) * hw(k) * hHw(q, j) * F[idx(3 + c, k, q, i)] +
                        hw(i) * hw(j) * hHw(q, k) * F[idx(6 + c, q, j, i)];
          r0(e, k, j, i) = acc[0];
          r1(e, k, j, i) = acc[1];
          r2(e, k, j, i) = acc[2];
        }
  }
}

double check(Fields d, int Echk, V4H r0, V4H r1, V4H r2) {
  auto   g0    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d.f0);
  auto   g1    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d.f1);
  auto   g2    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d.f2);
  double worst = 0.0, scale = 1e-30;
  for (int e = 0; e < Echk; ++e)
    for (int k = 0; k < cfg::N; ++k)
      for (int j = 0; j < cfg::N; ++j)
        for (int i = 0; i < cfg::N; ++i) {
          const double a[3] = {g0(e, k, j, i), g1(e, k, j, i), g2(e, k, j, i)};
          const double b[3] = {r0(e, k, j, i), r1(e, k, j, i), r2(e, k, j, i)};
          for (int c = 0; c < 3; ++c) {
            worst = std::max(worst, std::abs(a[c] - b[c]));
            scale = std::max(scale, std::abs(b[c]));
          }
        }
  return worst / scale;
}

template <class Fn>
double best_ms(Fn&& fn, int warmup, int reps) {
  for (int i = 0; i < warmup; ++i) fn();
  Kokkos::fence();
  double best = 1e300;
  for (int r = 0; r < reps; ++r) {
    Kokkos::Timer timer;
    fn();
    Kokkos::fence();
    best = std::min(best, timer.seconds());
  }
  return best * 1e3;
}

}  // namespace

// TE as a compile-time fold, since it is a template parameter of both kernels.
// TE=1 is the row the sibling benchmark CANNOT express (its pinned NR=2 needs
// SB even); TE=16 is the row that will not launch (264 KB of scratch).
template <typename Fn>
void for_each_TE(Fn&& fn) {
  fn(std::integral_constant<int, 1>{});
  fn(std::integral_constant<int, 2>{});
  fn(std::integral_constant<int, 4>{});
  fn(std::integral_constant<int, 8>{});
}

double useful_flops(int E) {
  const double P = cfg::N * cfg::N * cfg::N;
  const double per_point =
      9.0 * 2.0 * cfg::N + 45.0 + 33.0 + 54.0 + 9.0 * 2.0 * cfg::N + 27.0;
  return double(E) * P * per_point;
}

struct Best {
  double      ms = 1e300;
  int         TE = 0, team = 0;
  double      err     = 0.0;
  std::size_t scratch = 0;
};

// The body lives in its own function so every View destructs before
// Kokkos::finalize() runs -- returning out of a scope inside main() leaves the
// host mirrors alive past finalize and aborts at exit.
int run(int argc, char* argv[]) {
  {
    const int E_in   = argc > 1 ? std::atoi(argv[1]) : cfg::E_default;
    const int reps   = argc > 2 ? std::atoi(argv[2]) : 5;
    const int warmup = argc > 3 ? std::atoi(argv[3]) : 2;
    const int E      = (E_in / cfg::TE_max) * cfg::TE_max;

    Fields d;
    d.u0  = V4("u0", E);
    d.u1  = V4("u1", E);
    d.u2  = V4("u2", E);
    d.xix = V4("xix", E);
    d.xiy = V4("xiy", E);
    d.xiz = V4("xiz", E);
    d.etx = V4("etx", E);
    d.ety = V4("ety", E);
    d.etz = V4("etz", E);
    d.gmx = V4("gmx", E);
    d.gmy = V4("gmy", E);
    d.gmz = V4("gmz", E);
    d.l2m = V4("l2m", E);
    d.mu  = V4("mu", E);
    d.jac = V4("jac", E);
    d.f0  = V4("f0", E);
    d.f1  = V4("f1", E);
    d.f2  = V4("f2", E);
    d.H   = V2("H", cfg::N, cfg::N);
    d.Hw  = V2("Hw", cfg::N, cfg::N);

    int s = 0;
    for (V4 v : {d.u0, d.u1, d.u2, d.xix, d.xiy, d.xiz, d.etx, d.ety, d.etz,
                 d.gmx, d.gmy, d.gmz, d.l2m, d.mu, d.jac})
      fill(v, E, s++);

    Kokkos::View<float*, Kokkos::LayoutRight> wv("w", cfg::N);
    {
      auto hH = Kokkos::create_mirror_view(d.H);
      auto hW = Kokkos::create_mirror_view(d.Hw);
      auto hw = Kokkos::create_mirror_view(wv);
      for (int a = 0; a < cfg::N; ++a) {
        hw(a) = 0.2f + 0.05f * a;
        for (int b = 0; b < cfg::N; ++b)
          hH(a, b) = 0.3f * std::sin(1.3f * a + 0.7f * b) + 0.1f * (a == b);
      }
      // Hw is H's WEIGHTED transpose: hprimewgll. This is where the quadrature
      // weight at the SUMMED index lives, which is why levels 3 and 4 between
      // them apply all three weights with only the summed one indexed by q.
      for (int a = 0; a < cfg::N; ++a)
        for (int b = 0; b < cfg::N; ++b) hW(a, b) = hH(b, a) * hw(b);
      Kokkos::deep_copy(d.H, hH);
      Kokkos::deep_copy(d.Hw, hW);
      Kokkos::deep_copy(wv, hw);
    }
    d.w = wv;

    const int Echk = std::min(E, 256);
    V4H       r0("r0", Echk), r1("r1", Echk), r2("r2", Echk);
    reference(d, Echk, r0, r1, r2);

    std::printf("3D SEM stiffness on the LEVEL GRAPH: NGLL=%d, E=%d, %s\n",
                cfg::N, E, ES::name());
    std::printf(
        "declarative 4-level graph vs a hand-written kernel of the "
        "same 4 stages\n\n");

    // --- scratch, before anything launches ------------------------------
    // Printed rather than derived: LevelGraph carves one buffer per slot with
    // no liveness pooling, and shmem_size rounds each allocation up, so a
    // closed-form estimate is a lower bound and not the number that decides
    // whether a tile launches at all.
    std::printf("%4s %12s %12s %9s\n", "TE", "graph(B)", "hand(B)", "ratio");
    std::printf("---- ------------ ------------ ---------\n");
    for_each_TE([&](auto tag) {
      constexpr int     TE = decltype(tag)::value;
      const std::size_t g  = levelgraph_scratch<TE>(d);
      const std::size_t h  = hand4_scratch<TE>();
      std::printf("%4d %12zu %12zu %8.2fx\n", TE, g, h, double(g) / double(h));
    });
    std::printf("\n");
    std::fflush(stdout);

    // --- profile mode: one launch each, for ncu -------------------------
    if (argc > 5) {
      const int  pTE = std::atoi(argv[4]), pteam = std::atoi(argv[5]);
      const auto one = [&](auto tag) {
        constexpr int TE = decltype(tag)::value;
        if (TE != pTE) return;
        if (levelgraph_scratch<TE>(d) > cfg::kScratchCap) {
          std::printf("TE=%d needs %zu B of scratch, over the %zu B cap\n", TE,
                      levelgraph_scratch<TE>(d), cfg::kScratchCap);
          return;
        }
        levelgraph_sem3d<TE>(d, pteam);
        Kokkos::fence();
        const double ge = check(d, Echk, r0, r1, r2);
        hand4_sem3d<TE>(d, E, pteam);
        Kokkos::fence();
        const double he = check(d, Echk, r0, r1, r2);
        std::printf("profile TE=%d team=%d: graph err %.1e, hand err %.1e\n",
                    TE, pteam, ge, he);
      };
      for_each_TE(one);
      return 0;
    }

    // --- the sweep ------------------------------------------------------
    // Best-vs-best across TE x team size, never at a fixed team size. The team
    // sweep is CAPPED at the iteration space: every level's range is exactly
    // 125*TE items, so a larger team idles threads in all four phases.
    Best graph, hand;
    std::printf("%4s %6s %10s %10s %11s %7s\n", "TE", "team", "graph(ms)",
                "hand(ms)", "gscratch", "blk/SM");
    std::printf("---- ------ ---------- ---------- ----------- -------\n");

    const auto sweep = [&](auto tag) {
      constexpr int     TE  = decltype(tag)::value;
      const std::size_t gsc = levelgraph_scratch<TE>(d);
      const std::size_t hsc = hand4_scratch<TE>();
      if (gsc > cfg::kScratchCap || hsc > cfg::kScratchCap) {
        std::printf("%4d %6s %10s %10s %11zu %7s  over %zu B cap\n", TE, "-",
                    "-", "-", gsc, "-", cfg::kScratchCap);
        return;
      }
      // The Serial backend permits a team of 1 only, so it is a correctness
      // oracle rather than a performance one -- but it IS one, which the
      // sibling GPU-only benchmark cannot offer: at TE=1 the graph needs
      // 16.7 KB, inside Serial's scratch cap.
      const std::vector<int> teams =
          cfg::kIsGPU ? std::vector<int>{32, 64, 128, 256, 512}
                      : std::vector<int>{1};
      for (int team : teams) {
        if (cfg::kIsGPU && team > 125 * TE) continue;
        const double gms =
            best_ms([&] { levelgraph_sem3d<TE>(d, team); }, warmup, reps);
        const double gerr = check(d, Echk, r0, r1, r2);
        const double hms =
            best_ms([&] { hand4_sem3d<TE>(d, E, team); }, warmup, reps);
        const double herr = check(d, Echk, r0, r1, r2);
        std::printf("%4d %6d %10.3f %10.3f %11zu %7zu\n", TE, team, gms, hms,
                    gsc, gsc ? (227u * 1024u) / gsc : 0u);
        if (gms < graph.ms) graph = Best{gms, TE, team, gerr, gsc};
        if (hms < hand.ms) hand = Best{hms, TE, team, herr, hsc};
      }
    };
    for_each_TE(sweep);

    const double flops = useful_flops(E);
    std::printf("\n%-12s %10s %6s %6s %11s %10s %12s\n", "impl", "time(ms)",
                "TE", "team", "scratch", "GFLOP/s", "check");
    std::printf(
        "------------ ---------- ------ ------ ----------- ---------- "
        "------------\n");
    const auto row = [&](const char* name, const Best& b) {
      std::printf("%-12s %10.3f %6d %6d %11zu %10.1f %8s(%.0e)\n", name, b.ms,
                  b.TE, b.team, b.scratch, flops / (b.ms * 1e-3) / 1e9,
                  b.err < 1e-4 ? "PASS" : "FAIL", b.err);
    };
    row("levelgraph", graph);
    row("hand4", hand);
    std::printf("\nlevelgraph vs hand (best against best): %.3fx\n",
                hand.ms / graph.ms);

    // --- source size ----------------------------------------------------
    // Two columns, not one total: the claim under test is that the graph's
    // PHYSICS lines match the hand kernel's while its PLUMBING lines do not.
    // Both call the same shared physics functions, counted against neither.
    std::printf(
        "\nSource size (lines; the %d shared physics lines both call "
        "are excluded):\n",
        kPhysEnd - kPhysBegin - 1);
    std::printf("%-12s %9s %9s   %s\n", "impl", "physics", "plumbing", "what");
    std::printf("%-12s %9d %9d   %s\n", "levelgraph", kFnEnd - kFnBegin - 1,
                kGraphEnd - kGraphBegin - 1,
                "two functors; stages + 4 add() calls");
    std::printf("%-12s %9d %9d   %s\n", "hand4", 0, kHandEnd - kHandBegin - 1,
                "scratch views, 4 ranges, 4 barriers, index decode");
    std::printf(
        "\nFor reference, the sibling benchmark spends 210 lines "
        "(library, Tag1 DAG)\nand 353 (library2, caller-driven Tag2) "
        "on this same pipeline.\n");

    return (graph.err < 1e-4 && hand.err < 1e-4) ? 0 : 1;
  }
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  const int rc = run(argc, argv);
  Kokkos::finalize();
  return rc;
}
