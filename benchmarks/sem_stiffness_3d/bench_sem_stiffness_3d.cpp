// ===========================================================================
// bench_sem_stiffness_3d.cpp
//
// The 3D spectral-element stiffness kernel (Komatitsch & Tromp 1999, eqns
// A2/A7) at the PRODUCTION order NGLL=5, three ways, on the same data:
//
//   1. library  — the whole pipeline as ONE TensorOperations DAG with
//                 multi-output combine nodes. 24 nodes, one launch, every
//                 intermediate in team scratch.
//   2. hand     — a hand-written fused Kokkos team kernel: stage into scratch,
//                 gradients + stress + integrands, barrier, divergence.
//                 SPECFEM3D's structure, written for readability.
//   3. naive    — the obvious first kernel: two MDRange passes over global
//                 memory, no scratch, no barriers, integrands round-tripped
//                 through nine global temporaries.
//
// All three call the SAME compute_stress / chain_rule / project, so the physics
// work is identical and the comparison is about data movement and about how
// much code it costs to say the same thing.
//
// THE CLAIM UNDER TEST is two-sided, and the naive row is what makes it a
// claim rather than a tautology:
//
//   * naive is readable but slow      — simple code, no reuse
//   * hand  is fast but not readable  — the speed comes from scratch, index
//                                       arithmetic and barriers, which is most
//                                       of what the source says
//   * library is meant to be both     — the graph states the physics and the
//                                       library supplies the staging
//
// MEASURED RESULT: AT THE PRODUCTION ORDER THE CLAIM FAILS. H100 NVL,
// best-against-best over a TE x team sweep, correctness ~1e-07 against the
// host reference in every row:
//
//   NGLL=5, E=786432        naive 100.4 ms   hand  3.94 ms   library 10.49 ms
//   NGLL=8, E=196608        naive  78.5 ms   hand 13.67 ms   library 11.74 ms
//
// So the library is 2.7x SLOWER than the hand kernel at NGLL=5 and 1.16x
// faster at N=8 -- and it is 9.6x / 6.7x faster than naive at the two orders.
// On source size it is the LONGEST of the three (188 lines against hand's 90
// and naive's 41, excluding the shared physics). Both halves of the claim are
// therefore unsupported at the production order.
//
// WHY, from ncu (thread-instructions per output point, N=5, TE=2, team=96):
//
//   library 1696, hand 648.  The 1048 gap is:
//     58%  address / integer ALU     (IMAD +305, LEA +79, SHF +70, IADD3 +59)
//     19%  uniform / scalar datapath
//     13%  branch + barrier + sync   (BAR 26.5 vs 0.8 -- 24 nodes x ~2 each)
//      7%  float: 3 integrand nodes each rebuild the stress tensor
//      3%  global: those 3 nodes each re-read all 12 metric/material fields
//      1%  shared memory
//
// It is NOT losing on data movement: DRAM is 71.7 bytes/point for BOTH, and
// the library issues FEWER shared loads than the hand kernel (102 vs 120). It
// loses on computing addresses -- integer:float instruction ratio 3.06 against
// the hand kernel's 1.18. Most of that is the flat->coordinate DECODE that
// every element of every node pays (TiledLayout/TileLayout, ~366 inst/point,
// 22% of the kernel), which at N=5 is a magic-multiply IMAD chain per rank
// because the extents are not powers of two. An attempt to hoist it as
// redundant offset arithmetic was MEASURED AND FAILED: -19 instructions of
// 776, -0.3% runtime. Removing it needs a strength-reduced traversal that
// increments the coordinate instead of decoding it.
//
// THE N=8 WIN IS NOT A LIBRARY WIN, AND SHOULD NOT BE QUOTED AS ONE. The
// library's cost per output point is FLAT across the two orders (0.2275 ->
// 0.2257 ns, 0.99x); the comparison inverts because the HAND kernel degrades
// 3.38x per point (0.0697 -> 0.2355 ns), with its shared-memory traffic up 5x
// and stall_short_scoreboard up 26x. That degradation is UNEXPLAINED: ncu
// reports 12.8x the shared bank conflicts at N=8, but padding the innermost
// scratch extent -- the standard fix, -DSEM3D_PAD=1 below -- moved the time by
// 0.01% (13.649 -> 13.651 ms), and a static analysis of the three gradient
// reads says they are broadcast or unit-stride at both orders. A hand kernel
// that avoids whatever this is would plausibly beat the library at N=8 too.
//
// Per element the pipeline is:
//
//   A2  gradient    du_c/d(xi), du_c/d(eta), du_c/d(gamma)   9 contractions
//   --  chain rule  ds[c][d] = du_c/dx_d                     pointwise
//   --  stress      T = compute_stress(ds)                   BLACK BOX
//   --  integrand   F^r_c = J * sum_d T[c][d] * dr/dx_d      3 dirs x 3 comps
//   A7  divergence  t^r_c = sum_q Hw[q,.] F^r_c              9 contractions
//   --  force       f_c = w_j w_k t^xi + w_i w_k t^eta + w_i w_j t^gam
//
// WHY NGLL=5 IS THE POINT, AND WHY IT NEEDS A BUILD-TIME OVERRIDE. It is what
// production spectral-element codes run. The team GEMM register-blocks by
// (MT, NT, NR) and has no remainder path, so every staged extent must be a
// multiple of its factor; this kernel stages SA = SK = N and SB = TE*N*N. At
// N=5 those are prime and the library defaults (MT=4, NT=2) divide neither, so
// THE BENCHMARK DOES NOT COMPILE without -DTENSOR_GEMM_MT=5 -DTENSOR_GEMM_NT=5
// -DTENSOR_GEMM_NR=2, which its CMake target supplies (and must NOT supply at
// N=8, where the defaults already divide). Read that as the finding it is: as
// shipped, this library cannot express a production-order SEM kernel without
// retuning its GEMM per target.
//
// LABELS. e = element; i/j/k = the x/y/z axes; p = the gradient's contracted
// index; q = the divergence's contracted index and the "new" axis a gradient
// writes. Each gradient buffer is named differently by each of its three
// consumers -- same bytes, three label assignments, no data movement.
//
// GPU ONLY. At NGLL=5 the CPU backend's register blocking (MT=NT=8, NR=2*W,
// none of them overridable) divides none of the staged extents, and the graph's
// scratch does not fit Serial's 32 KB cap at any legal tile. Both would need
// the GEMM remainder path that does not exist yet.
//
// Usage: bench_sem_stiffness_3d [nspec [reps [warmup]]]
// ===========================================================================

#include <Kokkos_Core.hpp>
#include <TensorOperations/DagGraph.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/Tiling.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace TensorOperations;

namespace cfg {
inline constexpr bool kIsGPU =
    !Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace,
                                Kokkos::HostSpace>::accessible;

// FIXED AT THE PRODUCTION ORDER. Every tile extent, the legal TE list and the
// GEMM register blocking are all keyed to this, and the blocking override the
// CMake target applies (5/5/2) is only valid here -- see the header. The N=8
// comparison quoted above was taken by temporarily changing this constant and
// dropping that override; reproducing it means doing the same.
inline constexpr int N = 5;  // GLL points per direction (production NGLL)
inline constexpr int C = 3;  // displacement components

// Elements per team tile. SWEPT -- see main(). A slot is TE*N^3*4 bytes and the
// DAG pools 30 slots into 18 buffers, so TE sets shared memory per block and
// therefore blocks per SM; team size sets warps per block. The two are
// independent and are swept together.
//
// 18 pools * TE * N^3 * 4 bytes must stay under the H100's 227 KB opt-in
// shared memory; at N=5 that leaves TE=8 the largest usable tile.
inline constexpr int TE_max = 8;

// Sized so every timed row clears ~10 ms: 98M output points. 27 device arrays
// of E*N^3 floats (15 inputs, 3 outputs, 9 naive temporaries) is ~11 GB.
inline constexpr int E_default = 786432;
}  // namespace cfg

// Element-local fields are [E, k, j, i] = (element, z, y, x).
using V4  = Kokkos::View<float * [cfg::N][cfg::N][cfg::N], Kokkos::LayoutRight>;
using V2  = Kokkos::View<float**, Kokkos::LayoutRight>;
using V1  = Kokkos::View<const float*, Kokkos::LayoutRight>;
using V4H = V4::host_mirror_type;
// The naive path's nine integrand temporaries, as ONE view: nine separate
// views would have to be indexed through a Kokkos::Array in the kernel, and a
// dynamically indexed Array member cannot stay in registers.
using V5 =
    Kokkos::View<float* [9][cfg::N][cfg::N][cfg::N], Kokkos::LayoutRight>;

// ===========================================================================
// SHARED PHYSICS -- the black box. Identical for all three implementations, so
// none of them can drift from the others and the arithmetic is the same work.
// Counted once in the source-size table, against nobody.
// ===========================================================================
inline constexpr int kPhysBegin = __LINE__;
struct Mat3 {
  float m[3][3];
};

// ds[c][d] = du_c/dx_d, from the three reference gradients of each component
// and the inverse Jacobian rows J[r][d] = d(ref_r)/d(x_d).
KOKKOS_INLINE_FUNCTION Mat3 chain_rule(const float dxi[3], const float det[3],
                                       const float dgm[3], const Mat3& J) {
  Mat3 ds{};
  for (int c = 0; c < 3; ++c)
    for (int d = 0; d < 3; ++d)
      ds.m[c][d] = dxi[c] * J.m[0][d] + det[c] * J.m[1][d] + dgm[c] * J.m[2][d];
  return ds;
}

// Isotropic elastic stress. The library never looks inside it.
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

// F^r_c = J * sum_d sigma[c][d] * d(ref_r)/d(x_d), all three components at
// once.
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

// The inverse Jacobian at one point, gathered from the nine metric views.
// Shared plumbing, used by all three implementations.
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

// Every implementation takes the same 18 views; bundling them keeps the three
// signatures identical and out of the line counts.
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
// Implementation 1: THE LIBRARY. One DAG, one launch, 24 nodes.
//
// 9 gradients (each computed once), 3 multi-output integrand nodes (9 operands
// -> 3 outputs each: one chain rule and one stress evaluation per point serves
// all three force components), 9 divergences, 3 weighted sums.
//
// THE LOAD-BEARING RULE: a fused node is NAMED in its consumer's canonical
// order (a contraction's B operand is canonically contracted ++ freeB), which
// makes the operand permutation the identity and lets the consumer read the
// producer's buffer in place instead of staging a reorder.
// ===========================================================================
template <int TE>
struct LibTiles {
  using H_   = StaticTile<cfg::N, cfg::N>;              // [5,5]
  using E_   = StaticTile<TE, cfg::N, cfg::N, cfg::N>;  // (e,k,j,i)
  using Q_   = StaticTile<cfg::N, TE, cfg::N, cfg::N>;  // (q,e,.,.)
  using Grad = Tile<H_, E_, Q_>;                        // H  x u -> grad
  using Div  = Tile<H_, Q_, E_>;                        // Hw x F -> div
  using Comb = CombineTile<Q_, Q_, Q_, Q_, Q_, Q_, Q_, Q_, Q_, Q_>;
  using Sum  = CombineTile<E_, E_, E_, E_>;
};

inline constexpr int kLibBegin = __LINE__;
template <int TE>
void library_sem3d(Fields d, int team) {
  using T = LibTiles<TE>;
  // --- the three integrand functors, one per reference direction ----------
  // 9 gradients in, 3 force components out. The seven per-point fields are
  // CAPTURED, not passed as operands: a combine reads each operand once, so
  // staging them into scratch would buy nothing, and w is rank-1 and could not
  // be an operand at all (every operand carries the output's full label set).
  const auto integrand_xi =
      KOKKOS_LAMBDA(int q, int e, int k, int j, float x0, float x1, float x2,
                    float e0, float e1, float e2, float g0, float g1, float g2)
          ->Kokkos::Array<float, 3> {
    const int   z = k, y = j, x = q;  // node modes {q,e,k,j}
    const float dxi[3] = {x0, x1, x2}, det[3] = {e0, e1, e2},
                dgm[3] = {g0, g1, g2};
    const Mat3  J   = metric_at(d.xix, d.xiy, d.xiz, d.etx, d.ety, d.etz, d.gmx,
                                d.gmy, d.gmz, e, z, y, x);
    const Mat3  sig = compute_stress(chain_rule(dxi, det, dgm, J),
                                     d.l2m(e, z, y, x), d.mu(e, z, y, x));
    return project(sig, J.m[0], d.jac(e, z, y, x));
  };
  const auto integrand_eta =
      KOKKOS_LAMBDA(int q, int e, int k, int i, float x0, float x1, float x2,
                    float e0, float e1, float e2, float g0, float g1, float g2)
          ->Kokkos::Array<float, 3> {
    const int   z = k, y = q, x = i;  // node modes {q,e,k,i}
    const float dxi[3] = {x0, x1, x2}, det[3] = {e0, e1, e2},
                dgm[3] = {g0, g1, g2};
    const Mat3  J   = metric_at(d.xix, d.xiy, d.xiz, d.etx, d.ety, d.etz, d.gmx,
                                d.gmy, d.gmz, e, z, y, x);
    const Mat3  sig = compute_stress(chain_rule(dxi, det, dgm, J),
                                     d.l2m(e, z, y, x), d.mu(e, z, y, x));
    return project(sig, J.m[1], d.jac(e, z, y, x));
  };
  const auto integrand_gam =
      KOKKOS_LAMBDA(int q, int e, int j, int i, float x0, float x1, float x2,
                    float e0, float e1, float e2, float g0, float g1, float g2)
          ->Kokkos::Array<float, 3> {
    const int   z = q, y = j, x = i;  // node modes {q,e,j,i}
    const float dxi[3] = {x0, x1, x2}, det[3] = {e0, e1, e2},
                dgm[3] = {g0, g1, g2};
    const Mat3  J   = metric_at(d.xix, d.xiy, d.xiz, d.etx, d.ety, d.etz, d.gmx,
                                d.gmy, d.gmz, e, z, y, x);
    const Mat3  sig = compute_stress(chain_rule(dxi, det, dgm, J),
                                     d.l2m(e, z, y, x), d.mu(e, z, y, x));
    return project(sig, J.m[2], d.jac(e, z, y, x));
  };
  // f_c = w_j w_k t^xi + w_i w_k t^eta + w_i w_j t^gam. The weights ride in the
  // capture as a View, never a Kokkos::Array: a dynamically indexed array
  // member cannot stay in registers and spills the closure to the thread stack.
  const auto weighted_sum =
      KOKKOS_LAMBDA(int, int k, int j, int i, float t1, float t2, float t3)
          ->float {
    return d.w(j) * d.w(k) * t1 + d.w(i) * d.w(k) * t2 + d.w(i) * d.w(j) * t3;
  };

  // --- A2: nine gradients, each computed once -----------------------------
  // xi contracts x, eta contracts y, gamma contracts z.
  auto [d0, gx0] = make_dag<float>().add(
      make_contraction_node<'q', 'e', 'k', 'j'>(
          make_input_node(make_handle<'q', 'p'>(d.H)),
          make_input_node(make_handle<'e', 'k', 'j', 'p'>(d.u0))),
      typename T::Grad{});
  auto [d1, gx1] =
      d0.add(make_contraction_node<'q', 'e', 'k', 'j'>(
                 make_input_node(make_handle<'q', 'p'>(d.H)),
                 make_input_node(make_handle<'e', 'k', 'j', 'p'>(d.u1))),
             typename T::Grad{});
  auto [d2, gx2] =
      d1.add(make_contraction_node<'q', 'e', 'k', 'j'>(
                 make_input_node(make_handle<'q', 'p'>(d.H)),
                 make_input_node(make_handle<'e', 'k', 'j', 'p'>(d.u2))),
             typename T::Grad{});
  auto [d3, ge0] =
      d2.add(make_contraction_node<'q', 'e', 'k', 'i'>(
                 make_input_node(make_handle<'q', 'p'>(d.H)),
                 make_input_node(make_handle<'e', 'k', 'p', 'i'>(d.u0))),
             typename T::Grad{});
  auto [d4, ge1] =
      d3.add(make_contraction_node<'q', 'e', 'k', 'i'>(
                 make_input_node(make_handle<'q', 'p'>(d.H)),
                 make_input_node(make_handle<'e', 'k', 'p', 'i'>(d.u1))),
             typename T::Grad{});
  auto [d5, ge2] =
      d4.add(make_contraction_node<'q', 'e', 'k', 'i'>(
                 make_input_node(make_handle<'q', 'p'>(d.H)),
                 make_input_node(make_handle<'e', 'k', 'p', 'i'>(d.u2))),
             typename T::Grad{});
  auto [d6, gg0] =
      d5.add(make_contraction_node<'q', 'e', 'j', 'i'>(
                 make_input_node(make_handle<'q', 'p'>(d.H)),
                 make_input_node(make_handle<'e', 'p', 'j', 'i'>(d.u0))),
             typename T::Grad{});
  auto [d7, gg1] =
      d6.add(make_contraction_node<'q', 'e', 'j', 'i'>(
                 make_input_node(make_handle<'q', 'p'>(d.H)),
                 make_input_node(make_handle<'e', 'p', 'j', 'i'>(d.u1))),
             typename T::Grad{});
  auto [d8, gg2] =
      d7.add(make_contraction_node<'q', 'e', 'j', 'i'>(
                 make_input_node(make_handle<'q', 'p'>(d.H)),
                 make_input_node(make_handle<'e', 'p', 'j', 'i'>(d.u2))),
             typename T::Grad{});

  // --- integrands: three nodes, three outputs each -------------------------
  // Every node reads all nine gradients; only the label assignment differs.
  // gx* is physically (new-x, e, z, y), ge* is (new-y, e, z, x), gg* is
  // (new-z, e, y, x) -- so each consumer just says which of its own axes those
  // are.
  auto [d9, fx0, fx1, fx2] =
      d8.add(make_combine_node<'q', 'e', 'k', 'j'>(
                 gx0.template as<'q', 'e', 'k', 'j'>(),
                 gx1.template as<'q', 'e', 'k', 'j'>(),
                 gx2.template as<'q', 'e', 'k', 'j'>(),
                 ge0.template as<'j', 'e', 'k', 'q'>(),
                 ge1.template as<'j', 'e', 'k', 'q'>(),
                 ge2.template as<'j', 'e', 'k', 'q'>(),
                 gg0.template as<'k', 'e', 'j', 'q'>(),
                 gg1.template as<'k', 'e', 'j', 'q'>(),
                 gg2.template as<'k', 'e', 'j', 'q'>(), integrand_xi),
             typename T::Comb{});
  auto [d10, fe0, fe1, fe2] =
      d9.add(make_combine_node<'q', 'e', 'k', 'i'>(
                 gx0.template as<'i', 'e', 'k', 'q'>(),
                 gx1.template as<'i', 'e', 'k', 'q'>(),
                 gx2.template as<'i', 'e', 'k', 'q'>(),
                 ge0.template as<'q', 'e', 'k', 'i'>(),
                 ge1.template as<'q', 'e', 'k', 'i'>(),
                 ge2.template as<'q', 'e', 'k', 'i'>(),
                 gg0.template as<'k', 'e', 'q', 'i'>(),
                 gg1.template as<'k', 'e', 'q', 'i'>(),
                 gg2.template as<'k', 'e', 'q', 'i'>(), integrand_eta),
             typename T::Comb{});
  auto [d11, fg0, fg1, fg2] =
      d10.add(make_combine_node<'q', 'e', 'j', 'i'>(
                  gx0.template as<'i', 'e', 'q', 'j'>(),
                  gx1.template as<'i', 'e', 'q', 'j'>(),
                  gx2.template as<'i', 'e', 'q', 'j'>(),
                  ge0.template as<'j', 'e', 'q', 'i'>(),
                  ge1.template as<'j', 'e', 'q', 'i'>(),
                  ge2.template as<'j', 'e', 'q', 'i'>(),
                  gg0.template as<'q', 'e', 'j', 'i'>(),
                  gg1.template as<'q', 'e', 'j', 'i'>(),
                  gg2.template as<'q', 'e', 'j', 'i'>(), integrand_gam),
              typename T::Comb{});

  // --- A7: nine divergences ------------------------------------------------
  auto [d12, tx0] = d11.add(make_contraction_node<'e', 'k', 'j', 'i'>(
                                make_input_node(make_handle<'q', 'i'>(d.Hw)),
                                fx0.template as<'q', 'e', 'k', 'j'>()),
                            typename T::Div{});
  auto [d13, tx1] = d12.add(make_contraction_node<'e', 'k', 'j', 'i'>(
                                make_input_node(make_handle<'q', 'i'>(d.Hw)),
                                fx1.template as<'q', 'e', 'k', 'j'>()),
                            typename T::Div{});
  auto [d14, tx2] = d13.add(make_contraction_node<'e', 'k', 'j', 'i'>(
                                make_input_node(make_handle<'q', 'i'>(d.Hw)),
                                fx2.template as<'q', 'e', 'k', 'j'>()),
                            typename T::Div{});
  auto [d15, te0] = d14.add(make_contraction_node<'e', 'k', 'j', 'i'>(
                                make_input_node(make_handle<'q', 'j'>(d.Hw)),
                                fe0.template as<'q', 'e', 'k', 'i'>()),
                            typename T::Div{});
  auto [d16, te1] = d15.add(make_contraction_node<'e', 'k', 'j', 'i'>(
                                make_input_node(make_handle<'q', 'j'>(d.Hw)),
                                fe1.template as<'q', 'e', 'k', 'i'>()),
                            typename T::Div{});
  auto [d17, te2] = d16.add(make_contraction_node<'e', 'k', 'j', 'i'>(
                                make_input_node(make_handle<'q', 'j'>(d.Hw)),
                                fe2.template as<'q', 'e', 'k', 'i'>()),
                            typename T::Div{});
  auto [d18, tg0] = d17.add(make_contraction_node<'e', 'k', 'j', 'i'>(
                                make_input_node(make_handle<'q', 'k'>(d.Hw)),
                                fg0.template as<'q', 'e', 'j', 'i'>()),
                            typename T::Div{});
  auto [d19, tg1] = d18.add(make_contraction_node<'e', 'k', 'j', 'i'>(
                                make_input_node(make_handle<'q', 'k'>(d.Hw)),
                                fg1.template as<'q', 'e', 'j', 'i'>()),
                            typename T::Div{});
  auto [d20, tg2] = d19.add(make_contraction_node<'e', 'k', 'j', 'i'>(
                                make_input_node(make_handle<'q', 'k'>(d.Hw)),
                                fg2.template as<'q', 'e', 'j', 'i'>()),
                            typename T::Div{});

  // --- the weighted sums ---------------------------------------------------
  // A divergence writes freeA ++ freeB: {i,e,k,j} for xi, {j,e,k,i} for eta,
  // {k,e,j,i} for gamma.
  auto [d21, r0] =
      d20.add(make_combine_node<'e', 'k', 'j', 'i'>(
                  tx0.template as<'i', 'e', 'k', 'j'>(),
                  te0.template as<'j', 'e', 'k', 'i'>(),
                  tg0.template as<'k', 'e', 'j', 'i'>(), weighted_sum),
              typename T::Sum{});
  auto [d22, r1] =
      d21.add(make_combine_node<'e', 'k', 'j', 'i'>(
                  tx1.template as<'i', 'e', 'k', 'j'>(),
                  te1.template as<'j', 'e', 'k', 'i'>(),
                  tg1.template as<'k', 'e', 'j', 'i'>(), weighted_sum),
              typename T::Sum{});
  auto [d23, r2] =
      d22.add(make_combine_node<'e', 'k', 'j', 'i'>(
                  tx2.template as<'i', 'e', 'k', 'j'>(),
                  te2.template as<'j', 'e', 'k', 'i'>(),
                  tg2.template as<'k', 'e', 'j', 'i'>(), weighted_sum),
              typename T::Sum{});

  d23.outputs(r0, r1, r2)
      .team_size(team)
      .execute(TeamPolicyTag<>{}, d.f0, d.f1, d.f2);
}
inline constexpr int kLibEnd = __LINE__;

// ===========================================================================
// Implementation 2: HAND-WRITTEN FUSED KERNEL, SPECFEM3D-style.
//
// One launch, one team per TE elements. Stage the displacement and the two
// operators into team scratch; compute the nine gradients, the stress and the
// nine integrands per point; barrier; contract the integrands back down and
// apply the quadrature weights.
//
// Written for readability -- but note what "readable" costs here: the scratch
// views have to be declared and sized, the flat thread index has to be
// decoded into (element, k, j, i) twice, and the barrier between the two
// phases is load-bearing and invisible in the physics.
// ===========================================================================
inline constexpr int kHandBegin = __LINE__;
template <int TE>
void hand_sem3d(Fields d, int E, int team) {
  using member_t = Kokkos::TeamPolicy<>::member_type;
  // -DSEM3D_PAD=1 pads the innermost scratch extent, which makes the j and k
  // strides odd (N+1 and N*(N+1)) instead of powers of two at N=8. The standard
  // shared-memory bank-conflict fix.
  //
  // KEPT BECAUSE IT MEASURED NEGATIVE, and that is the informative part: ncu
  // reports 207M shared bank conflicts for this kernel at N=8 against 16M at
  // N=5, but padding moves the time by 0.01% (13.649 -> 13.651 ms at N=8,
  // 3.969 -> 3.996 at N=5). Whatever costs the hand kernel 3.4x per point at
  // N=8, it is not a stride/bank pathology that padding fixes -- see the
  // header. Leave this knob here so the next person does not spend the
  // afternoon re-deriving that.
#ifndef SEM3D_PAD
#define SEM3D_PAD 0
#endif
  constexpr int SP = cfg::N + SEM3D_PAD;
  using Scratch =
      Kokkos::View<float* * [cfg::N][cfg::N][SP],
                   Kokkos::DefaultExecutionSpace::scratch_memory_space,
                   Kokkos::MemoryUnmanaged>;
  using ScratchOp =
      Kokkos::View<float * [cfg::N][cfg::N],
                   Kokkos::DefaultExecutionSpace::scratch_memory_space,
                   Kokkos::MemoryUnmanaged>;

  constexpr int     P     = cfg::N * cfg::N * cfg::N;  // points per element
  const int         teams = E / TE;
  const std::size_t bytes = Scratch::shmem_size(3, TE)    // staged u
                            + Scratch::shmem_size(9, TE)  // integrands
                            + ScratchOp::shmem_size(2);   // H, Hw

  Kokkos::parallel_for(
      "hand_sem3d",
      Kokkos::TeamPolicy<>(teams, team)
          .set_scratch_size(0, Kokkos::PerTeam(static_cast<int>(bytes))),
      KOKKOS_LAMBDA(const member_t& t) {
        const int e0 = t.league_rank() * TE;

        Scratch   us(t.team_scratch(0), 3, TE);  // [c][le][k][j][i]
        Scratch   Fs(t.team_scratch(0), 9, TE);  // [3*dir+c][le][k][j][i]
        ScratchOp op(t.team_scratch(0), 2);      // [0]=H, [1]=Hw

        // --- stage: displacement and both operators into scratch -----------
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(t, 3 * TE * P), [&](int n) {
              const int i = n % cfg::N, j = (n / cfg::N) % cfg::N;
              const int k  = (n / (cfg::N * cfg::N)) % cfg::N;
              const int le = (n / P) % TE, c = n / (P * TE);
              const V4& src      = (c == 0) ? d.u0 : (c == 1) ? d.u1 : d.u2;
              us(c, le, k, j, i) = src(e0 + le, k, j, i);
            });
        Kokkos::parallel_for(Kokkos::TeamVectorRange(t, cfg::N * cfg::N),
                             [&](int n) {
                               const int a = n / cfg::N, b = n % cfg::N;
                               op(0, a, b) = d.H(a, b);
                               op(1, a, b) = d.Hw(a, b);
                             });
        t.team_barrier();

        // --- gradients, stress, integrands ---------------------------------
        Kokkos::parallel_for(Kokkos::TeamVectorRange(t, TE * P), [&](int n) {
          const int i = n % cfg::N, j = (n / cfg::N) % cfg::N;
          const int k  = (n / (cfg::N * cfg::N)) % cfg::N;
          const int le = n / P;
          const int e  = e0 + le;

          float dxi[3] = {0, 0, 0}, det[3] = {0, 0, 0}, dgm[3] = {0, 0, 0};
          for (int p = 0; p < cfg::N; ++p)
            for (int c = 0; c < 3; ++c) {
              dxi[c] += op(0, i, p) * us(c, le, k, j, p);
              det[c] += op(0, j, p) * us(c, le, k, p, i);
              dgm[c] += op(0, k, p) * us(c, le, p, j, i);
            }

          const Mat3 J   = metric_at(d.xix, d.xiy, d.xiz, d.etx, d.ety, d.etz,
                                     d.gmx, d.gmy, d.gmz, e, k, j, i);
          const Mat3 sig = compute_stress(chain_rule(dxi, det, dgm, J),
                                          d.l2m(e, k, j, i), d.mu(e, k, j, i));
          for (int dir = 0; dir < 3; ++dir) {
            const auto F = project(sig, J.m[dir], d.jac(e, k, j, i));
            for (int c = 0; c < 3; ++c) Fs(3 * dir + c, le, k, j, i) = F[c];
          }
        });
        t.team_barrier();

        // --- divergence and quadrature weights -----------------------------
        Kokkos::parallel_for(Kokkos::TeamVectorRange(t, TE * P), [&](int n) {
          const int i = n % cfg::N, j = (n / cfg::N) % cfg::N;
          const int k  = (n / (cfg::N * cfg::N)) % cfg::N;
          const int le = n / P;
          const int e  = e0 + le;

          float acc[3] = {0, 0, 0};
          for (int q = 0; q < cfg::N; ++q)
            for (int c = 0; c < 3; ++c)
              acc[c] += d.w(j) * d.w(k) * op(1, q, i) * Fs(0 + c, le, k, j, q) +
                        d.w(i) * d.w(k) * op(1, q, j) * Fs(3 + c, le, k, q, i) +
                        d.w(i) * d.w(j) * op(1, q, k) * Fs(6 + c, le, q, j, i);
          d.f0(e, k, j, i) = acc[0];
          d.f1(e, k, j, i) = acc[1];
          d.f2(e, k, j, i) = acc[2];
        });
      });
}
inline constexpr int kHandEnd = __LINE__;

// ===========================================================================
// Implementation 3: NAIVE. The obvious first kernel.
//
// Two flat passes over global memory: compute the nine integrands and write
// them out, then read them back and contract. No scratch, no barriers, no team
// structure -- and no reuse either: every point re-reads its whole line of the
// displacement from global memory, and the integrands make a full round trip
// through DRAM between the passes.
// ===========================================================================
inline constexpr int kNaiveBegin = __LINE__;
void                 naive_sem3d(Fields d, int E, V5 F) {
  using Range = Kokkos::MDRangePolicy<Kokkos::Rank<4>>;
  const Range points({0, 0, 0, 0}, {E, cfg::N, cfg::N, cfg::N});

  Kokkos::parallel_for(
      "naive_integrands", points, KOKKOS_LAMBDA(int e, int k, int j, int i) {
        float dxi[3] = {0, 0, 0}, det[3] = {0, 0, 0}, dgm[3] = {0, 0, 0};
        for (int p = 0; p < cfg::N; ++p) {
          dxi[0] += d.H(i, p) * d.u0(e, k, j, p);
          dxi[1] += d.H(i, p) * d.u1(e, k, j, p);
          dxi[2] += d.H(i, p) * d.u2(e, k, j, p);
          det[0] += d.H(j, p) * d.u0(e, k, p, i);
          det[1] += d.H(j, p) * d.u1(e, k, p, i);
          det[2] += d.H(j, p) * d.u2(e, k, p, i);
          dgm[0] += d.H(k, p) * d.u0(e, p, j, i);
          dgm[1] += d.H(k, p) * d.u1(e, p, j, i);
          dgm[2] += d.H(k, p) * d.u2(e, p, j, i);
        }
        const Mat3 J   = metric_at(d.xix, d.xiy, d.xiz, d.etx, d.ety, d.etz,
                                   d.gmx, d.gmy, d.gmz, e, k, j, i);
        const Mat3 sig = compute_stress(chain_rule(dxi, det, dgm, J),
                                        d.l2m(e, k, j, i), d.mu(e, k, j, i));
        for (int dir = 0; dir < 3; ++dir) {
          const auto f = project(sig, J.m[dir], d.jac(e, k, j, i));
          for (int c = 0; c < 3; ++c) F(e, 3 * dir + c, k, j, i) = f[c];
        }
      });

  Kokkos::parallel_for(
      "naive_divergence", points, KOKKOS_LAMBDA(int e, int k, int j, int i) {
        float acc[3] = {0, 0, 0};
        for (int q = 0; q < cfg::N; ++q)
          for (int c = 0; c < 3; ++c)
            acc[c] += d.w(j) * d.w(k) * d.Hw(q, i) * F(e, 0 + c, k, j, q) +
                      d.w(i) * d.w(k) * d.Hw(q, j) * F(e, 3 + c, k, q, i) +
                      d.w(i) * d.w(j) * d.Hw(q, k) * F(e, 6 + c, q, j, i);
        d.f0(e, k, j, i) = acc[0];
        d.f1(e, k, j, i) = acc[1];
        d.f2(e, k, j, i) = acc[2];
      });
}
inline constexpr int kNaiveEnd = __LINE__;

// ===========================================================================
// Setup, host reference, timing
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

// The pipeline transcribed straight onto the host, for validation.
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

// Max |diff| over the checked elements, relative to the reference's own scale.
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

// The tile sizes swept, as a compile-time list: TE is a template parameter of
// both kernels, so this is a fold, not a loop.
//
// TE=1 is absent and that is a HARD constraint, not a choice: SB = TE*N*N = 25
// is odd, so it is not a multiple of the register column block NR=2 that the
// CMake target pins, and the GEMM has no remainder path. TE=16 would need
// 18 pools x 16 x 125 x 4 = 144 KB of the H100's 227 KB, which runs but at one
// block per SM.
//
// Anything instantiating a TE outside this list fails to COMPILE, so the sweep
// and the profile mode have to agree on it -- hence one definition.
template <typename Fn>
void for_each_TE(Fn&& fn) {
  fn(std::integral_constant<int, 2>{});
  fn(std::integral_constant<int, 4>{});
  fn(std::integral_constant<int, 8>{});
}

// Useful FLOPs: what the hand kernel retires per element.
//   gradients   9 sums of N mul-adds        = 9 * 2N * P
//   chain rule  9 * (3 mul + 2 add)         = 45 * P
//   stress      9 * 3 + 3 * 2               = 33 * P
//   integrands  9 * (3 mul + 2 add + 1 mul) = 54 * P
//   divergence  9 sums of N mul-adds        = 9 * 2N * P
//   weights     3 * (3 mul + 2 add) * ...   = 27 * P
double useful_flops(int E) {
  const double P = cfg::N * cfg::N * cfg::N;
  const double per_point =
      9.0 * 2.0 * cfg::N + 45.0 + 33.0 + 54.0 + 9.0 * 2.0 * cfg::N + 27.0;
  return double(E) * P * per_point;
}

// The body is a function so that every View destructs before Kokkos::finalize()
// runs -- an early return from profile mode would otherwise outlive it.
static int run(int argc, char* argv[]) {
  int rc = 0;
  {
    int E      = (argc > 1) ? std::atoi(argv[1]) : cfg::E_default;
    int reps   = (argc > 2) ? std::atoi(argv[2]) : 5;
    int warmup = (argc > 3) ? std::atoi(argv[3]) : 2;
    E          = (E / cfg::TE_max) * cfg::TE_max;

    std::printf(
        "\n=== 3D SEM stiffness (K&T 1999 A2/A7) ====================\n"
        "execution space : %s\n"
        "order           : NGLL=%d, %d components, %d points/element\n"
        "elements        : %d   (reps=%d warmup=%d, min wall-clock)\n\n",
        Kokkos::DefaultExecutionSpace::name(), cfg::N, cfg::C,
        cfg::N * cfg::N * cfg::N, E, reps, warmup);

    if (!cfg::kIsGPU) {
      std::printf(
          "This benchmark is GPU-only. At NGLL=5 the CPU register blocking\n"
          "(MT=NT=8, NR=2*simd_width) divides none of the staged extents and\n"
          "cannot be overridden; the GEMM has no remainder path.\n\n");
      return 0;
    }

    Fields d{};
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
    d.H   = V2("H", cfg::N, cfg::N);
    d.Hw  = V2("Hw", cfg::N, cfg::N);
    d.f0  = V4("f0", E);
    d.f1  = V4("f1", E);
    d.f2  = V4("f2", E);

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
          hH(a, b) = 0.3f * std::cos(0.9f * a + 0.4f * b);
      }
      for (int a = 0; a < cfg::N; ++a)
        for (int b = 0; b < cfg::N; ++b) hW(a, b) = hH(b, a) * hw(b);
      Kokkos::deep_copy(d.H, hH);
      Kokkos::deep_copy(d.Hw, hW);
      Kokkos::deep_copy(wv, hw);
    }
    d.w = V1(wv);
    Kokkos::fence();

    // The naive path's global round-trip buffer.
    V5 Ftmp("F", E);

    // --- host reference, on a slice ---------------------------------------
    const int Echk = std::min(E, 256);
    V4H       r0("r0", Echk), r1("r1", Echk), r2("r2", Echk);
    reference(d, Echk, r0, r1, r2);

    const double flops = useful_flops(E);

    // --- sweep: library and hand over TE x team, naive has neither ---------
    // The TE values to sweep, as a compile-time list: TE is a template
    // parameter of both kernels, so the sweep is a fold, not a loop.
    struct Best {
      double ms = 1e300;
      int    TE = 0, team = 0, err_ok = 1;
      double err = 0.0;
    };
    Best lib, hand;

    // PROFILE MODE: `bench nspec reps warmup TE team` runs each implementation
    // exactly once at that configuration and stops. The sweep issues thousands
    // of launches, which ncu cannot replay in reasonable time.
    if (argc > 5) {
      const int  pTE = std::atoi(argv[4]), pteam = std::atoi(argv[5]);
      const auto one = [&](auto tag) {
        constexpr int TE = decltype(tag)::value;
        if (TE != pTE) return;
        library_sem3d<TE>(d, pteam);
        Kokkos::fence();
        const double le = check(d, Echk, r0, r1, r2);
        hand_sem3d<TE>(d, E, pteam);
        Kokkos::fence();
        const double he = check(d, Echk, r0, r1, r2);
        std::printf(
            "profile: N=%d E=%d TE=%d team=%d  lib_err=%.1e hand_err=%.1e\n",
            cfg::N, E, TE, pteam, le, he);
      };
      for_each_TE(one);
      return 0;
    }

    std::printf("sweep (ms, min over %d reps):\n", reps);
    std::printf("%6s %8s %12s %12s\n", "TE", "team", "library", "hand");
    const auto sweep_TE = [&](auto tag) {
      constexpr int TE = decltype(tag)::value;
      for (int team : {32, 64, 96, 128, 192, 256, 384, 512}) {
        const double lms =
            best_ms([&] { library_sem3d<TE>(d, team); }, warmup, reps);
        const double lerr = check(d, Echk, r0, r1, r2);
        const double hms =
            best_ms([&] { hand_sem3d<TE>(d, E, team); }, warmup, reps);
        const double herr = check(d, Echk, r0, r1, r2);

        std::printf("%6d %8d %12.3f %12.3f\n", TE, team, lms, hms);
        if (lms < lib.ms) lib = Best{lms, TE, team, lerr < 1e-4, lerr};
        if (hms < hand.ms) hand = Best{hms, TE, team, herr < 1e-4, herr};
      }
    };
    for_each_TE(sweep_TE);

    const double nms  = best_ms([&] { naive_sem3d(d, E, Ftmp); }, warmup, reps);
    const double nerr = check(d, Echk, r0, r1, r2);

    // --- results ----------------------------------------------------------
    std::printf("\n%-10s %10s %8s %8s %12s %10s %10s\n", "impl", "time(ms)",
                "TE", "team", "GFLOP/s", "vs naive", "check");
    std::printf(
        "---------- ---------- -------- -------- ------------ ---------- "
        "----------\n");
    auto row = [&](const char* name, double ms, int TE, int team, double err) {
      char te_s[8] = "-", tm_s[8] = "-";
      if (TE) std::snprintf(te_s, sizeof(te_s), "%d", TE);
      if (team) std::snprintf(tm_s, sizeof(tm_s), "%d", team);
      std::printf("%-10s %10.3f %8s %8s %12.1f %9.2fx %9s(%.0e)\n", name, ms,
                  te_s, tm_s, flops / (ms * 1e-3) / 1e9, nms / ms,
                  err < 1e-4 ? "PASS" : "FAIL", err);
    };
    row("naive", nms, 0, 0, nerr);
    row("hand", hand.ms, hand.TE, hand.team, hand.err);
    row("library", lib.ms, lib.TE, lib.team, lib.err);
    std::printf("\nlibrary vs hand (best against best): %.3fx\n",
                hand.ms / lib.ms);

    // --- source size ------------------------------------------------------
    // Lines of each implementation, excluding the shared physics every one of
    // them calls. What the library spends its lines on is naming modes; what
    // the hand kernel spends them on is scratch, index decoding and barriers.
    std::printf(
        "\nSource size (lines, excluding the %d shared physics lines all three "
        "call):\n",
        kPhysEnd - kPhysBegin - 1);
    std::printf("%-10s %8s   %s\n", "impl", "lines", "what the lines are");
    std::printf("%-10s %8d   %s\n", "naive", kNaiveEnd - kNaiveBegin - 1,
                "the physics, twice, straight over global memory");
    std::printf("%-10s %8d   %s\n", "hand", kHandEnd - kHandBegin - 1,
                "physics + scratch views, index decode, barriers, team policy");
    std::printf(
        "%-10s %8d   %s\n", "library", kLibEnd - kLibBegin - 1,
        "physics + mode labels; staging and barriers are the library's");
    std::printf("\n");

    rc = (lib.err < 1e-4 && hand.err < 1e-4 && nerr < 1e-4) ? 0 : 1;
  }
  return rc;
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  const int rc = run(argc, argv);
  Kokkos::finalize();
  return rc;
}
