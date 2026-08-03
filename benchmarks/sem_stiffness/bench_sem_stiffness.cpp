// ===========================================================================
// bench_sem_stiffness.cpp
//
// The SPECFEM++ spectral-element stiffness kernel (Komatitsch & Tromp 1999,
// eqns A2 and A7) as a FULLY FUSED TensorOperations graph, against a
// hand-written fused Kokkos kernel doing identical physics.
//
// 2D elastic P-SV, NGLL N=8, C=2 components, float. Rows 1 and 7 of the
// pipeline (indirect gather / atomic scatter through the mesh index map) are
// OUT OF SCOPE: both paths consume an element-local field and produce an
// element-local force.
//
// The pipeline, per element:
//
//   A2  gradient    dxi_c = d(u_c)/d(xi),  dgm_c = d(u_c)/d(gamma)
//                   -- 2 contractions per component against hprime
//   --  chain rule  ds[c][d] = du_c/dx_d, from the reference gradients and
//                   the inverse Jacobian                          (pointwise)
//   --  stress      T = compute_stress(ds)          BLACK BOX, shared by both
//                   implementations so the physics work is identical
//   --  integrand   F^xi_c, F^gamma_c = J * sum_d T[c][d] * dref/dx_d
//   A7  divergence  t1_c = sum_q Hw[q,i] F^xi_c,  t2_c = sum_q Hw[q,j] F^gam_c
//                   force_c = w[j]*t1_c + w[i]*t2_c
//
// WHY THIS BENCHMARK EXISTS. Making a combine node usable as a contraction
// operand made the above expressible in ONE kernel with every intermediate in
// team scratch. Whether that is FAST was never measured. Both paths are
// credited the same useful FLOP count (the hand kernel's work) so the columns
// compare directly; the library additionally reports what it really retires,
// which is ~2.5x more because each force component's tree holds two F nodes,
// each with all four gradients under it.
//
// MEASURED RESULT. The library is 4.26x slower than the hand kernel on H100
// and 3.69x slower on Serial. Those two headline rows are NOT comparable on
// their own: the library retires 2.52x the FLOPs and ~4x the auxiliary global
// traffic, so a raw ratio conflates DOING MORE WORK with DOING WORK LESS
// EFFICIENTLY. Separating the two is what the CONTROL rows below are for.
//
// CONTROL is a hand-written kernel reproducing the library's EXACT work profile
// -- 2 launches, 16 gradient sums, 4 stress evals, 4 divergences, 4x aux reads
// -- at hand-written efficiency. CONTROL-B is the same with one constexpr flag
// hoisting the gradient stage, which isolates gradient recompute from the rest
// of the redundancy. CONTROL-C goes further: 4 gradient sums shared by every
// consumer and one launch for both components -- the work profile a fan-out
// deduplicating (DAG) evaluator would produce, which bounds that change's
// payoff before any library work is done. All three validate against the host
// reference. The gap factors multiplicatively:
//
//            gap  =  redundancy (baseline -> CONTROL)  x  residual (-> library)
//   H100     4.26x =            2.89x                  x        1.48x
//   Serial   3.69x =            3.21x                  x        1.11x
//
// THE RESIDUALS DO NOT AGREE, and that is the most informative number here.
// An earlier revision of this comment, and of SEM_STIFFNESS_GAP_ANALYSIS.md,
// claimed they did (1.46x vs 1.47x) and concluded the whole remaining gap was
// algorithmic and backend-independent. That rested on ESTIMATING the CPU's
// redundancy as the 2.52x executed-FLOP ratio. Now that CONTROL actually runs
// on Serial it MEASURES 3.21x -- redundancy costs more than its FLOPs, because
// it also carries 4x the aux traffic. So the honest split is ~1.11x of
// backend-independent library cost, times a further ~1.33x that appears only
// on the GPU. Team utilisation is therefore back on the table, not ruled out.
//
// Probes behind that, all printed as [diag] rows:
//   * one gradient contraction alone runs at ~1300 GF/s, so the [8x8]x[8x32]
//     GEMM is healthy. The library's marginal gradient sweep is actually
//     CHEAPER than the hand kernel's.
//   * the fused-operand passthrough costs nothing: a divergence GEMM reading a
//     materialized View operand is within noise of a gradient GEMM of the same
//     shape. Fusion itself is not what is expensive.
//   * 4x [1 F node] accounts for most, but not all, of the library's runtime;
//     the remainder looks fixed-per-tree rather than proportional to work.
//   * halving TE halves scratch but DOUBLES the time (per-team cost is flat).
//
// What already got fixed: the seven auxiliary arrays were originally combine
// OPERANDS, which forced the library to stage all of them into scratch. Since
// a combine reads each operand exactly once, that was pure overhead. Capturing
// them in the functor instead (see StressIntegrand) was worth 1.47x and cut
// scratch 38760 -> 24312 bytes -- which is also what brought the Serial
// backend under its hardcoded 32 KB cap and made a CPU number possible at all.
//
// What is left, in order: the redundancy term is the larger one, and a
// multi-output combine as a fused operand collapses 16 gradient sums to 8 and
// 2 launches to 1. The residual is the subtler one -- see
// SEM_STIFFNESS_GAP_ANALYSIS.md for the ordered statement of work.
//
// A library SLOWDOWN here is a valid deliverable: the point is the number and
// what it points at, not a win.
//
// LABELS. e = element, i = x axis, j = z axis, p = the gradient's contracted
// index, q = the divergence's contracted index. Five are needed because the
// x-axis appears at three generations (input -> gradient output -> divergence
// output) inside one node tree, and each generation needs its own letter.
//
// THE LOAD-BEARING RULE. Every fused node is declared with its output labels
// in its CONSUMER's canonical order (a contraction's B operand is canonically
// `contracted ++ freeB`). Get this wrong and the consumer must stage the fused
// operand, which is an in-place reorder, which
// Impl::transpositions_equal_extent permits only when every transposed axis
// pair has equal extents -- false for a [TE,N,N] tile under a 3-cycle.
// Declaring in the consumer's order makes the permutation the identity and
// takes the zero-copy passthrough instead.
//
// 3D would add a third reference direction: 3 gradients x 3 components = 9
// contractions per F node and ~3x the live scratch. The sizes and counts below
// are the only things that would change shape; compute_stress and the functors
// are already independent of it. Nothing here is templated over dimension --
// that machinery is not worth building before the 2D number exists.
//
// Usage: bench_sem_stiffness [nspec [reps [warmup]]]
// ===========================================================================

#include <Kokkos_Core.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/Graph.hpp>
#include <TensorOperations/Tiling.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace TensorOperations;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
namespace cfg {
inline constexpr bool kIsGPU =
    !Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace,
                                Kokkos::HostSpace>::accessible;

inline constexpr int N  = 8;  // GLL points per direction
inline constexpr int C  = 2;  // displacement components (unrolled, see below)
inline constexpr int TE = 4;  // elements per team

// N=8 is padding-free for the register kernel: every GEMM below is
//   SA = N = 8            (free-A: the hprime output index)
//   SK = N = 8            (contracted)
//   SB = TE * N = 32      (free-B: element x the spectator GLL axis)
// GPU blocks are MT=4, NT=2, NR=2; CPU are MT=8, NT=8, NR=2*simd_width (32 on
// AVX-512). All three dims divide on both.
static_assert(N % 8 == 0, "SA/SK must satisfy CPU MT=NT=8 and GPU MT=4,NT=2");
static_assert((TE * N) % 32 == 0,
              "SB must satisfy CPU NR=2*simd_width (32 on AVX-512)");

// TE=4, not 32. Scratch is additive down the whole tree with no recycling, and
// every intermediate is live at once. Per force-component kernel, in floats:
//   gradient contraction = 128*TE + 64      (A stage, B stage, C out)
//   F combine            = 576*TE + 256      (out + 4 gradients)
//   divergence           = 640*TE + 320
//   force combine        = 1344*TE + 640     => 5376*TE + 2560 bytes
// TE=4 -> 24312 B measured. TE=8 would be ~46 KB: fits the GPU's 48 KB but not
// the Serial backend's hardcoded 32 KB, and TE=2 fails the CPU SB % NR rule
// above -- so TE=4 is the only value that runs on both.

// Element count, per backend. Must be a multiple of TE.
//
// Sized so that EVERY timed run exceeds 10 ms, which sub-millisecond timings
// cannot do reliably. The binding case is the single-gradient diagnostic --
// the fastest thing measured -- so it sets the count and everything else lands
// well above the floor:
//
//   GPU  E=12582912: diagnostic ~10 ms, baseline ~49 ms, fused ~4.5 s
//   CPU  E=16384:     diagnostic ~11 ms, baseline ~46 ms, fused ~160 ms
//
// The GPU count is ~750x the CPU's because Serial has no team parallelism to
// saturate: per-element cost is flat in E there, so a small count measures the
// same thing.
//
// Cost of clearing 10 ms on the DIAGNOSTIC specifically: it is by far the
// fastest row, so it alone drives E up ~1.7x beyond what the two compared
// implementations need, to a ~45 GB device footprint (14 views) and a ~60 s
// run. Pass a smaller nspec on the command line if you only care about the
// baseline-vs-library comparison; both clear 10 ms by E=2500000.
inline constexpr int E_default = kIsGPU ? 12582912 : 16384;
}  // namespace cfg

// Element-local tensors are [E, N, N] = (element, z, x). The user's framing:
// 3D in 2D physics, 4D in 3D physics.
// EXPERIMENT: static trailing extents. A Kokkos::View with fully dynamic rank
// carries its extents at runtime (48 B); pinning the two GLL axes at compile
// time shrinks it to a pointer plus the one dynamic extent. Every node handle
// in the graph stores a View by value, and the evaluator tree duplicates each
// one per nesting level, so this multiplies through the whole per-thread state.
using V3    = Kokkos::View<float * [cfg::N][cfg::N], Kokkos::LayoutRight>;
using V3dyn = Kokkos::View<float***, Kokkos::LayoutRight>;
using V2    = Kokkos::View<float**, Kokkos::LayoutRight>;
using V3H   = V3::host_mirror_type;
using V2H   = V2::host_mirror_type;

// ---------------------------------------------------------------------------
// compute_stress -- THE BLACK BOX.
//
// Takes the physical displacement gradient ds[c][d] = d(u_c)/d(x_d) and the
// material, returns the stress tensor. Isotropic P-SV here, but the library
// never looks inside: it is an opaque call in the middle of a user functor.
// Called by BOTH implementations, so the physics work is identical and the
// comparison is about data movement, not arithmetic.
// ---------------------------------------------------------------------------
struct Mat2 {
  float m[2][2];
};

KOKKOS_INLINE_FUNCTION Mat2 compute_stress(const Mat2& ds, float l2m,
                                           float mu) {
  const float lambda = l2m - 2.0f * mu;
  Mat2        s{};
  s.m[0][0] = l2m * ds.m[0][0] + lambda * ds.m[1][1];  // sigma_xx
  s.m[1][1] = lambda * ds.m[0][0] + l2m * ds.m[1][1];  // sigma_zz
  s.m[0][1] = mu * (ds.m[0][1] + ds.m[1][0]);          // sigma_xz
  s.m[1][0] = s.m[0][1];
  return s;
}

// Chain rule + stress + one integrand slot, shared by both paths so neither
// can drift from the other.
//   Dir 0 = xi (uses xix, xiz), Dir 1 = gamma (uses gammax, gammaz)
template <int Comp, int Dir>
KOKKOS_INLINE_FUNCTION float integrand_slot(float dxi0, float dxi1, float dgm0,
                                            float dgm1, float xix, float xiz,
                                            float gx, float gz, float l2m,
                                            float mu, float jac) {
  Mat2 ds{};
  ds.m[0][0]      = dxi0 * xix + dgm0 * gx;  // du_x/dx
  ds.m[0][1]      = dxi0 * xiz + dgm0 * gz;  // du_x/dz
  ds.m[1][0]      = dxi1 * xix + dgm1 * gx;  // du_z/dx
  ds.m[1][1]      = dxi1 * xiz + dgm1 * gz;  // du_z/dz
  const Mat2  sig = compute_stress(ds, l2m, mu);
  const float dx  = (Dir == 0) ? xix : gx;
  const float dz  = (Dir == 0) ? xiz : gz;
  return jac * (sig.m[Comp][0] * dx + sig.m[Comp][1] * dz);
}

// ---------------------------------------------------------------------------
// Combine functors. Named structs, not lambdas: GoogleTest-style private
// bodies aside, a device lambda cannot be used as a stored functor here, and
// fn must be const-callable (the evaluator invokes it through a const kernel
// capture).
//
// fn receives the node's global output coordinate followed by one value per
// operand. Operand order is fixed by the make_combine_node call below:
//   dxi_0, dxi_1, dgm_0, dgm_1, xix, xiz, gammax, gammaz, l2m, mu, jac
// The seven auxiliary arrays are CAPTURED by the functor rather than passed
// to make_combine_node.
//
// A combine reads each operand exactly once per element, so staging an input
// operand into scratch buys nothing -- unlike a contraction's GEMM, which
// re-reads its staged operands N times. Capturing the Views lets the functor
// read them straight from global memory using the coordinate it is already
// handed, exactly as the hand-written kernel does. Measured: 1.47x faster and
// 37% less scratch than passing them as operands.
//
// No library change is needed -- the escape hatch is just "do not make it an
// operand". The trade is that those arrays become OPAQUE to the graph: the
// library can no longer fuse, reuse, or reason about them, and their access
// pattern becomes the functor author's problem.
// ---------------------------------------------------------------------------
template <int Comp, int Dir>
struct StressIntegrand {
  V3                    xix, xiz, gx, gz, l2m, mu, jac;
  KOKKOS_FUNCTION float operator()(int c0, int c1, int c2, float dxi0,
                                   float dxi1, float dgm0, float dgm1) const {
    // Node modes are {q,e,j} when Dir==0 and {q,e,i} when Dir==1; the views are
    // (e, z, x). q is the x role in the first case and the z role in the
    // second.
    const int e = c1;
    const int z = (Dir == 0) ? c2 : c0;
    const int x = (Dir == 0) ? c0 : c2;
    return integrand_slot<Comp, Dir>(dxi0, dxi1, dgm0, dgm1, xix(e, z, x),
                                     xiz(e, z, x), gx(e, z, x), gz(e, z, x),
                                     l2m(e, z, x), mu(e, z, x), jac(e, z, x));
  }
};

// force_c[e,j,i] = w[j]*t1_c + w[i]*t2_c.
//
// The GLL weights are rank-1 (modes {j} alone), so they cannot be a combine
// operand -- every operand must carry the output's exact label set. They ride
// inside the functor instead, indexed off the coordinates fn already receives.
// This is the general answer for rank-deficient auxiliaries.
struct WeightedSum {
  Kokkos::Array<float, cfg::N> w;
  KOKKOS_FUNCTION float        operator()(int, int j, int i, float t1,
                                          float t2) const {
    return w[j] * t1 + w[i] * t2;
  }
};

// ===========================================================================
// Implementation 1: the library. One fully-fused graph per force component.
//
// Tile bundles, bottom-up. Two shapes recur: element-major (e,.,.) for a
// node declared in user order, and GLL-major (.,e,.) for a fused node declared
// in its consumer's canonical order.
// ===========================================================================
using TileH   = StaticTile<cfg::N, cfg::N>;           // H / Hw        [8,8]
using TileEJI = StaticTile<cfg::TE, cfg::N, cfg::N>;  // (e,.,.)     [4,8,8]
using TileQEJ = StaticTile<cfg::N, cfg::TE, cfg::N>;  // (.,e,.)     [8,4,8]
using BGrad   = Tile<TileH, TileEJI, TileQEJ>;        // one gradient
inline constexpr int kLibSrcBegin = __LINE__;
// The F combine takes FOUR operands -- the gradients. Everything else the
// physics needs is captured by StressIntegrand (see above).
using CombF   = CombineTile<TileQEJ, BGrad, BGrad, BGrad, BGrad>;
using BDiv    = Tile<TileH, CombF, TileEJI>;
using CombFrc = CombineTile<TileEJI, BDiv, BDiv>;

inline CombFrc library_tile() {
  const auto cf =
      make_combine_tile(TileQEJ{}, BGrad{}, BGrad{}, BGrad{}, BGrad{});
  const auto bd = BDiv{TileH{}, cf, TileEJI{}};
  return make_combine_tile(TileEJI{}, bd, bd);
}

template <int Comp>
auto library_node(V3 u0, V3 u1, V3 xix, V3 xiz, V3 gx, V3 gz, V3 l2m, V3 mu,
                  V3 jac, V2 H, V2 Hw, const Kokkos::Array<float, cfg::N>& w) {
  auto F0 = make_combine_node<'q', 'e', 'j'>(
      make_contraction_node<'q', 'e', 'j'>(
          make_input_node(make_handle<'q', 'p'>(H)),
          make_input_node(make_handle<'e', 'j', 'p'>(u0))),
      make_contraction_node<'q', 'e', 'j'>(
          make_input_node(make_handle<'q', 'p'>(H)),
          make_input_node(make_handle<'e', 'j', 'p'>(u1))),
      make_contraction_node<'j', 'e', 'q'>(
          make_input_node(make_handle<'j', 'p'>(H)),
          make_input_node(make_handle<'e', 'p', 'q'>(u0))),
      make_contraction_node<'j', 'e', 'q'>(
          make_input_node(make_handle<'j', 'p'>(H)),
          make_input_node(make_handle<'e', 'p', 'q'>(u1))),
      StressIntegrand<Comp, 0>{xix, xiz, gx, gz, l2m, mu, jac});

  auto F1 = make_combine_node<'q', 'e', 'i'>(
      make_contraction_node<'i', 'e', 'q'>(
          make_input_node(make_handle<'i', 'p'>(H)),
          make_input_node(make_handle<'e', 'q', 'p'>(u0))),
      make_contraction_node<'i', 'e', 'q'>(
          make_input_node(make_handle<'i', 'p'>(H)),
          make_input_node(make_handle<'e', 'q', 'p'>(u1))),
      make_contraction_node<'q', 'e', 'i'>(
          make_input_node(make_handle<'q', 'p'>(H)),
          make_input_node(make_handle<'e', 'p', 'i'>(u0))),
      make_contraction_node<'q', 'e', 'i'>(
          make_input_node(make_handle<'q', 'p'>(H)),
          make_input_node(make_handle<'e', 'p', 'i'>(u1))),
      StressIntegrand<Comp, 1>{xix, xiz, gx, gz, l2m, mu, jac});

  return make_combine_node<'e', 'j', 'i'>(
      make_contraction_node<'e', 'j', 'i'>(
          make_input_node(make_handle<'q', 'i'>(Hw)), F0),
      make_contraction_node<'e', 'j', 'i'>(
          make_input_node(make_handle<'q', 'j'>(Hw)), F1),
      WeightedSum{w});
}

template <int Comp>
void library_force(V3 u0, V3 u1, V3 xix, V3 xiz, V3 gx, V3 gz, V3 l2m, V3 mu,
                   V3 jac, V2 H, V2 Hw, const Kokkos::Array<float, cfg::N>& w,
                   V3 force) {
  auto g                          = make_graph();
  [[maybe_unused]] auto [g1, out] = g.ops(
      library_node<Comp>(u0, u1, xix, xiz, gx, gz, l2m, mu, jac, H, Hw, w));
  g1.execute(TeamPolicyTag<>{}, library_tile(), force);
}
inline constexpr int kLibSrcEnd = __LINE__;
// DIAG-B: one F node on its own -- 4 gradient contractions + StressIntegrand,
// written to global. This is exactly one quarter of what the full library run
// executes (2 kernels x 2 F nodes), so 4x this row is the whole gradient+stress
// half of the tree, measured rather than extrapolated.
void library_one_F(V3 u0, V3 u1, V3 xix, V3 xiz, V3 gx, V3 gz, V3 l2m, V3 mu,
                   V3 jac, V2 H, V3dyn out) {
  auto g                        = make_graph();
  [[maybe_unused]] auto [g1, o] = g.ops(make_combine_node<'q', 'e', 'j'>(
      make_contraction_node<'q', 'e', 'j'>(
          make_input_node(make_handle<'q', 'p'>(H)),
          make_input_node(make_handle<'e', 'j', 'p'>(u0))),
      make_contraction_node<'q', 'e', 'j'>(
          make_input_node(make_handle<'q', 'p'>(H)),
          make_input_node(make_handle<'e', 'j', 'p'>(u1))),
      make_contraction_node<'j', 'e', 'q'>(
          make_input_node(make_handle<'j', 'p'>(H)),
          make_input_node(make_handle<'e', 'p', 'q'>(u0))),
      make_contraction_node<'j', 'e', 'q'>(
          make_input_node(make_handle<'j', 'p'>(H)),
          make_input_node(make_handle<'e', 'p', 'q'>(u1))),
      StressIntegrand<0, 0>{xix, xiz, gx, gz, l2m, mu, jac}));
  g1.execute(TeamPolicyTag<>{},
             make_combine_tile(TileQEJ{}, BGrad{}, BGrad{}, BGrad{}, BGrad{}),
             out);
}

// DIAG-C: the same four gradients, but the combine does nothing but add them --
// no auxiliary View reads at all. DIAG-B minus DIAG-C is the cost of the seven
// global aux reads the StressIntegrand does; the remainder is the GEMMs.
struct SumFour {
  KOKKOS_FUNCTION float operator()(int, int, int, float a, float b, float c,
                                   float d) const {
    return a + b + c + d;
  }
};
void library_four_gradients(V3 u0, V3 u1, V2 H, V3dyn out) {
  auto g                        = make_graph();
  [[maybe_unused]] auto [g1, o] = g.ops(make_combine_node<'q', 'e', 'j'>(
      make_contraction_node<'q', 'e', 'j'>(
          make_input_node(make_handle<'q', 'p'>(H)),
          make_input_node(make_handle<'e', 'j', 'p'>(u0))),
      make_contraction_node<'q', 'e', 'j'>(
          make_input_node(make_handle<'q', 'p'>(H)),
          make_input_node(make_handle<'e', 'j', 'p'>(u1))),
      make_contraction_node<'j', 'e', 'q'>(
          make_input_node(make_handle<'j', 'p'>(H)),
          make_input_node(make_handle<'e', 'p', 'q'>(u0))),
      make_contraction_node<'j', 'e', 'q'>(
          make_input_node(make_handle<'j', 'p'>(H)),
          make_input_node(make_handle<'e', 'p', 'q'>(u1))),
      SumFour{}));
  g1.execute(TeamPolicyTag<>{},
             make_combine_tile(TileQEJ{}, BGrad{}, BGrad{}, BGrad{}, BGrad{}),
             out);
}

// DIAG-D: the divergence contraction with a MATERIALIZED operand -- same GEMM
// shape the tree runs, but B is a plain View instead of a fused combine node.
// Compare against the tree's implied per-divergence cost to price the fused
// operand path itself.
void library_one_divergence(V3dyn Fin, V2 Hw, V3 out) {
  auto g                        = make_graph();
  [[maybe_unused]] auto [g1, o] = g.ops(make_contraction_node<'e', 'j', 'i'>(
      make_input_node(make_handle<'q', 'i'>(Hw)),
      make_input_node(make_handle<'q', 'e', 'j'>(Fin))));
  g1.execute(TeamPolicyTag<>{}, Tile<TileH, TileQEJ, TileEJI>{}, out);
}

// DIAGNOSTIC (not part of the comparison): one gradient contraction on its own,
// same tile shape the fused tree uses internally. This separates two very
// different explanations for a slow fused path -- "the [8x8]x[8x32] GEMM is
// simply small" versus "the fused tree's serialized stages are the cost". If
// this row is also slow, the tile is the floor and fusion is not the culprit.
void library_one_gradient(V3 u0, V2 H, V3dyn out) {
  auto g                        = make_graph();
  [[maybe_unused]] auto [g1, o] = g.ops(make_contraction_node<'q', 'e', 'j'>(
      make_input_node(make_handle<'q', 'p'>(H)),
      make_input_node(make_handle<'e', 'j', 'p'>(u0))));
  g1.execute(TeamPolicyTag<>{}, BGrad{}, out);
}

// ===========================================================================
// Implementation 2: hand-written fused kernel, SPECFEM-style.
//
// One team per chunk of TE elements; BOTH force components in ONE launch.
// That is a structural advantage over the library (two launches, and the
// gradient stage evaluated 4x rather than once) and it is deliberate -- it is
// what SPECFEM actually does, and measuring against it is the whole point.
//
// Auxiliary arrays are read straight from global memory, not staged: each
// element is touched exactly once, so staging them would be pure overhead.
// The library has no way to express that today, which is why its scratch
// budget is ~40% aux staging.
// ===========================================================================
inline constexpr int kBaseSrcBegin = __LINE__;
void baseline_force(V3 u0, V3 u1, V3 xix, V3 xiz, V3 gx, V3 gz, V3 l2m, V3 mu,
                    V3 jac, V2 H, V2 Hw, Kokkos::Array<float, cfg::N> w,
                    V3 force0, V3 force1, int E) {
  using ExecSpace    = Kokkos::DefaultExecutionSpace;
  using ScratchSpace = ExecSpace::scratch_memory_space;
  using Sc3          = Kokkos::View<float***, ScratchSpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using member_t     = Kokkos::TeamPolicy<ExecSpace>::member_type;

  constexpr int N = cfg::N, TE = cfg::TE, NP = cfg::N * cfg::N;

  // u0,u1 staged; 4 reference gradients; 4 integrand slots.
  const size_t bytes = 10 * Sc3::shmem_size(TE, N, N);

  Kokkos::parallel_for(
      "baseline_sem",
      Kokkos::TeamPolicy<ExecSpace>(E / TE, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const member_t& team) {
        const int e0 = team.league_rank() * TE;

        Sc3 us0(team.team_scratch(0), TE, N, N);
        Sc3 us1(team.team_scratch(0), TE, N, N);
        Sc3 dxi0(team.team_scratch(0), TE, N, N);
        Sc3 dxi1(team.team_scratch(0), TE, N, N);
        Sc3 dgm0(team.team_scratch(0), TE, N, N);
        Sc3 dgm1(team.team_scratch(0), TE, N, N);
        Sc3 Fxi0(team.team_scratch(0), TE, N, N);
        Sc3 Fxi1(team.team_scratch(0), TE, N, N);
        Sc3 Fgm0(team.team_scratch(0), TE, N, N);
        Sc3 Fgm1(team.team_scratch(0), TE, N, N);

        // Stage the element-local displacement.
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team, TE * NP),
                             [&](int t) {
                               const int de = t / NP, r = t % NP;
                               const int j = r / N, i = r % N;
                               us0(de, j, i) = u0(e0 + de, j, i);
                               us1(de, j, i) = u1(e0 + de, j, i);
                             });
        team.team_barrier();

        // A2: reference-frame gradients, sum-factorized.
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team, TE * NP), [&](int t) {
              const int de = t / NP, r = t % NP;
              const int j = r / N, i = r % N;
              float     ax0 = 0.0f, ax1 = 0.0f, ag0 = 0.0f, ag1 = 0.0f;
              for (int p = 0; p < N; ++p) {
                const float hi = H(i, p), hj = H(j, p);
                ax0 += hi * us0(de, j, p);  // d/d(xi)
                ax1 += hi * us1(de, j, p);
                ag0 += hj * us0(de, p, i);  // d/d(gamma)
                ag1 += hj * us1(de, p, i);
              }
              dxi0(de, j, i) = ax0;
              dxi1(de, j, i) = ax1;
              dgm0(de, j, i) = ag0;
              dgm1(de, j, i) = ag1;
            });
        team.team_barrier();

        // Chain rule + stress + integrand. Computed ONCE per point; all four
        // slots fall out of the same stress tensor.
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team, TE * NP), [&](int t) {
              const int   de = t / NP, r = t % NP;
              const int   j = r / N, i = r % N;
              const int   e  = e0 + de;
              const float ax = xix(e, j, i), az = xiz(e, j, i);
              const float px = gx(e, j, i), pz = gz(e, j, i);
              const float lm = l2m(e, j, i), m = mu(e, j, i), J = jac(e, j, i);
              const float a0 = dxi0(de, j, i), a1 = dxi1(de, j, i);
              const float g0 = dgm0(de, j, i), g1 = dgm1(de, j, i);

              Mat2 ds{};
              ds.m[0][0]     = a0 * ax + g0 * px;
              ds.m[0][1]     = a0 * az + g0 * pz;
              ds.m[1][0]     = a1 * ax + g1 * px;
              ds.m[1][1]     = a1 * az + g1 * pz;
              const Mat2 sig = compute_stress(ds, lm, m);

              Fxi0(de, j, i) = J * (sig.m[0][0] * ax + sig.m[0][1] * az);
              Fgm0(de, j, i) = J * (sig.m[0][0] * px + sig.m[0][1] * pz);
              Fxi1(de, j, i) = J * (sig.m[1][0] * ax + sig.m[1][1] * az);
              Fgm1(de, j, i) = J * (sig.m[1][0] * px + sig.m[1][1] * pz);
            });
        team.team_barrier();

        // A7: weighted divergence, straight to global.
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team, TE * NP),
                             [&](int t) {
                               const int de = t / NP, r = t % NP;
                               const int j = r / N, i = r % N;
                               float s0 = 0.0f, s1 = 0.0f, g0 = 0.0f, g1 = 0.0f;
                               for (int q = 0; q < N; ++q) {
                                 const float wi = Hw(q, i), wj = Hw(q, j);
                                 s0 += wi * Fxi0(de, j, q);
                                 s1 += wi * Fxi1(de, j, q);
                                 g0 += wj * Fgm0(de, q, i);
                                 g1 += wj * Fgm1(de, q, i);
                               }
                               force0(e0 + de, j, i) = w[j] * s0 + w[i] * g0;
                               force1(e0 + de, j, i) = w[j] * s1 + w[i] * g1;
                             });
      });
}
inline constexpr int kBaseSrcEnd = __LINE__;

// ===========================================================================
// Implementation 3: THE CONTROL. A hand-written kernel that reproduces the
// library's REDUNDANCY exactly, at hand-written efficiency.
//
// Per force component (2 launches, like the library):
//   for each of the 2 F nodes:
//       stage u0 and u1 (twice each -- the library's four contractions each
//                        stage their own operand copy; fan-out isn't deduped)
//       compute ALL FOUR reference gradients   <-- recomputed for both F nodes
//       chain rule + stress -> ONE integrand slot (reads all 7 aux from global)
//   2 divergence sums + the weighted combine
//
// That is 8 gradient sums, 2 stress evaluations and 2 divergence sums per
// launch: the library's executed FLOP profile and the library's global-memory
// traffic, byte for byte.
//
// baseline -> control  = the price of the redundancy itself.
// control  -> library  = everything else the library costs.
// ===========================================================================
template <int Comp, bool DedupGrad = false>
void control_recompute_force(V3 u0, V3 u1, V3 xix, V3 xiz, V3 gx, V3 gz, V3 l2m,
                             V3 mu, V3 jac, V2 H, V2 Hw,
                             Kokkos::Array<float, cfg::N> w, V3 force, int E) {
  using ExecSpace    = Kokkos::DefaultExecutionSpace;
  using ScratchSpace = ExecSpace::scratch_memory_space;
  using Sc3          = Kokkos::View<float***, ScratchSpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using member_t     = Kokkos::TeamPolicy<ExecSpace>::member_type;

  constexpr int N = cfg::N, TE = cfg::TE, NP = cfg::N * cfg::N;
  const size_t  bytes = 8 * Sc3::shmem_size(TE, N, N);

  Kokkos::parallel_for(
      "control_recompute",
      Kokkos::TeamPolicy<ExecSpace>(E / TE, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const member_t& team) {
        const int e0 = team.league_rank() * TE;
        Sc3       us0(team.team_scratch(0), TE, N, N);
        Sc3       us1(team.team_scratch(0), TE, N, N);
        Sc3       a0v(team.team_scratch(0), TE, N, N);
        Sc3       a1v(team.team_scratch(0), TE, N, N);
        Sc3       g0v(team.team_scratch(0), TE, N, N);
        Sc3       g1v(team.team_scratch(0), TE, N, N);
        Sc3       F0(team.team_scratch(0), TE, N, N);
        Sc3       F1(team.team_scratch(0), TE, N, N);

        // Two F nodes; each redoes the staging and all four gradients --
        // unless DedupGrad, in which case the gradient stage runs once and
        // both F nodes read the same buffers. DedupGrad is the ONLY difference
        // between the two control rows.
        for (int f = 0; f < 2; ++f) {
          if (!DedupGrad || f == 0) {
            // Four operand stagings, matching the library's four contractions.
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team, TE * NP), [&](int t) {
                  const int de = t / NP, r = t % NP, j = r / N, i = r % N;
                  us0(de, j, i) = u0(e0 + de, j, i);
                  us1(de, j, i) = u1(e0 + de, j, i);
                  if (!DedupGrad) {
                    us0(de, j, i) = u0(e0 + de, j, i);
                    us1(de, j, i) = u1(e0 + de, j, i);
                  }
                });
            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team, TE * NP), [&](int t) {
                  const int de = t / NP, r = t % NP, j = r / N, i = r % N;
                  float     ax0 = 0.f, ax1 = 0.f, ag0 = 0.f, ag1 = 0.f;
                  for (int p = 0; p < N; ++p) {
                    const float hi = H(i, p), hj = H(j, p);
                    ax0 += hi * us0(de, j, p);
                    ax1 += hi * us1(de, j, p);
                    ag0 += hj * us0(de, p, i);
                    ag1 += hj * us1(de, p, i);
                  }
                  a0v(de, j, i) = ax0;
                  a1v(de, j, i) = ax1;
                  g0v(de, j, i) = ag0;
                  g1v(de, j, i) = ag1;
                });
            team.team_barrier();
          }

          // ONE integrand slot per F node, from a freshly rebuilt stress.
          Kokkos::parallel_for(
              Kokkos::TeamVectorRange(team, TE * NP), [&](int t) {
                const int   de = t / NP, r = t % NP, j = r / N, i = r % N;
                const int   e = e0 + de;
                const float v =
                    (f == 0) ? integrand_slot<Comp, 0>(
                                   a0v(de, j, i), a1v(de, j, i), g0v(de, j, i),
                                   g1v(de, j, i), xix(e, j, i), xiz(e, j, i),
                                   gx(e, j, i), gz(e, j, i), l2m(e, j, i),
                                   mu(e, j, i), jac(e, j, i))
                             : integrand_slot<Comp, 1>(
                                   a0v(de, j, i), a1v(de, j, i), g0v(de, j, i),
                                   g1v(de, j, i), xix(e, j, i), xiz(e, j, i),
                                   gx(e, j, i), gz(e, j, i), l2m(e, j, i),
                                   mu(e, j, i), jac(e, j, i));
                if (f == 0)
                  F0(de, j, i) = v;
                else
                  F1(de, j, i) = v;
              });
          team.team_barrier();
        }

        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team, TE * NP), [&](int t) {
              const int de = t / NP, r = t % NP, j = r / N, i = r % N;
              float     s = 0.f, g = 0.f;
              for (int q = 0; q < N; ++q) {
                s += Hw(q, i) * F0(de, j, q);
                g += Hw(q, j) * F1(de, q, i);
              }
              force(e0 + de, j, i) = w[j] * s + w[i] * g;
            });
      });
}

// ===========================================================================
// Implementation 4: CONTROL-C. The DAG's work profile, at hand-written
// efficiency.
//
// The spike for "would a topologically-ordered DAG evaluator pay?". This is
// what the library would execute if fan-out were deduplicated: the four
// DISTINCT reference gradients computed once and shared by every consumer, and
// both force components in ONE launch -- but still four separate F nodes, each
// rebuilding the stress tensor from scratch and re-reading all seven auxiliary
// arrays. Deduplicating shared subtrees does not merge the four single-output
// combines into one multi-output node, so the 4x stress and 4x aux traffic
// survive.
//
//              launches  grad sums  stress evals  div sums  aux reads
//   baseline       1          4           1           4        1x
//   CONTROL-C      1          4           4           4        4x
//   CONTROL-B      2          8           4           4        4x
//   CONTROL        2         16           4           4        4x
//   library        2         16           4           4        4x
//
// So CONTROL-C sits between the baseline and CONTROL-B:
//   baseline  -> CONTROL-C  = the redundancy a DAG evaluator CANNOT remove on
//                             its own (4x stress + 4x aux traffic); closing it
//                             needs multi-output combine, not sharing.
//   CONTROL-C -> CONTROL    = the redundancy it CAN remove. That ratio is the
//                             ceiling on the DAG's payoff, measured rather
//                             than extrapolated from the marginal-sweep fit.
//
// The four F passes are deliberately four separate parallel_for's separated by
// barriers rather than one pass calling integrand_slot four times: fusing them
// would let the compiler share one stress tensor across the four slots, which
// is exactly the redundancy this row exists to preserve.
//
// Scratch is 10 tiles -- the same as the baseline. A DAG needs 4 gradient
// slots and 4 F slots live at once, which is what the baseline already holds;
// sharing does not add storage here, it removes duplicates.
// ===========================================================================
void control_dag_force(V3 u0, V3 u1, V3 xix, V3 xiz, V3 gx, V3 gz, V3 l2m,
                       V3 mu, V3 jac, V2 H, V2 Hw,
                       Kokkos::Array<float, cfg::N> w, V3 force0, V3 force1,
                       int E) {
  using ExecSpace    = Kokkos::DefaultExecutionSpace;
  using ScratchSpace = ExecSpace::scratch_memory_space;
  using Sc3          = Kokkos::View<float***, ScratchSpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using member_t     = Kokkos::TeamPolicy<ExecSpace>::member_type;

  constexpr int N = cfg::N, TE = cfg::TE, NP = cfg::N * cfg::N;
  const size_t  bytes = 10 * Sc3::shmem_size(TE, N, N);

  Kokkos::parallel_for(
      "control_dag",
      Kokkos::TeamPolicy<ExecSpace>(E / TE, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const member_t& team) {
        const int e0 = team.league_rank() * TE;
        Sc3       us0(team.team_scratch(0), TE, N, N);
        Sc3       us1(team.team_scratch(0), TE, N, N);
        Sc3       a0v(team.team_scratch(0), TE, N, N);
        Sc3       a1v(team.team_scratch(0), TE, N, N);
        Sc3       g0v(team.team_scratch(0), TE, N, N);
        Sc3       g1v(team.team_scratch(0), TE, N, N);
        Sc3       Fxi0(team.team_scratch(0), TE, N, N);
        Sc3       Fgm0(team.team_scratch(0), TE, N, N);
        Sc3       Fxi1(team.team_scratch(0), TE, N, N);
        Sc3       Fgm1(team.team_scratch(0), TE, N, N);

        // Four operand stagings, matching the library's four gradient
        // contractions -- sharing a node does not merge its operands' staging
        // buffers, so this cost survives the DAG exactly as in CONTROL.
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team, TE * NP), [&](int t) {
              const int de = t / NP, r = t % NP, j = r / N, i = r % N;
              us0(de, j, i) = u0(e0 + de, j, i);
              us1(de, j, i) = u1(e0 + de, j, i);
              us0(de, j, i) = u0(e0 + de, j, i);
              us1(de, j, i) = u1(e0 + de, j, i);
            });
        team.team_barrier();

        // The four distinct gradients, computed ONCE for the whole graph. This
        // is the line the DAG buys: CONTROL runs this block four times (twice
        // per launch, two launches), CONTROL-B twice, the baseline once.
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team, TE * NP), [&](int t) {
              const int de = t / NP, r = t % NP, j = r / N, i = r % N;
              float     ax0 = 0.f, ax1 = 0.f, ag0 = 0.f, ag1 = 0.f;
              for (int p = 0; p < N; ++p) {
                const float hi = H(i, p), hj = H(j, p);
                ax0 += hi * us0(de, j, p);
                ax1 += hi * us1(de, j, p);
                ag0 += hj * us0(de, p, i);
                ag1 += hj * us1(de, p, i);
              }
              a0v(de, j, i) = ax0;
              a1v(de, j, i) = ax1;
              g0v(de, j, i) = ag0;
              g1v(de, j, i) = ag1;
            });
        team.team_barrier();

        // Four F nodes, one integrand slot each, every one rebuilding the
        // stress tensor and re-reading all seven auxiliary arrays. Separate
        // passes so the stress cannot be shared between them.
        for (int f = 0; f < 4; ++f) {
          Kokkos::parallel_for(
              Kokkos::TeamVectorRange(team, TE * NP), [&](int t) {
                const int   de = t / NP, r = t % NP, j = r / N, i = r % N;
                const int   e  = e0 + de;
                const float a0 = a0v(de, j, i), a1 = a1v(de, j, i);
                const float G0 = g0v(de, j, i), G1 = g1v(de, j, i);
                const float ax = xix(e, j, i), az = xiz(e, j, i);
                const float px = gx(e, j, i), pz = gz(e, j, i);
                const float lm = l2m(e, j, i), m = mu(e, j, i);
                const float J = jac(e, j, i);
                switch (f) {
                  case 0:
                    Fxi0(de, j, i) = integrand_slot<0, 0>(a0, a1, G0, G1, ax,
                                                          az, px, pz, lm, m, J);
                    break;
                  case 1:
                    Fgm0(de, j, i) = integrand_slot<0, 1>(a0, a1, G0, G1, ax,
                                                          az, px, pz, lm, m, J);
                    break;
                  case 2:
                    Fxi1(de, j, i) = integrand_slot<1, 0>(a0, a1, G0, G1, ax,
                                                          az, px, pz, lm, m, J);
                    break;
                  default:
                    Fgm1(de, j, i) = integrand_slot<1, 1>(a0, a1, G0, G1, ax,
                                                          az, px, pz, lm, m, J);
                    break;
                }
              });
          team.team_barrier();
        }

        // A7: both components' divergences and weighted sums, one launch.
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team, TE * NP),
                             [&](int t) {
                               const int de = t / NP, r = t % NP;
                               const int j = r / N, i = r % N;
                               float s0 = 0.0f, s1 = 0.0f, g0 = 0.0f, g1 = 0.0f;
                               for (int q = 0; q < N; ++q) {
                                 const float wi = Hw(q, i), wj = Hw(q, j);
                                 s0 += wi * Fxi0(de, j, q);
                                 s1 += wi * Fxi1(de, j, q);
                                 g0 += wj * Fgm0(de, q, i);
                                 g1 += wj * Fgm1(de, q, i);
                               }
                               force0(e0 + de, j, i) = w[j] * s0 + w[i] * g0;
                               force1(e0 + de, j, i) = w[j] * s1 + w[i] * g1;
                             });
      });
}

// ---------------------------------------------------------------------------
// Inputs. Deterministic, bounded, index-dependent and non-symmetric -- an
// all-ones fill cannot catch a transposed index, and this pipeline is nothing
// but transposed indices.
// ---------------------------------------------------------------------------
void fill_field(V3 v, int E, int salt) {
  Kokkos::parallel_for(
      "fill",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {E, cfg::N, cfg::N}),
      KOKKOS_LAMBDA(int e, int j, int i) {
        v(e, j, i) =
            static_cast<float>(((e + 3 * j + 5 * i + salt) % 7) + 1) * 0.125f;
      });
}

// hprime(a,b) = l'_b(xi_a); hprimewgll(a,b) = w[a] * hprime(a,b). Synthetic but
// non-symmetric, which is what matters for catching index bugs.
void fill_operators(V2 H, V2 Hw, Kokkos::Array<float, cfg::N>& w) {
  auto Hh  = Kokkos::create_mirror_view(H);
  auto Hwh = Kokkos::create_mirror_view(Hw);
  for (int a = 0; a < cfg::N; ++a)
    w[a] = 0.5f + 0.125f * static_cast<float>((a % 3));
  for (int a = 0; a < cfg::N; ++a)
    for (int b = 0; b < cfg::N; ++b) {
      Hh(a, b)  = static_cast<float>(((3 * a + 2 * b) % 5) - 2) * 0.5f;
      Hwh(a, b) = w[a] * Hh(a, b);
    }
  Kokkos::deep_copy(H, Hh);
  Kokkos::deep_copy(Hw, Hwh);
}

// ---------------------------------------------------------------------------
// Host reference: the same pipeline, straightforwardly, on the host.
// ---------------------------------------------------------------------------
void host_reference(V3 u0, V3 u1, V3 xix, V3 xiz, V3 gx, V3 gz, V3 l2m, V3 mu,
                    V3 jac, V2 H, V2 Hw, const Kokkos::Array<float, cfg::N>& w,
                    V3H f0, V3H f1, int E) {
  constexpr int N = cfg::N;
  auto          h = [](V3 v) {
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
  };
  auto U0 = h(u0), U1 = h(u1), AX = h(xix), AZ = h(xiz), PX = h(gx), PZ = h(gz),
       LM = h(l2m), MU = h(mu), JA = h(jac);
  auto Hh  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, H);
  auto Hwh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Hw);

  std::vector<float> Fx0(N * N), Fg0(N * N), Fx1(N * N), Fg1(N * N);
  for (int e = 0; e < E; ++e) {
    for (int j = 0; j < N; ++j)
      for (int i = 0; i < N; ++i) {
        float a0 = 0, a1 = 0, g0 = 0, g1 = 0;
        for (int p = 0; p < N; ++p) {
          a0 += Hh(i, p) * U0(e, j, p);
          a1 += Hh(i, p) * U1(e, j, p);
          g0 += Hh(j, p) * U0(e, p, i);
          g1 += Hh(j, p) * U1(e, p, i);
        }
        const float ax = AX(e, j, i), az = AZ(e, j, i);
        const float px = PX(e, j, i), pz = PZ(e, j, i);
        Mat2        ds{};
        ds.m[0][0]     = a0 * ax + g0 * px;
        ds.m[0][1]     = a0 * az + g0 * pz;
        ds.m[1][0]     = a1 * ax + g1 * px;
        ds.m[1][1]     = a1 * az + g1 * pz;
        const Mat2  sg = compute_stress(ds, LM(e, j, i), MU(e, j, i));
        const float J  = JA(e, j, i);
        Fx0[j * N + i] = J * (sg.m[0][0] * ax + sg.m[0][1] * az);
        Fg0[j * N + i] = J * (sg.m[0][0] * px + sg.m[0][1] * pz);
        Fx1[j * N + i] = J * (sg.m[1][0] * ax + sg.m[1][1] * az);
        Fg1[j * N + i] = J * (sg.m[1][0] * px + sg.m[1][1] * pz);
      }
    for (int j = 0; j < N; ++j)
      for (int i = 0; i < N; ++i) {
        float s0 = 0, s1 = 0, t0 = 0, t1 = 0;
        for (int q = 0; q < N; ++q) {
          s0 += Hwh(q, i) * Fx0[j * N + q];
          s1 += Hwh(q, i) * Fx1[j * N + q];
          t0 += Hwh(q, j) * Fg0[q * N + i];
          t1 += Hwh(q, j) * Fg1[q * N + i];
        }
        f0(e, j, i) = w[j] * s0 + w[i] * t0;
        f1(e, j, i) = w[j] * s1 + w[i] * t1;
      }
  }
}

double max_rel_diff(V3 X, V3H ref, int E) {
  auto   xh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, X);
  double m  = 0.0;
  for (int e = 0; e < E; ++e)
    for (int j = 0; j < cfg::N; ++j)
      for (int i = 0; i < cfg::N; ++i) {
        const double a = xh(e, j, i), b = ref(e, j, i);
        m = std::max(m, std::abs(a - b) / (std::abs(b) + 1e-6));
      }
  return m;
}

template <class Fn>
double seconds_of(Fn&& fn, int warmup, int reps) {
  for (int w = 0; w < warmup; ++w) fn();
  Kokkos::fence();
  double best = 1e300;
  for (int r = 0; r < reps; ++r) {
    Kokkos::Timer t;
    fn();
    Kokkos::fence();
    best = std::min(best, t.seconds());
  }
  return best;
}

// ---------------------------------------------------------------------------
// FLOP accounting, per element. Both paths are credited the SAME useful count
// (the hand kernel's work), so the two useful-GFLOP/s columns are directly
// comparable. The library's executed count additionally charges what it really
// retires: the gradient stage 4x (each force kernel holds two F nodes, each
// with all four gradients under it) and the chain-rule+stress 4x (once per F
// node, each producing a single slot, where the hand kernel derives all four
// slots from one stress tensor).
// ---------------------------------------------------------------------------
namespace flopcount {
inline constexpr double NP    = cfg::N * cfg::N;          // points/element
inline constexpr double kGrad = 4.0 * NP * cfg::N * 2.0;  // 4 sums
inline constexpr double kDiv  = 4.0 * NP * cfg::N * 2.0;  // 4 sums
inline constexpr double kSum  = 2.0 * 3.0 * NP;           // w[j]*t1 + w[i]*t2
inline constexpr double kStressShared =
    32.0 * NP;  // chain 12 + stress 8 + 4 slots
inline constexpr double kStressPerF =
    23.0 * NP;  // chain 12 + stress 8 + 1 slot

inline constexpr double kUseful = kGrad + kStressShared + kDiv + kSum;
inline constexpr double kLibExec =
    4.0 * kGrad + 4.0 * kStressPerF + kDiv + kSum;
// What a fan-out-deduplicated graph would retire: the gradient stage once
// (shared by every consumer) but still four single-output F nodes.
inline constexpr double kDagExec = kGrad + 4.0 * kStressPerF + kDiv + kSum;
}  // namespace flopcount

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    int E = (argc > 1) ? std::atoi(argv[1]) : cfg::E_default;
    // Every measurement is now >=10 ms, so a min over 5 is as tight as a min
    // over 20 was on the old sub-millisecond timings -- and 20 reps of a 2.6 s
    // fused kernel would put the GPU run over a minute.
    int reps   = (argc > 2) ? std::atoi(argv[2]) : 5;
    int warmup = (argc > 3) ? std::atoi(argv[3]) : 2;
    E          = (E / cfg::TE) * cfg::TE;  // must tile evenly

    const bool gpu = cfg::kIsGPU;
    std::printf(
        "SEM stiffness (K&T 1999 A2/A7), 2D P-SV | N=%d C=%d TE=%d E=%d | %s\n",
        cfg::N, cfg::C, cfg::TE, E, gpu ? "GPU" : "CPU (Serial)");

    V3 u0("u0", E, cfg::N, cfg::N), u1("u1", E, cfg::N, cfg::N);
    V3 xix("xix", E, cfg::N, cfg::N), xiz("xiz", E, cfg::N, cfg::N);
    V3 gx("gx", E, cfg::N, cfg::N), gz("gz", E, cfg::N, cfg::N);
    V3 l2m("l2m", E, cfg::N, cfg::N), mu("mu", E, cfg::N, cfg::N);
    V3 jac("jac", E, cfg::N, cfg::N);
    V2 H("H", cfg::N, cfg::N), Hw("Hw", cfg::N, cfg::N);
    V3 fb0("fb0", E, cfg::N, cfg::N), fb1("fb1", E, cfg::N, cfg::N);
    V3 fl0("fl0", E, cfg::N, cfg::N), fl1("fl1", E, cfg::N, cfg::N);

    fill_field(u0, E, 0);
    fill_field(u1, E, 1);
    fill_field(xix, E, 2);
    fill_field(xiz, E, 3);
    fill_field(gx, E, 4);
    fill_field(gz, E, 5);
    fill_field(l2m, E, 6);
    fill_field(mu, E, 7);
    fill_field(jac, E, 8);
    Kokkos::Array<float, cfg::N> w{};
    fill_operators(H, Hw, w);
    Kokkos::fence();

    // Scratch, measured rather than estimated: the fused tree's real cost.
    auto node =
        library_node<0>(u0, u1, xix, xiz, gx, gz, l2m, mu, jac, H, Hw, w);
    using LibEval = Evaluator<TeamPolicyTag<>, decltype(node), CombFrc>;
    const std::size_t sbytes = LibEval::scratch_size_per_team(library_tile());
    const std::size_t smax   = gpu ? 48u * 1024u : 32u * 1024u;
    std::printf("library scratch/team: %zu bytes (limit ~%zu)%s\n", sbytes,
                smax, sbytes > smax ? "  <-- OVER" : "");
    // Per-THREAD state, which is what actually limits this kernel on GPU. The
    // whole node graph is stored BY VALUE inside the evaluator, and every
    // nesting level stores its operands' nodes AGAIN: the parent keeps `node`,
    // and its ScratchAllocator keeps an inner Evaluator holding another copy.
    // A leaf View is therefore duplicated once per level of the tree.
    std::printf(
        "per-thread: node graph %zu B, evaluator %zu B (%zu regs if resident)"
        " | one View = %zu B\n",
        sizeof(decltype(node)), sizeof(LibEval), sizeof(LibEval) / 4,
        sizeof(V3));
    if (sbytes > smax) {
      std::printf(
          "The fused graph does not fit. On CPU there is no valid TE at all:\n"
          "  GEMM needs   TE*%d %% (2*simd_width) == 0  -> TE >= 4\n"
          "  scratch needs 8960*TE + 2560 <= 32768      -> TE <= 3\n"
          "Fixing it needs a GEMM remainder path or scratch recycling.\n",
          cfg::N);
      Kokkos::finalize();
      return 0;
    }

    // What team size does Kokkos::AUTO give each kernel? The fused tree's
    // stages are TeamVectorRange loops over TE*N*N points; if scratch pressure
    // forces a small team, every stage serializes.
    {
      using ES = Kokkos::DefaultExecutionSpace;
      Kokkos::TeamPolicy<ES> pf(E / cfg::TE, Kokkos::AUTO);
      pf.set_scratch_size(0, Kokkos::PerTeam(static_cast<int>(sbytes)));
      Kokkos::TeamPolicy<ES> pg(E / cfg::TE, Kokkos::AUTO);
      pg.set_scratch_size(0, Kokkos::PerTeam(2368));
      std::printf(
          "team size (AUTO): fused=%d  single-gradient=%d  (tile has %d "
          "points)\n\n",
          pf.team_size_recommended(
              [] KOKKOS_FUNCTION(
                  const typename Kokkos::TeamPolicy<ES>::member_type&) {},
              Kokkos::ParallelForTag{}),
          pg.team_size_recommended(
              [] KOKKOS_FUNCTION(
                  const typename Kokkos::TeamPolicy<ES>::member_type&) {},
              Kokkos::ParallelForTag{}),
          cfg::TE * cfg::N * cfg::N);
    }

    // Reference + correctness.
    // The host reference is a serial scalar loop, so check a bounded PREFIX
    // rather than all E: every element is computed independently and by the
    // same code path, so a prefix exercises every index permutation the full
    // range does. Timing still runs over the full E.
    const int Echk = std::min(E, 65536);
    V3H ref0("ref0", Echk, cfg::N, cfg::N), ref1("ref1", Echk, cfg::N, cfg::N);
    host_reference(u0, u1, xix, xiz, gx, gz, l2m, mu, jac, H, Hw, w, ref0, ref1,
                   Echk);

    auto run_base = [&] {
      baseline_force(u0, u1, xix, xiz, gx, gz, l2m, mu, jac, H, Hw, w, fb0, fb1,
                     E);
    };
    auto run_lib = [&] {
      library_force<0>(u0, u1, xix, xiz, gx, gz, l2m, mu, jac, H, Hw, w, fl0);
      library_force<1>(u0, u1, xix, xiz, gx, gz, l2m, mu, jac, H, Hw, w, fl1);
    };
    run_base();
    run_lib();
    Kokkos::fence();

    const double db =
        std::max(max_rel_diff(fb0, ref0, Echk), max_rel_diff(fb1, ref1, Echk));
    const double dl =
        std::max(max_rel_diff(fl0, ref0, Echk), max_rel_diff(fl1, ref1, Echk));

    const double tb = seconds_of(run_base, warmup, reps);
    const double tl = seconds_of(run_lib, warmup, reps);
    const double Ed = static_cast<double>(E);

    std::printf("%-30s %10s %14s %16s %8s\n", "impl", "time(ms)", "useful GF/s",
                "executed GF/s", "check");
    std::printf("%-30s %10.3f %14.1f %16s %8s\n", "baseline (hand, 1 kernel)",
                tb * 1e3, flopcount::kUseful * Ed / tb / 1e9, "-",
                db < 1e-2 ? "PASS" : "FAIL");
    std::printf(
        "%-30s %10.3f %14.1f %16.1f %8s\n", "library (fused, 2 kernels)",
        tl * 1e3, flopcount::kUseful * Ed / tl / 1e9,
        flopcount::kLibExec * Ed / tl / 1e9, dl < 1e-2 ? "PASS" : "FAIL");
    // THE CONTROL: hand-written, but with the library's exact redundancy.
    V3   fc0("fc0", E, cfg::N, cfg::N), fc1("fc1", E, cfg::N, cfg::N);
    auto run_ctl = [&] {
      control_recompute_force<0>(u0, u1, xix, xiz, gx, gz, l2m, mu, jac, H, Hw,
                                 w, fc0, E);
      control_recompute_force<1>(u0, u1, xix, xiz, gx, gz, l2m, mu, jac, H, Hw,
                                 w, fc1, E);
    };
    run_ctl();
    Kokkos::fence();
    const double dc =
        std::max(max_rel_diff(fc0, ref0, Echk), max_rel_diff(fc1, ref1, Echk));
    const double tc = seconds_of(run_ctl, warmup, reps);
    std::printf(
        "%-30s %10.3f %14.1f %16.1f %8s\n", "CONTROL (hand, lib's redund.)",
        tc * 1e3, flopcount::kUseful * Ed / tc / 1e9,
        flopcount::kLibExec * Ed / tc / 1e9, dc < 1e-2 ? "PASS" : "FAIL");

    // Same control, gradients computed ONCE per kernel. Isolates the gradient
    // recompute from the rest of the redundancy (2 kernels, 4 stress evals).
    auto run_ctl2 = [&] {
      control_recompute_force<0, true>(u0, u1, xix, xiz, gx, gz, l2m, mu, jac,
                                       H, Hw, w, fc0, E);
      control_recompute_force<1, true>(u0, u1, xix, xiz, gx, gz, l2m, mu, jac,
                                       H, Hw, w, fc1, E);
    };
    run_ctl2();
    Kokkos::fence();
    const double dc2 =
        std::max(max_rel_diff(fc0, ref0, Echk), max_rel_diff(fc1, ref1, Echk));
    const double tc2 = seconds_of(run_ctl2, warmup, reps);
    std::printf("%-30s %10.3f %14.1f %16s %8s\n", "CONTROL-B (grads deduped)",
                tc2 * 1e3, flopcount::kUseful * Ed / tc2 / 1e9, "-",
                dc2 < 1e-2 ? "PASS" : "FAIL");

    // CONTROL-C: the DAG's work profile. Fan-out deduplicated (4 gradient sums
    // shared by every consumer) and both components in one launch, but still
    // four single-output F nodes with 4x stress and 4x aux traffic.
    auto run_ctl3 = [&] {
      control_dag_force(u0, u1, xix, xiz, gx, gz, l2m, mu, jac, H, Hw, w, fc0,
                        fc1, E);
    };
    run_ctl3();
    Kokkos::fence();
    const double dc3 =
        std::max(max_rel_diff(fc0, ref0, Echk), max_rel_diff(fc1, ref1, Echk));
    const double tc3 = seconds_of(run_ctl3, warmup, reps);
    std::printf("%-30s %10.3f %14.1f %16.1f %8s\n", "CONTROL-C (DAG profile)",
                tc3 * 1e3, flopcount::kUseful * Ed / tc3 / 1e9,
                flopcount::kDagExec * Ed / tc3 / 1e9,
                dc3 < 1e-2 ? "PASS" : "FAIL");

    // Diagnostic row: one gradient contraction, output modes {q,e,j}.
    V3dyn        go("go", cfg::N, E, cfg::N);
    auto         run_one = [&] { library_one_gradient(u0, H, go); };
    const double t1g     = seconds_of(run_one, warmup, reps);
    const double f1g     = flopcount::NP * cfg::N * 2.0 * Ed;  // one gradient
    std::printf("%-30s %10.3f %14.1f %16s %8s\n", "  [diag] 1 gradient only",
                t1g * 1e3, f1g / t1g / 1e9, "-", "-");

    auto         run_4g = [&] { library_four_gradients(u0, u1, H, go); };
    const double t4g    = seconds_of(run_4g, warmup, reps);
    std::printf("%-30s %10.3f %14.1f %16s %8s\n",
                "  [diag] 4 gradients, no aux", t4g * 1e3,
                4.0 * f1g / t4g / 1e9, "-", "-");

    auto run_1F = [&] {
      library_one_F(u0, u1, xix, xiz, gx, gz, l2m, mu, jac, H, go);
    };
    const double t1F = seconds_of(run_1F, warmup, reps);
    std::printf("%-30s %10.3f %14.1f %16s %8s\n",
                "  [diag] 1 F node (4grad+stress)", t1F * 1e3,
                4.0 * f1g / t1F / 1e9, "-", "-");
    // ONE force component: 1 launch, 8 gradient sums, 2 stress evals, 2
    // divergences. This is the closest measurable proxy for what a 2-output
    // fused combine operand would produce (which adds back 2 divergences and a
    // second output, but shares everything below them).
    auto run_lib1 = [&] {
      library_force<0>(u0, u1, xix, xiz, gx, gz, l2m, mu, jac, H, Hw, w, fl0);
    };
    const double tl1 = seconds_of(run_lib1, warmup, reps);
    std::printf("%-30s %10.3f %14s %16s %8s\n", "  [diag] library, 1 component",
                tl1 * 1e3, "-", "-", "-");

    auto         run_1d = [&] { library_one_divergence(go, Hw, fl0); };
    const double t1d    = seconds_of(run_1d, warmup, reps);
    std::printf("%-30s %10.3f %14.1f %16s %8s\n",
                "  [diag] 1 divergence (mat. B)", t1d * 1e3, f1g / t1d / 1e9,
                "-", "-");

    std::printf(
        "\n  4 x [1 F node] = %.3f ms  (the tree holds 4 of them) vs library "
        "%.3f ms\n"
        "  aux-read cost per F node = %.3f ms; 4 grads vs 4 x 1 grad = %.3f / "
        "%.3f ms\n",
        4.0 * t1F * 1e3, tl * 1e3, (t1F - t4g) * 1e3, t4g * 1e3,
        4.0 * t1g * 1e3);

    std::printf(
        "\nspeedup (base/lib): %.4fx   |   library recompute factor: %.2fx\n",
        tb / tl, flopcount::kLibExec / flopcount::kUseful);
    std::printf(
        "DECOMPOSITION of the %.2fx gap:\n"
        "  redundancy    (baseline -> control): %.2fx\n"
        "     of which gradient recompute:      %.2fx  (control-B -> control)\n"
        "     of which 2 kernels + 4x stress:   %.2fx  (baseline  -> "
        "control-B)\n"
        "  library cost  (control  -> library): %.2fx\n",
        tl / tb, tc / tb, tc / tc2, tc2 / tb, tl / tc);

    // What a fan-out-deduplicating DAG evaluator would be worth. The first two
    // rows are MEASURED (CONTROL-C is a real kernel validated against the host
    // reference); the third is a PROJECTION that assumes the library's measured
    // residual carries over unchanged to a restructured evaluator, which is
    // exactly the kind of assumption this benchmark exists to stop trusting.
    {
      const double residual = tl / tc;         // library / CONTROL, measured
      const double proj     = tc3 * residual;  // projected library-on-a-DAG
      std::printf(
          "\nDAG PAYOFF (CONTROL-C = 4 grad sums, 4 stress, 1 launch):\n"
          "  removable by sharing  (control-C -> control): %.2fx  <- the "
          "ceiling\n"
          "  NOT removable by sharing (baseline -> control-C): %.2fx  (4x "
          "stress + 4x aux)\n"
          "  PROJECTED library on a DAG = %.3f ms x %.2f residual = %.3f ms"
          "  -> gap %.2fx (from %.2fx)\n"
          "  [projection only: assumes the residual is unchanged by the "
          "restructuring]\n",
          tc / tc3, tc3 / tb, tc3 * 1e3, residual, proj * 1e3, proj / tb,
          tl / tb);
    }
    std::printf("source lines: library %d, hand-written %d\n",
                kLibSrcEnd - kLibSrcBegin - 1, kBaseSrcEnd - kBaseSrcBegin - 1);
    std::printf(
        "max rel diff vs host reference: baseline %.2e, library %.2e, "
        "control-C %.2e\n",
        db, dl, dc3);
    if (db >= 1e-2 || dl >= 1e-2) rc = 1;
  }
  Kokkos::finalize();
  return rc;
}
