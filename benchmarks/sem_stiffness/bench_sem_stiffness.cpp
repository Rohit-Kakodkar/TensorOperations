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
// MEASURED RESULT. The library TREE is 3.41x slower than the hand kernel on
// H100 and 2.56x slower on Serial (it was 4.26x / 3.69x before the index-math
// work of task 5). Those two headline rows are NOT comparable on
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
//   H100     3.41x =            2.89x                  x        1.18x
//   Serial   2.56x =            3.22x                  x        0.80x
//
// (Was 1.48x / 1.11x on the residual before task 5. On Serial the residual is
// now BELOW one: the library is faster than the hand-written kernel doing its
// exact work profile.)
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
// What is left is the residual -- see SEM_STIFFNESS_GAP_ANALYSIS.md for the
// ordered statement of work. Both halves of the redundancy term have now been
// paid off: fan-out deduplication by the DAG row, and per-node duplicated work
// by the DAG-MO row below.
//
// THE DAG ROW, added 2026-08-03, is the library with fan-out deduplicated: the
// four DISTINCT gradients computed once, both components in one launch, 14
// nodes against the tree's 26. It is the same library and the same physics --
// only the graph is spelled flat, with consumers NAMING earlier results instead
// of nesting them.
//
//   GPU  33.2 -> 14.979 ms  (2.22x), gap to baseline 3.41x -> 1.53x
//   CPU  107.1 -> 58.486 ms (1.83x), gap to baseline 2.57x -> 1.40x
//   scratch 24312 B PER COMPONENT -> 20688 B for BOTH
//
// THE DAG-MO ROW, added 2026-08-03, is the same DAG with a MULTI-OUTPUT F
// stage: the four stress integrands merged into two 2-output combine nodes, one
// per reference direction, each emitting both force components from one chain
// rule and one compute_stress. 12 nodes owning the same 14 slots.
//
//   GPU  14.979 -> 12.068 ms  (1.24x), gap to baseline 1.53x -> 1.24x
//   CPU  58.486 -> 45.204 ms  (1.29x), gap to baseline 1.40x -> 1.08x
//   scratch 20688 B, UNCHANGED (two outputs are two buffers either way)
//
// This is the redundancy SHARING COULD NOT REACH, and the distinction is worth
// keeping straight: the DAG collapses duplicated SUBTREES, and the four F nodes
// were never that -- they were four different functions of one shared gradient
// set, each rebuilding the same stress tensor and re-reading all seven
// auxiliary arrays. Only a node that emits more than one tensor removes it.
// CONTROL-C still carries four stress evals, so it is no longer this row's
// floor: on both backends DAG-MO is FASTER than the hand-written kernel with
// the DAG's work profile (0.72x on CPU, 0.96x on GPU).
//
// (Both library rows moved again when the subview delinearize was given
// compile-time extents -- see SEM_STIFFNESS_GAP_ANALYSIS.md task 5. The tree
// gained 1.25x on GPU and 1.41x on CPU from that alone, so the DAG's speedup
// over it reads smaller here than the 2.27x first measured.)
//
// TEAM SIZE IS WORTH 1.7x HERE, AND Kokkos::AUTO GETS IT WRONG. AUTO sizes the
// team from occupancy alone and picks 512 threads for a tile of TE*N*N = 256
// points, so every TeamVectorRange leaves half the team idle: 42.5 ms at 512,
// 24.0 at 256, 16.8 at 128, **15.0 at 64**, 18.3 at 32. The optimum is interior
// and it is LOW occupancy (21.6%), because doubling the team to 128 doubles
// resident warps and buys 0.5% more fp32 work while adding 32% more
// warp-instructions and 3.6x the barrier stall -- see DagGraph.hpp's
// execute_dag_team for the ncu table. An earlier revision of this comment
// reported the AUTO number as the DAG's result and drew a wrong conclusion.
//
// It is SWEPT in every run rather than trusted, which is how 128 (tuned before
// the multi-output F stage existed) was caught as stale.
//
// THE RESIDUAL DOES SURVIVE THE RESTRUCTURING. CONTROL-C -> DAG measured 1.45x
// against the tree's CONTROL -> library 1.48x, so the projection was right.
// (Task 5 has since taken it to 1.31x on GPU and 0.93x on CPU -- on Serial the
// library now beats the hand-written kernel of the same work profile.)
// ncu attributes it to instruction count, as for the tree: 2.71x CONTROL-C's
// integer thread-instructions for 1.50x the fp32 work, at an integer:fp32 ratio
// of 3.05 against 1.68. What it is NOT is per-thread state -- registers are 32,
// the SAME as the hand-written control and a third of the tree's 96 -- nor
// occupancy, since the DAG runs at 43% and beats its own 98%-occupancy
// configuration by 1.42x. See SEM_STIFFNESS_GAP_ANALYSIS.md, Task 4.
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
#include <TensorOperations/DagGraph.hpp>
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

inline constexpr int N = 8;  // GLL points per direction
inline constexpr int C = 2;  // displacement components (unrolled, see below)
// Elements per team. PER BACKEND, because the two have different scratch
// ceilings and the optimum is nowhere near the same: see the block below.
// Each implementation runs at ITS OWN best tile, because they do not share one.
// Forcing a single TE on all of them does not just leave performance on the
// table -- it distorts the comparison. At TE=16 the hand-written baseline
// degrades to 12.768 ms from its own best of 9.719, so a single-TE run at 16
// would report DAG-MO as 1.42x faster than "the baseline" when the honest
// best-against-best figure is 1.08x. Measured bests (H100 NVL, E=2.5M,
// best-of-7 with warmup): baseline TE=8, tree TE=8, DAG TE=16, DAG-MO TE=16.
//
// Serial's 32 KB scratch cap rejects anything above TE=4 for the DAG, so the
// CPU side stays at 4 throughout.
inline constexpr int TE_ctl  = kIsGPU ? 8 : 4;   // baseline + CONTROL family
inline constexpr int TE_tree = kIsGPU ? 8 : 4;   // library, fused tree
inline constexpr int TE_dag  = kIsGPU ? 16 : 4;  // library DAG and DAG-MO

// The legacy single knob, kept only for E's divisibility check and the header
// line. Every kernel now names the constant it actually uses.
inline constexpr int TE = TE_dag;

// N=8 is padding-free for the register kernel: every GEMM below is
//   SA = N = 8            (free-A: the hprime output index)
//   SK = N = 8            (contracted)
//   SB = TE * N = 32      (free-B: element x the spectator GLL axis)
// GPU blocks are MT=4, NT=2, NR=2; CPU are MT=8, NT=8, NR=2*simd_width (32 on
// AVX-512). All three dims divide on both.
static_assert(N % 8 == 0, "SA/SK must satisfy CPU MT=NT=8 and GPU MT=4,NT=2");
static_assert((TE * N) % 32 == 0,
              "SB must satisfy CPU NR=2*simd_width (32 on AVX-512)");

// GPU TE=16, CPU TE=4, and the split is forced -- there is no value that is
// both legal on Serial and good on an H100.
//
// The TREE's scratch is additive with no recycling, every intermediate live at
// once. Per force-component kernel, in floats:
//   gradient contraction = 128*TE + 64      (A stage, B stage, C out)
//   F combine            = 576*TE + 256      (out + 4 gradients)
//   divergence           = 640*TE + 320
//   force combine        = 1344*TE + 640     => 5376*TE + 2560 bytes
// So the tree is 24312 B at TE=4 and 88824 B at TE=16 -- over the 48 KB every
// generation before Hopper had, which is why this file used to say TE=4 was
// "the only value that runs on both". That is now false in both halves:
//
//   * the DAG does NOT scale that way any more. Slots are pooled by live range
//     (8 buffers, not 14) and operand staging is one arena sized by the largest
//     node, not the sum over nodes, so the DAG needs 37208 B at TE=16 where the
//     tree needs 88824.
//   * an H100 accepts opt-in shared memory to 227 KB and Kokkos honours it, so
//     even the tree fits. See the smax guard in main().
//
// Serial is the real constraint, and it is unmoved: its 32 KB cap rejects the
// DAG's 37208 B outright, so the CPU side stays at TE=4. TE=2 fails the CPU
// SB % NR rule above, so 4 is still the CPU's only option.
//
// GPU TE=16 with team 128 is the measured optimum over TE in {4,8,16,32} x team
// in {32..1024}, best-of-7 with warmup, uncontended H100 NVL, E=2.5M:
//
//   TE      4      8     16     32     (DAG-MO at its own best team size)
//   best 10.009  9.248  8.981 10.023 ms
//   team     64     96    128    256
//
// The optimum holds ~8 tile points per thread at every TE, so team size must
// scale with TE -- see kDagTeam.

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
// The GLL weights as a device View. See WeightedSum for why this is not a
// Kokkos::Array. `const` so the read is known non-aliasing; the hand-written
// implementations keep the Array, which makes them the control for the change.
using V1 = Kokkos::View<const float*, Kokkos::LayoutRight>;

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

// Chain rule + stress + BOTH integrand slots for one reference direction,
// shared by every path so none can drift from the others.
//   Dir 0 = xi (uses xix, xiz), Dir 1 = gamma (uses gammax, gammaz)
//
// Both components come out of ONE chain rule and ONE compute_stress: that is
// the physical fact a 2-output combine expresses and four single-output nodes
// throw away. integrand_slot below selects one component from it, so the
// scalar callers pay nothing extra -- the discarded half is dead code at a
// compile-time index.
template <int Dir>
KOKKOS_INLINE_FUNCTION Kokkos::Array<float, 2> integrand_slots(
    float dxi0, float dxi1, float dgm0, float dgm1, float xix, float xiz,
    float gx, float gz, float l2m, float mu, float jac) {
  Mat2 ds{};
  ds.m[0][0]      = dxi0 * xix + dgm0 * gx;  // du_x/dx
  ds.m[0][1]      = dxi0 * xiz + dgm0 * gz;  // du_x/dz
  ds.m[1][0]      = dxi1 * xix + dgm1 * gx;  // du_z/dx
  ds.m[1][1]      = dxi1 * xiz + dgm1 * gz;  // du_z/dz
  const Mat2  sig = compute_stress(ds, l2m, mu);
  const float dx  = (Dir == 0) ? xix : gx;
  const float dz  = (Dir == 0) ? xiz : gz;
  return {jac * (sig.m[0][0] * dx + sig.m[0][1] * dz),
          jac * (sig.m[1][0] * dx + sig.m[1][1] * dz)};
}

template <int Comp, int Dir>
KOKKOS_INLINE_FUNCTION float integrand_slot(float dxi0, float dxi1, float dgm0,
                                            float dgm1, float xix, float xiz,
                                            float gx, float gz, float l2m,
                                            float mu, float jac) {
  return integrand_slots<Dir>(dxi0, dxi1, dgm0, dgm1, xix, xiz, gx, gz, l2m, mu,
                              jac)[Comp];
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

// The SAME node, emitting BOTH force components for one reference direction.
//
// Returning Kokkos::Array<float,2> makes this a multi-output combine: NumOut is
// deduced from fn's return type, the node owns two slots, and one pass over the
// four gradients produces both. Against StressIntegrand that halves the stress
// evaluations (4 -> 2) and the auxiliary global traffic (4x -> 2x) for the
// whole graph, which is the redundancy fan-out deduplication cannot touch --
// the four F nodes are not a shared subtree, they are four different functions
// of one.
//
// Both components share modes by construction (they differ only in which row of
// the stress tensor they contract), which is what makes them legal outputs of
// ONE node. The xi and gamma directions do NOT -- {q,e,j} vs {q,e,i} -- so they
// stay two nodes. A single 4-output node would force a reorder on the B slot of
// the divergence contraction; see THE LOAD-BEARING RULE in the header.
template <int Dir>
struct StressIntegrand2 {
  V3              xix, xiz, gx, gz, l2m, mu, jac;
  KOKKOS_FUNCTION Kokkos::Array<float, 2> operator()(int c0, int c1, int c2,
                                                     float dxi0, float dxi1,
                                                     float dgm0,
                                                     float dgm1) const {
    const int e = c1;
    const int z = (Dir == 0) ? c2 : c0;
    const int x = (Dir == 0) ? c0 : c2;
    return integrand_slots<Dir>(dxi0, dxi1, dgm0, dgm1, xix(e, z, x),
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
// A VIEW, not a Kokkos::Array, and the difference is worth 28% of this kernel's
// L1TEX sectors. An Array member indexed dynamically (j and i are runtime
// coordinates) cannot live in registers, so nvcc copies the whole enclosing
// node-handle struct out of the kernel parameter bank onto the per-thread LOCAL
// stack -- ~23 STL per node -- and then reads w back with LDL at 18 sectors per
// warp. Profiled: 100% of the kernel's local loads were these two reads.
//
// As a View, w is a pointer: `w(j)` is an LDG off the param bank with no object
// to materialize. And it costs almost nothing to read -- the output is (e,j,i)
// traversed i-fastest, so a warp spans i=0..7 and 4 values of j, which puts
// BOTH reads inside a single 32 B sector. The same 32 bytes are read by every
// warp of every block, so they sit in L1 permanently.
//
// This is the general answer for a rank-deficient auxiliary that has to ride
// inside the functor: hand it in as a View, never as a value member you index.
struct WeightedSum {
  V1                    w;
  KOKKOS_FUNCTION float operator()(int, int j, int i, float t1,
                                   float t2) const {
    return w(j) * t1 + w(i) * t2;
  }
};

// ===========================================================================
// Implementation 1: the library. One fully-fused graph per force component.
//
// Tile bundles, bottom-up. Two shapes recur: element-major (e,.,.) for a
// node declared in user order, and GLL-major (.,e,.) for a fused node declared
// in its consumer's canonical order.
// ===========================================================================
using TileH = StaticTile<cfg::N, cfg::N>;  // H / Hw        [8,8]
template <int TE>
using TileEJI = StaticTile<TE, cfg::N, cfg::N>;  // (e,.,.)     [4,8,8]
template <int TE>
using TileQEJ = StaticTile<cfg::N, TE, cfg::N>;  // (.,e,.)     [8,4,8]
template <int TE>
using BGrad = Tile<TileH, TileEJI<TE>, TileQEJ<TE>>;  // one gradient
inline constexpr int kLibSrcBegin = __LINE__;
// The F combine takes FOUR operands -- the gradients. Everything else the
// physics needs is captured by StressIntegrand (see above).
template <int TE>
using CombF =
    CombineTile<TileQEJ<TE>, BGrad<TE>, BGrad<TE>, BGrad<TE>, BGrad<TE>>;
template <int TE>
using BDiv = Tile<TileH, CombF<TE>, TileEJI<TE>>;
template <int TE>
using CombFrc = CombineTile<TileEJI<TE>, BDiv<TE>, BDiv<TE>>;

template <int TE>
CombFrc<TE> library_tile() {
  const auto cf = make_combine_tile(TileQEJ<TE>{}, BGrad<TE>{}, BGrad<TE>{},
                                    BGrad<TE>{}, BGrad<TE>{});
  const auto bd = BDiv<TE>{TileH{}, cf, TileEJI<TE>{}};
  return make_combine_tile(TileEJI<TE>{}, bd, bd);
}

template <int Comp, int TE>
auto library_node(V3 u0, V3 u1, V3 xix, V3 xiz, V3 gx, V3 gz, V3 l2m, V3 mu,
                  V3 jac, V2 H, V2 Hw, V1 w) {
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

template <int Comp, int TE>
void library_force(V3 u0, V3 u1, V3 xix, V3 xiz, V3 gx, V3 gz, V3 l2m, V3 mu,
                   V3 jac, V2 H, V2 Hw, V1 w, V3 force) {
  auto g                          = make_graph();
  [[maybe_unused]] auto [g1, out] = g.ops(
      library_node<Comp, TE>(u0, u1, xix, xiz, gx, gz, l2m, mu, jac, H, Hw, w));
  g1.execute(TeamPolicyTag<>{}, library_tile<TE>(), force);
}
inline constexpr int kLibSrcEnd = __LINE__;

// ===========================================================================
// Implementation 5: the library as a DAG. ONE launch, BOTH components, each
// distinct gradient computed ONCE.
//
// The tree spelling above nests operands, so a subtree reachable from two
// consumers is evaluated once per consumer: 8 gradient sums per component
// kernel, 16 across the two. There are only FOUR distinct gradients.
//
// Contract u's x-axis, or its z-axis, for each of the two displacement
// components. Every consumer wants one of those four -- what differs is only
// what it CALLS the axes:
//
//   gx_u0 is physically (new-x, e, z).  F0 calls it {q,e,j}; F1 calls it
//                                       {i,e,q}. Same buffer, two names.
//   gz_u0 is physically (new-z, e, x).  F0 calls it {j,e,q}; F1 calls it
//                                       {q,e,i}.
//
// That is the whole trick, and it is why no relabel node was needed: a slot
// handle's `as<labels...>()` names the buffer's axes, so naming one twice costs
// source text and nothing else. `plans/specfem-kernel-graph.md` argued
// relabeling is free BECAUSE there is no memoization; the conclusion survives
// memoization, the reasoning does not.
//
// 14 nodes: 4 gradients, 4 stress integrands (2 components x 2 directions),
// 4 divergences, 2 weighted sums. Against the tree's 26 across two launches.
//
// WHAT THIS DOES NOT COLLAPSE. The four F nodes still each rebuild the stress
// tensor and re-read all seven auxiliary arrays, because they are four separate
// single-output combines. Deduplicating THAT needs multi-output combine -- see
// the MULTI-OUTPUT F STAGE below, which is the same graph with those four nodes
// merged into two 2-output ones. CONTROL-C measures the 4-stress profile -- 4
// gradient sums, 4 stress evals, one launch -- so it is the floor THIS row is
// aimed at, and the multi-output row is expected to go below it.
// ===========================================================================
inline constexpr int kDagSrcBegin = __LINE__;
template <int TE>
using CombF2 = CombineTile<TileQEJ<TE>, TileQEJ<TE>, TileQEJ<TE>, TileQEJ<TE>,
                           TileQEJ<TE>>;
template <int TE>
using BDiv2 = Tile<TileH, TileQEJ<TE>, TileEJI<TE>>;
template <int TE>
using CombR2 = CombineTile<TileEJI<TE>, TileEJI<TE>, TileEJI<TE>>;

// --- the four distinct gradients, each computed once -----------------------
// Shared by both F-stage spellings, so a difference between the two rows can
// only come from the F stage.
template <int TE>
auto sem_dag_gradients(V3 u0, V3 u1, V2 H) {
  auto [d0, gxu0] = make_dag<float>().add(
      make_contraction_node<'q', 'e', 'j'>(
          make_input_node(make_handle<'q', 'p'>(H)),
          make_input_node(make_handle<'e', 'j', 'p'>(u0))),
      BGrad<TE>{});
  auto [d1, gxu1] = d0.add(make_contraction_node<'q', 'e', 'j'>(
                               make_input_node(make_handle<'q', 'p'>(H)),
                               make_input_node(make_handle<'e', 'j', 'p'>(u1))),
                           BGrad<TE>{});
  auto [d2, gzu0] = d1.add(make_contraction_node<'j', 'e', 'q'>(
                               make_input_node(make_handle<'j', 'p'>(H)),
                               make_input_node(make_handle<'e', 'p', 'q'>(u0))),
                           BGrad<TE>{});
  auto [d3, gzu1] = d2.add(make_contraction_node<'j', 'e', 'q'>(
                               make_input_node(make_handle<'j', 'p'>(H)),
                               make_input_node(make_handle<'e', 'p', 'q'>(u1))),
                           BGrad<TE>{});
  return std::make_tuple(d3, gxu0, gxu1, gzu0, gzu1);
}

// --- four divergences and two weighted sums --------------------------------
// Also shared. f<Comp><Dir> is the xi-direction (Dir 0) or gamma-direction
// (Dir 1) integrand for component Comp, whichever node produced it.
template <int TE, typename Dag, typename F00, typename F01, typename F10,
          typename F11>
auto sem_dag_tail(const Dag& d7, F00 f00, F01 f01, F10 f10, F11 f11, V2 Hw,
                  V1 w) {
  // Each F slot is named in its consumer's canonical B order (contracted ++
  // freeB), so the operand is an identity permutation and stays zero-copy --
  // the LOAD-BEARING RULE, unchanged by the DAG.
  auto [d8, t10]  = d7.add(make_contraction_node<'e', 'j', 'i'>(
                               make_input_node(make_handle<'q', 'i'>(Hw)),
                               f00.template as<'q', 'e', 'j'>()),
                           BDiv2<TE>{});
  auto [d9, t20]  = d8.add(make_contraction_node<'e', 'j', 'i'>(
                               make_input_node(make_handle<'q', 'j'>(Hw)),
                               f01.template as<'q', 'e', 'i'>()),
                           BDiv2<TE>{});
  auto [d10, t11] = d9.add(make_contraction_node<'e', 'j', 'i'>(
                               make_input_node(make_handle<'q', 'i'>(Hw)),
                               f10.template as<'q', 'e', 'j'>()),
                           BDiv2<TE>{});
  auto [d11, t21] = d10.add(make_contraction_node<'e', 'j', 'i'>(
                                make_input_node(make_handle<'q', 'j'>(Hw)),
                                f11.template as<'q', 'e', 'i'>()),
                            BDiv2<TE>{});

  // A divergence's canonical output is freeA ++ freeB: {i,e,j} for the xi
  // slots, {j,e,i} for the gamma ones. That is the order its buffer is
  // written in, so that is how the slot is named.
  auto [d12, r0] =
      d11.add(make_combine_node<'e', 'j', 'i'>(t10.template as<'i', 'e', 'j'>(),
                                               t20.template as<'j', 'e', 'i'>(),
                                               WeightedSum{w}),
              CombR2<TE>{});
  auto [d13, r1] =
      d12.add(make_combine_node<'e', 'j', 'i'>(t11.template as<'i', 'e', 'j'>(),
                                               t21.template as<'j', 'e', 'i'>(),
                                               WeightedSum{w}),
              CombR2<TE>{});

  return std::make_tuple(d13, r0, r1);
}

// Builds the graph and hands back its two roots. Separate from the launch so
// the host can size it without running it -- scratch is the number that decided
// against the previous attempt at this, so it must be printable.
template <int TE>
auto sem_dag_graph(V3 u0, V3 u1, V3 xix, V3 xiz, V3 gx, V3 gz, V3 l2m, V3 mu,
                   V3 jac, V2 H, V2 Hw, V1 w) {
  const StressIntegrand<0, 0> si00{xix, xiz, gx, gz, l2m, mu, jac};
  const StressIntegrand<0, 1> si01{xix, xiz, gx, gz, l2m, mu, jac};
  const StressIntegrand<1, 0> si10{xix, xiz, gx, gz, l2m, mu, jac};
  const StressIntegrand<1, 1> si11{xix, xiz, gx, gz, l2m, mu, jac};

  auto [d3, gxu0, gxu1, gzu0, gzu1] = sem_dag_gradients<TE>(u0, u1, H);

  // --- four stress integrands, all four naming the SAME four gradients ----
  // The xi-direction nodes read them as {q,e,j}/{j,e,q}; the gamma-direction
  // nodes read the identical buffers as {i,e,q}/{q,e,i}.
  auto [d4, f00] = d3.add(
      make_combine_node<'q', 'e', 'j'>(gxu0.template as<'q', 'e', 'j'>(),
                                       gxu1.template as<'q', 'e', 'j'>(),
                                       gzu0.template as<'j', 'e', 'q'>(),
                                       gzu1.template as<'j', 'e', 'q'>(), si00),
      CombF2<TE>{});
  auto [d5, f01] = d4.add(
      make_combine_node<'q', 'e', 'i'>(gxu0.template as<'i', 'e', 'q'>(),
                                       gxu1.template as<'i', 'e', 'q'>(),
                                       gzu0.template as<'q', 'e', 'i'>(),
                                       gzu1.template as<'q', 'e', 'i'>(), si01),
      CombF2<TE>{});
  auto [d6, f10] = d5.add(
      make_combine_node<'q', 'e', 'j'>(gxu0.template as<'q', 'e', 'j'>(),
                                       gxu1.template as<'q', 'e', 'j'>(),
                                       gzu0.template as<'j', 'e', 'q'>(),
                                       gzu1.template as<'j', 'e', 'q'>(), si10),
      CombF2<TE>{});
  auto [d7, f11] = d6.add(
      make_combine_node<'q', 'e', 'i'>(gxu0.template as<'i', 'e', 'q'>(),
                                       gxu1.template as<'i', 'e', 'q'>(),
                                       gzu0.template as<'q', 'e', 'i'>(),
                                       gzu1.template as<'q', 'e', 'i'>(), si11),
      CombF2<TE>{});

  return sem_dag_tail<TE>(d7, f00, f01, f10, f11, Hw, w);
}

// --- MULTI-OUTPUT F STAGE --------------------------------------------------
//
// The same 4 gradients and the same tail, with the four single-output F nodes
// merged into TWO 2-output ones -- one per reference direction, each emitting
// both force components from one chain rule and one compute_stress.
//
// 12 nodes owning 14 slots, against 14 nodes owning 14 slots. The store is
// unchanged (two outputs are two buffers however they are spelled); what falls
// is the WORK: 4 stress evaluations -> 2, and 4x auxiliary global traffic ->
// 2x. Sharing could never have removed it, because the four F nodes were not a
// duplicated subtree -- they were four different functions of one.
//
// Two 2-output nodes and not one 4-output node: the xi slots are {q,e,j} and
// the gamma slots {q,e,i}, and all outputs of a combine necessarily share its
// modes. Merging across directions would force a reorder on the B slot of the
// divergence contraction, which the header's LOAD-BEARING RULE exists to avoid.
template <int TE>
auto sem_dag_graph_mo(V3 u0, V3 u1, V3 xix, V3 xiz, V3 gx, V3 gz, V3 l2m, V3 mu,
                      V3 jac, V2 H, V2 Hw, V1 w) {
  const StressIntegrand2<0> si0{xix, xiz, gx, gz, l2m, mu, jac};
  const StressIntegrand2<1> si1{xix, xiz, gx, gz, l2m, mu, jac};

  auto [d3, gxu0, gxu1, gzu0, gzu1] = sem_dag_gradients<TE>(u0, u1, H);

  // add() hands back one handle PER OUTPUT: [component 0, component 1].
  auto [d4, f00, f10] = d3.add(
      make_combine_node<'q', 'e', 'j'>(gxu0.template as<'q', 'e', 'j'>(),
                                       gxu1.template as<'q', 'e', 'j'>(),
                                       gzu0.template as<'j', 'e', 'q'>(),
                                       gzu1.template as<'j', 'e', 'q'>(), si0),
      CombF2<TE>{});
  auto [d5, f01, f11] = d4.add(
      make_combine_node<'q', 'e', 'i'>(gxu0.template as<'i', 'e', 'q'>(),
                                       gxu1.template as<'i', 'e', 'q'>(),
                                       gzu0.template as<'q', 'e', 'i'>(),
                                       gzu1.template as<'q', 'e', 'i'>(), si1),
      CombF2<TE>{});

  return sem_dag_tail<TE>(d5, f00, f01, f10, f11, Hw, w);
}

// 128 on GPU, and it is TIED TO TE -- re-sweep both together or neither.
//
// The optimum holds roughly 8 tile points per thread, so it scales with the
// tile. Measured best-of-7 with warmup on an uncontended H100 NVL, E=2.5M:
//
//   TE            4     8    16    32
//   tile points 256   512  1024  2048
//   best team    64    96   128   256      (~8 points/thread throughout)
//   DAG-MO   10.009 9.248 8.981 10.023 ms
//
// This value has now been 128, then 64, then 128 again, each time because
// something else moved: the multi-output F stage, then the slot pool and the
// operand arena, which cut the DAG's scratch enough that a larger tile became
// affordable and dragged the team size up with it. That history is the argument
// for the sweep in main() being unconditional rather than something to run once
// -- a hardcoded team size is invisible when it goes stale.
//
// NOT Kokkos::AUTO, which is 2-4x too large at every TE and gets worse as TE
// grows. Measured against the real functor (team_size_recommended), it picks
// 128/256/512/1024 for TE=4/8/16/32 -- consistently tile_points/2, about 2
// points per thread. At the shipping TE=16 that costs ~1.25-1.34x. AUTO's
// answer also MOVES when scratch changes, so it is not even a stable property
// of the kernel: it gave the DAG 256 at TE=4 before the slot pool and 128
// after.
//
// Serial caps team size at 1 and THROWS on anything larger, so the choice has
// to be per backend -- a negative value means Kokkos::AUTO, which is right
// there.
inline constexpr int kDagTeam = cfg::kIsGPU ? 128 : -1;

// `team` is a parameter and not just kDagTeam so main() can SWEEP it: a value
// that was optimal for one graph is not automatically optimal for the next, and
// this one is worth 1.4x when it is wrong.
template <int TE>
void library_dag_force(V3 u0, V3 u1, V3 xix, V3 xiz, V3 gx, V3 gz, V3 l2m,
                       V3 mu, V3 jac, V2 H, V2 Hw, V1 w, V3 force0, V3 force1,
                       int team = kDagTeam) {
  auto [g, r0, r1] = sem_dag_graph<cfg::TE_dag>(u0, u1, xix, xiz, gx, gz, l2m,
                                                mu, jac, H, Hw, w);
  g.outputs(r0, r1).team_size(team).execute(TeamPolicyTag<>{}, force0, force1);
}

template <int TE>
void library_dag_mo_force(V3 u0, V3 u1, V3 xix, V3 xiz, V3 gx, V3 gz, V3 l2m,
                          V3 mu, V3 jac, V2 H, V2 Hw, V1 w, V3 force0,
                          V3 force1, int team = kDagTeam) {
  auto [g, r0, r1] = sem_dag_graph_mo<cfg::TE_dag>(u0, u1, xix, xiz, gx, gz,
                                                   l2m, mu, jac, H, Hw, w);
  g.outputs(r0, r1).team_size(team).execute(TeamPolicyTag<>{}, force0, force1);
}
inline constexpr int kDagSrcEnd = __LINE__;
// DIAG-B: one F node on its own -- 4 gradient contractions + StressIntegrand,
// written to global. This is exactly one quarter of what the full library run
// executes (2 kernels x 2 F nodes), so 4x this row is the whole gradient+stress
// half of the tree, measured rather than extrapolated.
template <int TE>
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
             make_combine_tile(TileQEJ<TE>{}, BGrad<TE>{}, BGrad<TE>{},
                               BGrad<TE>{}, BGrad<TE>{}),
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
template <int TE>
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
             make_combine_tile(TileQEJ<TE>{}, BGrad<TE>{}, BGrad<TE>{},
                               BGrad<TE>{}, BGrad<TE>{}),
             out);
}

// DIAG-D: the divergence contraction with a MATERIALIZED operand -- same GEMM
// shape the tree runs, but B is a plain View instead of a fused combine node.
// Compare against the tree's implied per-divergence cost to price the fused
// operand path itself.
template <int TE>
void library_one_divergence(V3dyn Fin, V2 Hw, V3 out) {
  auto g                        = make_graph();
  [[maybe_unused]] auto [g1, o] = g.ops(make_contraction_node<'e', 'j', 'i'>(
      make_input_node(make_handle<'q', 'i'>(Hw)),
      make_input_node(make_handle<'q', 'e', 'j'>(Fin))));
  g1.execute(TeamPolicyTag<>{}, Tile<TileH, TileQEJ<TE>, TileEJI<TE>>{}, out);
}

// DIAGNOSTIC (not part of the comparison): one gradient contraction on its own,
// same tile shape the fused tree uses internally. This separates two very
// different explanations for a slow fused path -- "the [8x8]x[8x32] GEMM is
// simply small" versus "the fused tree's serialized stages are the cost". If
// this row is also slow, the tile is the floor and fusion is not the culprit.
template <int TE>
void library_one_gradient(V3 u0, V2 H, V3dyn out) {
  auto g                        = make_graph();
  [[maybe_unused]] auto [g1, o] = g.ops(make_contraction_node<'q', 'e', 'j'>(
      make_input_node(make_handle<'q', 'p'>(H)),
      make_input_node(make_handle<'e', 'j', 'p'>(u0))));
  g1.execute(TeamPolicyTag<>{}, BGrad<TE>{}, out);
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
template <int TE>
void baseline_force(V3 u0, V3 u1, V3 xix, V3 xiz, V3 gx, V3 gz, V3 l2m, V3 mu,
                    V3 jac, V2 H, V2 Hw, Kokkos::Array<float, cfg::N> w,
                    V3 force0, V3 force1, int E) {
  using ExecSpace    = Kokkos::DefaultExecutionSpace;
  using ScratchSpace = ExecSpace::scratch_memory_space;
  using Sc3          = Kokkos::View<float***, ScratchSpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using member_t     = Kokkos::TeamPolicy<ExecSpace>::member_type;

  constexpr int N = cfg::N, NP = cfg::N * cfg::N;

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
template <int Comp, int TE, bool DedupGrad = false>
void control_recompute_force(V3 u0, V3 u1, V3 xix, V3 xiz, V3 gx, V3 gz, V3 l2m,
                             V3 mu, V3 jac, V2 H, V2 Hw,
                             Kokkos::Array<float, cfg::N> w, V3 force, int E) {
  using ExecSpace    = Kokkos::DefaultExecutionSpace;
  using ScratchSpace = ExecSpace::scratch_memory_space;
  using Sc3          = Kokkos::View<float***, ScratchSpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using member_t     = Kokkos::TeamPolicy<ExecSpace>::member_type;

  constexpr int N = cfg::N, NP = cfg::N * cfg::N;
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
template <int TE>
void control_dag_force(V3 u0, V3 u1, V3 xix, V3 xiz, V3 gx, V3 gz, V3 l2m,
                       V3 mu, V3 jac, V2 H, V2 Hw,
                       Kokkos::Array<float, cfg::N> w, V3 force0, V3 force1,
                       int E) {
  using ExecSpace    = Kokkos::DefaultExecutionSpace;
  using ScratchSpace = ExecSpace::scratch_memory_space;
  using Sc3          = Kokkos::View<float***, ScratchSpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using member_t     = Kokkos::TeamPolicy<ExecSpace>::member_type;

  constexpr int N = cfg::N, NP = cfg::N * cfg::N;
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
inline constexpr double kStressPer2F =
    26.0 * NP;  // chain 12 + stress 8 + 2 slots

inline constexpr double kUseful = kGrad + kStressShared + kDiv + kSum;
inline constexpr double kLibExec =
    4.0 * kGrad + 4.0 * kStressPerF + kDiv + kSum;
// What a fan-out-deduplicated graph would retire: the gradient stage once
// (shared by every consumer) but still four single-output F nodes.
inline constexpr double kDagExec = kGrad + 4.0 * kStressPerF + kDiv + kSum;
// The same with the F stage merged into two 2-output nodes: the chain rule and
// the stress tensor are evaluated twice instead of four times. This is the
// closest the library gets to kUseful -- the remaining excess is that the two
// reference directions cannot share a node (their modes differ).
inline constexpr double kDagMoExec = kGrad + 2.0 * kStressPer2F + kDiv + kSum;
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
        "SEM stiffness (K&T 1999 A2/A7), 2D P-SV | N=%d C=%d E=%d | %s\n"
        "TE per implementation: baseline/CONTROL %d, tree %d, DAG %d "
        "(each at its own measured best)\n",
        cfg::N, cfg::C, E, gpu ? "GPU" : "CPU (Serial)", cfg::TE_ctl,
        cfg::TE_tree, cfg::TE_dag);

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
    // The same weights as a device View, for the library implementations only.
    // The hand-written kernels keep the Array, so they are an unchanged control
    // for the Array-vs-View change inside WeightedSum.
    Kokkos::View<float*, Kokkos::LayoutRight> wv_("w", cfg::N);
    auto wvh = Kokkos::create_mirror_view(wv_);
    for (int a = 0; a < cfg::N; ++a) wvh(a) = w[a];
    Kokkos::deep_copy(wv_, wvh);
    const V1 wv = wv_;
    Kokkos::fence();

    // Scratch, measured rather than estimated: the fused tree's real cost.
    auto node = library_node<0, cfg::TE_tree>(u0, u1, xix, xiz, gx, gz, l2m, mu,
                                              jac, H, Hw, wv);
    using LibEval =
        Evaluator<TeamPolicyTag<>, decltype(node), CombFrc<cfg::TE_tree>>;
    const std::size_t sbytes =
        LibEval::scratch_size_per_team(library_tile<cfg::TE_tree>());
    // 227 KB, not 48: an H100 accepts opt-in shared memory to that and Kokkos
    // honours it. The old 48 KB constant was not describing the machine, and at
    // the shipping TE=16 it would bail this benchmark out before a single
    // kernel ran -- the TREE needs 88824 B there (the DAG needs 37208).
    // Serial's 32 KB is a real hardcoded cap and stays.
    const std::size_t smax = gpu ? 227u * 1024u : 32u * 1024u;
    std::printf("library scratch/team: %zu bytes (limit ~%zu)%s\n", sbytes,
                smax, sbytes > smax ? "  <-- OVER" : "");
    // The DAG covers BOTH components in one launch, so this is against the
    // tree's PER-COMPONENT figure -- it is doing twice the work for less.
    {
      auto [dg, dr0, dr1] = sem_dag_graph<cfg::TE_dag>(u0, u1, xix, xiz, gx, gz,
                                                       l2m, mu, jac, H, Hw, wv);
      const auto outs     = dg.outputs(dr0, dr1);
      std::printf(
          "DAG scratch/team:     %zu bytes (both components, one launch)%s\n",
          outs.scratch_bytes(),
          dg.index_consistent() ? "" : "  <-- INCONSISTENT");
    }
    // The multi-output F stage owns the same 14 slots from 12 nodes, so the
    // store is unchanged and only the operand side can move. Printed next to
    // the row above because "did merging the F nodes cost scratch?" is the
    // question that killed the previous attempt at this.
    {
      auto [dg, dr0, dr1] = sem_dag_graph_mo<cfg::TE_dag>(
          u0, u1, xix, xiz, gx, gz, l2m, mu, jac, H, Hw, wv);
      const auto outs = dg.outputs(dr0, dr1);
      // Slots are POOLED by live range, so the store is no longer one buffer
      // per node output. The unpooled figure is printed beside it because the
      // ratio is what liveness bought, and because "14 slots" is still the
      // right way to read the graph even when they occupy 8 buffers.
      std::printf(
          "DAG-MO scratch/team:  %zu bytes (slots %zu in %zu pools, was %zu, "
          "operands %zu)%s\n",
          outs.scratch_bytes(), outs.slot_bytes(),
          static_cast<std::size_t>(decltype(outs)::num_pools), dg.slot_bytes(),
          outs.operand_bytes(),
          dg.index_consistent() ? "" : "  <-- INCONSISTENT");
    }
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
      baseline_force<cfg::TE_ctl>(u0, u1, xix, xiz, gx, gz, l2m, mu, jac, H, Hw,
                                  w, fb0, fb1, E);
    };
    auto run_lib = [&] {
      library_force<0, cfg::TE_tree>(u0, u1, xix, xiz, gx, gz, l2m, mu, jac, H,
                                     Hw, wv, fl0);
      library_force<1, cfg::TE_tree>(u0, u1, xix, xiz, gx, gz, l2m, mu, jac, H,
                                     Hw, wv, fl1);
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
    // THE DAG: the same library, fan-out deduplicated. One launch, both
    // components, 4 gradient sums instead of 16.
    V3   fd0("fd0", E, cfg::N, cfg::N), fd1("fd1", E, cfg::N, cfg::N);
    auto run_dag = [&] {
      library_dag_force<cfg::TE_dag>(u0, u1, xix, xiz, gx, gz, l2m, mu, jac, H,
                                     Hw, wv, fd0, fd1);
    };
    run_dag();
    Kokkos::fence();
    const double dd =
        std::max(max_rel_diff(fd0, ref0, Echk), max_rel_diff(fd1, ref1, Echk));
    const double tdag = seconds_of(run_dag, warmup, reps);
    std::printf("%-30s %10.3f %14.1f %16.1f %8s\n", "library DAG (1 kernel)",
                tdag * 1e3, flopcount::kUseful * Ed / tdag / 1e9,
                flopcount::kDagExec * Ed / tdag / 1e9,
                dd < 1e-2 ? "PASS" : "FAIL");

    // THE DAG WITH A MULTI-OUTPUT F STAGE: the same graph, the four stress
    // integrands merged into two 2-output nodes. 2 stress evals instead of 4,
    // 2x auxiliary traffic instead of 4x. Same slots, same launch count.
    V3   fm0("fm0", E, cfg::N, cfg::N), fm1("fm1", E, cfg::N, cfg::N);
    auto run_mo = [&] {
      library_dag_mo_force<cfg::TE_dag>(u0, u1, xix, xiz, gx, gz, l2m, mu, jac,
                                        H, Hw, wv, fm0, fm1);
    };
    run_mo();
    Kokkos::fence();
    const double dm =
        std::max(max_rel_diff(fm0, ref0, Echk), max_rel_diff(fm1, ref1, Echk));
    const double tmo = seconds_of(run_mo, warmup, reps);
    std::printf("%-30s %10.3f %14.1f %16.1f %8s\n", "library DAG-MO (2-out F)",
                tmo * 1e3, flopcount::kUseful * Ed / tmo / 1e9,
                flopcount::kDagMoExec * Ed / tmo / 1e9,
                dm < 1e-2 ? "PASS" : "FAIL");

    // TEAM SIZE IS NOT A CONSTANT OF THE PROBLEM. It is worth 1.42x on this
    // kernel and the optimum moves with the graph, so it is swept here rather
    // than inherited: a value tuned for one node list is evidence about that
    // node list only. Cheap enough (2 reps, no warmup) to leave in every run,
    // which is the point -- a stale hardcoded team size is invisible otherwise.
    if (cfg::kIsGPU) {
      std::printf("\nteam-size sweep (%d warmup, %d reps):  DAG      DAG-MO\n",
                  warmup, reps);
      for (int ts : {32, 64, 96, 128, 192, 256, 384, 512, 768, 1024}) {
        const double a = seconds_of(
            [&] {
              library_dag_force<cfg::TE_dag>(u0, u1, xix, xiz, gx, gz, l2m, mu,
                                             jac, H, Hw, wv, fd0, fd1, ts);
            },
            warmup, reps);
        const double b = seconds_of(
            [&] {
              library_dag_mo_force<cfg::TE_dag>(u0, u1, xix, xiz, gx, gz, l2m,
                                                mu, jac, H, Hw, wv, fm0, fm1,
                                                ts);
            },
            warmup, reps);
        std::printf("  %4d threads            %10.3f  %10.3f ms%s\n", ts,
                    a * 1e3, b * 1e3, ts == kDagTeam ? "   <-- in use" : "");
      }
      std::printf("\n");
    }

    // THE CONTROL: hand-written, but with the library's exact redundancy.
    V3   fc0("fc0", E, cfg::N, cfg::N), fc1("fc1", E, cfg::N, cfg::N);
    auto run_ctl = [&] {
      control_recompute_force<0, cfg::TE_ctl>(u0, u1, xix, xiz, gx, gz, l2m, mu,
                                              jac, H, Hw, w, fc0, E);
      control_recompute_force<1, cfg::TE_ctl>(u0, u1, xix, xiz, gx, gz, l2m, mu,
                                              jac, H, Hw, w, fc1, E);
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
      control_recompute_force<0, cfg::TE_ctl, true>(
          u0, u1, xix, xiz, gx, gz, l2m, mu, jac, H, Hw, w, fc0, E);
      control_recompute_force<1, cfg::TE_ctl, true>(
          u0, u1, xix, xiz, gx, gz, l2m, mu, jac, H, Hw, w, fc1, E);
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
      control_dag_force<cfg::TE_ctl>(u0, u1, xix, xiz, gx, gz, l2m, mu, jac, H,
                                     Hw, w, fc0, fc1, E);
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
    V3dyn go("go", cfg::N, E, cfg::N);
    auto  run_one    = [&] { library_one_gradient<cfg::TE_tree>(u0, H, go); };
    const double t1g = seconds_of(run_one, warmup, reps);
    const double f1g = flopcount::NP * cfg::N * 2.0 * Ed;  // one gradient
    std::printf("%-30s %10.3f %14.1f %16s %8s\n", "  [diag] 1 gradient only",
                t1g * 1e3, f1g / t1g / 1e9, "-", "-");

    auto run_4g = [&] { library_four_gradients<cfg::TE_tree>(u0, u1, H, go); };
    const double t4g = seconds_of(run_4g, warmup, reps);
    std::printf("%-30s %10.3f %14.1f %16s %8s\n",
                "  [diag] 4 gradients, no aux", t4g * 1e3,
                4.0 * f1g / t4g / 1e9, "-", "-");

    auto run_1F = [&] {
      library_one_F<cfg::TE_tree>(u0, u1, xix, xiz, gx, gz, l2m, mu, jac, H,
                                  go);
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
      library_force<0, cfg::TE_tree>(u0, u1, xix, xiz, gx, gz, l2m, mu, jac, H,
                                     Hw, wv, fl0);
    };
    const double tl1 = seconds_of(run_lib1, warmup, reps);
    std::printf("%-30s %10.3f %14s %16s %8s\n", "  [diag] library, 1 component",
                tl1 * 1e3, "-", "-", "-");

    auto run_1d = [&] { library_one_divergence<cfg::TE_tree>(go, Hw, fl0); };
    const double t1d = seconds_of(run_1d, warmup, reps);
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
    std::printf(
        "source lines: library tree %d, library DAG %d (BOTH F spellings, "
        "which share the gradient and tail stages), hand-written %d\n",
        kLibSrcEnd - kLibSrcBegin - 1, kDagSrcEnd - kDagSrcBegin - 1,
        kBaseSrcEnd - kBaseSrcBegin - 1);
    std::printf(
        "\nDAG RESULT (measured, not projected):\n"
        "  library tree -> DAG:            %.2fx faster\n"
        "  gap to hand-written baseline:   %.2fx  (was %.2fx)\n"
        "  CONTROL-C is the floor for this work profile: %.3f ms vs DAG %.3f "
        "ms = %.2fx of library overhead left\n",
        tl / tdag, tdag / tb, tl / tb, tc3 * 1e3, tdag * 1e3, tdag / tc3);
    // MULTI-OUTPUT F STAGE. CONTROL-C carries FOUR stress evals, so it is no
    // longer the floor for this row -- the DAG-MO does strictly less work than
    // any hand-written control here, and the baseline is the only bound left.
    std::printf(
        "\nMULTI-OUTPUT F STAGE (4 stress evals -> 2, 4x aux -> 2x):\n"
        "  DAG -> DAG-MO:                  %.2fx faster\n"
        "  gap to hand-written baseline:   %.2fx  (was %.2fx)\n"
        "  vs CONTROL-C (which still does 4 stress evals): %.3f ms vs %.3f ms "
        "= %.2fx\n",
        tdag / tmo, tmo / tb, tdag / tb, tc3 * 1e3, tmo * 1e3, tmo / tc3);
    std::printf(
        "max rel diff vs host reference: baseline %.2e, library %.2e, "
        "control-C %.2e, DAG %.2e, DAG-MO %.2e\n",
        db, dl, dc3, dd, dm);
    if (db >= 1e-2 || dl >= 1e-2 || dd >= 1e-2 || dm >= 1e-2) rc = 1;
  }
  Kokkos::finalize();
  return rc;
}
