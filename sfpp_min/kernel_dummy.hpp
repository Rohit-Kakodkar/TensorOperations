#pragma once

#include <access_model.hpp>
#include <config.hpp>
#include <data.hpp>
#include <layout.hpp>

#include <Kokkos_Core.hpp>

#include <cstddef>

// ============================================================================
// A DELIBERATELY UNOPTIMISED TRANSCRIPTION. DO NOT "FIX" IT.
//
// This is a literal stand-in for SPECFEM++'s
//   gather_kernel<5, tags::Tags<1,6,0,0,0,0,0>>
// (dim3, elastic, forward, isotropic, attenuation::none, boundary::none),
// traced from SPECFEMPP
// core/specfem/compute/impl/stiffness_kernels.hpp:101-251.
//
// Its only purpose is to be a BASELINE that behaves like the real kernel, so
// that a new kernel measured against it is measured against something real.
// Every "obvious" cleanup below makes it FASTER and therefore WRONG:
//
//   * the metric tensor is loaded THREE times per point, and not identically:
//     9 components inside the gradient (gradient.hpp:359-362, store_jacobian
//     == false) and 10 components at each of the two kernel-level loads
//     (stiffness_kernels.hpp:136-138 and :196-198).
//   * the gradient callback re-loads displacement from GLOBAL memory
//     (:150-152) even though it is already staged in scratch.
//   * the divergence callback loads properties, velocity, weights and the
//     metric tensor (:181-198) to feed compute_damping_force, which is a
//     no-op under attenuation::none.
//   * scratch is LayoutLeft, contradicting this repo's standing
//     LayoutRight-on-scratch rule. LayoutLeft is what SPECFEM++ uses
//     (chunk_ndim_view.hpp:92) and it is what makes ielement = i % 4 coalesce.
//   * lambda is computed as (kappa + 4/3 mu) - 2 mu, not kappa - 2/3 mu.
//     Same value, two extra flops, different rounding.
//
// CALIBRATION STATUS (accepted 2026-08-21): 13 of 14 ground-truth counters are
// reproduced -- FFMA 136.0000, registers 48, shmem 26,188 B, 5 blocks/SM, block
// 256, and all five memory-traffic counters inside 1%. The one open residual is
// warp-inst/GLL point, 21.316 against 22.388 (-4.8%).
//
// That residual is ATTRIBUTED, not unexplained. Per-SASS-instruction profiling
// of both binaries shows every memory and floating-point opcode identical to
// four decimals DYNAMICALLY (LDG 0.8020=0.8020, LDS 3.8400=3.8400, FFMA
// 4.3520=4.3520, FMUL/FADD/STS/RED/BAR all zero delta). The entire gap is
// upstream's iterator bookkeeping: uniform-datapath address work (39%, upstream
// passes a 15,832 B `assembly` by value against our 1,416 B parameter block),
// branch/reconvergence (27%), and vector integer (34%).
//
// Accepted rather than closed, for two reasons: the deficit is COMMON-MODE (the
// kernel this baseline exists to measure will also lack SPECFEM++'s iterator
// machinery, so it cancels in that comparison), and it points the SAFE way --
// 4.8% fewer instructions but 1.017x SLOWER than the reference, so this
// baseline is marginally weak, not unrepresentatively strong.
//
// RE-OPEN THIS if any of the following becomes true:
//   - a new kernel changes the iteration/indexing machinery, so the deficit
//     stops being common-mode;
//   - the kernel becomes issue-bound rather than latency-bound, putting those
//     uniform/branch instructions on the critical path;
//   - upstream changes stiffness_kernels.hpp, domain_view.hpp or
//     chunked_domain_iterator.hpp -- then recalibrate from scratch.
//
// The no-op consumers below are NOT placeholders. They mirror the empty
// functions the real tag combination resolves to, so that nvcc is handed the
// same dead-code-elimination decision it is handed upstream. Forcing the loads
// to survive would overshoot the target just as deleting them undershoots it.
// ============================================================================

namespace sfpp_min {

// ONE mapping, ONE flat index, reused across all ten arrays -- exactly as
// assembly/jacobian_matrix/dim3/impl_load.hpp:108-123 does. Giving each array
// its own offset policy instead recomputes the address math ten times per load
// and keeps ten copies of the runtime extents live; that measured +7.8% warp
// instructions and 121 registers against a target of 48.
template <typename Offset>
struct MetricsAccessor {
  using view_type = Kokkos::View<real_t*>;
  view_type xix, xiy, xiz;
  view_type etax, etay, etaz;
  view_type gammax, gammay, gammaz;
  view_type jacobian;
  Offset    off;
};

template <typename Offset>
struct PropertiesAccessor {
  using view_type = Kokkos::View<real_t*>;
  view_type kappa, mu, rho;
  Offset    off;
};

template <typename Offset>
MetricsAccessor<Offset> make_accessor(const Metrics<Offset>& m) {
  return {
      m.xix.device_view(),      m.xiy.device_view(),    m.xiz.device_view(),
      m.etax.device_view(),     m.etay.device_view(),   m.etaz.device_view(),
      m.gammax.device_view(),   m.gammay.device_view(), m.gammaz.device_view(),
      m.jacobian.device_view(), Offset(m.xix.nspec())};
}

template <typename Offset>
PropertiesAccessor<Offset> make_accessor(const Properties<Offset>& p) {
  return {p.kappa.device_view(), p.mu.device_view(), p.rho.device_view(),
          Offset(p.kappa.nspec())};
}

struct PointMetric {
  real_t xix, xiy, xiz;
  real_t etax, etay, etaz;
  real_t gammax, gammay, gammaz;
  real_t jacobian;
};

struct PointProperty {
  real_t kappa, mu, rho;

  KOKKOS_INLINE_FUNCTION real_t lambdaplus2mu() const {
    return kappa + static_cast<real_t>(4.0 / 3.0) * mu;
  }
  KOKKOS_INLINE_FUNCTION real_t lambda() const {
    return lambdaplus2mu() - static_cast<real_t>(2.0) * mu;
  }
};

// store_jacobian == false: the gradient's own load reads 9 components, not 10.
template <typename Offset>
KOKKOS_INLINE_FUNCTION void load_metric(const MetricsAccessor<Offset>& m,
                                        int ispec, int iz, int iy, int ix,
                                        PointMetric& out, bool load_jacobian) {
  const std::size_t _index = m.off(ispec, iz, iy, ix);
  out.xix                  = m.xix.data()[_index];
  out.xiy                  = m.xiy.data()[_index];
  out.xiz                  = m.xiz.data()[_index];
  out.etax                 = m.etax.data()[_index];
  out.etay                 = m.etay.data()[_index];
  out.etaz                 = m.etaz.data()[_index];
  out.gammax               = m.gammax.data()[_index];
  out.gammay               = m.gammay.data()[_index];
  out.gammaz               = m.gammaz.data()[_index];
  if (load_jacobian) out.jacobian = m.jacobian.data()[_index];
}

template <typename Offset>
KOKKOS_INLINE_FUNCTION void load_property(const PropertiesAccessor<Offset>& p,
                                          int ispec, int iz, int iy, int ix,
                                          PointProperty& out) {
  const std::size_t _index = p.off(ispec, iz, iy, ix);
  out.kappa                = p.kappa.data()[_index];
  out.mu                   = p.mu.data()[_index];
  out.rho                  = p.rho.data()[_index];
}

// medium_physics/compute_attenuation.hpp:14-23 -- empty for attenuation::none.
KOKKOS_INLINE_FUNCTION void compute_attenuation(real_t (&)[3][3]) {}

// medium_physics/compute_cosserat_stress.hpp:73-81 -- has_cosserat_stress is
// false for dim3 elastic, so point_displacement is loaded and never read.
KOKKOS_INLINE_FUNCTION void compute_cosserat_stress(const PointProperty&,
                                                    const real_t (&)[3],
                                                    real_t (&)[3][3]) {}

// medium_physics/compute_damping_force.hpp:41-42 -- has_damping_force is false.
KOKKOS_INLINE_FUNCTION void compute_damping_force(real_t, const PointProperty&,
                                                  const real_t (&)[3],
                                                  real_t (&)[3]) {}

// boundary_conditions/none/none.hpp:29-41 -- do nothing.
KOKKOS_INLINE_FUNCTION void apply_boundary_conditions(const PointProperty&,
                                                      const real_t (&)[3],
                                                      real_t (&)[3]) {}

using ScratchSpace = Kokkos::DefaultExecutionSpace::scratch_memory_space;
using Unmanaged    = Kokkos::MemoryTraits<Kokkos::Unmanaged>;

// LayoutLeft: element index stride-1, so the four lanes sharing a point are
// contiguous. chunk_ndim_view.hpp:92. NOT this repo's usual LayoutRight.
using DisplacementPack =
    Kokkos::View<real_t[kExecChunk][NGLL][NGLL][NGLL][3], Kokkos::LayoutLeft,
                 ScratchSpace, Unmanaged>;
using StressIntegrand =
    Kokkos::View<real_t[kExecChunk][NGLL][NGLL][NGLL][3][3], Kokkos::LayoutLeft,
                 ScratchSpace, Unmanaged>;
// quadrature/lagrange_derivative.hpp:37-38 -- this one really is LayoutRight.
using LagrangeDerivative = Kokkos::View<real_t[NGLL][NGLL], Kokkos::LayoutRight,
                                        ScratchSpace, Unmanaged>;

inline std::size_t dummy_shmem_size() {
  return DisplacementPack::shmem_size() + StressIntegrand::shmem_size() +
         LagrangeDerivative::shmem_size();
}

using GlobalHPrime  = Kokkos::View<real_t**, Kokkos::LayoutRight>;
using GlobalWeights = Kokkos::View<real_t*>;

template <typename Offset>
struct DummyKernelArgs {
  MetricsAccessor<Offset>    metrics;
  PropertiesAccessor<Offset> properties;
  IglobMap::view_type        iglob;
  Fields::view_type          displacement;
  Fields::view_type          velocity;
  Fields::view_type          acceleration;
  GlobalHPrime               hprime;
  GlobalWeights              weights;
  int                        nspec;
};

// Registers are the binding occupancy constraint on the real kernel: it is
// capped at 5 blocks/SM by its 48 registers, 6 by shared memory. Without launch
// bounds ptxas picks its own occupancy target and settles this kernel at 32
// registers / 8 blocks per SM -- a DIFFERENT machine, however similar the
// instruction counts.
//
// PROVEN 2026-08-21: the register count here is a ptxas scheduling budget, not
// a measure of required live state. The identical source compiles to 32 with no
// bounds, 48 with <256,5>, 64 with <256,4>, 77 with <256,3>. 65536/(5*256) =
// 51.2 -> 48, exactly the real kernel's count. The extra registers are spent on
// FFMA/LDS/FMUL -- deeper software pipelining of shared loads -- not on more
// global traffic (only 7 of them touch LDG).
//
// Upstream reaches 48 WITHOUT launch bounds; we reach it by pinning the budget.
// Same destination, different route, so occupancy and stalls must be
// re-measured rather than assumed. See plans/minimal-sfpp-library-sprints.md.
using DummyLaunchBounds = Kokkos::LaunchBounds<kTeamSize, 5>;

template <typename Offset>
int dummy_stiffness(const DummyKernelArgs<Offset>& args, int team_size = -1) {
  using policy_t    = Kokkos::TeamPolicy<DummyLaunchBounds>;
  using member_type = typename policy_t::member_type;

  const int nspec  = args.nspec;
  const int league = num_teams(nspec);

  const auto metrics      = args.metrics;
  const auto properties   = args.properties;
  const auto iglob        = args.iglob;
  const auto displacement = args.displacement;
  const auto velocity     = args.velocity;
  const auto acceleration = args.acceleration;
  const auto hprime_g     = args.hprime;
  const auto weights_g    = args.weights;

  const std::size_t bytes = dummy_shmem_size();

  // chunked_domain_iterator.hpp:668-674 passes Kokkos::AUTO. The config's
  // num_threads = 512 is dead code upstream; 256 is what AUTO returns given
  // this kernel's register and scratch footprint. Letting AUTO choose here
  // makes "team size == 256" a derived confirmation of that footprint.
  auto policy = (team_size > 0) ? policy_t(league, team_size)
                                : policy_t(league, Kokkos::AUTO);
  policy.set_scratch_size(0, Kokkos::PerTeam(static_cast<int>(bytes)));

  const auto closure = KOKKOS_LAMBDA(const member_type& team) {
    const int base = team.league_rank() * kExecChunk;
    const int num_elements =
        (nspec - base) < kExecChunk ? (nspec - base) : kExecChunk;

    DisplacementPack   up(team.team_scratch(0));
    LagrangeDerivative hp(team.team_scratch(0));
    StressIntegrand    F(team.team_scratch(0));

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NGLL * NGLL),
                         [&](const int n) {
                           const int a = n / NGLL;
                           const int b = n % NGLL;
                           hp(a, b)    = hprime_g(a, b);
                         });

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, kWorkItemsPerTeam), [&](const int i) {
          const WorkItem w = decompose(i);
          if (w.ielement >= num_elements) return;
          const int ispec = base + w.ielement;
          const int ig    = iglob(ispec, w.iz, w.iy, w.ix);
          for (int c = 0; c < 3; ++c)
            up(w.ielement, w.iz, w.iy, w.ix, c) = displacement(ig, c);
        });

    team.team_barrier();

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, kWorkItemsPerTeam), [&](const int i) {
          const WorkItem w = decompose(i);
          if (w.ielement >= num_elements) return;
          const int ispec = base + w.ielement;

          // --- gradient.hpp:359-362: metric load #1, 9 components -------
          PointMetric grad_metric;
          load_metric(metrics, ispec, w.iz, w.iy, w.ix, grad_metric, false);

          real_t df_dxi[3]    = {0, 0, 0};
          real_t df_deta[3]   = {0, 0, 0};
          real_t df_dgamma[3] = {0, 0, 0};

          for (int l = 0; l < NGLL; ++l) {
            for (int c = 0; c < 3; ++c) {
              df_dxi[c] += hp(w.ix, l) * up(w.ielement, w.iz, w.iy, l, c);
              df_deta[c] += hp(w.iy, l) * up(w.ielement, w.iz, l, w.ix, c);
              df_dgamma[c] += hp(w.iz, l) * up(w.ielement, l, w.iy, w.ix, c);
            }
          }

          real_t du[3][3];
          for (int c = 0; c < 3; ++c) {
            du[c][0] = grad_metric.xix * df_dxi[c] +
                       grad_metric.etax * df_deta[c] +
                       grad_metric.gammax * df_dgamma[c];
            du[c][1] = grad_metric.xiy * df_dxi[c] +
                       grad_metric.etay * df_deta[c] +
                       grad_metric.gammay * df_dgamma[c];
            du[c][2] = grad_metric.xiz * df_dxi[c] +
                       grad_metric.etaz * df_deta[c] +
                       grad_metric.gammaz * df_dgamma[c];
          }

          // --- stiffness_kernels.hpp:136-138: metric load #2, 10 comps --
          PointMetric point_metric;
          load_metric(metrics, ispec, w.iz, w.iy, w.ix, point_metric, true);

          // --- :140-142: properties (live, feeds compute_stress) --------
          PointProperty point_property;
          load_property(properties, ispec, w.iz, w.iy, w.ix, point_property);

          // --- :150-152: displacement re-read from GLOBAL, already in
          //     scratch, consumed only by the cosserat no-op ------------
          const int ig_grad = iglob(ispec, w.iz, w.iy, w.ix);
          real_t    point_displacement[3];
          for (int c = 0; c < 3; ++c)
            point_displacement[c] = displacement(ig_grad, c);

          const real_t l2m = point_property.lambdaplus2mu();
          const real_t lam = point_property.lambda();
          const real_t mu  = point_property.mu;

          real_t T[3][3];
          T[0][0] = l2m * du[0][0] + lam * (du[1][1] + du[2][2]);
          T[1][1] = l2m * du[1][1] + lam * (du[0][0] + du[2][2]);
          T[2][2] = l2m * du[2][2] + lam * (du[0][0] + du[1][1]);
          T[0][1] = mu * (du[0][1] + du[1][0]);
          T[1][0] = T[0][1];
          T[0][2] = mu * (du[0][2] + du[2][0]);
          T[2][0] = T[0][2];
          T[1][2] = mu * (du[1][2] + du[2][1]);
          T[2][1] = T[1][2];

          compute_attenuation(T);
          compute_cosserat_stress(point_property, point_displacement, T);

          // --- :164-165 / point/stress.hpp:184-197: F = T * J ----------
          for (int c = 0; c < 3; ++c) {
            F(w.ielement, w.iz, w.iy, w.ix, c, 0) =
                point_metric.jacobian *
                (T[c][0] * point_metric.xix + T[c][1] * point_metric.xiy +
                 T[c][2] * point_metric.xiz);
            F(w.ielement, w.iz, w.iy, w.ix, c, 1) =
                point_metric.jacobian *
                (T[c][0] * point_metric.etax + T[c][1] * point_metric.etay +
                 T[c][2] * point_metric.etaz);
            F(w.ielement, w.iz, w.iy, w.ix, c, 2) =
                point_metric.jacobian *
                (T[c][0] * point_metric.gammax + T[c][1] * point_metric.gammay +
                 T[c][2] * point_metric.gammaz);
          }
        });

    team.team_barrier();

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, kWorkItemsPerTeam), [&](const int i) {
          const WorkItem w = decompose(i);
          if (w.ielement >= num_elements) return;
          const int ispec = base + w.ielement;

          // --- divergence.hpp:115-128: THREE separate component loops,
          //     weights(l) re-read from global inside each -------------
          real_t temp1l[3] = {0, 0, 0};
          real_t temp2l[3] = {0, 0, 0};
          real_t temp3l[3] = {0, 0, 0};

          for (int l = 0; l < NGLL; ++l) {
            for (int c = 0; c < 3; ++c)
              temp1l[c] += F(w.ielement, w.iz, w.iy, l, c, 0) * hp(l, w.ix) *
                           weights_g(l);
            for (int c = 0; c < 3; ++c)
              temp2l[c] += F(w.ielement, w.iz, l, w.ix, c, 1) * hp(l, w.iy) *
                           weights_g(l);
            for (int c = 0; c < 3; ++c)
              temp3l[c] += F(w.ielement, l, w.iy, w.ix, c, 2) * hp(l, w.iz) *
                           weights_g(l);
          }

          real_t result[3];
          for (int c = 0; c < 3; ++c)
            result[c] = weights_g(w.iz) * weights_g(w.iy) * temp1l[c] +
                        weights_g(w.iz) * weights_g(w.ix) * temp2l[c] +
                        weights_g(w.iy) * weights_g(w.ix) * temp3l[c];

          // --- :178-179: negate ---------------------------------------
          real_t accel[3];
          for (int c = 0; c < 3; ++c)
            accel[c] = result[c] * static_cast<real_t>(-1.0);

          // --- :181-198: five loads feeding only no-ops ----------------
          PointProperty div_property;
          load_property(properties, ispec, w.iz, w.iy, w.ix, div_property);

          const int ig = iglob(ispec, w.iz, w.iy, w.ix);
          real_t    point_velocity[3];
          for (int c = 0; c < 3; ++c) point_velocity[c] = velocity(ig, c);

          // boundary is NOT loaded: boundaries.hpp:113-114 returns at
          // compile time under boundary_tag::none.

          const real_t wz = weights_g(w.iz);
          const real_t wy = weights_g(w.iy);
          const real_t wx = weights_g(w.ix);

          PointMetric div_metric;
          load_metric(metrics, ispec, w.iz, w.iy, w.ix, div_metric, true);

          const real_t factor = wx * wy * wz * div_metric.jacobian;

          compute_damping_force(factor, div_property, point_velocity, accel);
          apply_boundary_conditions(div_property, point_velocity, accel);

          // --- :247-248: scatter --------------------------------------
          for (int c = 0; c < 3; ++c)
            Kokkos::atomic_add(&acceleration(ig, c), accel[c]);
        });
  };

  const int resolved = (team_size > 0) ? team_size
                                       : policy.team_size_recommended(
                                             closure, Kokkos::ParallelForTag());

  Kokkos::parallel_for("sfpp_min::dummy_stiffness", policy, closure);
  return resolved;
}

}  // namespace sfpp_min
