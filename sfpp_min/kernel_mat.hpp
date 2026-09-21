#pragma once

// ============================================================================
// THE MATERIALIZATION AXIS. kernel_dummy.hpp with ONE knob: how many
// intermediate quantities are forced through team scratch.
//
// Sprint 5 measured the LevelGraph kernel 2.772x slower than the dummy and
// closed every load, register, occupancy and FFMA explanation. What survived is
// a STRUCTURAL claim: the graph materializes FOUR quantities where the geometry
// demands TWO. The dummy stages the displacement and the stress integrand, and
// carries the two Lagrange contractions -- the gradient and the divergence --
// in REGISTERS, because each is read back at exactly the coordinate it was
// written at. The graph gives each its own level, so each round-trips through
// scratch.
//
// That claim was a prediction about a kernel nobody had built. This file builds
// it. NMat is the number of materialized quantities:
//
//   NMat = 2  u, F                  the incumbent, verbatim -- THE CONTROL
//   NMat = 3  u, du, F              the gradient contraction gets its own pass
//   NMat = 4  u, du, F, T           the divergence contraction does too
//
// NMat = 2 must reproduce dummy_stiffness bit for bit: the loop bodies below
// are the same statements in the same order, and the if constexpr branches that
// split them are not taken. bench_mat.cpp fails the run if its time or its
// scratch request misses the calibrated incumbent, so 3 and 4 are only quotable
// from a cell that already reproduced the baseline.
//
// The gradient and the divergence buffers are ONE view: du dies at the end of
// the stress pass, T is written by the divergence pass after it. Aliasing them
// is what holds shared memory -- and therefore occupancy -- FIXED across
// NMat = 3 and NMat = 4, so the 3 -> 4 step prices a barrier and a scratch
// round trip with the occupancy confound removed. The 2 -> 3 step cannot hold
// it fixed; that step really does cost 9 more buffers, and it is reported with
// its byte count attached.
//
// What this file is NOT: a proposed kernel. Nobody would write NMat = 4 on
// purpose. It exists to price a structure the library currently forces.
// ============================================================================

#include <config.hpp>
#include <data.hpp>
#include <kernel_dummy.hpp>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <type_traits>

namespace sfpp_min {

// du(elem,iz,iy,ix,c,d): d = 0 xi, 1 eta, 2 gamma. Same shape and same layout
// as StressIntegrand, because it is the same nine-per-point quantity -- and
// reused as T, the divergence contraction's output, once du is dead.
template <typename ScratchLayout = Kokkos::LayoutLeft>
using ContractionPack = Kokkos::View<real_t[kExecChunk][NGLL][NGLL][NGLL][3][3],
                                     ScratchLayout, ScratchSpace, Unmanaged>;

template <int NMat, typename ScratchLayout = Kokkos::LayoutLeft>
inline std::size_t mat_shmem_size() {
  std::size_t bytes = dummy_shmem_size<ScratchLayout>();
  if constexpr (NMat >= 3)
    bytes += ContractionPack<ScratchLayout>::shmem_size();
  return bytes;
}

// MinBlocks = 0 means NO launch bounds, which is what the LevelGraph kernel
// runs with. It is not a free knob: the incumbent's <256,5> asks for five
// 24,124 B teams per SM, and nine more buffers put that target out of reach of
// an A100's 164 KB, so a bounded NMat >= 3 does not launch at all. The axis is
// therefore swept UNBOUNDED, with the bounded NMat = 2 cell kept alongside as
// the tie back to the calibrated incumbent.
template <int MinBlocks>
using MatLaunchBounds = std::conditional_t<
    MinBlocks == 0, Kokkos::LaunchBounds<>,
    Kokkos::LaunchBounds<kTeamSize, (MinBlocks > 0 ? MinBlocks : 1)>>;

template <int NMat, int MinBlocks = 0,
          typename ScratchLayout = Kokkos::LayoutLeft, typename MetricsAcc,
          typename PropertiesAcc, typename IglobView>
int mat_stiffness(
    const DummyKernelArgs<MetricsAcc, PropertiesAcc, IglobView>& args,
    int team_size = -1) {
  static_assert(NMat >= 2 && NMat <= 4,
                "NMat counts materialized quantities: 2 = the incumbent "
                "(u, F), 3 adds the gradient, 4 adds the divergence");

  using policy_t    = Kokkos::TeamPolicy<MatLaunchBounds<MinBlocks>>;
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

  const std::size_t bytes = mat_shmem_size<NMat, ScratchLayout>();

  auto policy = (team_size > 0) ? policy_t(league, team_size)
                                : policy_t(league, Kokkos::AUTO);
  policy.set_scratch_size(0, Kokkos::PerTeam(static_cast<int>(bytes)));

  const auto closure = KOKKOS_LAMBDA(const member_type& team) {
    const int base = team.league_rank() * kExecChunk;
    const int num_elements =
        (nspec - base) < kExecChunk ? (nspec - base) : kExecChunk;

    DisplacementPack<ScratchLayout> up(team.team_scratch(0));
    LagrangeDerivative              hp(team.team_scratch(0));
    StressIntegrand<ScratchLayout>  F(team.team_scratch(0));
    // Carved only when it is used: at NMat = 2 nothing bumps the scratch
    // cursor past F, so up/hp/F land at the incumbent's exact offsets.
    using CPack = ContractionPack<ScratchLayout>;
    CPack C     = [&]() -> CPack {
      if constexpr (NMat >= 3)
        return CPack(team.team_scratch(0));
      else
        return CPack();
    }();

    // nvcc forbids an extended lambda first-capturing a variable inside an
    // `if constexpr` body, and the NMat >= 4 pass is where weights_g would
    // first appear. Capturing it here, unconditionally, is the whole fix.
    const auto wg = weights_g;

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

    // NMat >= 3: the gradient contraction as its own pass, writing the nine
    // derivatives to scratch. This is the graph's level 3, and nothing else
    // moves out of the stress pass with it -- the chain rule and its metric
    // load stay where the incumbent puts them.
    if constexpr (NMat >= 3) {
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, kWorkItemsPerTeam), [&](const int i) {
            const WorkItem w = decompose(i);
            if (w.ielement >= num_elements) return;

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

            for (int c = 0; c < 3; ++c) {
              C(w.ielement, w.iz, w.iy, w.ix, c, 0) = df_dxi[c];
              C(w.ielement, w.iz, w.iy, w.ix, c, 1) = df_deta[c];
              C(w.ielement, w.iz, w.iy, w.ix, c, 2) = df_dgamma[c];
            }
          });

      team.team_barrier();
    }

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, kWorkItemsPerTeam), [&](const int i) {
          const WorkItem w = decompose(i);
          if (w.ielement >= num_elements) return;
          const int ispec = base + w.ielement;

          PointMetric grad_metric;
          load_metric(metrics, ispec, w.iz, w.iy, w.ix, grad_metric, false);

          real_t df_dxi[3]    = {0, 0, 0};
          real_t df_deta[3]   = {0, 0, 0};
          real_t df_dgamma[3] = {0, 0, 0};

          if constexpr (NMat >= 3) {
            for (int c = 0; c < 3; ++c) {
              df_dxi[c]    = C(w.ielement, w.iz, w.iy, w.ix, c, 0);
              df_deta[c]   = C(w.ielement, w.iz, w.iy, w.ix, c, 1);
              df_dgamma[c] = C(w.ielement, w.iz, w.iy, w.ix, c, 2);
            }
          } else {
            for (int l = 0; l < NGLL; ++l) {
              for (int c = 0; c < 3; ++c) {
                df_dxi[c] += hp(w.ix, l) * up(w.ielement, w.iz, w.iy, l, c);
                df_deta[c] += hp(w.iy, l) * up(w.ielement, w.iz, l, w.ix, c);
                df_dgamma[c] += hp(w.iz, l) * up(w.ielement, l, w.iy, w.ix, c);
              }
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

          PointMetric point_metric;
          load_metric(metrics, ispec, w.iz, w.iy, w.ix, point_metric, true);

          PointProperty point_property;
          load_property(properties, ispec, w.iz, w.iy, w.ix, point_property);

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

    // NMat >= 4: the divergence contraction as its own pass, writing the three
    // partial sums per component back into the buffer the gradient vacated.
    // This is the graph's level 5; the weight combination and the scatter stay
    // in the sink pass below, exactly as the graph splits them.
    if constexpr (NMat >= 4) {
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, kWorkItemsPerTeam), [&](const int i) {
            const WorkItem w = decompose(i);
            if (w.ielement >= num_elements) return;

            real_t temp1l[3] = {0, 0, 0};
            real_t temp2l[3] = {0, 0, 0};
            real_t temp3l[3] = {0, 0, 0};

            for (int l = 0; l < NGLL; ++l) {
              for (int c = 0; c < 3; ++c)
                temp1l[c] +=
                    F(w.ielement, w.iz, w.iy, l, c, 0) * hp(l, w.ix) * wg(l);
              for (int c = 0; c < 3; ++c)
                temp2l[c] +=
                    F(w.ielement, w.iz, l, w.ix, c, 1) * hp(l, w.iy) * wg(l);
              for (int c = 0; c < 3; ++c)
                temp3l[c] +=
                    F(w.ielement, l, w.iy, w.ix, c, 2) * hp(l, w.iz) * wg(l);
            }

            for (int c = 0; c < 3; ++c) {
              C(w.ielement, w.iz, w.iy, w.ix, c, 0) = temp1l[c];
              C(w.ielement, w.iz, w.iy, w.ix, c, 1) = temp2l[c];
              C(w.ielement, w.iz, w.iy, w.ix, c, 2) = temp3l[c];
            }
          });

      team.team_barrier();
    }

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, kWorkItemsPerTeam), [&](const int i) {
          const WorkItem w = decompose(i);
          if (w.ielement >= num_elements) return;
          const int ispec = base + w.ielement;

          real_t temp1l[3] = {0, 0, 0};
          real_t temp2l[3] = {0, 0, 0};
          real_t temp3l[3] = {0, 0, 0};

          if constexpr (NMat >= 4) {
            for (int c = 0; c < 3; ++c) {
              temp1l[c] = C(w.ielement, w.iz, w.iy, w.ix, c, 0);
              temp2l[c] = C(w.ielement, w.iz, w.iy, w.ix, c, 1);
              temp3l[c] = C(w.ielement, w.iz, w.iy, w.ix, c, 2);
            }
          } else {
            for (int l = 0; l < NGLL; ++l) {
              for (int c = 0; c < 3; ++c)
                temp1l[c] +=
                    F(w.ielement, w.iz, w.iy, l, c, 0) * hp(l, w.ix) * wg(l);
              for (int c = 0; c < 3; ++c)
                temp2l[c] +=
                    F(w.ielement, w.iz, l, w.ix, c, 1) * hp(l, w.iy) * wg(l);
              for (int c = 0; c < 3; ++c)
                temp3l[c] +=
                    F(w.ielement, l, w.iy, w.ix, c, 2) * hp(l, w.iz) * wg(l);
            }
          }

          real_t result[3];
          for (int c = 0; c < 3; ++c)
            result[c] = wg(w.iz) * wg(w.iy) * temp1l[c] +
                        wg(w.iz) * wg(w.ix) * temp2l[c] +
                        wg(w.iy) * wg(w.ix) * temp3l[c];

          real_t accel[3];
          for (int c = 0; c < 3; ++c)
            accel[c] = result[c] * static_cast<real_t>(-1.0);

          PointProperty div_property;
          load_property(properties, ispec, w.iz, w.iy, w.ix, div_property);

          const int ig = iglob(ispec, w.iz, w.iy, w.ix);
          real_t    point_velocity[3];
          for (int c = 0; c < 3; ++c) point_velocity[c] = velocity(ig, c);

          const real_t wz = wg(w.iz);
          const real_t wy = wg(w.iy);
          const real_t wx = wg(w.ix);

          PointMetric div_metric;
          load_metric(metrics, ispec, w.iz, w.iy, w.ix, div_metric, true);

          const real_t factor = wx * wy * wz * div_metric.jacobian;

          compute_damping_force(factor, div_property, point_velocity, accel);
          apply_boundary_conditions(div_property, point_velocity, accel);

          for (int c = 0; c < 3; ++c)
            Kokkos::atomic_add(&acceleration(ig, c), accel[c]);
        });
  };

  const int resolved = (team_size > 0) ? team_size
                                       : policy.team_size_recommended(
                                             closure, Kokkos::ParallelForTag());

  Kokkos::parallel_for("sfpp_min::mat_stiffness", policy, closure);
  return resolved;
}

}  // namespace sfpp_min
