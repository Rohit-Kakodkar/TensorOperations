#pragma once

#include <config.hpp>
#include <data.hpp>
#include <gll.hpp>
#include <mesh.hpp>

#include <vector>

namespace sfpp_min {

struct ReferenceOutput {
  std::vector<double> accel;
  std::vector<double> du;
  std::vector<double> stress;
  bool                record_point_data = false;

  int nglob = 0;

  double a(int comp, int iglob) const {
    return accel[static_cast<std::size_t>(comp) * nglob + iglob];
  }
  double du_at(int ispec, int p, int c, int d) const {
    return du[((static_cast<std::size_t>(ispec) * kPointsPerElement + p) * 3 +
               c) *
                  3 +
              d];
  }
  double stress_at(int ispec, int p, int c, int d) const {
    return stress
        [((static_cast<std::size_t>(ispec) * kPointsPerElement + p) * 3 + c) *
             3 +
         d];
  }
};

template <typename Offset>
ReferenceOutput reference_stiffness(const ElementSet&         set,
                                    const Metrics<Offset>&    m,
                                    const Properties<Offset>& p,
                                    const IglobMap& g, const Fields& f,
                                    int nglob, bool record = false) {
  ReferenceOutput out;
  out.nglob             = nglob;
  out.record_point_data = record;
  out.accel.assign(static_cast<std::size_t>(nglob) * 3, 0.0);
  if (record) {
    const std::size_t n =
        static_cast<std::size_t>(set.nspec()) * kPointsPerElement * 9;
    out.du.assign(n, 0.0);
    out.stress.assign(n, 0.0);
  }

  const auto& gm = g.h_map;
  const auto& uh = f.h_displacement;

  for (int ispec = 0; ispec < set.nspec(); ++ispec) {
    double u[3][NGLL][NGLL][NGLL];
    for (int iz = 0; iz < NGLL; ++iz)
      for (int iy = 0; iy < NGLL; ++iy)
        for (int ix = 0; ix < NGLL; ++ix) {
          const int ig = gm(ispec, iz, iy, ix);
          for (int c = 0; c < 3; ++c) u[c][iz][iy][ix] = uh(ig, c);
        }

    double F[3][3][NGLL][NGLL][NGLL];

    for (int iz = 0; iz < NGLL; ++iz)
      for (int iy = 0; iy < NGLL; ++iy)
        for (int ix = 0; ix < NGLL; ++ix) {
          double dxi[3] = {0, 0, 0};
          double det[3] = {0, 0, 0};
          double dgm[3] = {0, 0, 0};
          for (int l = 0; l < NGLL; ++l)
            for (int c = 0; c < 3; ++c) {
              dxi[c] += gll::hprime(ix, l) * u[c][iz][iy][l];
              det[c] += gll::hprime(iy, l) * u[c][iz][l][ix];
              dgm[c] += gll::hprime(iz, l) * u[c][l][iy][ix];
            }

          const double xix    = m.xix.host(ispec, iz, iy, ix);
          const double xiy    = m.xiy.host(ispec, iz, iy, ix);
          const double xiz    = m.xiz.host(ispec, iz, iy, ix);
          const double etax   = m.etax.host(ispec, iz, iy, ix);
          const double etay   = m.etay.host(ispec, iz, iy, ix);
          const double etaz   = m.etaz.host(ispec, iz, iy, ix);
          const double gammax = m.gammax.host(ispec, iz, iy, ix);
          const double gammay = m.gammay.host(ispec, iz, iy, ix);
          const double gammaz = m.gammaz.host(ispec, iz, iy, ix);
          const double jac    = m.jacobian.host(ispec, iz, iy, ix);

          double dudx[3][3];
          for (int c = 0; c < 3; ++c) {
            dudx[c][0] = xix * dxi[c] + etax * det[c] + gammax * dgm[c];
            dudx[c][1] = xiy * dxi[c] + etay * det[c] + gammay * dgm[c];
            dudx[c][2] = xiz * dxi[c] + etaz * det[c] + gammaz * dgm[c];
          }

          const double kappa = p.kappa.host(ispec, iz, iy, ix);
          const double mu    = p.mu.host(ispec, iz, iy, ix);
          const double l2m   = kappa + (4.0 / 3.0) * mu;
          const double lam   = l2m - 2.0 * mu;

          double T[3][3];
          T[0][0] = l2m * dudx[0][0] + lam * (dudx[1][1] + dudx[2][2]);
          T[1][1] = l2m * dudx[1][1] + lam * (dudx[0][0] + dudx[2][2]);
          T[2][2] = l2m * dudx[2][2] + lam * (dudx[0][0] + dudx[1][1]);
          T[0][1] = mu * (dudx[0][1] + dudx[1][0]);
          T[1][0] = T[0][1];
          T[0][2] = mu * (dudx[0][2] + dudx[2][0]);
          T[2][0] = T[0][2];
          T[1][2] = mu * (dudx[1][2] + dudx[2][1]);
          T[2][1] = T[1][2];

          if (record) {
            const int pt = (iz * NGLL + iy) * NGLL + ix;
            for (int c = 0; c < 3; ++c)
              for (int d = 0; d < 3; ++d) {
                out.du[((static_cast<std::size_t>(ispec) * kPointsPerElement +
                         pt) *
                            3 +
                        c) *
                           3 +
                       d]     = dudx[c][d];
                out.stress[((static_cast<std::size_t>(ispec) *
                                 kPointsPerElement +
                             pt) *
                                3 +
                            c) *
                               3 +
                           d] = T[c][d];
              }
          }

          for (int c = 0; c < 3; ++c) {
            F[c][0][iz][iy][ix] =
                jac * (T[c][0] * xix + T[c][1] * xiy + T[c][2] * xiz);
            F[c][1][iz][iy][ix] =
                jac * (T[c][0] * etax + T[c][1] * etay + T[c][2] * etaz);
            F[c][2][iz][iy][ix] =
                jac * (T[c][0] * gammax + T[c][1] * gammay + T[c][2] * gammaz);
          }
        }

    for (int iz = 0; iz < NGLL; ++iz)
      for (int iy = 0; iy < NGLL; ++iy)
        for (int ix = 0; ix < NGLL; ++ix) {
          double t1[3] = {0, 0, 0};
          double t2[3] = {0, 0, 0};
          double t3[3] = {0, 0, 0};
          for (int l = 0; l < NGLL; ++l)
            for (int c = 0; c < 3; ++c) {
              t1[c] += F[c][0][iz][iy][l] * gll::hprime(l, ix) * gll::weight(l);
              t2[c] += F[c][1][iz][l][ix] * gll::hprime(l, iy) * gll::weight(l);
              t3[c] += F[c][2][l][iy][ix] * gll::hprime(l, iz) * gll::weight(l);
            }
          const int ig = gm(ispec, iz, iy, ix);
          for (int c = 0; c < 3; ++c) {
            const double res = gll::weight(iz) * gll::weight(iy) * t1[c] +
                               gll::weight(iz) * gll::weight(ix) * t2[c] +
                               gll::weight(iy) * gll::weight(ix) * t3[c];
            out.accel[static_cast<std::size_t>(c) * nglob + ig] += -res;
          }
        }
  }
  return out;
}

}  // namespace sfpp_min
