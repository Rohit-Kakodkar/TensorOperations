#pragma once

#include <config.hpp>
#include <data.hpp>
#include <gll.hpp>
#include <mesh.hpp>

#include <cmath>
#include <vector>

namespace sfpp_min {

struct Mat3 {
  double m[3][3];
};

inline double det3(const Mat3& a) {
  return a.m[0][0] * (a.m[1][1] * a.m[2][2] - a.m[1][2] * a.m[2][1]) -
         a.m[0][1] * (a.m[1][0] * a.m[2][2] - a.m[1][2] * a.m[2][0]) +
         a.m[0][2] * (a.m[1][0] * a.m[2][1] - a.m[1][1] * a.m[2][0]);
}

inline Mat3 inverse3(const Mat3& a) {
  const double d = det3(a);
  Mat3         r{};
  r.m[0][0] = (a.m[1][1] * a.m[2][2] - a.m[1][2] * a.m[2][1]) / d;
  r.m[0][1] = (a.m[0][2] * a.m[2][1] - a.m[0][1] * a.m[2][2]) / d;
  r.m[0][2] = (a.m[0][1] * a.m[1][2] - a.m[0][2] * a.m[1][1]) / d;
  r.m[1][0] = (a.m[1][2] * a.m[2][0] - a.m[1][0] * a.m[2][2]) / d;
  r.m[1][1] = (a.m[0][0] * a.m[2][2] - a.m[0][2] * a.m[2][0]) / d;
  r.m[1][2] = (a.m[0][2] * a.m[1][0] - a.m[0][0] * a.m[1][2]) / d;
  r.m[2][0] = (a.m[1][0] * a.m[2][1] - a.m[1][1] * a.m[2][0]) / d;
  r.m[2][1] = (a.m[0][1] * a.m[2][0] - a.m[0][0] * a.m[2][1]) / d;
  r.m[2][2] = (a.m[0][0] * a.m[1][1] - a.m[0][1] * a.m[1][0]) / d;
  return r;
}

inline Mat3 shear_matrix() {
  return Mat3{{{1.00, 0.23, -0.17}, {-0.31, 1.10, 0.19}, {0.21, -0.13, 0.95}}};
}

struct Axis {
  std::vector<double> h;
  std::vector<double> centre;
};

inline Axis make_axis(int n, double phase) {
  Axis a;
  a.h.resize(n);
  a.centre.resize(n);
  double edge = 0.0;
  for (int i = 0; i < n; ++i) {
    a.h[i]      = 1.0 + 0.4 * std::sin(1.7 * i + phase);
    a.centre[i] = edge + 0.5 * a.h[i];
    edge += a.h[i];
  }
  return a;
}

struct Geometry {
  Axis   ax, ay, az;
  Mat3   S, Sinv;
  double detS = 0.0;
};

inline Geometry make_geometry(const MeshDims& d) {
  Geometry g;
  g.ax   = make_axis(d.nex, 0.3);
  g.ay   = make_axis(d.ney, 1.1);
  g.az   = make_axis(d.nez, 2.4);
  g.S    = shear_matrix();
  g.Sinv = inverse3(g.S);
  g.detS = det3(g.S);
  return g;
}

template <typename Offset>
struct Coordinates {
  DomainArray<Offset> x, y, z;

  Coordinates() = default;
  explicit Coordinates(int nspec)
      : x("coord_x", nspec), y("coord_y", nspec), z("coord_z", nspec) {}

  void to_device() {
    x.to_device();
    y.to_device();
    z.to_device();
  }
};

inline void element_map(const Geometry& g, int ex, int ey, int ez, Mat3& A,
                        Mat3& M, double& jac) {
  const double h[3] = {g.ax.h[ex], g.ay.h[ey], g.az.h[ez]};
  for (int i = 0; i < 3; ++i)
    for (int r = 0; r < 3; ++r) A.m[i][r] = g.S.m[i][r] * 0.5 * h[r];
  for (int r = 0; r < 3; ++r)
    for (int dd = 0; dd < 3; ++dd) M.m[r][dd] = (2.0 / h[r]) * g.Sinv.m[r][dd];
  jac = g.detS * 0.125 * h[0] * h[1] * h[2];
}

template <typename Offset>
void fill_affine(const MeshDims& d, const ElementSet& set, const Geometry& g,
                 Metrics<Offset>& m, Coordinates<Offset>& c) {
  for (int ispec = 0; ispec < set.nspec(); ++ispec) {
    int ex, ey, ez;
    grid_element_coords(d, set.to_grid[ispec], ex, ey, ez);
    Mat3   A{}, M{};
    double jac = 0.0;
    element_map(g, ex, ey, ez, A, M, jac);
    for (int iz = 0; iz < NGLL; ++iz)
      for (int iy = 0; iy < NGLL; ++iy)
        for (int ix = 0; ix < NGLL; ++ix) {
          m.xix.host(ispec, iz, iy, ix)      = static_cast<real_t>(M.m[0][0]);
          m.xiy.host(ispec, iz, iy, ix)      = static_cast<real_t>(M.m[0][1]);
          m.xiz.host(ispec, iz, iy, ix)      = static_cast<real_t>(M.m[0][2]);
          m.etax.host(ispec, iz, iy, ix)     = static_cast<real_t>(M.m[1][0]);
          m.etay.host(ispec, iz, iy, ix)     = static_cast<real_t>(M.m[1][1]);
          m.etaz.host(ispec, iz, iy, ix)     = static_cast<real_t>(M.m[1][2]);
          m.gammax.host(ispec, iz, iy, ix)   = static_cast<real_t>(M.m[2][0]);
          m.gammay.host(ispec, iz, iy, ix)   = static_cast<real_t>(M.m[2][1]);
          m.gammaz.host(ispec, iz, iy, ix)   = static_cast<real_t>(M.m[2][2]);
          m.jacobian.host(ispec, iz, iy, ix) = static_cast<real_t>(jac);

          const double zeta[3] = {
              g.ax.centre[ex] + 0.5 * g.ax.h[ex] * gll::node(ix),
              g.ay.centre[ey] + 0.5 * g.ay.h[ey] * gll::node(iy),
              g.az.centre[ez] + 0.5 * g.az.h[ez] * gll::node(iz)};
          double p[3] = {0.0, 0.0, 0.0};
          for (int i = 0; i < 3; ++i)
            for (int r = 0; r < 3; ++r) p[i] += g.S.m[i][r] * zeta[r];
          c.x.host(ispec, iz, iy, ix) = static_cast<real_t>(p[0]);
          c.y.host(ispec, iz, iy, ix) = static_cast<real_t>(p[1]);
          c.z.host(ispec, iz, iy, ix) = static_cast<real_t>(p[2]);
        }
  }
}

inline Mat3 curved_metric(const Mat3& base, double t) {
  static const double freq[3][3] = {
      {1.30, 0.77, 1.91}, {0.53, 1.67, 1.13}, {1.87, 0.97, 0.61}};
  static const double phase[3][3] = {
      {0.10, 1.20, 2.30}, {3.40, 4.50, 5.60}, {0.70, 1.80, 2.90}};
  Mat3 r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      r.m[i][j] =
          base.m[i][j] * (1.0 + 0.25 * std::sin(freq[i][j] * t + phase[i][j]));
  return r;
}

template <typename Offset>
void fill_curved(const MeshDims& d, const ElementSet& set, Metrics<Offset>& m) {
  const Geometry g = make_geometry(d);
  for (int ispec = 0; ispec < set.nspec(); ++ispec) {
    const int igrid = set.to_grid[ispec];
    for (int iz = 0; iz < NGLL; ++iz)
      for (int iy = 0; iy < NGLL; ++iy)
        for (int ix = 0; ix < NGLL; ++ix) {
          int ex, ey, ez;
          grid_element_coords(d, igrid, ex, ey, ez);
          const double t = 0.3714 * (DEG * ex + ix) + 0.2189 * (DEG * ey + iy) +
                           0.5473 * (DEG * ez + iz);
          const Mat3   M = curved_metric(g.Sinv, t);
          const double jac                   = 1.0 / det3(M);
          m.xix.host(ispec, iz, iy, ix)      = static_cast<real_t>(M.m[0][0]);
          m.xiy.host(ispec, iz, iy, ix)      = static_cast<real_t>(M.m[0][1]);
          m.xiz.host(ispec, iz, iy, ix)      = static_cast<real_t>(M.m[0][2]);
          m.etax.host(ispec, iz, iy, ix)     = static_cast<real_t>(M.m[1][0]);
          m.etay.host(ispec, iz, iy, ix)     = static_cast<real_t>(M.m[1][1]);
          m.etaz.host(ispec, iz, iy, ix)     = static_cast<real_t>(M.m[1][2]);
          m.gammax.host(ispec, iz, iy, ix)   = static_cast<real_t>(M.m[2][0]);
          m.gammay.host(ispec, iz, iy, ix)   = static_cast<real_t>(M.m[2][1]);
          m.gammaz.host(ispec, iz, iy, ix)   = static_cast<real_t>(M.m[2][2]);
          m.jacobian.host(ispec, iz, iy, ix) = static_cast<real_t>(jac);
        }
  }
}

template <typename Offset>
void fill_properties(const MeshDims& d, const ElementSet& set,
                     Properties<Offset>& p) {
  for (int ispec = 0; ispec < set.nspec(); ++ispec) {
    const int igrid = set.to_grid[ispec];
    int       ex, ey, ez;
    grid_element_coords(d, igrid, ex, ey, ez);
    for (int iz = 0; iz < NGLL; ++iz)
      for (int iy = 0; iy < NGLL; ++iy)
        for (int ix = 0; ix < NGLL; ++ix) {
          const double t = 0.4111 * (DEG * ex + ix) + 0.2699 * (DEG * ey + iy) +
                           0.6421 * (DEG * ez + iz);
          const double rho = 2300.0 * (1.0 + 0.05 * std::sin(t + 0.4));
          const double vp  = 2800.0 * (1.0 + 0.04 * std::sin(1.3 * t + 1.9));
          const double vs  = 1500.0 * (1.0 + 0.06 * std::sin(0.7 * t + 3.1));
          const double mu  = rho * vs * vs;
          p.rho.host(ispec, iz, iy, ix) = static_cast<real_t>(rho);
          p.mu.host(ispec, iz, iy, ix)  = static_cast<real_t>(mu);
          p.kappa.host(ispec, iz, iy, ix) =
              static_cast<real_t>(rho * vp * vp - (4.0 / 3.0) * mu);
        }
  }
}

template <typename Offset>
void fill_properties_uniform(const ElementSet& set, Properties<Offset>& p,
                             double rho, double vp, double vs) {
  const double mu = rho * vs * vs;
  for (int ispec = 0; ispec < set.nspec(); ++ispec)
    for (int iz = 0; iz < NGLL; ++iz)
      for (int iy = 0; iy < NGLL; ++iy)
        for (int ix = 0; ix < NGLL; ++ix) {
          p.rho.host(ispec, iz, iy, ix) = static_cast<real_t>(rho);
          p.mu.host(ispec, iz, iy, ix)  = static_cast<real_t>(mu);
          p.kappa.host(ispec, iz, iy, ix) =
              static_cast<real_t>(rho * vp * vp - (4.0 / 3.0) * mu);
        }
}

}  // namespace sfpp_min
