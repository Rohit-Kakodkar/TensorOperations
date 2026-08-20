#include <config.hpp>
#include <data.hpp>
#include <geometry.hpp>
#include <gll.hpp>
#include <mesh.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <set>

using namespace sfpp_min;

namespace {

using Off = ChunkTiledOffset;

constexpr MeshDims kD{4, 3, 2};

Mat3 metric_at(const Metrics<Off>& m, int ispec, int iz, int iy, int ix) {
  Mat3 M{};
  M.m[0][0] = m.xix.host(ispec, iz, iy, ix);
  M.m[0][1] = m.xiy.host(ispec, iz, iy, ix);
  M.m[0][2] = m.xiz.host(ispec, iz, iy, ix);
  M.m[1][0] = m.etax.host(ispec, iz, iy, ix);
  M.m[1][1] = m.etay.host(ispec, iz, iy, ix);
  M.m[1][2] = m.etaz.host(ispec, iz, iy, ix);
  M.m[2][0] = m.gammax.host(ispec, iz, iy, ix);
  M.m[2][1] = m.gammay.host(ispec, iz, iy, ix);
  M.m[2][2] = m.gammaz.host(ispec, iz, iy, ix);
  return M;
}

double min_abs_entry(const Mat3& a) {
  double v = std::abs(a.m[0][0]);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) v = std::min(v, std::abs(a.m[i][j]));
  return v;
}

}  // namespace

TEST(SfppMinGeometry, ShearIsFullNonSymmetricAndWellConditioned) {
  const Mat3 S  = shear_matrix();
  const Mat3 Si = inverse3(S);
  EXPECT_GT(min_abs_entry(S), 0.1);
  EXPECT_GT(min_abs_entry(Si), 0.1);
  EXPECT_NEAR(det3(S), 1.17903, 1e-4);
  bool symmetric = true;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      if (std::abs(S.m[i][j] - S.m[j][i]) > 1e-12) symmetric = false;
  EXPECT_FALSE(symmetric);
  Mat3 p{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      p.m[i][j] = 0.0;
      for (int k = 0; k < 3; ++k) p.m[i][j] += S.m[i][k] * Si.m[k][j];
      EXPECT_NEAR(p.m[i][j], i == j ? 1.0 : 0.0, 1e-12);
    }
}

TEST(SfppMinGeometry, AffineMetricsInvertTheForwardMap) {
  const auto       set = all_elements(kD);
  const Geometry   g   = make_geometry(kD);
  Metrics<Off>     m(set.nspec());
  Coordinates<Off> c(set.nspec());
  fill_affine(kD, set, g, m, c);

  for (int ispec = 0; ispec < set.nspec(); ++ispec) {
    int ex, ey, ez;
    grid_element_coords(kD, set.to_grid[ispec], ex, ey, ez);
    Mat3   A{}, Mexp{};
    double jac = 0.0;
    element_map(g, ex, ey, ez, A, Mexp, jac);
    const Mat3 M = metric_at(m, ispec, 1, 2, 3);
    for (int r = 0; r < 3; ++r)
      for (int dd = 0; dd < 3; ++dd) {
        double s = 0.0;
        for (int k = 0; k < 3; ++k) s += M.m[r][k] * A.m[k][dd];
        ASSERT_NEAR(s, r == dd ? 1.0 : 0.0, 1e-4)
            << "M*A != I at ispec=" << ispec << " r=" << r << " d=" << dd;
      }
    ASSERT_NEAR(m.jacobian.host(ispec, 1, 2, 3), det3(A), 1e-4 * std::abs(jac));
    ASSERT_GT(std::abs(jac), 0.02);
  }
}

TEST(SfppMinGeometry, AffineCoordinatesAgreeWithTheMetrics) {
  const auto       set = all_elements(kD);
  const Geometry   g   = make_geometry(kD);
  Metrics<Off>     m(set.nspec());
  Coordinates<Off> c(set.nspec());
  fill_affine(kD, set, g, m, c);

  for (int ispec = 0; ispec < set.nspec(); ++ispec) {
    int ex, ey, ez;
    grid_element_coords(kD, set.to_grid[ispec], ex, ey, ez);
    Mat3   A{}, M{};
    double jac = 0.0;
    element_map(g, ex, ey, ez, A, M, jac);
    for (int ix = 0; ix + 1 < NGLL; ++ix) {
      const double dxi = gll::node(ix + 1) - gll::node(ix);
      const double dx =
          c.x.host(ispec, 0, 0, ix + 1) - c.x.host(ispec, 0, 0, ix);
      const double dy =
          c.y.host(ispec, 0, 0, ix + 1) - c.y.host(ispec, 0, 0, ix);
      ASSERT_NEAR(dx, A.m[0][0] * dxi, 1e-4);
      ASSERT_NEAR(dy, A.m[1][0] * dxi, 1e-4);
    }
  }
}

TEST(SfppMinGeometry, AffineMeshIsGeometricallyConforming) {
  const auto       set = all_elements(kD);
  const Geometry   g   = make_geometry(kD);
  Metrics<Off>     m(set.nspec());
  Coordinates<Off> c(set.nspec());
  fill_affine(kD, set, g, m, c);

  for (int ez = 0; ez < kD.nez; ++ez)
    for (int ey = 0; ey < kD.ney; ++ey)
      for (int ex = 0; ex + 1 < kD.nex; ++ex) {
        const int a = grid_element_index(kD, ex, ey, ez);
        const int b = grid_element_index(kD, ex + 1, ey, ez);
        for (int iz = 0; iz < NGLL; ++iz)
          for (int iy = 0; iy < NGLL; ++iy) {
            ASSERT_NEAR(c.x.host(a, iz, iy, DEG), c.x.host(b, iz, iy, 0), 1e-4);
            ASSERT_NEAR(c.y.host(a, iz, iy, DEG), c.y.host(b, iz, iy, 0), 1e-4);
            ASSERT_NEAR(c.z.host(a, iz, iy, DEG), c.z.host(b, iz, iy, 0), 1e-4);
          }
      }
  for (int ez = 0; ez + 1 < kD.nez; ++ez)
    for (int ey = 0; ey < kD.ney; ++ey)
      for (int ex = 0; ex < kD.nex; ++ex) {
        const int a = grid_element_index(kD, ex, ey, ez);
        const int b = grid_element_index(kD, ex, ey, ez + 1);
        for (int iy = 0; iy < NGLL; ++iy)
          for (int ix = 0; ix < NGLL; ++ix)
            ASSERT_NEAR(c.z.host(a, DEG, iy, ix), c.z.host(b, 0, iy, ix), 1e-4);
      }
}

TEST(SfppMinGeometry, CurvedJacobianIsTheReciprocalDeterminant) {
  const auto   set = all_elements(kD);
  Metrics<Off> m(set.nspec());
  fill_curved(kD, set, m);
  for (int ispec = 0; ispec < set.nspec(); ++ispec)
    for (int iz = 0; iz < NGLL; ++iz)
      for (int iy = 0; iy < NGLL; ++iy)
        for (int ix = 0; ix < NGLL; ++ix) {
          const Mat3   M = metric_at(m, ispec, iz, iy, ix);
          const double d = det3(M);
          ASSERT_GT(std::abs(d), 0.2) << "curved metric is ill-conditioned";
          ASSERT_NEAR(m.jacobian.host(ispec, iz, iy, ix) * d, 1.0, 1e-4);
        }
}

TEST(SfppMinGeometry, NeitherFillHasStructuralZeros) {
  const auto       set = all_elements(kD);
  const Geometry   g   = make_geometry(kD);
  Metrics<Off>     ma(set.nspec());
  Coordinates<Off> c(set.nspec());
  fill_affine(kD, set, g, ma, c);
  Metrics<Off> mc(set.nspec());
  fill_curved(kD, set, mc);

  for (const Metrics<Off>* m : {&ma, &mc})
    for (int ispec = 0; ispec < set.nspec(); ++ispec)
      for (int iz = 0; iz < NGLL; ++iz)
        for (int iy = 0; iy < NGLL; ++iy)
          for (int ix = 0; ix < NGLL; ++ix) {
            const Mat3 M = metric_at(*m, ispec, iz, iy, ix);
            ASSERT_GT(min_abs_entry(M), 1e-3)
                << "a metric component is ~zero; a permutation bug could hide";
          }
}

TEST(SfppMinGeometry, AffineVariesPerElementCurvedVariesPerPoint) {
  const auto       set = all_elements(kD);
  const Geometry   g   = make_geometry(kD);
  Metrics<Off>     ma(set.nspec());
  Coordinates<Off> c(set.nspec());
  fill_affine(kD, set, g, ma, c);
  Metrics<Off> mc(set.nspec());
  fill_curved(kD, set, mc);

  std::set<float> across_elements;
  for (int ispec = 0; ispec < set.nspec(); ++ispec)
    across_elements.insert(ma.xix.host(ispec, 0, 0, 0));
  EXPECT_GT(across_elements.size(), 1u) << "affine fill is element-constant";

  for (int ispec = 0; ispec < set.nspec(); ++ispec) {
    const float a = ma.xix.host(ispec, 0, 0, 0);
    for (int iz = 0; iz < NGLL; ++iz)
      for (int iy = 0; iy < NGLL; ++iy)
        for (int ix = 0; ix < NGLL; ++ix)
          ASSERT_EQ(ma.xix.host(ispec, iz, iy, ix), a)
              << "affine fill must be constant WITHIN an element";
  }

  std::set<float> within_element;
  for (int iz = 0; iz < NGLL; ++iz)
    for (int iy = 0; iy < NGLL; ++iy)
      for (int ix = 0; ix < NGLL; ++ix)
        within_element.insert(mc.xix.host(0, iz, iy, ix));
  EXPECT_GT(within_element.size(), 100u)
      << "curved fill must vary per POINT or wrong-point reads are invisible";
}

TEST(SfppMinGeometry, PropertiesArePositiveAndVaryPerPoint) {
  const auto      set = all_elements(kD);
  Properties<Off> p(set.nspec());
  fill_properties(kD, set, p);
  std::set<float> seen;
  for (int ispec = 0; ispec < set.nspec(); ++ispec)
    for (int iz = 0; iz < NGLL; ++iz)
      for (int iy = 0; iy < NGLL; ++iy)
        for (int ix = 0; ix < NGLL; ++ix) {
          const double kappa = p.kappa.host(ispec, iz, iy, ix);
          const double mu    = p.mu.host(ispec, iz, iy, ix);
          const double rho   = p.rho.host(ispec, iz, iy, ix);
          ASSERT_GT(kappa, 0.0);
          ASSERT_GT(mu, 0.0);
          ASSERT_GT(rho, 0.0);
          ASSERT_GT(kappa + (4.0 / 3.0) * mu - 2.0 * mu, 0.0)
              << "lambda must stay positive";
          if (ispec == 0) seen.insert(p.mu.host(ispec, iz, iy, ix));
        }
  EXPECT_GT(seen.size(), 100u);
}

TEST(SfppMinGeometry, FillsAreDeterministic) {
  const auto       set = all_elements(kD);
  const Geometry   g   = make_geometry(kD);
  Metrics<Off>     a1(set.nspec()), a2(set.nspec());
  Coordinates<Off> c1(set.nspec()), c2(set.nspec());
  fill_affine(kD, set, g, a1, c1);
  fill_affine(kD, set, g, a2, c2);
  for (int ispec = 0; ispec < set.nspec(); ++ispec)
    for (int iz = 0; iz < NGLL; ++iz)
      for (int iy = 0; iy < NGLL; ++iy)
        for (int ix = 0; ix < NGLL; ++ix) {
          ASSERT_EQ(a1.xix.host(ispec, iz, iy, ix),
                    a2.xix.host(ispec, iz, iy, ix));
          ASSERT_EQ(c1.x.host(ispec, iz, iy, ix), c2.x.host(ispec, iz, iy, ix));
        }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
