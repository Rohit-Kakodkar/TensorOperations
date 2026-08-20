#pragma once

#include <config.hpp>

#include <Kokkos_Core.hpp>

namespace sfpp_min {

static_assert(DEG == 4, "the Legendre evaluation below is specialised to P_4");

namespace gll {

struct Table1D {
  double v[NGLL];
};

struct Table2D {
  double v[NGLL][NGLL];
};

constexpr Table1D make_nodes() {
  constexpr double s = 0.65465367070797714379829245624686;
  return Table1D{{-1.0, -s, 0.0, s, 1.0}};
}

constexpr Table1D make_weights() {
  return Table1D{
      {1.0 / 10.0, 49.0 / 90.0, 32.0 / 45.0, 49.0 / 90.0, 1.0 / 10.0}};
}

constexpr double pnleg(double x) {
  const double x2 = x * x;
  return (35.0 * x2 * x2 - 30.0 * x2 + 3.0) / 8.0;
}

constexpr double pndleg(double x) {
  return (35.0 * x * x * x - 15.0 * x) / 2.0;
}

constexpr Table2D make_hprime() {
  const Table1D    xi = make_nodes();
  constexpr double corner =
      static_cast<double>(DEG) * (static_cast<double>(DEG) + 1.0) * 0.25;
  Table2D h{};
  for (int point = 0; point < NGLL; ++point) {
    for (int poly = 0; poly < NGLL; ++poly) {
      if (point == 0 && poly == 0) {
        h.v[point][poly] = -corner;
      } else if (point == DEG && poly == DEG) {
        h.v[point][poly] = corner;
      } else if (point == poly) {
        h.v[point][poly] = 0.0;
      } else {
        const double xp = xi.v[point];
        const double xq = xi.v[poly];
        const double d  = xp - xq;
        h.v[point][poly] =
            pnleg(xp) / (pnleg(xq) * d) +
            (1.0 - xp * xp) * pndleg(xp) /
                (static_cast<double>(DEG) * (static_cast<double>(DEG) + 1.0) *
                 pnleg(xq) * d * d);
      }
    }
  }
  return h;
}

KOKKOS_INLINE_FUNCTION double node(int i) {
  constexpr Table1D t = make_nodes();
  return t.v[i];
}

KOKKOS_INLINE_FUNCTION double weight(int i) {
  constexpr Table1D t = make_weights();
  return t.v[i];
}

KOKKOS_INLINE_FUNCTION double hprime(int point, int poly) {
  constexpr Table2D t = make_hprime();
  return t.v[point][poly];
}

}  // namespace gll
}  // namespace sfpp_min
