#pragma once

#include <config.hpp>

#include <Kokkos_Core.hpp>

namespace sfpp_min {

namespace gll {

struct Table1D {
  double v[NGLL];
};

struct Table2D {
  double v[NGLL][NGLL];
};

// P_DEG(x) and P'_DEG(x) by the three-term recurrence. At DEG = 4 the closed
// forms below are kept so the calibrated NGLL = 5 tables stay bit-identical.
struct Legendre {
  double p;
  double dp;
};

constexpr Legendre legendre(double x) {
  double p0 = 1.0, p1 = x, d0 = 0.0, d1 = 1.0;
  if (DEG == 0) return {1.0, 0.0};
  for (int n = 1; n < DEG; ++n) {
    const double p2 = ((2.0 * n + 1.0) * x * p1 - n * p0) / (n + 1.0);
    const double d2 = d0 + (2.0 * n + 1.0) * p1;
    p0              = p1;
    p1              = p2;
    d0              = d1;
    d1              = d2;
  }
  return {p1, d1};
}

constexpr double pnleg(double x) {
  if constexpr (DEG == 4) {
    const double x2 = x * x;
    return (35.0 * x2 * x2 - 30.0 * x2 + 3.0) / 8.0;
  } else {
    return legendre(x).p;
  }
}

constexpr double pndleg(double x) {
  if constexpr (DEG == 4)
    return (35.0 * x * x * x - 15.0 * x) / 2.0;
  else
    return legendre(x).dp;
}

// cos on [0, pi] for the Chebyshev-Gauss-Lobatto starting guesses; std::cos
// is not constexpr in C++20.
constexpr double cos_0_pi(double a) {
  constexpr double pi   = 3.14159265358979323846264338327950288;
  const bool       flip = a > pi / 2.0;
  const double     t    = flip ? pi - a : a;
  double           term = 1.0, sum = 1.0;
  for (int k = 1; k < 30; ++k) {
    term *= -t * t / ((2.0 * k - 1.0) * (2.0 * k));
    sum += term;
  }
  return flip ? -sum : sum;
}

// The interior GLL nodes are the roots of P'_DEG. Newton on P'_DEG, using
// (1 - x^2) P''_N = 2x P'_N - N(N+1) P_N, from x_j = -cos(pi j / DEG).
constexpr Table1D make_nodes() {
  if constexpr (DEG == 4) {
    constexpr double s = 0.65465367070797714379829245624686;
    return Table1D{{-1.0, -s, 0.0, s, 1.0}};
  } else {
    constexpr double pi = 3.14159265358979323846264338327950288;
    constexpr double nn = static_cast<double>(DEG) * (DEG + 1.0);
    Table1D          t{};
    t.v[0]   = -1.0;
    t.v[DEG] = 1.0;
    for (int j = 1; j < DEG; ++j) {
      double x = -cos_0_pi(pi * j / DEG);
      for (int it = 0; it < 100; ++it) {
        const Legendre l   = legendre(x);
        const double   ddp = (2.0 * x * l.dp - nn * l.p) / (1.0 - x * x);
        const double   dx  = l.dp / ddp;
        x -= dx;
        if (dx < 1e-17 && dx > -1e-17) break;
      }
      t.v[j] = x;
    }
    if (DEG % 2 == 0) t.v[DEG / 2] = 0.0;
    return t;
  }
}

constexpr Table1D make_weights() {
  if constexpr (DEG == 4) {
    return Table1D{
        {1.0 / 10.0, 49.0 / 90.0, 32.0 / 45.0, 49.0 / 90.0, 1.0 / 10.0}};
  } else {
    const Table1D    xi = make_nodes();
    constexpr double nn = static_cast<double>(DEG) * (DEG + 1.0);
    Table1D          w{};
    for (int j = 0; j < NGLL; ++j) {
      const double p = pnleg(xi.v[j]);
      w.v[j]         = 2.0 / (nn * p * p);
    }
    return w;
  }
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
