#pragma once

#include <config.hpp>
#include <data.hpp>
#include <kernel_dummy.hpp>

#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/LevelPlan.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>

#include <cstddef>

namespace sfpp_min::simple {

using namespace TensorOperations;

inline constexpr int N = NGLL;
using ES               = Kokkos::DefaultExecutionSpace;
using N_               = cute::Int<N>;

template <int TE, int Sa, int Sb0, int Sb1>
using RowOfE = decltype(cute::make_tiled_mma(
    cute::UniversalFMA<real_t, real_t, real_t>{},
    cute::Layout<
        cute::Shape<N_, cute::Shape<N_, N_>, cute::_1>,
        cute::Stride<cute::Int<Sa>,
                     cute::Stride<cute::Int<Sb0>, cute::Int<Sb1>>, cute::_0>>{},
    cute::Tile<
        N_,
        cute::Layout<cute::Shape<N_, N_, cute::Int<TE>>,
                     cute::Stride<cute::Int<N * TE>, cute::Int<TE>, cute::_1>>,
        cute::_1>{}));

template <int TE>
struct RowOfEMmas {
  using xi    = RowOfE<TE, 1, N, N * N>;
  using eta   = RowOfE<TE, N, 1, N * N>;
  using gamma = RowOfE<TE, N * N, 1, N>;
};

template <int TE>
using PointTV =
    cute::Layout<cute::Shape<cute::Shape<N_, N_, N_>, cute::Int<TE>>,
                 cute::Stride<cute::Stride<cute::Int<N * N * TE>,
                                           cute::Int<N * TE>, cute::Int<TE>>,
                              cute::_1>>;

template <char... L, typename Node>
auto as(const Node& n) {
  return n.template as<L...>();
}

struct Stress {
  KOKKOS_FUNCTION Kokkos::Array<real_t, 9> operator()(
      int, int, int, int, real_t dxi0, real_t dxi1, real_t dxi2, real_t deta0,
      real_t deta1, real_t deta2, real_t dgam0, real_t dgam1, real_t dgam2,
      real_t xix, real_t xiy, real_t xiz, real_t etax, real_t etay, real_t etaz,
      real_t gamx, real_t gamy, real_t gamz, real_t jac, real_t kappa,
      real_t mu) const {
    const real_t dxi[3]  = {dxi0, dxi1, dxi2};
    const real_t deta[3] = {deta0, deta1, deta2};
    const real_t dgam[3] = {dgam0, dgam1, dgam2};

    real_t du[3][3];
    for (int c = 0; c < 3; ++c) {
      du[c][0] = xix * dxi[c] + etax * deta[c] + gamx * dgam[c];
      du[c][1] = xiy * dxi[c] + etay * deta[c] + gamy * dgam[c];
      du[c][2] = xiz * dxi[c] + etaz * deta[c] + gamz * dgam[c];
    }

    const real_t l2m = kappa + real_t(4.0 / 3.0) * mu;
    const real_t lam = l2m - real_t(2.0) * mu;

    real_t T[3][3];
    T[0][0] = l2m * du[0][0] + lam * (du[1][1] + du[2][2]);
    T[1][1] = l2m * du[1][1] + lam * (du[0][0] + du[2][2]);
    T[2][2] = l2m * du[2][2] + lam * (du[0][0] + du[1][1]);
    T[0][1] = T[1][0] = mu * (du[0][1] + du[1][0]);
    T[0][2] = T[2][0] = mu * (du[0][2] + du[2][0]);
    T[1][2] = T[2][1] = mu * (du[1][2] + du[2][1]);

    real_t Fxi[3], Feta[3], Fgam[3];
    for (int c = 0; c < 3; ++c) {
      Fxi[c]  = jac * (T[c][0] * xix + T[c][1] * xiy + T[c][2] * xiz);
      Feta[c] = jac * (T[c][0] * etax + T[c][1] * etay + T[c][2] * etaz);
      Fgam[c] = jac * (T[c][0] * gamx + T[c][1] * gamy + T[c][2] * gamz);
    }
    return {Fxi[0],  Fxi[1],  Fxi[2],  Feta[0], Feta[1],
            Feta[2], Fgam[0], Fgam[1], Fgam[2]};
  }
};

template <typename Weights>
struct Divergence {
  Weights         w;
  KOKKOS_FUNCTION Kokkos::Array<real_t, 3> operator()(
      int, int k, int j, int i, real_t sxi0, real_t sxi1, real_t sxi2,
      real_t seta0, real_t seta1, real_t seta2, real_t sgam0, real_t sgam1,
      real_t sgam2) const {
    const real_t wjk = w(j) * w(k);
    const real_t wik = w(i) * w(k);
    const real_t wij = w(i) * w(j);
    return {-(wjk * sxi0 + wik * seta0 + wij * sgam0),
            -(wjk * sxi1 + wik * seta1 + wij * sgam1),
            -(wjk * sxi2 + wik * seta2 + wij * sgam2)};
  }
};

template <int TE, typename Args, typename Policy>
int stiffness(const Args& a, GlobalHPrime hprimewgll, Policy policy,
              int team_size = -1) {
  using Tiles =
      LabelTiles<LabelTile<'e', TE>, LabelWhole<'k', N>, LabelWhole<'j', N>,
                 LabelWhole<'i', N>, LabelWhole<'p', N>, LabelWhole<'r', N>>;
  const Kokkos::Array<int, 4> extents{a.nspec, N, N, N};
  using Mma     = RowOfEMmas<TE>;
  const auto& m = a.metrics;
  const auto& p = a.properties;

  auto in_registers = [&](auto view) {
    using Field = Kokkos::View<real_t****, Kokkos::LayoutRight,
                               typename decltype(view)::device_type,
                               Kokkos::MemoryUnmanaged>;
    return make_register_node<'e', 'k', 'j', 'i'>(
        Field(view.data(), extents[0], extents[1], extents[2], extents[3]),
        PointTV<TE>{});
  };
  auto gather = [&](int c) {
    return make_gather_node<'e', 'k', 'j', 'i'>(
        a.iglob, Kokkos::subview(a.displacement, Kokkos::ALL, c));
  };
  auto scatter = [&](int c, auto r) {
    return make_scatter_add_node<'e', 'k', 'j', 'i'>(
        a.iglob, Kokkos::subview(a.acceleration, Kokkos::ALL, c), r);
  };

  auto g0 = make_level_graph<real_t, ES>(Tiles{});

  auto [g1, H, HW] = g0.add(
      make_stage_node(make_input_node(make_handle<'r', 'p'>(a.hprime))),
      make_stage_node(make_input_node(make_handle<'p', 'r'>(hprimewgll))));

  auto [g2, xix, xiy, xiz, etax, etay, etaz, gamx, gamy, gamz, jac, kappa, mu] =
      g1.add(in_registers(m.xix), in_registers(m.xiy), in_registers(m.xiz),
             in_registers(m.etax), in_registers(m.etay), in_registers(m.etaz),
             in_registers(m.gammax), in_registers(m.gammay),
             in_registers(m.gammaz), in_registers(m.jacobian),
             in_registers(p.kappa), in_registers(p.mu));

  auto [g3, u0, u1, u2] = g2.add(gather(0), gather(1), gather(2));

  auto d_dxi = [&](auto u) {
    return make_contraction_node<'e', 'k', 'j', 'i'>(
        as<'i', 'p'>(H), as<'e', 'k', 'j', 'p'>(u), NoHook{},
        typename Mma::xi{});
  };
  auto d_deta = [&](auto u) {
    return make_contraction_node<'e', 'k', 'j', 'i'>(
        as<'j', 'p'>(H), as<'e', 'k', 'p', 'i'>(u), NoHook{},
        typename Mma::eta{});
  };
  auto d_dgam = [&](auto u) {
    return make_contraction_node<'e', 'k', 'j', 'i'>(
        as<'k', 'p'>(H), as<'e', 'p', 'j', 'i'>(u), NoHook{},
        typename Mma::gamma{});
  };
  auto [g4, dxi0, dxi1, dxi2, deta0, deta1, deta2, dgam0, dgam1, dgam2] =
      g3.add(d_dxi(u0), d_dxi(u1), d_dxi(u2), d_deta(u0), d_deta(u1),
             d_deta(u2), d_dgam(u0), d_dgam(u1), d_dgam(u2));

  auto [g5, Fxi0, Fxi1, Fxi2, Feta0, Feta1, Feta2, Fgam0, Fgam1, Fgam2] =
      g4.add(make_combine_node<'e', 'k', 'j', 'i'>(
          dxi0, dxi1, dxi2, deta0, deta1, deta2, dgam0, dgam1, dgam2, xix, xiy,
          xiz, etax, etay, etaz, gamx, gamy, gamz, jac, kappa, mu, Stress{}));

  auto div_xi = [&](auto F) {
    return make_contraction_node<'e', 'k', 'j', 'i'>(
        as<'p', 'i'>(HW), as<'e', 'k', 'j', 'p'>(F), NoHook{},
        typename Mma::xi{});
  };
  auto div_eta = [&](auto F) {
    return make_contraction_node<'e', 'k', 'j', 'i'>(
        as<'p', 'j'>(HW), as<'e', 'k', 'p', 'i'>(F), NoHook{},
        typename Mma::eta{});
  };
  auto div_gam = [&](auto F) {
    return make_contraction_node<'e', 'k', 'j', 'i'>(
        as<'p', 'k'>(HW), as<'e', 'p', 'j', 'i'>(F), NoHook{},
        typename Mma::gamma{});
  };
  auto [g6, sxi0, sxi1, sxi2, seta0, seta1, seta2, sgam0, sgam1, sgam2] =
      g5.add(div_xi(Fxi0), div_xi(Fxi1), div_xi(Fxi2), div_eta(Feta0),
             div_eta(Feta1), div_eta(Feta2), div_gam(Fgam0), div_gam(Fgam1),
             div_gam(Fgam2));

  auto [g7, r0, r1, r2] = g6.add(make_combine_node<'e', 'k', 'j', 'i'>(
      sxi0, sxi1, sxi2, seta0, seta1, seta2, sgam0, sgam1, sgam2,
      Divergence<GlobalWeights>{a.weights}));

  auto g8 = g7.add(scatter(0, r0), scatter(1, r1), scatter(2, r2));

  return g8.outputs().team_size(team_size).execute(policy);
}

}  // namespace sfpp_min::simple
