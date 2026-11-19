#pragma once

/**
 * @file stiffness_generic.hpp
 * @brief The sfpp_min stiffness level graph, written once for any medium.
 *
 * `generic::stiffness<Physics, Data, TE>` builds and launches the SEM
 * stiffness graph of a field with `Physics::components` components (C):
 *
 *   C gathers | 3C gradients | one 3C-output stress integrand |
 *   3C divergences | one C-output negated sum | C scatter-adds
 *
 * No node is enumerated by hand. Every per-direction list is ordered
 * `flat = direction * C + component`, direction 0, 1, 2 = xi, eta, zeta, and
 * `Physics::stress` is the only medium-aware code. `Elastic` (C = 3) and
 * `Acoustic` (C = 1) are provided.
 *
 * The labels and names follow SPECFEM++'s `stiffness_graph.cpp` so the graph
 * body can move there verbatim: `'e'` is the element, `('g', 'b', 'a')` are
 * (zeta, eta, xi), and `'s'` is the summed point. One derivative matrix,
 * hprime, serves the gradient as `(axis, s)` and the weak-form divergence as
 * its transpose `(s, axis)`, and the full quadrature weight `w_a w_b w_g` is
 * applied in the integrand.
 *
 * `PointData` selects where the metrics and material properties are read:
 * inside the integrand through the accessors, or as register nodes the graph
 * stages for the integrand. Both compute the same field.
 */

#include <config.hpp>
#include <data.hpp>
#include <kernel_dummy.hpp>
#include <kernel_new.hpp>
#include <layout.hpp>

#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/LevelPlan.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace sfpp_min::generic {

// ---------------------------------------------------------------------------
// Physics. A Physics type provides
//   components                 number of field components C;
//   num_properties             number of material properties P;
//   property_views(accessor)   the P property views, for PointData::Registers;
//   property_from(values)      a PointProperty from those P values;
//   stress(du, property, T)    T[c][d] from du[c][d] = du_c/dx_d.
// ---------------------------------------------------------------------------

/// @brief Isotropic elastic medium: three displacement components.
struct Elastic {
  static constexpr int components     = 3;
  static constexpr int num_properties = 2;

  template <typename PropertiesAcc>
  static auto property_views(const PropertiesAcc& properties) {
    return std::make_tuple(properties.kappa, properties.mu);
  }

  KOKKOS_INLINE_FUNCTION static PointProperty property_from(
      const real_t (&values)[num_properties]) {
    return {values[0], values[1], real_t{0}};
  }

  KOKKOS_INLINE_FUNCTION static void stress(const real_t (&du)[3][3],
                                            const PointProperty& property,
                                            real_t (&T)[3][3]) {
    const real_t l2m = property.lambdaplus2mu();
    const real_t lam = property.lambda();
    const real_t mu  = property.mu;
    T[0][0]          = l2m * du[0][0] + lam * (du[1][1] + du[2][2]);
    T[1][1]          = l2m * du[1][1] + lam * (du[0][0] + du[2][2]);
    T[2][2]          = l2m * du[2][2] + lam * (du[0][0] + du[1][1]);
    T[0][1]          = mu * (du[0][1] + du[1][0]);
    T[1][0]          = T[0][1];
    T[0][2]          = mu * (du[0][2] + du[2][0]);
    T[2][0]          = T[0][2];
    T[1][2]          = mu * (du[1][2] + du[2][1]);
    T[2][1]          = T[1][2];
  }
};

/// @brief Acoustic medium: one scalar potential with flux grad(u) / rho.
struct Acoustic {
  static constexpr int components     = 1;
  static constexpr int num_properties = 1;

  template <typename PropertiesAcc>
  static auto property_views(const PropertiesAcc& properties) {
    return std::make_tuple(properties.rho);
  }

  KOKKOS_INLINE_FUNCTION static PointProperty property_from(
      const real_t (&values)[num_properties]) {
    return {real_t{0}, real_t{0}, values[0]};
  }

  KOKKOS_INLINE_FUNCTION static void stress(const real_t (&du)[1][3],
                                            const PointProperty& property,
                                            real_t (&T)[1][3]) {
    const real_t rho_inverse = real_t{1} / property.rho;
    for (int d = 0; d < 3; ++d) T[0][d] = rho_inverse * du[0][d];
  }
};

/// @brief Where the stress integrand reads metrics and material properties.
enum class PointData {
  Integrand,  ///< The integrand loads them through the accessors.
  Registers   ///< The graph stages them as register nodes, one per field.
};

namespace graph_impl {

inline constexpr int kDimensions = 3;
inline constexpr int kMetrics    = 10;  // xix..gammaz, then the jacobian

// Adds one level and returns {next graph, tuple of the level's handles}.
// LevelGraph::add is variadic and returns std::tuple<graph, handles...>.
template <typename Graph, typename... Nodes>
auto add_level(const Graph& graph, const std::tuple<Nodes...>& nodes) {
  auto graph_and_handles = std::apply(
      [&](const auto&... node) { return graph.add(node...); }, nodes);
  return std::apply(
      [](const auto& next_graph, const auto&... handle) {
        return std::make_pair(next_graph, std::make_tuple(handle...));
      },
      graph_and_handles);
}

template <int Direction, typename Mmas>
using direction_mma_t = std::tuple_element_t<
    Direction,
    std::tuple<typename Mmas::x, typename Mmas::eta, typename Mmas::gamma>>;

// d(field)/d(direction) when Transposed is false. When true, the weak-form
// divergence along `direction`: the derivative matrix read as its transpose.
template <int Direction, bool Transposed, typename Mma, typename Matrix,
          typename Field>
auto differentiate(const Matrix& derivative_matrix, const Field& field) {
  namespace TOps           = TensorOperations;
  constexpr int32_t axis   = Direction == 0 ? 'a' : Direction == 1 ? 'b' : 'g';
  const auto        matrix = [&] {
    if constexpr (Transposed)
      return derivative_matrix.template as<'s', axis>();
    else
      return derivative_matrix.template as<axis, 's'>();
  }();
  const auto summed = [&] {
    if constexpr (Direction == 0)
      return field.template as<'e', 'g', 'b', 's'>();
    else if constexpr (Direction == 1)
      return field.template as<'e', 'g', 's', 'a'>();
    else
      return field.template as<'e', 's', 'b', 'a'>();
  }();
  return TOps::make_contraction_node<'e', 'g', 'b', 'a'>(matrix, summed,
                                                         TOps::NoHook{}, Mma{});
}

// All 3C derivative nodes of one level, in flat order. The gradient of
// component c along r reads field c; the divergence term (r, c) reads flux
// r * C + c.
template <bool Transposed, int Components, typename Mmas, typename Matrix,
          typename Operands, std::size_t... Flat>
auto derivative_nodes(const Matrix& derivative_matrix, const Operands& operands,
                      std::index_sequence<Flat...>) {
  return std::make_tuple(
      differentiate<static_cast<int>(Flat / Components), Transposed,
                    direction_mma_t<static_cast<int>(Flat / Components), Mmas>>(
          derivative_matrix,
          std::get<Transposed ? Flat : Flat % Components>(operands))...);
}

template <typename Table, typename Field, std::size_t... Component>
auto gather_nodes(const Table& dof_table, const Field& field,
                  std::index_sequence<Component...>) {
  return std::make_tuple(TensorOperations::make_gather_node<'e', 'g', 'b', 'a'>(
      dof_table,
      Kokkos::subview(field, Kokkos::ALL, static_cast<int>(Component)))...);
}

// One pointwise combine over every handle in `operands`.
template <typename Operands, typename Fn>
auto combine(const Operands& operands, const Fn& fn) {
  return std::apply(
      [&](const auto&... operand) {
        return TensorOperations::make_combine_node<'e', 'g', 'b', 'a'>(
            operand..., fn);
      },
      operands);
}

// Reference gradient -> weighted stress integrand at one point.
// integrand[r * C + c] = weight * jacobian * sum_d T[c][d] * dr/dx_d.
template <typename Physics>
KOKKOS_INLINE_FUNCTION Kokkos::Array<real_t, kDimensions * Physics::components>
point_integrand(const real_t* reference_derivatives, const PointMetric& metric,
                const PointProperty& property, const real_t weight) {
  constexpr int C = Physics::components;
  // dr_dx[r][d] = d(direction r)/dx_d.
  const real_t dr_dx[kDimensions][3] = {
      {metric.xix, metric.xiy, metric.xiz},
      {metric.etax, metric.etay, metric.etaz},
      {metric.gammax, metric.gammay, metric.gammaz}};

  real_t du[C][3];
  for (int c = 0; c < C; ++c)
    for (int d = 0; d < 3; ++d)
      du[c][d] = dr_dx[0][d] * reference_derivatives[0 * C + c] +
                 dr_dx[1][d] * reference_derivatives[1 * C + c] +
                 dr_dx[2][d] * reference_derivatives[2 * C + c];

  real_t T[C][3];
  Physics::stress(du, property, T);

  const real_t                           scale = weight * metric.jacobian;
  Kokkos::Array<real_t, kDimensions * C> integrand;
  for (int r = 0; r < kDimensions; ++r)
    for (int c = 0; c < C; ++c)
      integrand[r * C + c] =
          scale * (T[c][0] * dr_dx[r][0] + T[c][1] * dr_dx[r][1] +
                   T[c][2] * dr_dx[r][2]);
  return integrand;
}

// PointData::Integrand: the metrics and properties are loaded here, at the
// point's global coordinate, through the same accessors kernel_dummy uses.
template <typename Physics, typename MetricsAcc, typename PropertiesAcc>
struct StressIntegrand {
  static constexpr int C = Physics::components;
  MetricsAcc           metrics;
  PropertiesAcc        properties;
  GlobalWeights        weights;

  template <typename... ReferenceDerivatives>
  KOKKOS_FUNCTION Kokkos::Array<real_t, kDimensions * C> operator()(
      const int e, const int g, const int b, const int a,
      const ReferenceDerivatives... reference_derivatives) const {
    static_assert(sizeof...(ReferenceDerivatives) == kDimensions * C,
                  "the stress integrand reads one reference derivative per "
                  "(direction, component)");
    const real_t flat[] = {reference_derivatives...};
    PointMetric  metric;
    load_metric(metrics, e, g, b, a, metric, true);
    PointProperty property;
    load_property(properties, e, g, b, a, property);
    return point_integrand<Physics>(flat, metric, property,
                                    weights(a) * weights(b) * weights(g));
  }
};

// PointData::Registers: the metrics and properties arrive as trailing operands
// after the 3C reference derivatives.
template <typename Physics>
struct RegisterStressIntegrand {
  static constexpr int C = Physics::components;
  static constexpr int P = Physics::num_properties;
  GlobalWeights        weights;

  template <typename... Values>
  KOKKOS_FUNCTION Kokkos::Array<real_t, kDimensions * C> operator()(
      const int, const int g, const int b, const int a,
      const Values... values) const {
    static_assert(sizeof...(Values) == kDimensions * C + kMetrics + P,
                  "the register stress integrand reads the 3C reference "
                  "derivatives, the ten metrics, then the properties");
    const real_t      flat[] = {values...};
    const real_t*     m      = flat + kDimensions * C;
    const PointMetric metric{m[0], m[1], m[2], m[3], m[4],
                             m[5], m[6], m[7], m[8], m[9]};
    real_t            property_values[P];
    for (int p = 0; p < P; ++p) property_values[p] = m[kMetrics + p];
    return point_integrand<Physics>(flat, metric,
                                    Physics::property_from(property_values),
                                    weights(a) * weights(b) * weights(g));
  }
};

/// -(sum over directions) per component.
template <int Components>
struct NegatedSum {
  template <typename... DivergenceTerms>
  KOKKOS_FUNCTION Kokkos::Array<real_t, Components> operator()(
      const int, const int, const int, const int,
      const DivergenceTerms... divergence_terms) const {
    static_assert(sizeof...(DivergenceTerms) == kDimensions * Components,
                  "one divergence term per (direction, component)");
    const real_t                      flat[] = {divergence_terms...};
    Kokkos::Array<real_t, Components> force;
    for (int c = 0; c < Components; ++c)
      force[c] = -(flat[0 * Components + c] + flat[1 * Components + c] +
                   flat[2 * Components + c]);
    return force;
  }
};

// The contraction MMAs a policy runs with by default: the row-of-e map on
// CuTe; the team backend ignores the MMA.
template <int TE, typename Policy>
struct DefaultMmas {
  using type = ContractionMmas<TensorOperations::DefaultMma<>>;
};
#if defined(TENSOR_OPS_ENABLE_CUTE)
template <int TE, typename ES>
struct DefaultMmas<TE, TensorOperations::CutePolicyTag<ES>> {
  using type = RowOfEMmas<TE>;
};
#endif

template <int TE, typename Policy, typename Mmas>
using resolved_mmas_t =
    std::conditional_t<std::is_void_v<Mmas>,
                       typename DefaultMmas<TE, Policy>::type, Mmas>;

// The DOF table's thread-value layout. CuTe scatters from the slot in the
// row-of-e thread map; the team backend reads it at the point's coordinate.
#if defined(TENSOR_OPS_ENABLE_CUTE)
template <int TE>
using DofTableTV = RowOfETV<TE>;
#else
template <int TE>
using DofTableTV = TensorOperations::DefaultTV;
#endif

}  // namespace graph_impl

/**
 * @brief Builds the stiffness graph for `Physics` without launching it.
 *
 * The graph adds `-K u` to `args.acceleration` for the `Physics::components`
 * leading columns of `args.displacement`, through `args.iglob`.
 *
 * @tparam Physics A physics type, such as `Elastic` or `Acoustic`.
 * @tparam Data Where the integrand reads metrics and properties.
 * @tparam TE Elements per team; must divide `args.nspec`.
 * @tparam Mmas A `ContractionMmas` triple for the xi, eta and zeta
 * contractions; the CuTe policy reads it, the team policy ignores it.
 * @param[in] args The sfpp_min kernel arguments. `PointData::Registers`
 * requires layout_right_dynamic metric and property storage.
 * @return The complete graph; call `.outputs()` on it to launch or to query
 * its scratch.
 * @par Constraints
 * `LevelPlan` assertions fix the level and slot counts, so a helper that drops
 * or duplicates a node fails to compile.
 */
template <typename Physics, PointData Data, int TE, typename Mmas,
          typename MetricsAcc, typename PropertiesAcc, typename IglobView>
auto build_graph(
    const DummyKernelArgs<MetricsAcc, PropertiesAcc, IglobView>& args) {
  namespace TOps  = TensorOperations;
  using ExecSpace = KernelES;

  constexpr int  components     = Physics::components;
  constexpr int  dimensions     = graph_impl::kDimensions;
  constexpr auto each_component = std::make_index_sequence<components>{};
  constexpr auto each_derivative =
      std::make_index_sequence<components * dimensions>{};

  using Tiles =
      TOps::LabelTiles<TOps::LabelTile<'e', TE>, TOps::LabelWhole<'g', NGLL>,
                       TOps::LabelWhole<'b', NGLL>, TOps::LabelWhole<'a', NGLL>,
                       TOps::LabelWhole<'s', NGLL>>;

  const auto empty = TOps::make_level_graph<real_t, ExecSpace>(Tiles{});

  const auto [with_matrix, derivative_matrix] = empty.add(TOps::make_stage_node(
      TOps::make_input_node(TOps::make_handle<'a', 's'>(args.hprime))));

  const auto [with_dofs, dof_table] =
      with_matrix.add(TOps::make_index_node<'e', 'g', 'b', 'a'>(
          args.iglob, graph_impl::DofTableTV<TE>{}));

  // PointData::Registers stages the metrics and properties here; the
  // integrand variant adds no level.
  const auto [with_point_data, point_values] = [&] {
    if constexpr (Data == PointData::Registers) {
      static_assert(
          std::is_same_v<std::decay_t<decltype(args.metrics.off)>,
                         LayoutRightDynamicOffset> &&
              std::is_same_v<std::decay_t<decltype(args.properties.off)>,
                             LayoutRightDynamicOffset>,
          "register point data reads the metric and property views as "
          "(e, g, b, a) LayoutRight arrays: use layout_right_dynamic storage");
      const auto& metrics      = args.metrics;
      const auto  in_registers = [&](const auto& view) {
        using Field =
            Kokkos::View<real_t****, Kokkos::LayoutRight,
                         typename std::decay_t<decltype(view)>::device_type,
                         Kokkos::MemoryUnmanaged>;
        return TOps::make_register_node<'e', 'g', 'b', 'a'>(
            Field(view.data(), args.nspec, NGLL, NGLL, NGLL), MetricTV<TE>{});
      };
      const auto views = std::tuple_cat(
          std::make_tuple(metrics.xix, metrics.xiy, metrics.xiz, metrics.etax,
                          metrics.etay, metrics.etaz, metrics.gammax,
                          metrics.gammay, metrics.gammaz, metrics.jacobian),
          Physics::property_views(args.properties));
      return graph_impl::add_level(
          with_dofs, std::apply(
                         [&](const auto&... view) {
                           return std::make_tuple(in_registers(view)...);
                         },
                         views));
    } else {
      return std::make_pair(with_dofs, std::tuple<>{});
    }
  }();

  const auto [with_displacement, displacement] = graph_impl::add_level(
      with_point_data,
      graph_impl::gather_nodes(dof_table, args.displacement, each_component));

  const auto [with_gradient, reference_gradient] = graph_impl::add_level(
      with_displacement, graph_impl::derivative_nodes<false, components, Mmas>(
                             derivative_matrix, displacement, each_derivative));

  const auto [with_stress, stress_integrand] = [&] {
    if constexpr (Data == PointData::Registers)
      return graph_impl::add_level(
          with_gradient,
          std::make_tuple(graph_impl::combine(
              std::tuple_cat(reference_gradient, point_values),
              graph_impl::RegisterStressIntegrand<Physics>{args.weights})));
    else
      return graph_impl::add_level(
          with_gradient,
          std::make_tuple(graph_impl::combine(
              reference_gradient,
              graph_impl::StressIntegrand<Physics, MetricsAcc, PropertiesAcc>{
                  args.metrics, args.properties, args.weights})));
  }();

  const auto [with_divergence, divergence_terms] = graph_impl::add_level(
      with_stress, graph_impl::derivative_nodes<true, components, Mmas>(
                       derivative_matrix, stress_integrand, each_derivative));

  const auto [with_force, internal_force] = graph_impl::add_level(
      with_divergence,
      std::make_tuple(graph_impl::combine(
          divergence_terms, graph_impl::NegatedSum<components>{})));

  // A level of sinks returns the bare graph, so no add_level here.
  const auto complete =
      [&]<std::size_t... Component>(std::index_sequence<Component...>) {
        return with_force.add(TOps::make_scatter_add_node<'e', 'g', 'b', 'a'>(
            dof_table,
            Kokkos::subview(args.acceleration, Kokkos::ALL,
                            static_cast<int>(Component)),
            std::get<Component>(internal_force))...);
      }(each_component);

  constexpr bool registers = Data == PointData::Registers;
  using Plan = TOps::LevelPlan<std::decay_t<decltype(complete.levels)>>;
  static_assert(Plan::num_levels == 8 + (registers ? 1 : 0),
                "matrix, DOF table, [point data,] gathers, gradients, "
                "integrand, divergences, forces, scatters");
  static_assert(
      Plan::num_slots ==
          2 + (registers ? graph_impl::kMetrics + Physics::num_properties : 0) +
              components * (1 + 3 * dimensions + 1),
      "one slot each for the matrix, the DOF table, [the point data,] the "
      "fields, gradients, integrands, divergence terms and forces");
  return complete;
}

/**
 * @brief Adds `-K u` for `Physics` to `args.acceleration`.
 *
 * @tparam Physics A physics type, such as `Elastic` or `Acoustic`.
 * @tparam Data Where the integrand reads metrics and properties.
 * @tparam TE Elements per team; must divide `args.nspec`.
 * @tparam Mmas Contraction MMAs; `void` picks the policy's default
 * (`RowOfEMmas<TE>` on CuTe).
 * @param[in] args The sfpp_min kernel arguments; acceleration is accumulated
 * into atomically, so zero it first for a fresh result.
 * @param[in] policy `TeamPolicyTag` or `CutePolicyTag` over `KernelES`.
 * @param[in] team_size Team size for the team policy; `-1` lets Kokkos choose.
 * The CuTe block size comes from the MMAs.
 * @return The league size the launch used.
 * @note Asynchronous: fence before reading acceleration.
 */
template <typename Physics, PointData Data, int TE, typename Mmas = void,
          typename Args, typename Policy>
int stiffness(const Args& args, Policy policy, int team_size = -1) {
  using M      = graph_impl::resolved_mmas_t<TE, Policy, Mmas>;
  auto outputs = build_graph<Physics, Data, TE, M>(args).outputs();
  if constexpr (std::is_same_v<Policy,
                               TensorOperations::TeamPolicyTag<KernelES>>)
    return outputs.team_size(team_size).execute(policy);
  else
    return outputs.execute(policy);
}

/// @brief The scratch `stiffness` would request, without launching.
template <typename Physics, PointData Data, int TE, typename Mmas = void,
          typename Args, typename Policy>
NewFootprint footprint(const Args& args, Policy) {
  using M            = graph_impl::resolved_mmas_t<TE, Policy, Mmas>;
  const auto outputs = build_graph<Physics, Data, TE, M>(args).outputs();
#if defined(TENSOR_OPS_ENABLE_CUTE)
  if constexpr (std::is_same_v<Policy,
                               TensorOperations::CutePolicyTag<KernelES>>)
    return {outputs.cute_smem_bytes(), outputs.cute_unpooled_smem_bytes(),
            outputs.cute_num_threads()};
  else
#endif
    return {outputs.scratch_bytes(), outputs.slot_bytes(), -1,
            outputs.scratch_level()};
}

}  // namespace sfpp_min::generic
