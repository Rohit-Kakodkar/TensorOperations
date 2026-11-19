// ===========================================================================
// test_stiffness_generic.cpp -- the component-agnostic stiffness graph
// (stiffness_generic.hpp) against the serial oracles.
//
// Elastic runs against reference_stiffness, the oracle every sfpp_min kernel
// is judged by; acoustic against reference_acoustic_stiffness, the same loop
// with one component. Both PointData variants run on every policy the build
// has. On CuTe the generic graph is also compared with the hand-enumerated
// graphs it replaces: stiffness_simple (register point data) and kernel_new's
// index-slot gather/scatter configuration (point data read in the integrand).
// ===========================================================================
#include "kernel_case.hpp"

#include <kernel_new.hpp>
#include <stiffness_generic.hpp>
#if defined(TENSOR_OPS_ENABLE_CUTE)
#include <stiffness_simple.hpp>
#endif

#include <string>
#include <type_traits>
#include <vector>

using namespace sfpp_min;
using namespace sfpp_min::test;
namespace TOps = TensorOperations;

namespace {

using generic::Acoustic;
using generic::Elastic;
using generic::PointData;

// The host backend's 32 KB per-team scratch admits TE = 2; the device runs the
// ground-truth TE = 4. See test_kernel_new.cpp.
constexpr int kTestTE =
    std::is_same_v<KernelES, Kokkos::DefaultHostExecutionSpace> ? 2
                                                                : kExecChunk;

using TeamPolicy = TOps::TeamPolicyTag<KernelES>;
#if defined(TENSOR_OPS_ENABLE_CUTE)
using CutePolicy = TOps::CutePolicyTag<KernelES>;
#endif

// Register point data reads the views as LayoutRight (e, g, b, a) arrays.
using RegisterOffset = LayoutRightDynamicOffset;

template <typename Physics, PointData Data, int TE, typename Policy,
          typename Off>
void launch(Case<Off>& k) {
  generic::stiffness<Physics, Data, TE>(make_args(k, make_gll_views()),
                                        Policy{});
}

struct AcousticReport {
  ErrorReport error;
  double      other_columns = 0.0;  // max |acceleration| in columns 1 and 2
};

template <typename Off>
AcousticReport compare_to_acoustic_oracle(Case<Off>& k) {
  fetch(k);
  const std::vector<double> want =
      reference_acoustic_stiffness(k.set, k.m, k.p, k.g, k.f, k.nglob);
  AcousticReport r;
  for (int ig = 0; ig < k.nglob; ++ig) {
    const double got  = k.f.h_acceleration(ig, 0);
    r.error.worst_abs = std::max(r.error.worst_abs, std::abs(got - want[ig]));
    r.error.scale     = std::max(r.error.scale, std::abs(want[ig]));
    for (int c = 1; c < 3; ++c)
      r.other_columns =
          std::max(r.other_columns,
                   std::abs(static_cast<double>(k.f.h_acceleration(ig, c))));
  }
  return r;
}

// The rigid-body ratio of kernel_case.hpp, on the given storage: a constant
// field is in the operator's null space, so its output must sit at float
// roundoff relative to the output on the linear field.
template <typename Off, typename Launch>
double rigid_body_ratio_on(Launch launch_on) {
  auto         k    = make_case<Off>(/*ix_fastest=*/true);
  const double c[3] = {1.3e-3, -0.7e-3, 2.1e-3};
  set_constant_field(k, c);
  launch_on(k);
  fetch(k);
  double residual = 0.0;
  for (int ig = 0; ig < k.nglob; ++ig)
    for (int comp = 0; comp < 3; ++comp)
      residual =
          std::max(residual,
                   std::abs(static_cast<double>(k.f.h_acceleration(ig, comp))));

  auto k2 = make_linear_case<Off>(/*ix_fastest=*/true);
  launch_on(k2);
  fetch(k2);
  double scale = 0.0;
  for (int ig = 0; ig < k2.nglob; ++ig)
    for (int comp = 0; comp < 3; ++comp)
      scale = std::max(
          scale, std::abs(static_cast<double>(k2.f.h_acceleration(ig, comp))));
  std::printf(
      "[ INFO     ] rigid body: residual = %.3e, scale = %.3e, ratio = %.3e\n",
      residual, scale, residual / scale);
  EXPECT_GT(scale, 0.0) << "no output to compare against; test is void";
  return residual / scale;
}

constexpr double kRigidBound = NGLL == 5 ? 1e-5 : 1e-4;

// Relative max difference between the accelerations of two launches.
template <typename Off>
double relative_difference(const Case<Off>& a, const Case<Off>& b) {
  double worst = 0.0, scale = 0.0;
  for (int ig = 0; ig < a.nglob; ++ig)
    for (int c = 0; c < 3; ++c) {
      const double x = a.f.h_acceleration(ig, c);
      scale          = std::max(scale, std::abs(x));
      worst          = std::max(worst, std::abs(x - b.f.h_acceleration(ig, c)));
    }
  EXPECT_GT(scale, 0.0) << "void comparison: the field is zero";
  return scale > 0.0 ? worst / scale : worst;
}

template <typename Policy>
class SfppMinStiffnessGeneric : public ::testing::Test {};

struct PolicyName {
  template <typename Policy>
  static std::string GetName(int) {
    return std::is_same_v<Policy, TeamPolicy> ? "Team" : "Cute";
  }
};

#if defined(TENSOR_OPS_ENABLE_CUTE)
using Policies = ::testing::Types<TeamPolicy, CutePolicy>;
#else
using Policies = ::testing::Types<TeamPolicy>;
#endif
TYPED_TEST_SUITE(SfppMinStiffnessGeneric, Policies, PolicyName);

}  // namespace

// --- elastic ---------------------------------------------------------------

// The integrand reads through the accessors, so every storage layout works.
TYPED_TEST(SfppMinStiffnessGeneric, ElasticIntegrandMatchesTheOracle) {
  auto k = make_linear_case<ChunkTiledDynamicOffset>(/*ix_fastest=*/true);
  launch<Elastic, PointData::Integrand, kTestTE, TypeParam>(k);
  expect_matches_oracle(compare_to_oracle(k), "elastic integrand chunked");

  auto kr = make_linear_case<RegisterOffset>(/*ix_fastest=*/true);
  launch<Elastic, PointData::Integrand, kTestTE, TypeParam>(kr);
  expect_matches_oracle(compare_to_oracle(kr), "elastic integrand LR");
}

TYPED_TEST(SfppMinStiffnessGeneric, ElasticRegistersMatchesTheOracle) {
  auto k = make_linear_case<RegisterOffset>(/*ix_fastest=*/true);
  launch<Elastic, PointData::Registers, kTestTE, TypeParam>(k);
  expect_matches_oracle(compare_to_oracle(k), "elastic registers");
}

TYPED_TEST(SfppMinStiffnessGeneric, ElasticRigidBodyIsAtTheRoundoffFloor) {
  EXPECT_LT(rigid_body_ratio_on<RegisterOffset>([](auto& k) {
              launch<Elastic, PointData::Integrand, kTestTE, TypeParam>(k);
            }),
            kRigidBound);
  EXPECT_LT(rigid_body_ratio_on<RegisterOffset>([](auto& k) {
              launch<Elastic, PointData::Registers, kTestTE, TypeParam>(k);
            }),
            kRigidBound);
}

// --- acoustic --------------------------------------------------------------

template <PointData Data, typename Policy>
void expect_acoustic_oracle(const char* name) {
  auto k = make_linear_case<RegisterOffset>(/*ix_fastest=*/true);
  launch<Acoustic, Data, kTestTE, Policy>(k);
  const AcousticReport r = compare_to_acoustic_oracle(k);
  expect_matches_oracle(r.error, name);
  EXPECT_EQ(r.other_columns, 0.0)
      << name << ": acoustic wrote outside the potential's column";
}

TYPED_TEST(SfppMinStiffnessGeneric, AcousticMatchesTheOracle) {
  expect_acoustic_oracle<PointData::Integrand, TypeParam>("acoustic integrand");
  expect_acoustic_oracle<PointData::Registers, TypeParam>("acoustic registers");
}

TYPED_TEST(SfppMinStiffnessGeneric, AcousticRigidBodyIsAtTheRoundoffFloor) {
  EXPECT_LT(rigid_body_ratio_on<RegisterOffset>([](auto& k) {
              launch<Acoustic, PointData::Integrand, kTestTE, TypeParam>(k);
            }),
            kRigidBound);
  EXPECT_LT(rigid_body_ratio_on<RegisterOffset>([](auto& k) {
              launch<Acoustic, PointData::Registers, kTestTE, TypeParam>(k);
            }),
            kRigidBound);
}

#if defined(TENSOR_OPS_ENABLE_CUTE)
// --- CuTe: every tile width, and the hand-enumerated controls ---------------

template <typename Physics, PointData Data, int TE>
void expect_cute_tile(const char* name) {
  auto k = make_linear_case<RegisterOffset>(/*ix_fastest=*/true);
  launch<Physics, Data, TE, CutePolicy>(k);
  const std::string label = std::string(name) + " TE=" + std::to_string(TE);
  if constexpr (Physics::components == 3)
    expect_matches_oracle(compare_to_oracle(k), label);
  else
    expect_matches_oracle(compare_to_acoustic_oracle(k).error, label);
}

template <typename Physics, PointData Data>
void expect_cute_every_tile(const char* name) {
  expect_cute_tile<Physics, Data, 1>(name);
  expect_cute_tile<Physics, Data, 2>(name);
  expect_cute_tile<Physics, Data, 4>(name);
  if constexpr (NGLL == 5) expect_cute_tile<Physics, Data, 8>(name);
}

TEST(SfppMinStiffnessGenericCute, MatchesTheOracleAtEveryTE) {
  expect_cute_every_tile<Elastic, PointData::Integrand>("elastic integrand");
  expect_cute_every_tile<Elastic, PointData::Registers>("elastic registers");
  expect_cute_every_tile<Acoustic, PointData::Integrand>("acoustic integrand");
  expect_cute_every_tile<Acoustic, PointData::Registers>("acoustic registers");
}

TEST(SfppMinStiffnessGenericCute, SimpleMatchesTheOracle) {
  auto       k    = make_linear_case<RegisterOffset>(/*ix_fastest=*/true);
  const auto args = make_args(k, make_gll_views());
  simple::stiffness<kTestTE>(args, make_hprimewgll(args.hprime, args.weights),
                             CutePolicy{});
  expect_matches_oracle(compare_to_oracle(k), "simple");
}

// Same graph shape, same register point data: only the weight placement
// (hprimewgll and transverse weights in simple, the point weight in the
// generic integrand) and the atomic order differ.
TEST(SfppMinStiffnessGenericCute, RegistersMatchesSimple) {
  auto       ks   = make_linear_case<RegisterOffset>(/*ix_fastest=*/true);
  const auto args = make_args(ks, make_gll_views());
  simple::stiffness<kTestTE>(args, make_hprimewgll(args.hprime, args.weights),
                             CutePolicy{});
  fetch(ks);

  auto kg = make_linear_case<RegisterOffset>(/*ix_fastest=*/true);
  launch<Elastic, PointData::Registers, kTestTE, CutePolicy>(kg);
  fetch(kg);
  const double relative = relative_difference(ks, kg);
  std::printf("[ INFO     ] generic registers vs simple: relative = %.3e\n",
              relative);
  EXPECT_LT(relative, 1e-5);
}

// kernel_new's row-of-e graph with the point data in the integrand and the
// gather and scatter through the index slot: the generic integrand's control.
TEST(SfppMinStiffnessGenericCute, IntegrandMatchesKernelNew) {
  auto kn = make_linear_case<RegisterOffset>(/*ix_fastest=*/true);
  new_stiffness<kTestTE, RowOfEMmas<kTestTE>, GatherMode::IndexSlot,
                ScatterMode::IndexSlot>(make_args(kn, make_gll_views()),
                                        CutePolicy{});
  fetch(kn);

  auto kg = make_linear_case<RegisterOffset>(/*ix_fastest=*/true);
  launch<Elastic, PointData::Integrand, kTestTE, CutePolicy>(kg);
  fetch(kg);
  const double relative = relative_difference(kn, kg);
  std::printf("[ INFO     ] generic integrand vs kernel_new: relative = %.3e\n",
              relative);
  EXPECT_LT(relative, 1e-5);
}

// The row-of-e MMAs run one thread per point of an element (NGLL^3) whatever
// the tile width.
TEST(SfppMinStiffnessGenericCute, FootprintFitsAndIsPooled) {
  auto       k    = make_case<RegisterOffset>(/*ix_fastest=*/true);
  const auto args = make_args(k, make_gll_views());
  const auto integrand =
      generic::footprint<Elastic, PointData::Integrand, kTestTE>(args,
                                                                 CutePolicy{});
  const auto registers =
      generic::footprint<Elastic, PointData::Registers, kTestTE>(args,
                                                                 CutePolicy{});
  std::printf(
      "[ INFO     ] cute TE=%d: integrand pooled %zu B (unpooled %zu), "
      "registers pooled %zu B (unpooled %zu), block %d\n",
      kTestTE, integrand.pooled, integrand.unpooled, registers.pooled,
      registers.unpooled, integrand.threads);
  EXPECT_LE(integrand.pooled, integrand.unpooled);
  EXPECT_LE(registers.pooled, registers.unpooled);
  EXPECT_EQ(integrand.threads, NGLL * NGLL * NGLL);
  EXPECT_EQ(registers.threads, NGLL * NGLL * NGLL);
}
#endif

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  const int rc = RUN_ALL_TESTS();
  Kokkos::finalize();
  return rc;
}
