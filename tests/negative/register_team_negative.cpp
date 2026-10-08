#include <TensorOperations/LabelTiles.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>

using namespace TensorOperations;
using ES = Kokkos::DefaultExecutionSpace;

#ifndef REGISTER_NEG_CASE
#error "define REGISTER_NEG_CASE"
#endif

namespace {

constexpr int kN = 5, kTE = 2;

using Map =
    LabelTiles<LabelTile<'e', kTE>, LabelWhole<'a', kN>, LabelWhole<'b', kN>>;
using ValV = Kokkos::View<float***, Kokkos::LayoutRight, ES>;

struct SomeTV {};

struct Id3 {
  KOKKOS_FUNCTION float operator()(int, int, int, float v) const { return v; }
};

[[maybe_unused]] void build(const ValV& u, const ValV& out) {
  auto g0 = make_level_graph<float, ES>(Map{});
#if REGISTER_NEG_CASE == 0
  auto [g1, s] = g0.add(make_stage_node(
      make_input_node(make_handle<'e', 'a', 'b'>(u)), SomeTV{}));
#elif REGISTER_NEG_CASE == 1 || REGISTER_NEG_CASE == 2
  auto [g1, s] = g0.add(make_register_node(
      make_input_node(make_handle<'e', 'a', 'b'>(u)), SomeTV{}));
#else
#error "REGISTER_NEG_CASE must be 0..2"
#endif
  auto [g2, r]   = g1.add(make_combine_node<'e', 'a', 'b'>(s, Id3{}));
  const auto res = g2.outputs(r);
#if REGISTER_NEG_CASE == 2
  (void)res.scratch_bytes();
#else
  (void)res.execute(TeamPolicyTag<ES>{}, out);
#endif
}

}  // namespace

int main() { return 0; }
