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

// On the team backend a register node is read by the combines that consume it,
// at their own coordinate (LevelGraph/Team.hpp, RegisterBoundFn). Cases 0-2
// and 5 must compile; 3 and 4 are the reads that need its tile in memory and
// must not.
namespace {

constexpr int kN = 5, kTE = 2;

using Map  = LabelTiles<LabelTile<'e', kTE>, LabelWhole<'a', kN>,
                        LabelWhole<'b', kN>, LabelWhole<'q', kN>>;
using ValV = Kokkos::View<float***, Kokkos::LayoutRight, ES>;
using OpV  = Kokkos::View<float**, Kokkos::LayoutRight, ES>;

struct SomeTV {};

struct Id3 {
  KOKKOS_FUNCTION float operator()(int, int, int, float v) const { return v; }
};
struct Add3 {
  KOKKOS_FUNCTION float operator()(int, int, int, float a, float b) const {
    return a + b;
  }
};

[[maybe_unused]] void build(const ValV& u, const ValV& w, const OpV& h,
                            const ValV& out) {
  auto g0 = make_level_graph<float, ES>(Map{});
#if REGISTER_NEG_CASE == 0
  // A plain stage node that happens to carry a thread-value layout.
  auto [g1, s] = g0.add(make_stage_node(
      make_input_node(make_handle<'e', 'a', 'b'>(u)), SomeTV{}));
  auto [g2, r] = g1.add(make_combine_node<'e', 'a', 'b'>(s, Id3{}));
  (void)g2.outputs(r).execute(TeamPolicyTag<ES>{}, out);
#elif REGISTER_NEG_CASE == 1 || REGISTER_NEG_CASE == 2
  // A register node read by a combine next to a staged operand.
  auto [g1, s, t] =
      g0.add(make_register_node(make_input_node(make_handle<'e', 'a', 'b'>(u)),
                                SomeTV{}),
             make_stage_node(make_input_node(make_handle<'e', 'a', 'b'>(w))));
  auto [g2, r]   = g1.add(make_combine_node<'e', 'a', 'b'>(s, t, Add3{}));
  const auto res = g2.outputs(r);
#if REGISTER_NEG_CASE == 1
  (void)res.execute(TeamPolicyTag<ES>{}, out);
#else
  (void)res.scratch_bytes();
#endif
#elif REGISTER_NEG_CASE == 3
  // A contraction reading a register node.
  auto [g1, hs, s] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'a'>(h))),
             make_register_node(make_input_node(make_handle<'e', 'a', 'b'>(u)),
                                SomeTV{}));
  auto [g2, c] = g1.add(make_contraction_node<'q', 'e', 'b'>(hs, s));
  (void)g2.outputs(c).execute(TeamPolicyTag<ES>{}, out);
#elif REGISTER_NEG_CASE == 4
  // A register node as a graph output.
  auto [g1, s] = g0.add(make_register_node(
      make_input_node(make_handle<'e', 'a', 'b'>(u)), SomeTV{}));
  (void)g1.outputs(s).execute(TeamPolicyTag<ES>{}, out);
#elif REGISTER_NEG_CASE == 5
  // A combine whose only operand is a register node: no operand left in
  // memory, the output tile drives the loop.
  auto [g1, s] = g0.add(make_register_node(
      make_input_node(make_handle<'e', 'a', 'b'>(u)), SomeTV{}));
  auto [g2, r] = g1.add(make_combine_node<'e', 'a', 'b'>(s, Id3{}));
  (void)g2.outputs(r).execute(TeamPolicyTag<ES>{}, out);
#else
#error "REGISTER_NEG_CASE must be 0..5"
#endif
}

}  // namespace

int main() { return 0; }
