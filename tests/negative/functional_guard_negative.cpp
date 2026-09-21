// ===========================================================================
// functional_guard_negative.cpp — functional inputs used where they cannot
// work.
//
// A functional input has no address. It answers at a coordinate, which is
// exactly what a STAGE needs and exactly what nothing else can use: a
// contraction and a combine both read their operands through regroup_view /
// slice, i.e. affine arithmetic over data() and stride(). Handing one straight
// to either would not fail at the factory without these guards -- it would fail
// deep inside evaluator instantiation, in a diagnostic that names layout
// internals and not the mistake.
//
// Asserting the guards in an ordinary test would prove the predicates. It would
// NOT prove the factories consult them, which is the failure this file exists
// to catch. So each case builds the forbidden call and ctest asserts the build
// fails WITH THAT GUARD'S OWN DIAGNOSTIC.
//
// Selected by -DFUNCTIONAL_NEG_CASE=<n>; exactly one case per target.
//   1  a functional input straight into make_contraction_node
//   2  a functional input straight into make_combine_node
//   0  the control: the same functional input, STAGED, which must COMPILE.
// ===========================================================================
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/NodeHandle.hpp>

#include <Kokkos_Core.hpp>

using namespace TensorOperations;

#ifndef FUNCTIONAL_NEG_CASE
#error "define FUNCTIONAL_NEG_CASE"
#endif

namespace {

constexpr int kQ = 4, kA = 5, kB = 3, kTE = 2, kE = 6;

using ES    = Kokkos::DefaultExecutionSpace;
using ViewH = Kokkos::View<float**, Kokkos::LayoutRight, ES>;

using Map = LabelTiles<LabelWhole<'q', kQ>, LabelWhole<'a', kA>,
                       LabelTile<'e', kTE>, LabelWhole<'b', kB>>;

struct Source {
  KOKKOS_INLINE_FUNCTION float operator()(int e, int a, int b) const {
    return 1.0f * e + 0.5f * a + 0.25f * b;
  }
};

struct Passthrough {
  KOKKOS_INLINE_FUNCTION float operator()(int, int, int, float v) const {
    return v;
  }
};

auto source_node() {
  return make_functional_input_node<'e', 'a', 'b'>(
      Kokkos::Array<int, 3>{kE, kA, kB}, Source{});
}

}  // namespace

int main() {
  ViewH Hd("H", kQ, kA);

#if FUNCTIONAL_NEG_CASE == 0
  // CONTROL: staged first, then contracted. This is the supported spelling.
  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, h] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'a'>(Hd))));
  auto [g2, u] = g1.add(make_stage_node(source_node()));
  auto [g3, c] = g2.add(make_contraction_node<'q', 'e', 'b'>(h, u));
  (void)g3;
  return 0;

#elif FUNCTIONAL_NEG_CASE == 1
  // A functional input as a contraction operand: there is no strided operand
  // for the contraction to read.
  auto node = make_contraction_node<'q', 'e', 'b'>(
      make_input_node(make_handle<'q', 'a'>(Hd)), source_node());
  (void)node;
  return 0;

#elif FUNCTIONAL_NEG_CASE == 2
  // Same for a combine operand.
  auto node = make_combine_node<'e', 'a', 'b'>(source_node(), Passthrough{});
  (void)node;
  return 0;

#else
#error "unknown FUNCTIONAL_NEG_CASE"
#endif
}
