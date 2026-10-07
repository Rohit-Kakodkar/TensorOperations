#include <TensorOperations/LabelTiles.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>

using namespace TensorOperations;
using ES = Kokkos::DefaultExecutionSpace;

#ifndef INDEX_NEG_CASE
#error "define INDEX_NEG_CASE"
#endif

namespace {

constexpr int kN = 5, kTE = 2;

using Map = LabelTiles<LabelTile<'e', kTE>, LabelWhole<'q', kN>,
                       LabelWhole<'a', kN>, LabelWhole<'b', kN>>;

using IdxV = Kokkos::View<int***, Kokkos::LayoutRight, ES>;
using ValV = Kokkos::View<float***, Kokkos::LayoutRight, ES>;
using HV   = Kokkos::View<float**, Kokkos::LayoutRight, ES>;

struct Id3 {
  KOKKOS_FUNCTION float operator()(int, int, int, float v) const { return v; }
};

[[maybe_unused]] void build() {
  auto g0 = make_level_graph<float, ES>(Map{});
#if INDEX_NEG_CASE == 0
  auto [g1, ig] = g0.add(make_index_node<'e', 'a', 'b'>(IdxV{}));
  auto [g2, h] =
      g1.add(make_stage_node(make_input_node(make_handle<'q', 'a'>(HV{}))));
  auto [g3, u] = g2.add(
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b'>(ValV{}))));
  auto [g4, c] = g3.add(make_contraction_node<'q', 'e', 'b'>(h, u));
  auto [g5, r] = g4.add(make_combine_node<'e', 'a', 'b'>(u, Id3{}));
  (void)ig;
  (void)c;
  (void)g5.outputs(r);
#elif INDEX_NEG_CASE == 1
  auto [g1, ig] = g0.add(make_index_node<'e', 'a', 'b'>(IdxV{}));
  (void)g1.add(make_combine_node<'e', 'a', 'b'>(ig, Id3{}));
#elif INDEX_NEG_CASE == 2
  auto [g1, ig] = g0.add(make_index_node<'e', 'a', 'b'>(IdxV{}));
  auto [g2, h] =
      g1.add(make_stage_node(make_input_node(make_handle<'q', 'a'>(HV{}))));
  (void)g2.add(make_contraction_node<'q', 'e', 'b'>(h, ig));
#elif INDEX_NEG_CASE == 3
  (void)make_index_node<'e', 'a', 'b'>(ValV{});
#elif INDEX_NEG_CASE == 4
  (void)make_index_node<'e', 'a'>(IdxV{});
#else
#error "INDEX_NEG_CASE must be 0..4"
#endif
}

}  // namespace

int main() { return 0; }
