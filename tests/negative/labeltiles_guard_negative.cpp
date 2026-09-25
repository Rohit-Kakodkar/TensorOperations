// ===========================================================================
// labeltiles_guard_negative.cpp — tile maps that MUST NOT COMPILE.
//
// Both guards protect against a SILENT wrong answer rather than a crash. A
// duplicate label makes the lookup depend on entry order, and a label absent
// from the map has no extent to give -- defaulting one would quietly change the
// iteration space of every node carrying it. Neither shows up as a failure at
// runtime; they show up as different numbers.
//
// Asserting the predicates in an ordinary test would prove they are correct. It
// would NOT prove TileFromLabels consults them, which is the failure this file
// exists to catch. So each case instantiates the real lookup on a map it
// forbids, and ctest asserts the build fails WITH THAT GUARD'S OWN DIAGNOSTIC.
//
// Selected by -DLABELTILES_NEG_CASE=<n>; exactly one case per target.
//   1  a map naming the same label twice
//   2  a node carrying a label the map does not mention
//   3  a contraction whose contracted label is gridded (LabelTile)
//   0  the control: a well-formed map, which must COMPILE.
// ===========================================================================
#include <TensorOperations/LabelTiles.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/NodeHandle.hpp>

#include <Kokkos_Core.hpp>

using namespace TensorOperations;

#ifndef LABELTILES_NEG_CASE
#error "define LABELTILES_NEG_CASE"
#endif

namespace {

template <int32_t... M>
using Modes = std::integer_sequence<int32_t, M...>;

constexpr int kN = 5, kTE = 8;

}  // namespace

int main() {
#if LABELTILES_NEG_CASE == 0
  // CONTROL: every label the node carries is in the map, each exactly once.
  using Map  = LabelTiles<LabelTile<'e', kTE>, LabelTile<'i', kN>>;
  using Tile = tile_from_labels_t<Map, Modes<'e', 'i'>>;
  return static_cast<int>(Tile::rank == 2);

#elif LABELTILES_NEG_CASE == 1
  // 'i' twice. Which extent wins would depend on entry order.
  using Map  = LabelTiles<LabelTile<'e', kTE>, LabelTile<'i', kN>,
                          LabelTile<'i', kN + 1>>;
  using Tile = tile_from_labels_t<Map, Modes<'e', 'i'>>;
  return static_cast<int>(Tile::rank == 2);

#elif LABELTILES_NEG_CASE == 2
  // The node carries 'j', which the map never mentions.
  using Map  = LabelTiles<LabelTile<'e', kTE>, LabelTile<'i', kN>>;
  using Tile = tile_from_labels_t<Map, Modes<'e', 'j', 'i'>>;
  return static_cast<int>(Tile::rank == 3);

#elif LABELTILES_NEG_CASE == 3
  // 'a' is summed over but gridded: each block would own a partial sum.
  using ES = Kokkos::DefaultExecutionSpace;
  using Map =
      LabelTiles<LabelWhole<'q', kN>, LabelTile<'a', 1>, LabelTile<'e', kTE>>;
  using View = Kokkos::View<float**, Kokkos::LayoutRight, ES>;
  View h("h", kN, kN), u("u", kTE, kN);
  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, sh] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'a'>(h))));
  auto [g2, su] =
      g1.add(make_stage_node(make_input_node(make_handle<'e', 'a'>(u))));
  auto g3 = g2.add(make_contraction_node<'q', 'e'>(sh, su));
  return static_cast<int>(sizeof(g3) > 0);

#else
#error "unknown LABELTILES_NEG_CASE"
#endif
}
