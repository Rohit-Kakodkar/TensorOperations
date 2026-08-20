// ===========================================================================
// test_label_tiles.cpp — one tile extent per LABEL, for the whole graph.
//
// The change this file exists to justify is a SUBSTITUTION: a node's tile stops
// being derived from its operands (CanonTileOf) or declared at its call site
// (stage), and becomes a lookup in a graph-wide label -> extent map. That is
// only sound if the lookup returns exactly what the derivation returns, for
// every member shape the level graph can build.
//
// So the load-bearing content here is the EQUIVALENCE asserts: for contraction
// and combine members shaped like the real SEM3D pipeline,
//
//     tile_from_labels_t<Map, canonical labels>  ==  member_out_tile_t<Member>
//
// checked at compile time against the machinery that exists today. If those
// hold, routing tiles through the map cannot change a single extent. Everything
// else in the file is the map's own behaviour: order sensitivity, and the two
// guards.
//
// Compile-time by construction -- the runtime test only pins the SEM3D tiles a
// reader would want to see spelled out.
// ===========================================================================
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LabelTiles.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <type_traits>

using namespace TensorOperations;
using ES = Kokkos::DefaultExecutionSpace;

namespace {

template <int32_t... M>
using Modes = std::integer_sequence<int32_t, M...>;

template <std::size_t I, typename ModesSeq, typename Tile>
using Slot = decltype(make_slot_node_seq<I, ModesSeq>(
    std::declval<SlotView<float, ES, Tile>>(),
    std::declval<Kokkos::Array<int, ModesSeq::size()>>()));

// The SEM3D map: NGLL=5 everywhere, elements blocked by TE.
constexpr int kN = 5, kTE = 8;

using SemMap =
    LabelTiles<LabelTile<'e', kTE>, LabelTile<'k', kN>, LabelTile<'j', kN>,
               LabelTile<'i', kN>, LabelTile<'p', kN>, LabelTile<'r', kN>>;

using TileH = StaticTile<kN, kN>;           // the 5x5 operators
using TileE = StaticTile<kTE, kN, kN, kN>;  // a frame tile, <e,k,j,i>

// --- the map's own behaviour ----------------------------------------------

static_assert(
    std::is_same_v<tile_from_labels_t<SemMap, Modes<'r', 'p'>>, TileH>,
    "H is <r,p>, both 5");
static_assert(std::is_same_v<
                  tile_from_labels_t<SemMap, Modes<'e', 'k', 'j', 'i'>>, TileE>,
              "a frame tensor is <e,k,j,i>");

// ORDER matters, and it has to: the same label set in a different order is a
// different tile. That is exactly what lets one contraction's declared and
// canonical output tiles both come out of this map.
static_assert(
    std::is_same_v<tile_from_labels_t<SemMap, Modes<'i', 'e', 'k', 'j'>>,
                   StaticTile<kN, kTE, kN, kN>>,
    "<i,e,k,j> is not <e,k,j,i>");
static_assert(
    !std::is_same_v<tile_from_labels_t<SemMap, Modes<'i', 'e', 'k', 'j'>>,
                    tile_from_labels_t<SemMap, Modes<'e', 'k', 'j', 'i'>>>);

// A label appearing twice in one node is legal for the MAP (it is the node's
// business, not the map's) and simply yields the extent twice.
static_assert(std::is_same_v<tile_from_labels_t<SemMap, Modes<'p', 'p'>>,
                             StaticTile<kN, kN>>);

// --- THE EQUIVALENCE: contraction ------------------------------------------
//
// The nine SEM3D gradients are three shapes, differing in which axis of the one
// staged u is summed. Each declares <e,k,j,i> but stores CANONICAL order, which
// is freeA-in-A's-order ++ freeB-in-B's-order -- so the declared and canonical
// tiles genuinely differ here, and both must come from the map.

using H_ip   = Slot<0, Modes<'i', 'p'>, TileH>;
using H_jp   = Slot<0, Modes<'j', 'p'>, TileH>;
using H_kp   = Slot<0, Modes<'k', 'p'>, TileH>;
using U_ekjp = Slot<1, Modes<'e', 'k', 'j', 'p'>, TileE>;
using U_ekpi = Slot<1, Modes<'e', 'k', 'p', 'i'>, StaticTile<kTE, kN, kN, kN>>;
using U_epji = Slot<1, Modes<'e', 'p', 'j', 'i'>, StaticTile<kTE, kN, kN, kN>>;

using Gx = decltype(make_contraction_node<'e', 'k', 'j', 'i'>(
    std::declval<H_ip>(), std::declval<U_ekjp>()));
using Ge = decltype(make_contraction_node<'e', 'k', 'j', 'i'>(
    std::declval<H_jp>(), std::declval<U_ekpi>()));
using Gg = decltype(make_contraction_node<'e', 'k', 'j', 'i'>(
    std::declval<H_kp>(), std::declval<U_epji>()));

// The canonical label order is already computable from the operands' labels --
// canonC_modes_seq_t is what permC is defined against -- so the substitution is
// "look the canonical labels up" rather than "derive extents from layouts".
template <typename Node, typename A, typename B>
inline constexpr bool canon_tile_agrees = std::is_same_v<
    tile_from_labels_t<SemMap, Impl::canonC_modes_seq_t<Node::Rank, A, B>>,
    member_out_tile_t<Node>>;

static_assert(canon_tile_agrees<Gx, Modes<'i', 'p'>, Modes<'e', 'k', 'j', 'p'>>,
              "d/d(xi): the map must reproduce CanonTileOf exactly");
static_assert(canon_tile_agrees<Ge, Modes<'j', 'p'>, Modes<'e', 'k', 'p', 'i'>>,
              "d/d(eta)");
static_assert(canon_tile_agrees<Gg, Modes<'k', 'p'>, Modes<'e', 'p', 'j', 'i'>>,
              "d/d(gamma)");

// NON-VACUITY: for these members the canonical and declared tiles are genuinely
// different objects, so an assert that the map reproduces both is saying
// something. If they ever coincide, the equivalence checks above would pass
// while proving only half of what they claim.
static_assert(!std::is_same_v<member_out_tile_t<Gx>,
                              typename Impl::lg_member_decl_tile<Gx>::type>,
              "canonical <i,e,k,j> must differ from declared <e,k,j,i>");
static_assert(
    std::is_same_v<member_out_tile_t<Gx>, StaticTile<kN, kTE, kN, kN>>,
    "canonical order spelled out");

// And the DECLARED-order tile, which is what a consumer reads the slot as.
static_assert(
    std::is_same_v<typename Impl::lg_member_decl_tile<Gx>::type,
                   tile_from_labels_t<SemMap, Modes<'e', 'k', 'j', 'i'>>>,
    "declared order comes from the same map, by the node's own labels");

// --- THE EQUIVALENCE: a staged member --------------------------------------
//
// A stage is the one member kind whose tile cannot be derived from operands, so
// it is carried on the node and resolved from the map by LevelGraph::add. This
// pins that the carried tile is the SAME thing the map would give, which is
// what lets MemberOutTile treat all three member kinds alike.
using View4    = Kokkos::View<float****, Kokkos::LayoutRight, ES>;
using RawStage = decltype(make_stage_node(
    make_input_node(make_handle<'e', 'k', 'j', 'i'>(std::declval<View4>()))));
using Staged   = Impl::lg_resolve_member_t<SemMap, RawStage>;

static_assert(
    std::is_same_v<typename RawStage::tile_type, void>,
    "make_stage_node leaves the tile unresolved -- it cannot know it");
static_assert(
    std::is_same_v<member_out_tile_t<Staged>,
                   tile_from_labels_t<SemMap, Modes<'e', 'k', 'j', 'i'>>>,
    "a resolved stage reports exactly the map's tile");
static_assert(std::is_same_v<member_out_tile_t<Staged>, TileE>,
              "spelled out, so the assert above cannot pass vacuously");

// --- THE EQUIVALENCE: combine ----------------------------------------------

struct Id4 {
  KOKKOS_FUNCTION float operator()(int, int, int, int, float v) const {
    return v;
  }
};

using SlotEKJI = Slot<2, Modes<'e', 'k', 'j', 'i'>, TileE>;

using CmbId = decltype(make_combine_node<'e', 'k', 'j', 'i'>(
    std::declval<SlotEKJI>(), std::declval<Id4>()));
static_assert(
    std::is_same_v<member_out_tile_t<CmbId>,
                   tile_from_labels_t<SemMap, Modes<'e', 'k', 'j', 'i'>>>,
    "an identity-labelled combine keeps its operand's tile");

// A permuted combine, where the operand's tile must be gathered into the
// output's label order -- the case that would catch a lookup done against the
// wrong sequence.
using CmbPerm = decltype(make_combine_node<'k', 'e', 'i', 'j'>(
    std::declval<SlotEKJI>(), std::declval<Id4>()));
static_assert(
    std::is_same_v<member_out_tile_t<CmbPerm>,
                   tile_from_labels_t<SemMap, Modes<'k', 'e', 'i', 'j'>>>,
    "a permuted combine's tile follows its DECLARED labels");
static_assert(
    std::is_same_v<member_out_tile_t<CmbPerm>, StaticTile<kN, kTE, kN, kN>>,
    "spelled out, so the assert above cannot pass vacuously");

}  // namespace

// The map is only worth anything if it is exhaustive over the pipeline it has
// to describe. Naming the SEM3D tiles here means a reader can see the whole
// substitution without reconstructing it from the asserts above.
TEST(LabelTilesTest, ReproducesEverySem3dTile) {
  EXPECT_TRUE((std::is_same_v<tile_from_labels_t<SemMap, Modes<'r', 'p'>>,
                              StaticTile<5, 5>>));
  EXPECT_TRUE((std::is_same_v<tile_from_labels_t<SemMap, Modes<'p', 'r'>>,
                              StaticTile<5, 5>>));
  EXPECT_TRUE(
      (std::is_same_v<tile_from_labels_t<SemMap, Modes<'e', 'k', 'j', 'i'>>,
                      StaticTile<8, 5, 5, 5>>));
  // The canonical storage order of every gradient member.
  EXPECT_TRUE(
      (std::is_same_v<tile_from_labels_t<SemMap, Modes<'i', 'e', 'k', 'j'>>,
                      StaticTile<5, 8, 5, 5>>));
}

TEST(LabelTilesTest, GuardsAcceptTheWellFormedMap) {
  EXPECT_TRUE(Impl::label_tiles_distinct_v<SemMap>);
  EXPECT_EQ((Impl::label_tile_v<SemMap, 'e'>), kTE);
  EXPECT_EQ((Impl::label_tile_v<SemMap, 'p'>), kN);
  // Absent labels report -1 rather than a plausible extent, which is what the
  // TileFromLabels guard turns into a diagnosable error.
  EXPECT_EQ((Impl::label_tile_v<SemMap, 'z'>), -1);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
