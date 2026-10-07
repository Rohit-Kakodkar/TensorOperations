#include <TensorOperations/LabelTiles.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/LevelPlan.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>

using namespace TensorOperations;
using ES = Kokkos::DefaultExecutionSpace;

#ifndef GATHER_NEG_CASE
#error "define GATHER_NEG_CASE"
#endif

namespace {

constexpr int kN = 5, kTE = 2;

using Map =
    LabelTiles<LabelTile<'e', kTE>, LabelWhole<'a', kN>, LabelWhole<'b', kN>>;

using IdxV = Kokkos::View<int***, Kokkos::LayoutRight, ES>;
using FltV = Kokkos::View<float***, Kokkos::LayoutRight, ES>;
using Src  = Kokkos::View<float*, ES>;
using Src2 = Kokkos::View<float**, ES>;

struct NotATV {};

[[maybe_unused]] void build() {
  auto g0 = make_level_graph<float, ES>(Map{});
#if GATHER_NEG_CASE == 0
  auto [g1, ig] = g0.add(make_index_node<'e', 'a', 'b'>(IdxV{}));
  auto [g2, u]  = g1.add(make_gather_node<'e', 'a', 'b'>(ig, Src{}));
  auto [g3, w]  = g2.add(make_gather_node<'e', 'a', 'b'>(IdxV{}, Src{}));
  (void)g3.outputs(u, w);
#elif GATHER_NEG_CASE == 1
  auto [g1, ig] = g0.add(make_index_node<'e', 'a', 'b'>(IdxV{}));
  (void)g1.add(
      make_gather_node<'e', 'b', 'a'>(ig.template as<'e', 'b', 'a'>(), Src{}));
  (void)g1.add(make_gather_node<'e', 'b', 'a'>(ig, Src{}));
#elif GATHER_NEG_CASE == 2
  (void)make_gather_node<'e', 'a', 'b'>(IdxV{}, Src2{});
#elif GATHER_NEG_CASE == 3
  (void)make_gather_node<'e', 'a', 'b'>(FltV{}, Src{});
#elif GATHER_NEG_CASE == 4
  auto [g1, ig] = g0.add(make_index_node<'e', 'a', 'b'>(IdxV{}));
  (void)make_gather_node<'e', 'a', 'b'>(ig, Src{}, NotATV{});
#elif GATHER_NEG_CASE == 5
  using IdxSlot = decltype(make_slot_node_seq<
                           0, std::integer_sequence<int32_t, 'e', 'a', 'b'>>(
      std::declval<SlotView<int, ES, StaticTile<kTE, kN, kN>>>(),
      std::declval<Kokkos::Array<int, 3>>()));
  using Raw     = decltype(make_index_node<'e', 'a', 'b'>(IdxV{}));
  using Stg     = typename Impl::lg_resolve_member<Map, Raw>::type;
  using Gth     = typename Impl::lg_resolve_member<
      Map, decltype(make_gather_node<'e', 'a', 'b'>(std::declval<IdxSlot>(),
                                                    Src{}))>::type;
  using Plan = LevelPlan<DeviceTuple<DeviceTuple<Stg, Gth>>>;
  static_assert(Plan::num_members > 0, "forces the guards to instantiate");
#else
#error "GATHER_NEG_CASE must be 0..5"
#endif
}

}  // namespace

int main() { return 0; }
