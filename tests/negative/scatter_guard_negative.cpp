#include <TensorOperations/LabelTiles.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/LevelPlan.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>

using namespace TensorOperations;
using ES = Kokkos::DefaultExecutionSpace;

#ifndef SCATTER_NEG_CASE
#error "define SCATTER_NEG_CASE"
#endif

namespace {

constexpr int kN = 5, kTE = 2;

using Map =
    LabelTiles<LabelTile<'e', kTE>, LabelWhole<'a', kN>, LabelWhole<'b', kN>>;

using IdxV = Kokkos::View<int***, Kokkos::LayoutRight, ES>;
using FltV = Kokkos::View<float***, Kokkos::LayoutRight, ES>;
using Dst  = Kokkos::View<float*, ES>;
using CDst = Kokkos::View<const float*, ES>;

auto stage(FltV v) {
  return make_stage_node(make_input_node(make_handle<'e', 'a', 'b'>(v)));
}

[[maybe_unused]] void build() {
  auto g0 = make_level_graph<float, ES>(Map{});
#if SCATTER_NEG_CASE == 0
  auto [g1, ig, u] =
      g0.add(make_index_node<'e', 'a', 'b'>(IdxV{}), stage(FltV{}));
  auto g2 = g1.add(make_scatter_add_node<'e', 'a', 'b'>(ig, Dst{}, u),
                   make_scatter_add_node<'e', 'a', 'b'>(IdxV{}, Dst{}, u));
  (void)g2.outputs();
#elif SCATTER_NEG_CASE == 1
  auto [g1, ig, u] =
      g0.add(make_index_node<'e', 'a', 'b'>(IdxV{}), stage(FltV{}));
  (void)make_scatter_add_node<'e', 'b', 'a'>(ig, Dst{}, u);
#elif SCATTER_NEG_CASE == 2
  auto [g1, u] = g0.add(stage(FltV{}));
  (void)make_scatter_add_node<'e', 'a', 'b'>(IdxV{}, CDst{}, u);
#elif SCATTER_NEG_CASE == 3
  auto [g1, u] = g0.add(stage(FltV{}));
  (void)make_scatter_add_node<'e', 'a', 'b'>(FltV{}, Dst{}, u);
#elif SCATTER_NEG_CASE == 4
  auto [g1, ig, u] =
      g0.add(make_index_node<'e', 'a', 'b'>(IdxV{}), stage(FltV{}));
  auto g2 = g1.add(make_scatter_add_node<'e', 'a', 'b'>(ig, Dst{}, u));
  (void)g2.outputs().execute(CutePolicyTag<>{});
#elif SCATTER_NEG_CASE == 5
  using Val =
      typename Impl::lg_resolve_member<Map, decltype(stage(FltV{}))>::type;
  using ValSlot = decltype(make_slot_node_seq<
                           0, std::integer_sequence<int32_t, 'e', 'a', 'b'>>(
      std::declval<SlotView<float, ES, StaticTile<kTE, kN, kN>>>(),
      std::declval<Kokkos::Array<int, 3>>()));
  using IdxSlot = decltype(make_slot_node_seq<
                           1, std::integer_sequence<int32_t, 'e', 'a', 'b'>>(
      std::declval<SlotView<int, ES, StaticTile<kTE, kN, kN>>>(),
      std::declval<Kokkos::Array<int, 3>>()));
  using Sct     = decltype(make_scatter_add_node<'e', 'a', 'b'>(
      std::declval<IdxSlot>(), Dst{}, std::declval<ValSlot>()));
  using Plan    = LevelPlan<DeviceTuple<DeviceTuple<Val>, DeviceTuple<Sct>>>;
  static_assert(Plan::num_members > 0, "forces the guards to instantiate");
#elif SCATTER_NEG_CASE == 6
  auto [g1, ig] = g0.add(make_index_node<'e', 'a', 'b'>(IdxV{}));
  (void)make_scatter_add_node<'e', 'a', 'b'>(IdxV{}, Dst{}, ig);
#else
#error "SCATTER_NEG_CASE must be 0..6"
#endif
}

}  // namespace

int main() { return 0; }
