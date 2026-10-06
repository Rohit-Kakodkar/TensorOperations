#pragma once
#include <TensorOperations/LevelGraph/Team.hpp>
#if defined(TENSOR_OPS_ENABLE_CUTE)
#include <TensorOperations/LevelGraph/Cute.hpp>
#endif

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace TensorOperations {

namespace Impl {

template <typename LT, typename Member>
constexpr bool lg_contracted_labels_whole() {
  if constexpr (!has_node_tag_v<ContractionTag, Member>) {
    return true;
  } else {
    constexpr auto a = seq_to_array(typename Member::node_a_type::modes_seq{});
    constexpr auto c = seq_to_array(typename Member::modes_seq{});
    for (std::size_t i = 0; i < a.size(); ++i)
      if (!arr_contains(c, a[i]) && label_gridded_of<LT>(a[i])) return false;
    return true;
  }
}

}  // namespace Impl

template <typename Graph, std::size_t... Roots>
struct LevelOutputs {
  Graph graph;
  int   team = -1;

  LevelOutputs team_size(int n) const { return {graph, n}; }

  using roots_seq = std::index_sequence<Roots...>;

  // Pooled (what the launch requests) against un-pooled. The ratio of the two
  // is what liveness bought, and both are printable before anything runs.
  std::size_t scratch_bytes() const {
    return graph.template scratch_bytes<roots_seq>();
  }
  std::size_t slot_bytes() const { return graph.slot_bytes(); }

  static constexpr std::size_t num_pools =
      Impl::lg_pool_count_v<typename Graph::levels_type, roots_seq>;

  template <typename ES, TensorLike... Ts>
  int execute(const TeamPolicyTag<ES>& tag, const Ts&... views) const {
    return graph.template launch<Roots...>(tag, team, views...);
  }

#if defined(TENSOR_OPS_ENABLE_CUTE)
  template <typename ES, int N, TensorLike... Ts>
  int execute(const CutePolicyTag<ES, N>& tag, const Ts&... views) const {
    return graph.template launch<Roots...>(tag, team, views...);
  }
#endif
};

template <typename ValueType, typename ExecSpace, typename LabelTilesT,
          typename LevelsT>
struct LevelGraph {
  using label_tiles_type                  = LabelTilesT;
  using levels_type                       = LevelsT;
  static constexpr std::size_t num_levels = tuple_size_v<LevelsT>;

  LevelsT levels;

  // A staged member arrives with its tile unresolved -- make_stage_node cannot
  // know it -- so this is where the map fills it in. Every other member kind
  // passes through untouched.
  template <typename... Members>
  auto add(const Members&... members) const {
    static_assert(
        (Impl::lg_contracted_labels_whole<LabelTilesT, Members>() && ...),
        "level graph: a contracted label must be LabelWhole -- gridding it "
        "would split the sum across blocks, and nothing combines the partial "
        "sums");
    auto level =
        DeviceTuple<Impl::lg_resolve_member_t<LabelTilesT, Members>...>{
            Impl::lg_resolve_member<LabelTilesT, Members>::get(members)...};
    return add_impl(level, std::make_index_sequence<sizeof...(Members)>{});
  }

  template <typename... Handles>
  auto outputs(const Handles&...) const {
    return LevelOutputs<LevelGraph, Handles::SlotIdx...>{*this};
  }

  // Root-dependent, and has to be: a designated output is read after every
  // level has run, so it outlives the whole graph and cannot share a pool with
  // anything. Which slots are roots therefore changes the plan.
  template <typename RootsSeq>
  std::size_t scratch_bytes() const {
    return Impl::lg_scratch_bytes<ValueType, ExecSpace, LevelsT, RootsSeq>(
        std::make_index_sequence<Impl::lg_num_slots_v<LevelsT>>{});
  }

  // One buffer per slot: what the store would cost with no liveness plan.
  std::size_t slot_bytes() const {
    return Impl::lg_unpooled_scratch_bytes<ValueType, ExecSpace, LevelsT>(
        std::make_index_sequence<Impl::lg_num_slots_v<LevelsT>>{});
  }

  bool index_consistent() const {
    return index_consistent_impl(
        std::index_sequence<>{},
        std::make_index_sequence<Impl::lg_total_members_v<LevelsT>>{});
  }

  template <std::size_t... Roots, typename ES, typename... ViewTs>
  int launch(const TeamPolicyTag<ES>&, int team_size,
             const ViewTs&... views) const {
    check_launch<ES, sizeof...(Roots), sizeof...(ViewTs)>();
    return Impl::lg_execute<ValueType, ExecSpace, LabelTilesT, LevelsT>(
        levels, scratch_bytes<std::index_sequence<Roots...>>(), team_size,
        std::index_sequence<Roots...>{}, views...);
  }

#if defined(TENSOR_OPS_ENABLE_CUTE)
  template <std::size_t... Roots, typename ES, int N, typename... ViewTs>
  int launch(const CutePolicyTag<ES, N>&, int team_size,
             const ViewTs&... views) const {
    check_launch<ES, sizeof...(Roots), sizeof...(ViewTs)>();
    if (team_size > 0 && team_size != N)
      Kokkos::abort(
          "LevelGraph::execute: with the CuTe backend the block size is the "
          "CutePolicyTag's NumThreads; team_size must match it or be unset");
    return Impl::lg_execute_cute<ValueType, ExecSpace, LabelTilesT, LevelsT, N>(
        levels, std::index_sequence<Roots...>{}, views...);
  }
#endif

 private:
  template <typename ES, std::size_t NumRoots, std::size_t NumViews>
  void check_launch() const {
    static_assert(NumRoots == NumViews,
                  "LevelGraph::execute needs one view per designated output");
    static_assert(std::is_same_v<ES, ExecSpace>,
                  "LevelGraph::execute policy tag must match the graph's "
                  "execution space");
    static_assert(
        Impl::lg_grid_modes_t<LabelTilesT, LevelsT>::size() > 0,
        "level graph: at least one label must be blocked (LabelTile) -- "
        "with every label declared LabelWhole the whole problem is one "
        "team, which is a tile map that forgot to block an axis");
    assert(index_consistent() &&
           "LevelGraph: a mode is tiled inconsistently with the grid, so its "
           "tile index cannot be gathered. Every blocked label must be tiled "
           "identically wherever it appears, and every label outside the grid "
           "must have exactly one tile.");
  }

  template <typename Level, std::size_t... Ms>
  auto add_impl(const Level& level, std::index_sequence<Ms...>) const {
    using NewLevels = decltype(tuple_append(levels, level));
    constexpr std::size_t                                    L = num_levels;
    LevelGraph<ValueType, ExecSpace, LabelTilesT, NewLevels> g{
        tuple_append(levels, level)};
    // A level of all sinks contributes zero handles; return the bare graph so
    // the caller writes `auto g = g0.add(...)` instead of destructuring a
    // 1-tuple. A mixed level still returns graph + its non-sink handles.
    constexpr std::size_t NH =
        (0 + ... + Impl::output_arity<tuple_element_t<Ms, Level>>::value);
    if constexpr (NH == 0)
      return g;
    else
      return std::tuple_cat(
          std::make_tuple(g),
          member_handles<NewLevels, L, Ms>(level.template get<Ms>())...);
  }

  template <typename NewLevels, std::size_t L, std::size_t M, typename Member>
  auto member_handles(const Member& m) const {
    return member_handles_impl<NewLevels, L, M>(
        m, std::make_index_sequence<static_cast<std::size_t>(
               Impl::output_arity<Member>::value)>{});
  }

  template <typename NewLevels, std::size_t L, std::size_t M, typename Member,
            std::size_t... Os>
  auto member_handles_impl(const Member& m, std::index_sequence<Os...>) const {
    constexpr std::size_t Base = Impl::lg_member_base_v<NewLevels, L, M>;
    using Tile = typename Impl::lg_member_decl_tile<Member>::type;
    return std::make_tuple(
        make_slot_node_seq<Base + Os,
                           typename Impl::lg_member_decl_modes<Member>::type>(
            SlotView<ValueType, ExecSpace, Tile>{},
            Impl::lg_member_decl_shape<Member>::get(m))...);
  }

  template <std::size_t F>
  bool member_index_consistent() const {
    using Member = Impl::lg_flat_member_t<LevelsT, F>;
    using Gather =
        Impl::gather_seq_t<typename Member::modes_seq,
                           Impl::lg_grid_modes_t<LabelTilesT, LevelsT>>;
    constexpr auto        g = Impl::seq_to_array(Gather{});
    constexpr std::size_t L = Impl::lg_flat_level_of<LevelsT>(F);
    constexpr std::size_t M = Impl::lg_flat_member_of<LevelsT>(F);
    return Impl::lg_index_ok<Member::Rank>(
        member_out_tile_t<Member>{},
        levels.template get<L>().template get<M>().shape(),
        Impl::lg_grid_tile_t<LabelTilesT, LevelsT>{},
        Impl::lg_grid_shape<LabelTilesT>(
            levels, std::make_index_sequence<num_levels>{}),
        g);
  }

  template <std::size_t... Fs>
  bool index_consistent_impl(std::index_sequence<Fs...>) const {
    return (member_index_consistent<Fs>() && ...);
  }
};

// The tile map is the graph's single source of truth for tile extents: one per
// label, supplied once here instead of once per stage() call.
//
// Taken as an ARGUMENT rather than a template parameter so it is deduced, which
// keeps ExecSpace's default reachable and puts the map where a reader looks for
// the graph's configuration. LabelTiles is stateless, so the value is discarded
// once its type has been read -- the same convention stage() used for its tile.
template <typename ValueType,
          typename ExecSpace = Kokkos::DefaultExecutionSpace,
          typename LabelTilesT>
auto make_level_graph(LabelTilesT) {
  return LevelGraph<ValueType, ExecSpace, LabelTilesT, DeviceTuple<>>{{}};
}

}  // namespace TensorOperations
