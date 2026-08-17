#pragma once
#include <TensorOperations/LevelPlan.hpp>

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

namespace TensorOperations {
namespace Impl {

template <typename LevelsT>
constexpr std::size_t lg_flat_level_of(std::size_t f) {
  const auto  c   = lg_level_member_counts<LevelsT>();
  std::size_t acc = 0;
  for (std::size_t l = 0; l < c.size(); ++l) {
    if (f < acc + c[l]) return l;
    acc += c[l];
  }
  return c.size();
}
template <typename LevelsT>
constexpr std::size_t lg_flat_member_of(std::size_t f) {
  const auto  c   = lg_level_member_counts<LevelsT>();
  std::size_t acc = 0;
  for (std::size_t l = 0; l < c.size(); ++l) {
    if (f < acc + c[l]) return f - acc;
    acc += c[l];
  }
  return 0;
}

template <typename LevelsT, std::size_t F>
struct lg_flat_member {
  static constexpr std::size_t L = lg_flat_level_of<LevelsT>(F);
  static constexpr std::size_t M = lg_flat_member_of<LevelsT>(F);
  using type = tuple_element_t<M, tuple_element_t<L, LevelsT>>;
};
template <typename LevelsT, std::size_t F>
using lg_flat_member_t = typename lg_flat_member<LevelsT, F>::type;

template <typename LevelsT, std::size_t NumStages, std::size_t GS>
struct lg_slot_member_node {
  static constexpr std::size_t L = lg_slot_level_v<LevelsT, NumStages, GS>;
  static constexpr std::size_t M = lg_slot_member_v<LevelsT, NumStages, GS>;
  using type = tuple_element_t<M, tuple_element_t<L, LevelsT>>;
};
template <typename LevelsT, std::size_t NumStages, std::size_t GS>
using lg_slot_tile_t = member_out_tile_t<
    typename lg_slot_member_node<LevelsT, NumStages, GS>::type>;

template <typename V, typename ES, typename LevelsT, std::size_t NumStages,
          typename Team, typename StageTilesT, std::size_t... Ss,
          std::size_t... Ls>
KOKKOS_FUNCTION auto lg_carve(const Team& team, const StageTilesT& stage_tiles,
                              std::index_sequence<Ss...>,
                              std::index_sequence<Ls...>) {
  return carve_slot_store<V, ES>(
      team, stage_tiles.template get<Ss>()...,
      lg_slot_tile_t<LevelsT, NumStages, NumStages + Ls>{}...);
}

template <typename V, typename ES, typename LevelsT, std::size_t NumStages,
          typename StageTilesT, std::size_t... Ss, std::size_t... Ls>
std::size_t lg_scratch_bytes(const StageTilesT& stage_tiles,
                             std::index_sequence<Ss...>,
                             std::index_sequence<Ls...>) {
  return slot_store_bytes<V, ES>(
      stage_tiles.template get<Ss>()...,
      lg_slot_tile_t<LevelsT, NumStages, NumStages + Ls>{}...);
}

template <typename V, typename ES, typename GridNode, std::size_t RootR,
          std::size_t S, typename StagesT, typename StageTilesT, typename Store,
          typename Team>
KOKKOS_FUNCTION void lg_stage_one(const StagesT&                   stages,
                                  const StageTilesT&               stage_tiles,
                                  const Store&                     store,
                                  const Kokkos::Array<int, RootR>& grid_idx,
                                  const Team&                      team) {
  using StageNode = tuple_element_t<S, StagesT>;
  using StageTile = tuple_element_t<S, StageTilesT>;
  using SEval     = Evaluator<TeamPolicyTag2<ES>, StageNode, StageTile>;
  using Gather    = dag_gather_seq_t<typename StageNode::modes_seq,
                                     typename GridNode::modes_seq>;

  const auto idx = dag_node_index<StageNode::Rank, RootR>(grid_idx, Gather{});
  SEval      sev(stages.template get<S>(), stage_tiles.template get<S>(),
                 store.template get<S>(), team);
  sev(idx);
}

template <typename V, typename ES, typename GridNode, std::size_t RootR,
          typename StagesT, typename StageTilesT, typename Store, typename Team,
          std::size_t... Ss>
KOKKOS_FUNCTION void lg_run_stages(const StagesT&                   stages,
                                   const StageTilesT&               stage_tiles,
                                   const Store&                     store,
                                   const Kokkos::Array<int, RootR>& grid_idx,
                                   const Team&                      team,
                                   std::index_sequence<Ss...>) {
  (lg_stage_one<V, ES, GridNode, RootR, Ss>(stages, stage_tiles, store,
                                            grid_idx, team),
   ...);
  team.team_barrier();
}

template <typename V, typename ES, typename LevelsT, std::size_t NumStages,
          std::size_t L, std::size_t M, typename Store, typename Team>
KOKKOS_FUNCTION auto lg_make_contraction_member(const LevelsT& levels,
                                                const Store&   store,
                                                const Team&    team) {
  using LevelT                = tuple_element_t<L, LevelsT>;
  using Node                  = tuple_element_t<M, LevelT>;
  constexpr std::size_t SlotA = Node::node_a_type::SlotIdx;
  constexpr std::size_t SlotB = Node::node_b_type::SlotIdx;
  constexpr std::size_t SlotC = lg_member_base_v<LevelsT, NumStages, L, M>;

  auto a =
      make_value_evaluator(make_interm_node(store.template get<SlotA>()), team);
  auto b =
      make_value_evaluator(make_interm_node(store.template get<SlotB>()), team);
  return contract_into(levels.template get<L>().template get<M>(), a, b,
                       make_interm_node(store.template get<SlotC>()), team);
}

template <typename EvalsT, std::size_t... Ms>
KOKKOS_FUNCTION void lg_store_ij(const EvalsT& evs, int i, int j,
                                 std::index_sequence<Ms...>) {
  (evs.template get<Ms>()(i, j), ...);
}

template <typename V, typename ES, typename LevelsT, std::size_t NumStages,
          std::size_t L, typename Store, typename Team, std::size_t... Ms>
KOKKOS_FUNCTION void lg_run_contraction_level(const LevelsT& levels,
                                              const Store&   store,
                                              const Team&    team,
                                              std::index_sequence<Ms...>) {
  using LevelT     = tuple_element_t<L, LevelsT>;
  using M0         = tuple_element_t<0, LevelT>;
  constexpr int SA = lg_member_sa_v<M0>;
  constexpr int SB = lg_member_sb_v<M0>;

  auto evs = DeviceTuple<
      decltype(lg_make_contraction_member<V, ES, LevelsT, NumStages, L, Ms>(
          levels, store, team))...>{
      lg_make_contraction_member<V, ES, LevelsT, NumStages, L, Ms>(
          levels, store, team)...};

  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, SA * SB), [=](int t) {
    lg_store_ij(evs, t / SB, t % SB, std::index_sequence<Ms...>{});
  });
  team.team_barrier();
}

template <typename V, typename ES, typename LevelsT, std::size_t NumStages,
          std::size_t L, std::size_t M, typename Store, typename Team,
          std::size_t... Ks, std::size_t... Os>
KOKKOS_FUNCTION auto lg_make_combine_member_impl(const LevelsT& levels,
                                                 const Store&   store,
                                                 const Team&    team,
                                                 std::index_sequence<Ks...>,
                                                 std::index_sequence<Os...>) {
  using Node                 = tuple_element_t<M, tuple_element_t<L, LevelsT>>;
  constexpr std::size_t Base = lg_member_base_v<LevelsT, NumStages, L, M>;
  using OutNode = decltype(make_interm_node(store.template get<Base>()));
  Kokkos::Array<OutNode, static_cast<std::size_t>(Node::NumOut)> outs{
      make_interm_node(store.template get<Base + Os>())...};
  auto ops = make_combine_operands(
      make_value_evaluator(
          make_interm_node(
              store.template get<
                  tuple_element_t<Ks, typename Node::ops_tuple_t>::SlotIdx>()),
          team)...,
      outs);
  return make_evaluator<TeamPolicyTag2<ES>>(
      levels.template get<L>().template get<M>(), ops, team);
}

template <typename V, typename ES, typename LevelsT, std::size_t NumStages,
          std::size_t L, std::size_t M, typename Store, typename Team>
KOKKOS_FUNCTION auto lg_make_combine_member(const LevelsT& levels,
                                            const Store&   store,
                                            const Team&    team) {
  using Node = tuple_element_t<M, tuple_element_t<L, LevelsT>>;
  return lg_make_combine_member_impl<V, ES, LevelsT, NumStages, L, M>(
      levels, store, team,
      std::make_index_sequence<static_cast<std::size_t>(Node::NumOps)>{},
      std::make_index_sequence<static_cast<std::size_t>(Node::NumOut)>{});
}

template <typename EvalsT, typename Coord, std::size_t... Ms>
KOKKOS_FUNCTION void lg_store_coord(const EvalsT& evs, const Coord& coord,
                                    std::index_sequence<Ms...>) {
  (evs.template get<Ms>()(coord), ...);
}

template <typename V, typename ES, typename LevelsT, std::size_t NumStages,
          std::size_t L, typename Store, typename Team, std::size_t... Ms>
KOKKOS_FUNCTION void lg_run_combine_level(const LevelsT& levels,
                                          const Store& store, const Team& team,
                                          std::index_sequence<Ms...>) {
  constexpr std::size_t Base0 = lg_member_base_v<LevelsT, NumStages, L, 0>;

  auto evs =
      DeviceTuple<decltype(lg_make_combine_member<V, ES, LevelsT, NumStages, L,
                                                  Ms>(levels, store, team))...>{
          lg_make_combine_member<V, ES, LevelsT, NumStages, L, Ms>(
              levels, store, team)...};

  const auto out0 = store.template get<Base0>();
  team_for_each_coord(team, out0, [=](auto coord) {
    lg_store_coord(evs, coord, std::index_sequence<Ms...>{});
  });
  team.team_barrier();
}

template <typename V, typename ES, typename LevelsT, std::size_t NumStages,
          std::size_t L, typename Store, typename Team>
KOKKOS_FUNCTION void lg_run_level(const LevelsT& levels, const Store& store,
                                  const Team& team) {
  using LevelT             = tuple_element_t<L, LevelsT>;
  constexpr std::size_t NM = tuple_size_v<LevelT>;
  if constexpr (lg_all_contraction_v<LevelT>) {
    lg_run_contraction_level<V, ES, LevelsT, NumStages, L>(
        levels, store, team, std::make_index_sequence<NM>{});
  } else {
    lg_run_combine_level<V, ES, LevelsT, NumStages, L>(
        levels, store, team, std::make_index_sequence<NM>{});
  }
}

template <typename V, typename ES, typename LevelsT, std::size_t NumStages,
          typename Store, typename Team, std::size_t... Ls>
KOKKOS_FUNCTION void lg_run_all_levels(const LevelsT& levels,
                                       const Store& store, const Team& team,
                                       std::index_sequence<Ls...>) {
  (lg_run_level<V, ES, LevelsT, NumStages, Ls>(levels, store, team), ...);
}

template <typename V, typename ES, typename LevelsT, std::size_t NumStages,
          typename GridNode, std::size_t RootR, std::size_t R, typename Store,
          typename Team, typename ViewT>
KOKKOS_FUNCTION void lg_store_root(const LevelsT& levels, const Store& store,
                                   const Kokkos::Array<int, RootR>& grid_idx,
                                   const Team& team, const ViewT& view) {
  constexpr std::size_t L = lg_slot_level_v<LevelsT, NumStages, R>;
  constexpr std::size_t M = lg_slot_member_v<LevelsT, NumStages, R>;
  using Node              = tuple_element_t<M, tuple_element_t<L, LevelsT>>;
  using Gather =
      dag_gather_seq_t<typename Node::modes_seq, typename GridNode::modes_seq>;

  const auto idx   = dag_node_index<Node::Rank, RootR>(grid_idx, Gather{});
  auto       seval = make_evaluator<TeamPolicyTag<ES>>(
      make_interm_node(store.template get<R>()), member_out_tile_t<Node>{});
  seval(team, idx, view, output_perm_seq<Node>());
}

template <typename V, typename ES, typename LevelsT, std::size_t NumStages,
          typename GridNode, std::size_t RootR, typename Store, typename Team,
          typename ViewArr, std::size_t... Rs>
KOKKOS_FUNCTION void lg_store_roots(const LevelsT& levels, const Store& store,
                                    const Kokkos::Array<int, RootR>& grid_idx,
                                    const Team& team, const ViewArr& views,
                                    std::index_sequence<Rs...>) {
  int i = 0;
  ((lg_store_root<V, ES, LevelsT, NumStages, GridNode, RootR, Rs>(
       levels, store, grid_idx, team, views[i++])),
   ...);
}

template <typename V, typename ES, typename StagesT, typename StageTilesT,
          typename LevelsT, std::size_t NumStages, typename RootsSeq,
          typename... ViewTs>
int lg_execute(const StagesT& stages, const StageTilesT& stage_tiles,
               const LevelsT& levels, std::size_t bytes, int team_size,
               RootsSeq roots, const ViewTs&... views) {
  using member_t            = team_member_t<ES>;
  constexpr std::size_t NL  = tuple_size_v<LevelsT>;
  constexpr std::size_t NLS = lg_num_slots_v<LevelsT, NumStages> - NumStages;
  using GridNode            = tuple_element_t<NumStages - 1, StagesT>;
  constexpr int RootR       = GridNode::Rank;

  using stage_seq      = std::make_index_sequence<NumStages>;
  using level_slot_seq = std::make_index_sequence<NLS>;
  using level_seq      = std::make_index_sequence<NL>;

  const StagesT     sd = stages;
  const StageTilesT gt = stage_tiles;
  const LevelsT     ld = levels;

  const auto grid_node = stages.template get<NumStages - 1>();
  const auto grid_tile = tuple_element_t<NumStages - 1, StageTilesT>{};
  const int  wk        = work_items(grid_node, grid_tile);

  Kokkos::TeamPolicy<ES> policy =
      team_size > 0 ? Kokkos::TeamPolicy<ES>(wk, team_size)
                    : Kokkos::TeamPolicy<ES>(wk, Kokkos::AUTO);
  policy.set_scratch_size(0, Kokkos::PerTeam(static_cast<int>(bytes)));

  using ViewT = std::tuple_element_t<0, std::tuple<ViewTs...>>;
  static_assert((std::is_same_v<ViewT, ViewTs> && ...),
                "LevelGraph::execute: output views must share one type");
  const Kokkos::Array<ViewT, sizeof...(ViewTs)> varr{views...};

  Kokkos::parallel_for(
      "TensorOperations::execute_level_graph", policy,
      KOKKOS_LAMBDA(const member_t& team) {
        const auto grid_idx = decode_tile_index<RootR>(
            static_cast<int>(team.league_rank()), grid_node.shape(), grid_tile);

        auto store = lg_carve<V, ES, LevelsT, NumStages>(team, gt, stage_seq{},
                                                         level_slot_seq{});

        lg_run_stages<V, ES, GridNode, RootR>(sd, gt, store, grid_idx, team,
                                              stage_seq{});
        lg_run_all_levels<V, ES, LevelsT, NumStages>(ld, store, team,
                                                     level_seq{});
        lg_store_roots<V, ES, LevelsT, NumStages, GridNode, RootR>(
            ld, store, grid_idx, team, varr, roots);
      });
  return wk;
}

template <int Rank, typename NodeTile, typename GridTile, std::size_t GridRank>
bool lg_index_ok(const NodeTile& nt, const Kokkos::Array<int, Rank>& nshape,
                 const GridTile& gt, const Kokkos::Array<int, GridRank>& gshape,
                 const std::array<int, Rank>& g) {
  for (int i = 0; i < Rank; ++i) {
    if (g[i] < 0) {
      if (tile_count_along(nt, i, nshape[i]) != 1) return false;
    } else {
      if (nt.extent(i) != gt.extent(static_cast<std::size_t>(g[i])))
        return false;
      if (nshape[i] != gshape[static_cast<std::size_t>(g[i])]) return false;
    }
  }
  return true;
}

}  // namespace Impl

template <typename Graph, std::size_t... Roots>
struct LevelOutputs {
  Graph graph;
  int   team = -1;

  LevelOutputs team_size(int n) const { return {graph, n}; }

  std::size_t scratch_bytes() const { return graph.scratch_bytes(); }

  template <typename ES, TensorLike... Ts>
  int execute(const TeamPolicyTag2<ES>&, const Ts&... views) const {
    return graph.template launch<ES, Roots...>(team, views...);
  }
};

template <typename ValueType, typename ExecSpace, typename StagesT,
          typename StageTilesT, typename LevelsT>
struct LevelGraph {
  static constexpr std::size_t num_stages = tuple_size_v<StagesT>;
  static constexpr std::size_t num_levels = tuple_size_v<LevelsT>;

  StagesT     stages;
  StageTilesT stage_tiles;
  LevelsT     levels;

  template <typename Node, typename Tile>
  auto stage(const Node& input_node, const Tile&) const {
    static_assert(num_levels == 0,
                  "level graph: stage() every input before adding any level -- "
                  "a level reads stage slots, so the stage set must be closed "
                  "before the first level is added");
    constexpr std::size_t Idx  = num_stages;
    auto                  node = make_stage_node(input_node);
    using NewStages            = decltype(tuple_append(stages, node));
    using NewStageTiles        = decltype(tuple_append(stage_tiles, Tile{}));

    auto handle = make_slot_node_seq<Idx, typename Node::modes_seq>(
        SlotView<ValueType, ExecSpace, Tile>{}, input_node.shape());

    return std::make_tuple(
        LevelGraph<ValueType, ExecSpace, NewStages, NewStageTiles, LevelsT>{
            tuple_append(stages, node), tuple_append(stage_tiles, Tile{}),
            levels},
        handle);
  }

  template <typename... Members>
  auto add(const Members&... members) const {
    auto level = DeviceTuple<Members...>{members...};
    return add_impl(level, std::make_index_sequence<sizeof...(Members)>{});
  }

  template <typename... Handles>
  auto outputs(const Handles&...) const {
    return LevelOutputs<LevelGraph, Handles::SlotIdx...>{*this};
  }

  std::size_t scratch_bytes() const {
    return Impl::lg_scratch_bytes<ValueType, ExecSpace, LevelsT, num_stages>(
        stage_tiles, std::make_index_sequence<num_stages>{},
        std::make_index_sequence<Impl::lg_num_slots_v<LevelsT, num_stages> -
                                 num_stages>{});
  }

  bool index_consistent() const {
    return index_consistent_impl(
        std::make_index_sequence<num_stages>{},
        std::make_index_sequence<Impl::lg_total_members_v<LevelsT>>{});
  }

  template <typename ES, std::size_t... Roots, typename... ViewTs>
  int launch(int team_size, const ViewTs&... views) const {
    static_assert(sizeof...(Roots) == sizeof...(ViewTs),
                  "LevelGraph::execute needs one view per designated output");
    static_assert(std::is_same_v<ES, ExecSpace>,
                  "LevelGraph::execute policy tag must match the graph's "
                  "execution space");
    assert(index_consistent() &&
           "LevelGraph: a mode is tiled inconsistently with the grid node, so "
           "its tile index cannot be gathered. Retile so every shared mode "
           "matches and every unshared mode has one tile.");
    return Impl::lg_execute<ValueType, ExecSpace, StagesT, StageTilesT, LevelsT,
                            num_stages>(
        stages, stage_tiles, levels, scratch_bytes(), team_size,
        std::index_sequence<Roots...>{}, views...);
  }

 private:
  template <typename Level, std::size_t... Ms>
  auto add_impl(const Level& level, std::index_sequence<Ms...>) const {
    using NewLevels         = decltype(tuple_append(levels, level));
    constexpr std::size_t L = num_levels;
    return std::tuple_cat(
        std::make_tuple(
            LevelGraph<ValueType, ExecSpace, StagesT, StageTilesT, NewLevels>{
                stages, stage_tiles, tuple_append(levels, level)}),
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
    constexpr std::size_t Base =
        Impl::lg_member_base_v<NewLevels, num_stages, L, M>;
    using Tile = member_out_tile_t<Member>;
    return std::make_tuple(
        make_slot_node_seq<Base + Os, typename Member::modes_seq>(
            SlotView<ValueType, ExecSpace, Tile>{}, m.shape())...);
  }

  template <std::size_t S>
  bool stage_index_consistent() const {
    using StageNode  = tuple_element_t<S, StagesT>;
    using StageTile  = tuple_element_t<S, StageTilesT>;
    using GridNode   = tuple_element_t<num_stages - 1, StagesT>;
    using Gather     = Impl::dag_gather_seq_t<typename StageNode::modes_seq,
                                              typename GridNode::modes_seq>;
    constexpr auto g = Impl::seq_to_array(Gather{});
    return Impl::lg_index_ok<StageNode::Rank>(
        StageTile{}, stages.template get<S>().shape(),
        tuple_element_t<num_stages - 1, StageTilesT>{},
        stages.template get<num_stages - 1>().shape(), g);
  }

  template <std::size_t F>
  bool member_index_consistent() const {
    using Member   = Impl::lg_flat_member_t<LevelsT, F>;
    using GridNode = tuple_element_t<num_stages - 1, StagesT>;
    using Gather   = Impl::dag_gather_seq_t<typename Member::modes_seq,
                                            typename GridNode::modes_seq>;
    constexpr auto        g = Impl::seq_to_array(Gather{});
    constexpr std::size_t L = Impl::lg_flat_level_of<LevelsT>(F);
    constexpr std::size_t M = Impl::lg_flat_member_of<LevelsT>(F);
    return Impl::lg_index_ok<Member::Rank>(
        member_out_tile_t<Member>{},
        levels.template get<L>().template get<M>().shape(),
        tuple_element_t<num_stages - 1, StageTilesT>{},
        stages.template get<num_stages - 1>().shape(), g);
  }

  template <std::size_t... Ss, std::size_t... Fs>
  bool index_consistent_impl(std::index_sequence<Ss...>,
                             std::index_sequence<Fs...>) const {
    return (stage_index_consistent<Ss>() && ...) &&
           (member_index_consistent<Fs>() && ...);
  }
};

template <typename ValueType,
          typename ExecSpace = Kokkos::DefaultExecutionSpace>
auto make_level_graph() {
  return LevelGraph<ValueType, ExecSpace, DeviceTuple<>, DeviceTuple<>,
                    DeviceTuple<>>{{}, {}, {}};
}

}  // namespace TensorOperations
