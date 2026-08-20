#pragma once
#include <TensorOperations/LabelTiles.hpp>
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

template <typename Node, typename OpModes,
          typename Tag = typename Node::node_tag>
struct lg_canon_modes_of {
  using type = OpModes;
};
template <typename Node, typename OpModes>
struct lg_canon_modes_of<Node, OpModes, ContractionTag> {
  using type = gather_modes_seq_t<OpModes, typename Node::permC_seq>;
};

template <typename LevelsT, std::size_t NumStages, std::size_t S,
          typename OpModes, bool IsStage = (S < NumStages)>
struct lg_slot_canon_modes {
  using type = OpModes;
};
template <typename LevelsT, std::size_t NumStages, std::size_t S,
          typename OpModes>
struct lg_slot_canon_modes<LevelsT, NumStages, S, OpModes, false> {
  using type = typename lg_canon_modes_of<
      typename lg_slot_member_node<LevelsT, NumStages, S>::type, OpModes>::type;
};

template <typename Member, typename Tag = typename Member::node_tag>
struct lg_member_decl_modes {
  using type = typename Member::modes_seq;
};
template <typename Member>
struct lg_member_decl_modes<Member, ContractionTag> {
  using type =
      gather_modes_seq_t<typename Member::modes_seq,
                         inverse_perm_seq_t<typename Member::permC_seq>>;
};

template <typename Member, typename Tag = typename Member::node_tag>
struct lg_member_decl_tile {
  using type = member_out_tile_t<Member>;
};
template <typename Member>
struct lg_member_decl_tile<Member, ContractionTag> {
  using canon_layout = decltype(make_tile_layout(
      std::declval<member_out_tile_t<Member>>(), LayoutRight{}));
  using type = TileFromPerm<canon_layout,
                            inverse_perm_seq_t<typename Member::permC_seq>>;
};

template <typename Member, typename Tag = typename Member::node_tag>
struct lg_member_decl_shape {
  static auto get(const Member& m) { return m.shape(); }
};
template <typename Member>
struct lg_member_decl_shape<Member, ContractionTag> {
  static auto get(const Member& m) {
    return scatter_index(m.shape(), typename Member::permC_seq{});
  }
};

template <typename LevelsT, std::size_t NumStages, std::size_t S,
          typename OpModes, typename TargetModes, typename Store, typename Team>
KOKKOS_FUNCTION auto lg_read_slot(const Store& store, const Team& team) {
  using Canon =
      typename lg_slot_canon_modes<LevelsT, NumStages, S, OpModes>::type;
  static_assert(same_label_set_v<Canon, TargetModes>,
                "level graph: an operand's labels must be a permutation of the "
                "labels it is read as. The read permutation is DERIVED from "
                "labels, so a mismatched label set has no well-defined order");
  using Perm = label_perm_seq_t<TargetModes, Canon>;
  if constexpr (is_identity_v<Perm>)
    return make_value_evaluator(make_interm_node(store.template get<S>()),
                                team);
  else
    return make_value_evaluator(
        make_interm_node(reorder_view(store.template get<S>(), Perm{})), team);
}

// The pool assignment, as the SlotPools type the store wants: one entry per
// slot, in slot order. Named here rather than inline at the two call sites so
// sizing and carving cannot be handed different plans.
template <typename LevelsT, std::size_t NumStages, typename RootsSeq,
          typename SlotsSeq>
struct lg_pools_of;
template <typename LevelsT, std::size_t NumStages, typename RootsSeq,
          std::size_t... Ss>
struct lg_pools_of<LevelsT, NumStages, RootsSeq, std::index_sequence<Ss...>> {
  using type = SlotPools<lg_slot_pool_v<LevelsT, NumStages, RootsSeq, Ss>...>;
};
template <typename LevelsT, std::size_t NumStages, typename RootsSeq>
using lg_pools_t = typename lg_pools_of<
    LevelsT, NumStages, RootsSeq,
    std::make_index_sequence<lg_num_slots_v<LevelsT, NumStages>>>::type;

template <typename V, typename ES, typename LevelsT, std::size_t NumStages,
          typename RootsSeq, typename Team, typename StageTilesT,
          std::size_t... Ss, std::size_t... Ls>
KOKKOS_FUNCTION auto lg_carve(const Team& team, const StageTilesT& stage_tiles,
                              std::index_sequence<Ss...>,
                              std::index_sequence<Ls...>) {
  return carve_pooled_arena_slot_store<
      V, ES, lg_pools_t<LevelsT, NumStages, RootsSeq>>(
      team, stage_tiles.template get<Ss>()...,
      lg_slot_tile_t<LevelsT, NumStages, NumStages + Ls>{}...);
}

template <typename V, typename ES, typename LevelsT, std::size_t NumStages,
          typename RootsSeq, typename StageTilesT, std::size_t... Ss,
          std::size_t... Ls>
std::size_t lg_scratch_bytes(const StageTilesT& stage_tiles,
                             std::index_sequence<Ss...>,
                             std::index_sequence<Ls...>) {
  return pooled_arena_slot_store_bytes<
      V, ES, lg_pools_t<LevelsT, NumStages, RootsSeq>>(
      stage_tiles.template get<Ss>()...,
      lg_slot_tile_t<LevelsT, NumStages, NumStages + Ls>{}...);
}

// The same store WITHOUT pooling: one buffer per slot. Not what the launch
// requests -- it is the honest denominator for "what did pooling buy", and both
// are answerable on the host before anything runs, because scratch is the
// number that decides whether a tile size is viable at all.
template <typename V, typename ES, typename LevelsT, std::size_t NumStages,
          typename StageTilesT, std::size_t... Ss, std::size_t... Ls>
std::size_t lg_unpooled_scratch_bytes(const StageTilesT& stage_tiles,
                                      std::index_sequence<Ss...>,
                                      std::index_sequence<Ls...>) {
  return arena_slot_store_bytes<V, ES>(
      stage_tiles.template get<Ss>()...,
      lg_slot_tile_t<LevelsT, NumStages, NumStages + Ls>{}...);
}

template <typename V, typename ES, typename GridModes, std::size_t RootR,
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
  using Gather    = dag_gather_seq_t<typename StageNode::modes_seq, GridModes>;

  const auto idx = dag_node_index<StageNode::Rank, RootR>(grid_idx, Gather{});
  SEval      sev(stages.template get<S>(), stage_tiles.template get<S>(),
                 store.template get<S>(), team);
  sev(idx);
}

template <typename V, typename ES, typename GridModes, std::size_t RootR,
          typename StagesT, typename StageTilesT, typename Store, typename Team,
          std::size_t... Ss>
KOKKOS_FUNCTION void lg_run_stages(const StagesT&                   stages,
                                   const StageTilesT&               stage_tiles,
                                   const Store&                     store,
                                   const Kokkos::Array<int, RootR>& grid_idx,
                                   const Team&                      team,
                                   std::index_sequence<Ss...>) {
  (lg_stage_one<V, ES, GridModes, RootR, Ss>(stages, stage_tiles, store,
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

  using AModes = typename Node::node_a_type::modes_seq;
  using BModes = typename Node::node_b_type::modes_seq;

  auto a = lg_read_slot<LevelsT, NumStages, SlotA, AModes, AModes>(store, team);
  auto b = lg_read_slot<LevelsT, NumStages, SlotB, BModes, BModes>(store, team);
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
          typename GridModes, std::size_t RootR, std::size_t L, std::size_t M,
          typename Store, typename Team, std::size_t... Ks, std::size_t... Os>
KOKKOS_FUNCTION auto lg_make_combine_member_impl(
    const LevelsT& levels, const Store& store,
    const Kokkos::Array<int, RootR>& grid_idx, const Team& team,
    std::index_sequence<Ks...>, std::index_sequence<Os...>) {
  using Node                 = tuple_element_t<M, tuple_element_t<L, LevelsT>>;
  constexpr std::size_t Base = lg_member_base_v<LevelsT, NumStages, L, M>;

  using Gather   = dag_gather_seq_t<typename Node::modes_seq, GridModes>;
  using OutTile  = member_out_tile_t<Node>;
  const auto idx = dag_node_index<Node::Rank, RootR>(grid_idx, Gather{});

  Kokkos::Array<int, Node::Rank> origin{};
  for (int d = 0; d < Node::Rank; ++d) origin[d] = idx[d] * OutTile::extent(d);
  using OutNode = decltype(make_interm_node(store.template get<Base>()));
  Kokkos::Array<OutNode, static_cast<std::size_t>(Node::NumOut)> outs{
      make_interm_node(store.template get<Base + Os>())...};
  auto ops =
      make_combine_operands(
          lg_read_slot<LevelsT, NumStages,
                       tuple_element_t<Ks, typename Node::ops_tuple_t>::SlotIdx,
                       typename tuple_element_t<
                           Ks, typename Node::ops_tuple_t>::modes_seq,
                       typename Node::modes_seq>(store, team)...,
          outs)
          .at(origin);
  return make_evaluator<TeamPolicyTag2<ES>>(
      levels.template get<L>().template get<M>(), ops, team);
}

template <typename V, typename ES, typename LevelsT, std::size_t NumStages,
          typename GridModes, std::size_t RootR, std::size_t L, std::size_t M,
          typename Store, typename Team>
KOKKOS_FUNCTION auto lg_make_combine_member(
    const LevelsT& levels, const Store& store,
    const Kokkos::Array<int, RootR>& grid_idx, const Team& team) {
  using Node = tuple_element_t<M, tuple_element_t<L, LevelsT>>;
  return lg_make_combine_member_impl<V, ES, LevelsT, NumStages, GridModes,
                                     RootR, L, M>(
      levels, store, grid_idx, team,
      std::make_index_sequence<static_cast<std::size_t>(Node::NumOps)>{},
      std::make_index_sequence<static_cast<std::size_t>(Node::NumOut)>{});
}

template <typename EvalsT, typename Coord, std::size_t... Ms>
KOKKOS_FUNCTION void lg_store_coord(const EvalsT& evs, const Coord& coord,
                                    std::index_sequence<Ms...>) {
  (evs.template get<Ms>()(coord), ...);
}

template <typename V, typename ES, typename LevelsT, std::size_t NumStages,
          typename GridModes, std::size_t RootR, std::size_t L, typename Store,
          typename Team, std::size_t... Ms>
KOKKOS_FUNCTION void lg_run_combine_level(
    const LevelsT& levels, const Store& store,
    const Kokkos::Array<int, RootR>& grid_idx, const Team& team,
    std::index_sequence<Ms...>) {
  constexpr std::size_t Base0 = lg_member_base_v<LevelsT, NumStages, L, 0>;

  auto evs =
      DeviceTuple<decltype(lg_make_combine_member<V, ES, LevelsT, NumStages,
                                                  GridModes, RootR, L, Ms>(
          levels, store, grid_idx, team))...>{
          lg_make_combine_member<V, ES, LevelsT, NumStages, GridModes, RootR, L,
                                 Ms>(levels, store, grid_idx, team)...};

  const auto out0 = store.template get<Base0>();
  team_for_each_coord(team, out0, [=](auto coord) {
    lg_store_coord(evs, coord, std::index_sequence<Ms...>{});
  });
  team.team_barrier();
}

template <typename V, typename ES, typename LevelsT, std::size_t NumStages,
          typename GridModes, std::size_t RootR, std::size_t L, typename Store,
          typename Team>
KOKKOS_FUNCTION void lg_run_level(const LevelsT& levels, const Store& store,
                                  const Kokkos::Array<int, RootR>& grid_idx,
                                  const Team&                      team) {
  using LevelT             = tuple_element_t<L, LevelsT>;
  constexpr std::size_t NM = tuple_size_v<LevelT>;
  if constexpr (lg_all_contraction_v<LevelT>) {
    lg_run_contraction_level<V, ES, LevelsT, NumStages, L>(
        levels, store, team, std::make_index_sequence<NM>{});
  } else {
    lg_run_combine_level<V, ES, LevelsT, NumStages, GridModes, RootR, L>(
        levels, store, grid_idx, team, std::make_index_sequence<NM>{});
  }
}

template <typename V, typename ES, typename LevelsT, std::size_t NumStages,
          typename GridModes, std::size_t RootR, typename Store, typename Team,
          std::size_t... Ls>
KOKKOS_FUNCTION void lg_run_all_levels(
    const LevelsT& levels, const Store& store,
    const Kokkos::Array<int, RootR>& grid_idx, const Team& team,
    std::index_sequence<Ls...>) {
  (lg_run_level<V, ES, LevelsT, NumStages, GridModes, RootR, Ls>(
       levels, store, grid_idx, team),
   ...);
}

template <typename V, typename ES, typename LevelsT, std::size_t NumStages,
          typename GridModes, std::size_t RootR, std::size_t R, typename Store,
          typename Team, typename ViewT>
KOKKOS_FUNCTION void lg_store_root(const LevelsT& levels, const Store& store,
                                   const Kokkos::Array<int, RootR>& grid_idx,
                                   const Team& team, const ViewT& view) {
  constexpr std::size_t L = lg_slot_level_v<LevelsT, NumStages, R>;
  constexpr std::size_t M = lg_slot_member_v<LevelsT, NumStages, R>;
  using Node              = tuple_element_t<M, tuple_element_t<L, LevelsT>>;
  using Gather = dag_gather_seq_t<typename Node::modes_seq, GridModes>;

  const auto idx   = dag_node_index<Node::Rank, RootR>(grid_idx, Gather{});
  auto       seval = make_evaluator<TeamPolicyTag<ES>>(
      make_interm_node(store.template get<R>()),
      typename lg_member_decl_tile<Node>::type{});
  seval(team, idx, view, output_perm_seq<Node>());
}

template <typename V, typename ES, typename LevelsT, std::size_t NumStages,
          typename GridModes, std::size_t RootR, typename Store, typename Team,
          typename ViewArr, std::size_t... Rs>
KOKKOS_FUNCTION void lg_store_roots(const LevelsT& levels, const Store& store,
                                    const Kokkos::Array<int, RootR>& grid_idx,
                                    const Team& team, const ViewArr& views,
                                    std::index_sequence<Rs...>) {
  int i = 0;
  ((lg_store_root<V, ES, LevelsT, NumStages, GridModes, RootR, Rs>(
       levels, store, grid_idx, team, views[i++])),
   ...);
}

// The grid's MODES: the map's labels filtered to those this graph actually
// carries, in map order.
//
// Not simply "every label in the map" -- a map may legitimately be shared by
// graphs that use different subsets of it, and a label no input carries has no
// extent to give. Taking the map wholesale makes such a label's tile count zero
// and collapses the league to zero teams, which is a silent wrong answer rather
// than an error. (That is not hypothetical; it is what the first version of
// this did, and LevelGraphDeclaredOrder.CombineFnSeesGlobalCoordinate caught
// it.)
template <typename StagesT, std::size_t... Ss>
constexpr bool lg_label_carried(int32_t l, std::index_sequence<Ss...>) {
  bool found = false;
  ((found =
        found ||
        arr_contains(
            seq_to_array(typename tuple_element_t<Ss, StagesT>::modes_seq{}),
            l)),
   ...);
  return found;
}

template <typename LT, typename StagesT>
constexpr std::size_t lg_grid_rank() {
  constexpr auto all = seq_to_array(label_seq_t<LT>{});
  using stage_seq    = std::make_index_sequence<tuple_size_v<StagesT>>;
  std::size_t n      = 0;
  for (std::size_t i = 0; i < all.size(); ++i)
    if (label_gridded_of<LT>(all[i]) &&
        lg_label_carried<StagesT>(all[i], stage_seq{}))
      ++n;
  return n;
}

template <typename LT, typename StagesT>
constexpr auto lg_grid_labels() {
  constexpr auto all = seq_to_array(label_seq_t<LT>{});
  using stage_seq    = std::make_index_sequence<tuple_size_v<StagesT>>;
  std::array<int32_t, lg_grid_rank<LT, StagesT>()> out{};
  std::size_t                                      n = 0;
  for (std::size_t i = 0; i < all.size(); ++i)
    if (label_gridded_of<LT>(all[i]) &&
        lg_label_carried<StagesT>(all[i], stage_seq{}))
      out[n++] = all[i];
  return out;
}

template <typename LT, typename StagesT>
using lg_grid_modes_t = array_to_seq_t<lg_grid_labels<LT, StagesT>()>;

template <typename LT, typename StagesT>
using lg_grid_tile_t =
    typename Impl::TileFromLabels<LT, lg_grid_modes_t<LT, StagesT>>::type;

// The grid's per-label extents, gathered from the stage inputs.
//
// Tiles are compile-time (the map) but extents are not -- they come from the
// views the caller staged. So this walks the stages and records, for each label
// it carries, that input's extent along it. Two inputs disagreeing about a
// label is a graph-level error: the label would have two different tile counts
// and no single grid could index both.
template <typename LT, typename GridModes, std::size_t N, typename StageNode>
void lg_note_extents(std::array<int, N>& ext, std::array<bool, N>& seen,
                     const StageNode& n) {
  constexpr auto grid  = seq_to_array(GridModes{});
  constexpr auto modes = seq_to_array(typename StageNode::modes_seq{});
  const auto     shape = n.shape();
  for (std::size_t d = 0; d < modes.size(); ++d) {
    int slot = -1;
    for (std::size_t g = 0; g < grid.size(); ++g)
      if (grid[g] == modes[d]) slot = static_cast<int>(g);
    if (slot < 0) {
      // Declared LabelWhole, so it is not a grid mode and every team indexes 0
      // along it. If the tensor is actually longer than the tile, that is a
      // SILENT wrong answer -- every team would compute the first tile -- so
      // this is checked unconditionally rather than under assert, which a
      // release build would drop.
      if (shape[d] > label_tile_of<LT>(modes[d]))
        Kokkos::abort(
            "level graph: a label declared LabelWhole is longer than its tile, "
            "so it needs to be gridded -- declare it LabelTile instead");
      continue;
    }
    if (seen[static_cast<std::size_t>(slot)])
      assert(ext[static_cast<std::size_t>(slot)] == shape[d] &&
             "grid: two inputs disagree on a label's extent");
    ext[static_cast<std::size_t>(slot)]  = shape[d];
    seen[static_cast<std::size_t>(slot)] = true;
  }
}

template <typename LT, typename StagesT, std::size_t... Ss>
Kokkos::Array<int, lg_grid_rank<LT, StagesT>()> lg_grid_shape(
    const StagesT& stages, std::index_sequence<Ss...>) {
  constexpr std::size_t N = lg_grid_rank<LT, StagesT>();
  using GridModes         = lg_grid_modes_t<LT, StagesT>;
  std::array<int, N>  ext{};
  std::array<bool, N> seen{};
  (lg_note_extents<LT, GridModes, N>(ext, seen, stages.template get<Ss>()),
   ...);
  Kokkos::Array<int, N> out{};
  for (std::size_t d = 0; d < N; ++d) {
    assert(seen[d] && "grid: a grid mode has no extent");
    out[d] = ext[d];
  }
  return out;
}

// League size: the product of every label's tile count. A label whose extent
// equals its tile contributes exactly 1, which is how a mode that is not really
// gridded costs nothing but a factor of one.
template <typename GridTile, std::size_t N>
int lg_league_size(const Kokkos::Array<int, N>& shape) {
  int total = 1;
  for (std::size_t d = 0; d < N; ++d) {
    const int t = GridTile::extent(static_cast<int>(d));
    total *= (shape[d] + t - 1) / t;
  }
  return total;
}

template <typename V, typename ES, typename LT, typename StagesT,
          typename StageTilesT, typename LevelsT, std::size_t NumStages,
          typename RootsSeq, typename... ViewTs>
int lg_execute(const StagesT& stages, const StageTilesT& stage_tiles,
               const LevelsT& levels, std::size_t bytes, int team_size,
               RootsSeq roots, const ViewTs&... views) {
  using member_t            = team_member_t<ES>;
  constexpr std::size_t NL  = tuple_size_v<LevelsT>;
  constexpr std::size_t NLS = lg_num_slots_v<LevelsT, NumStages> - NumStages;
  // The grid is the LABEL SET, not a designated node: every label the map
  // knows is a grid mode, and a label whose extent equals its tile just has one
  // tile. Nothing has to be staged last, and reordering the stages cannot
  // silently change which modes are iterated.
  using GridModes = lg_grid_modes_t<LT, StagesT>;
  static_assert(
      GridModes::size() > 0,
      "level graph: at least one label must be blocked (LabelTile) -- "
      "with every label declared LabelWhole the whole problem is one "
      "team, which is a tile map that forgot to block an axis");
  using GridTile      = lg_grid_tile_t<LT, StagesT>;
  constexpr int RootR = static_cast<int>(GridModes::size());

  using stage_seq      = std::make_index_sequence<NumStages>;
  using level_slot_seq = std::make_index_sequence<NLS>;
  using level_seq      = std::make_index_sequence<NL>;

  const StagesT     sd = stages;
  const StageTilesT gt = stage_tiles;
  const LevelsT     ld = levels;

  const auto grid_shape = lg_grid_shape<LT>(stages, stage_seq{});
  const int  wk         = lg_league_size<GridTile>(grid_shape);

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
            static_cast<int>(team.league_rank()), grid_shape, GridTile{});

        auto store = lg_carve<V, ES, LevelsT, NumStages, RootsSeq>(
            team, gt, stage_seq{}, level_slot_seq{});

        lg_run_stages<V, ES, GridModes, RootR>(sd, gt, store, grid_idx, team,
                                               stage_seq{});
        lg_run_all_levels<V, ES, LevelsT, NumStages, GridModes, RootR>(
            ld, store, grid_idx, team, level_seq{});
        lg_store_roots<V, ES, LevelsT, NumStages, GridModes, RootR>(
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

  using roots_seq = std::index_sequence<Roots...>;

  // Pooled (what the launch requests) against un-pooled. The ratio of the two
  // is what liveness bought, and both are printable before anything runs.
  std::size_t scratch_bytes() const {
    return graph.template scratch_bytes<roots_seq>();
  }
  std::size_t slot_bytes() const { return graph.slot_bytes(); }

  static constexpr std::size_t num_pools =
      Impl::lg_pool_count_v<typename Graph::levels_type, Graph::num_stages,
                            roots_seq>;

  template <typename ES, TensorLike... Ts>
  int execute(const TeamPolicyTag2<ES>&, const Ts&... views) const {
    return graph.template launch<ES, Roots...>(team, views...);
  }
};

template <typename ValueType, typename ExecSpace, typename LabelTilesT,
          typename StagesT, typename StageTilesT, typename LevelsT>
struct LevelGraph {
  using label_tiles_type                  = LabelTilesT;
  using levels_type                       = LevelsT;
  static constexpr std::size_t num_stages = tuple_size_v<StagesT>;
  static constexpr std::size_t num_levels = tuple_size_v<LevelsT>;

  StagesT     stages;
  StageTilesT stage_tiles;
  LevelsT     levels;

  template <typename Node>
  auto stage(const Node& input_node) const {
    static_assert(num_levels == 0,
                  "level graph: stage() every input before adding any level -- "
                  "a level reads stage slots, so the stage set must be closed "
                  "before the first level is added");
    // The tile is no longer the caller's to choose: it follows from the labels
    // this input carries and the graph's one tile extent per label. Every
    // DOWNSTREAM tile then agrees with the map for free, because a member's
    // output tile is derived from its operands and that derivation is asserted
    // equivalent to the map in tests/test_label_tiles.cpp -- so the induction
    // runs from the stages outward.
    using Tile = tile_from_labels_t<LabelTilesT, typename Node::modes_seq>;
    constexpr std::size_t Idx  = num_stages;
    auto                  node = make_stage_node(input_node);
    using NewStages            = decltype(tuple_append(stages, node));
    using NewStageTiles        = decltype(tuple_append(stage_tiles, Tile{}));

    auto handle = make_slot_node_seq<Idx, typename Node::modes_seq>(
        SlotView<ValueType, ExecSpace, Tile>{}, input_node.shape());

    return std::make_tuple(
        LevelGraph<ValueType, ExecSpace, LabelTilesT, NewStages, NewStageTiles,
                   LevelsT>{tuple_append(stages, node),
                            tuple_append(stage_tiles, Tile{}), levels},
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

  // Root-dependent, and has to be: a designated output is read after every
  // level has run, so it outlives the whole graph and cannot share a pool with
  // anything. Which slots are roots therefore changes the plan.
  template <typename RootsSeq>
  std::size_t scratch_bytes() const {
    return Impl::lg_scratch_bytes<ValueType, ExecSpace, LevelsT, num_stages,
                                  RootsSeq>(
        stage_tiles, std::make_index_sequence<num_stages>{},
        std::make_index_sequence<Impl::lg_num_slots_v<LevelsT, num_stages> -
                                 num_stages>{});
  }

  // One buffer per slot: what the store would cost with no liveness plan.
  std::size_t slot_bytes() const {
    return Impl::lg_unpooled_scratch_bytes<ValueType, ExecSpace, LevelsT,
                                           num_stages>(
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
           "LevelGraph: a mode is tiled inconsistently with the grid, so its "
           "tile index cannot be gathered. Every blocked label must be tiled "
           "identically wherever it appears, and every label outside the grid "
           "must have exactly one tile.");
    return Impl::lg_execute<ValueType, ExecSpace, LabelTilesT, StagesT,
                            StageTilesT, LevelsT, num_stages>(
        stages, stage_tiles, levels,
        scratch_bytes<std::index_sequence<Roots...>>(), team_size,
        std::index_sequence<Roots...>{}, views...);
  }

 private:
  template <typename Level, std::size_t... Ms>
  auto add_impl(const Level& level, std::index_sequence<Ms...>) const {
    using NewLevels         = decltype(tuple_append(levels, level));
    constexpr std::size_t L = num_levels;
    return std::tuple_cat(
        std::make_tuple(LevelGraph<ValueType, ExecSpace, LabelTilesT, StagesT,
                                   StageTilesT, NewLevels>{
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
    using Tile = typename Impl::lg_member_decl_tile<Member>::type;
    return std::make_tuple(
        make_slot_node_seq<Base + Os,
                           typename Impl::lg_member_decl_modes<Member>::type>(
            SlotView<ValueType, ExecSpace, Tile>{},
            Impl::lg_member_decl_shape<Member>::get(m))...);
  }

  template <std::size_t S>
  bool stage_index_consistent() const {
    using StageNode = tuple_element_t<S, StagesT>;
    using StageTile = tuple_element_t<S, StageTilesT>;
    using Gather =
        Impl::dag_gather_seq_t<typename StageNode::modes_seq,
                               Impl::lg_grid_modes_t<LabelTilesT, StagesT>>;
    constexpr auto g = Impl::seq_to_array(Gather{});
    return Impl::lg_index_ok<StageNode::Rank>(
        StageTile{}, stages.template get<S>().shape(),
        Impl::lg_grid_tile_t<LabelTilesT, StagesT>{},
        Impl::lg_grid_shape<LabelTilesT>(
            stages, std::make_index_sequence<num_stages>{}),
        g);
  }

  template <std::size_t F>
  bool member_index_consistent() const {
    using Member = Impl::lg_flat_member_t<LevelsT, F>;
    using Gather =
        Impl::dag_gather_seq_t<typename Member::modes_seq,
                               Impl::lg_grid_modes_t<LabelTilesT, StagesT>>;
    constexpr auto        g = Impl::seq_to_array(Gather{});
    constexpr std::size_t L = Impl::lg_flat_level_of<LevelsT>(F);
    constexpr std::size_t M = Impl::lg_flat_member_of<LevelsT>(F);
    return Impl::lg_index_ok<Member::Rank>(
        member_out_tile_t<Member>{},
        levels.template get<L>().template get<M>().shape(),
        Impl::lg_grid_tile_t<LabelTilesT, StagesT>{},
        Impl::lg_grid_shape<LabelTilesT>(
            stages, std::make_index_sequence<num_stages>{}),
        g);
  }

  template <std::size_t... Ss, std::size_t... Fs>
  bool index_consistent_impl(std::index_sequence<Ss...>,
                             std::index_sequence<Fs...>) const {
    return (stage_index_consistent<Ss>() && ...) &&
           (member_index_consistent<Fs>() && ...);
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
  return LevelGraph<ValueType, ExecSpace, LabelTilesT, DeviceTuple<>,
                    DeviceTuple<>, DeviceTuple<>>{{}, {}, {}};
}

}  // namespace TensorOperations
