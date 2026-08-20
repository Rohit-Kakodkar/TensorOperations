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

template <typename LevelsT, std::size_t GS>
struct lg_slot_member_node {
  static constexpr std::size_t L = lg_slot_level_v<LevelsT, GS>;
  static constexpr std::size_t M = lg_slot_member_v<LevelsT, GS>;
  using type = tuple_element_t<M, tuple_element_t<L, LevelsT>>;
};
template <typename LevelsT, std::size_t GS>
using lg_slot_tile_t =
    member_out_tile_t<typename lg_slot_member_node<LevelsT, GS>::type>;

// A staged member with its tile resolved from the graph's label map; every
// other member kind passes through untouched.
//
// This is where a stage node stops being half-built. make_stage_node cannot
// know the tile -- it is a property of the graph, not the tensor -- so the node
// arrives with tile_type = void and the graph completes it.
template <typename LT, typename Member,
          typename Tag = typename Member::node_tag>
struct lg_resolve_member {
  using type = Member;
  static const Member& get(const Member& m) { return m; }
};
template <typename LT, typename Member>
struct lg_resolve_member<LT, Member, StagedTag> {
  using type = NodeHandle<StagedTag, typename Member::operand_type,
                          typename Member::modes_seq,
                          tile_from_labels_t<LT, typename Member::modes_seq>>;
  static type get(const Member& m) { return type{m.operand_}; }
};
template <typename LT, typename Member>
using lg_resolve_member_t = typename lg_resolve_member<LT, Member>::type;

template <typename Node, typename OpModes,
          typename Tag = typename Node::node_tag>
struct lg_canon_modes_of {
  using type = OpModes;
};
template <typename Node, typename OpModes>
struct lg_canon_modes_of<Node, OpModes, ContractionTag> {
  using type = gather_modes_seq_t<OpModes, typename Node::permC_seq>;
};

// Every slot is a level member's output, so a slot's canonical labels are just
// its producer's. There used to be a second branch here for stage slots, whose
// labels were the operand's as given; staging being a level removed it.
template <typename LevelsT, std::size_t S, typename OpModes>
struct lg_slot_canon_modes {
  using type =
      typename lg_canon_modes_of<typename lg_slot_member_node<LevelsT, S>::type,
                                 OpModes>::type;
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

template <typename LevelsT, std::size_t S, typename OpModes,
          typename TargetModes, typename Store, typename Team>
KOKKOS_FUNCTION auto lg_read_slot(const Store& store, const Team& team) {
  using Canon = typename lg_slot_canon_modes<LevelsT, S, OpModes>::type;
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
template <typename LevelsT, typename RootsSeq, typename SlotsSeq>
struct lg_pools_of;
template <typename LevelsT, typename RootsSeq, std::size_t... Ss>
struct lg_pools_of<LevelsT, RootsSeq, std::index_sequence<Ss...>> {
  using type = SlotPools<lg_slot_pool_v<LevelsT, RootsSeq, Ss>...>;
};
template <typename LevelsT, typename RootsSeq>
using lg_pools_t = typename lg_pools_of<
    LevelsT, RootsSeq, std::make_index_sequence<lg_num_slots_v<LevelsT>>>::type;

template <typename V, typename ES, typename LevelsT, typename RootsSeq,
          typename Team, std::size_t... Ls>
KOKKOS_FUNCTION auto lg_carve(const Team& team, std::index_sequence<Ls...>) {
  return carve_pooled_arena_slot_store<V, ES, lg_pools_t<LevelsT, RootsSeq>>(
      team, lg_slot_tile_t<LevelsT, Ls>{}...);
}

template <typename V, typename ES, typename LevelsT, typename RootsSeq,
          std::size_t... Ls>
std::size_t lg_scratch_bytes(std::index_sequence<Ls...>) {
  return pooled_arena_slot_store_bytes<V, ES, lg_pools_t<LevelsT, RootsSeq>>(
      lg_slot_tile_t<LevelsT, Ls>{}...);
}

// The same store WITHOUT pooling: one buffer per slot. Not what the launch
// requests -- it is the honest denominator for "what did pooling buy", and both
// are answerable on the host before anything runs, because scratch is the
// number that decides whether a tile size is viable at all.
template <typename V, typename ES, typename LevelsT, std::size_t... Ls>
std::size_t lg_unpooled_scratch_bytes(std::index_sequence<Ls...>) {
  return arena_slot_store_bytes<V, ES>(lg_slot_tile_t<LevelsT, Ls>{}...);
}

template <typename V, typename ES, typename LevelsT, std::size_t L,
          std::size_t M, typename Store, typename Team>
KOKKOS_FUNCTION auto lg_make_contraction_member(const LevelsT& levels,
                                                const Store&   store,
                                                const Team&    team) {
  using LevelT                = tuple_element_t<L, LevelsT>;
  using Node                  = tuple_element_t<M, LevelT>;
  constexpr std::size_t SlotA = Node::node_a_type::SlotIdx;
  constexpr std::size_t SlotB = Node::node_b_type::SlotIdx;
  constexpr std::size_t SlotC = lg_member_base_v<LevelsT, L, M>;

  using AModes = typename Node::node_a_type::modes_seq;
  using BModes = typename Node::node_b_type::modes_seq;

  auto a = lg_read_slot<LevelsT, SlotA, AModes, AModes>(store, team);
  auto b = lg_read_slot<LevelsT, SlotB, BModes, BModes>(store, team);
  return contract_into(levels.template get<L>().template get<M>(), a, b,
                       make_interm_node(store.template get<SlotC>()), team);
}

template <typename EvalsT, std::size_t... Ms>
KOKKOS_FUNCTION void lg_store_ij(const EvalsT& evs, int i, int j,
                                 std::index_sequence<Ms...>) {
  (evs.template get<Ms>()(i, j), ...);
}

template <typename V, typename ES, typename LevelsT, std::size_t L,
          typename Store, typename Team, std::size_t... Ms>
KOKKOS_FUNCTION void lg_run_contraction_level(const LevelsT& levels,
                                              const Store&   store,
                                              const Team&    team,
                                              std::index_sequence<Ms...>) {
  using LevelT     = tuple_element_t<L, LevelsT>;
  using M0         = tuple_element_t<0, LevelT>;
  constexpr int SA = lg_member_sa_v<M0>;
  constexpr int SB = lg_member_sb_v<M0>;

  auto evs =
      DeviceTuple<decltype(lg_make_contraction_member<V, ES, LevelsT, L, Ms>(
          levels, store, team))...>{
          lg_make_contraction_member<V, ES, LevelsT, L, Ms>(levels, store,
                                                            team)...};

  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, SA * SB), [=](int t) {
    lg_store_ij(evs, t / SB, t % SB, std::index_sequence<Ms...>{});
  });
  team.team_barrier();
}

template <typename V, typename ES, typename LevelsT, typename GridModes,
          std::size_t RootR, std::size_t L, std::size_t M, typename Store,
          typename Team, std::size_t... Ks, std::size_t... Os>
KOKKOS_FUNCTION auto lg_make_combine_member_impl(
    const LevelsT& levels, const Store& store,
    const Kokkos::Array<int, RootR>& grid_idx, const Team& team,
    std::index_sequence<Ks...>, std::index_sequence<Os...>) {
  using Node                 = tuple_element_t<M, tuple_element_t<L, LevelsT>>;
  constexpr std::size_t Base = lg_member_base_v<LevelsT, L, M>;

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
          lg_read_slot<LevelsT,
                       tuple_element_t<Ks, typename Node::ops_tuple_t>::SlotIdx,
                       typename tuple_element_t<
                           Ks, typename Node::ops_tuple_t>::modes_seq,
                       typename Node::modes_seq>(store, team)...,
          outs)
          .at(origin);
  return make_evaluator<TeamPolicyTag2<ES>>(
      levels.template get<L>().template get<M>(), ops, team);
}

template <typename V, typename ES, typename LevelsT, typename GridModes,
          std::size_t RootR, std::size_t L, std::size_t M, typename Store,
          typename Team>
KOKKOS_FUNCTION auto lg_make_combine_member(
    const LevelsT& levels, const Store& store,
    const Kokkos::Array<int, RootR>& grid_idx, const Team& team) {
  using Node = tuple_element_t<M, tuple_element_t<L, LevelsT>>;
  return lg_make_combine_member_impl<V, ES, LevelsT, GridModes, RootR, L, M>(
      levels, store, grid_idx, team,
      std::make_index_sequence<static_cast<std::size_t>(Node::NumOps)>{},
      std::make_index_sequence<static_cast<std::size_t>(Node::NumOut)>{});
}

template <typename EvalsT, typename Coord, std::size_t... Ms>
KOKKOS_FUNCTION void lg_store_coord(const EvalsT& evs, const Coord& coord,
                                    std::index_sequence<Ms...>) {
  (evs.template get<Ms>()(coord), ...);
}

template <typename V, typename ES, typename LevelsT, typename GridModes,
          std::size_t RootR, std::size_t L, typename Store, typename Team,
          std::size_t... Ms>
KOKKOS_FUNCTION void lg_run_combine_level(
    const LevelsT& levels, const Store& store,
    const Kokkos::Array<int, RootR>& grid_idx, const Team& team,
    std::index_sequence<Ms...>) {
  constexpr std::size_t Base0 = lg_member_base_v<LevelsT, L, 0>;

  auto evs = DeviceTuple<
      decltype(lg_make_combine_member<V, ES, LevelsT, GridModes, RootR, L, Ms>(
          levels, store, grid_idx, team))...>{
      lg_make_combine_member<V, ES, LevelsT, GridModes, RootR, L, Ms>(
          levels, store, grid_idx, team)...};

  const auto out0 = store.template get<Base0>();
  team_for_each_coord(team, out0, [=](auto coord) {
    lg_store_coord(evs, coord, std::index_sequence<Ms...>{});
  });
  team.team_barrier();
}

// One staged member's SOURCE view: the global subview it copies FROM. This is
// exactly what the staged evaluator builds internally before its own copy loop
// (Evaluator/Team2.hpp), lifted out so a whole level's sources can be built
// before any of them is stored.
template <typename V, typename ES, typename LevelsT, typename GridModes,
          std::size_t RootR, std::size_t L, std::size_t M, typename Team>
KOKKOS_FUNCTION auto lg_stage_src(const LevelsT&                   levels,
                                  const Kokkos::Array<int, RootR>& grid_idx,
                                  const Team&                      team) {
  using Node     = tuple_element_t<M, tuple_element_t<L, LevelsT>>;
  using Gather   = dag_gather_seq_t<typename Node::modes_seq, GridModes>;
  const auto idx = dag_node_index<Node::Rank, RootR>(grid_idx, Gather{});
  return make_evaluator<TeamPolicyTag2<ES>>(
             levels.template get<L>().template get<M>().operand_,
             member_out_tile_t<Node>{}, team)(idx)
      .node()
      .storage_;
}

template <typename SrcsT, typename Store, typename Coord, std::size_t... Bs,
          std::size_t... Ms>
KOKKOS_FUNCTION void lg_copy_coord(const SrcsT& srcs, const Store& store,
                                   Coord coord, std::index_sequence<Ms...>,
                                   std::index_sequence<Bs...>) {
  ((store.template get<Bs>()[coord] = srcs.template get<Ms>()[coord]), ...);
}

// A STAGE level: every member's global -> scratch copy, in ONE TeamVectorRange.
//
// This is the whole reason staging is a level. Run separately, each copy is its
// own range and the SASS comes out L S L S L S ... -- every store waiting on
// the load eight instructions ahead of it, because nvcc does not overlap
// consecutive ranges. Building all the sources FIRST and storing them after
// puts every LDG in flight together (L L L S S S) and turns the largest
// long_scoreboard site in the kernel into one round trip instead of N.
//
// Legal only because lg_layout_space_impl already forced every member of the
// level to share one output tile layout, which is what makes a single range
// able to drive all of them.
template <typename V, typename ES, typename LevelsT, typename GridModes,
          std::size_t RootR, std::size_t L, typename Store, typename Team,
          std::size_t... Ms>
KOKKOS_FUNCTION void lg_run_staged_level(
    const LevelsT& levels, const Store& store,
    const Kokkos::Array<int, RootR>& grid_idx, const Team& team,
    std::index_sequence<Ms...>) {
  const auto srcs =
      DeviceTuple<decltype(lg_stage_src<V, ES, LevelsT, GridModes, RootR, L,
                                        Ms>(levels, grid_idx, team))...>{
          lg_stage_src<V, ES, LevelsT, GridModes, RootR, L, Ms>(
              levels, grid_idx, team)...};

  using bases     = std::index_sequence<lg_member_base_v<LevelsT, L, Ms>...>;
  const auto src0 = srcs.template get<0>();
  team_for_each_coord(team, src0, [=](auto coord) {
    lg_copy_coord(srcs, store, coord, std::index_sequence<Ms...>{}, bases{});
  });
  team.team_barrier();
}

template <typename V, typename ES, typename LevelsT, typename GridModes,
          std::size_t RootR, std::size_t L, typename Store, typename Team>
KOKKOS_FUNCTION void lg_run_level(const LevelsT& levels, const Store& store,
                                  const Kokkos::Array<int, RootR>& grid_idx,
                                  const Team&                      team) {
  using LevelT             = tuple_element_t<L, LevelsT>;
  constexpr std::size_t NM = tuple_size_v<LevelT>;
  if constexpr (lg_all_staged_v<LevelT>) {
    lg_run_staged_level<V, ES, LevelsT, GridModes, RootR, L>(
        levels, store, grid_idx, team, std::make_index_sequence<NM>{});
  } else if constexpr (lg_all_contraction_v<LevelT>) {
    lg_run_contraction_level<V, ES, LevelsT, L>(levels, store, team,
                                                std::make_index_sequence<NM>{});
  } else {
    lg_run_combine_level<V, ES, LevelsT, GridModes, RootR, L>(
        levels, store, grid_idx, team, std::make_index_sequence<NM>{});
  }
}

template <typename V, typename ES, typename LevelsT, typename GridModes,
          std::size_t RootR, typename Store, typename Team, std::size_t... Ls>
KOKKOS_FUNCTION void lg_run_all_levels(
    const LevelsT& levels, const Store& store,
    const Kokkos::Array<int, RootR>& grid_idx, const Team& team,
    std::index_sequence<Ls...>) {
  (lg_run_level<V, ES, LevelsT, GridModes, RootR, Ls>(levels, store, grid_idx,
                                                      team),
   ...);
}

template <typename V, typename ES, typename LevelsT, typename GridModes,
          std::size_t RootR, std::size_t R, typename Store, typename Team,
          typename ViewT>
KOKKOS_FUNCTION void lg_store_root(const LevelsT& levels, const Store& store,
                                   const Kokkos::Array<int, RootR>& grid_idx,
                                   const Team& team, const ViewT& view) {
  constexpr std::size_t L = lg_slot_level_v<LevelsT, R>;
  constexpr std::size_t M = lg_slot_member_v<LevelsT, R>;
  using Node              = tuple_element_t<M, tuple_element_t<L, LevelsT>>;
  using Gather = dag_gather_seq_t<typename Node::modes_seq, GridModes>;

  const auto idx   = dag_node_index<Node::Rank, RootR>(grid_idx, Gather{});
  auto       seval = make_evaluator<TeamPolicyTag<ES>>(
      make_interm_node(store.template get<R>()),
      typename lg_member_decl_tile<Node>::type{});
  seval(team, idx, view, output_perm_seq<Node>());
}

template <typename V, typename ES, typename LevelsT, typename GridModes,
          std::size_t RootR, typename Store, typename Team, typename ViewArr,
          std::size_t... Rs>
KOKKOS_FUNCTION void lg_store_roots(const LevelsT& levels, const Store& store,
                                    const Kokkos::Array<int, RootR>& grid_idx,
                                    const Team& team, const ViewArr& views,
                                    std::index_sequence<Rs...>) {
  int i = 0;
  ((lg_store_root<V, ES, LevelsT, GridModes, RootR, Rs>(levels, store, grid_idx,
                                                        team, views[i++])),
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
template <typename LevelT, std::size_t... Ms>
constexpr bool lg_level_carries(int32_t l, std::index_sequence<Ms...>) {
  bool found = false;
  ((found =
        found ||
        (has_node_tag_v<StagedTag, tuple_element_t<Ms, LevelT>> &&
         arr_contains(
             seq_to_array(typename tuple_element_t<Ms, LevelT>::modes_seq{}),
             l))),
   ...);
  return found;
}

// Extents enter the graph through STAGED members -- they are the only ones that
// read a real tensor -- so the grid's labels and its extents both come from
// them, wherever in the level list they sit.
template <typename LevelsT, std::size_t... Ls>
constexpr bool lg_label_carried(int32_t l, std::index_sequence<Ls...>) {
  bool found = false;
  ((found = found || lg_level_carries<tuple_element_t<Ls, LevelsT>>(
                         l, std::make_index_sequence<
                                tuple_size_v<tuple_element_t<Ls, LevelsT>>>{})),
   ...);
  return found;
}

template <typename LT, typename LevelsT>
constexpr std::size_t lg_grid_rank() {
  constexpr auto all = seq_to_array(label_seq_t<LT>{});
  using stage_seq    = std::make_index_sequence<tuple_size_v<LevelsT>>;
  std::size_t n      = 0;
  for (std::size_t i = 0; i < all.size(); ++i)
    if (label_gridded_of<LT>(all[i]) &&
        lg_label_carried<LevelsT>(all[i], stage_seq{}))
      ++n;
  return n;
}

template <typename LT, typename LevelsT>
constexpr auto lg_grid_labels() {
  constexpr auto all = seq_to_array(label_seq_t<LT>{});
  using stage_seq    = std::make_index_sequence<tuple_size_v<LevelsT>>;
  std::array<int32_t, lg_grid_rank<LT, LevelsT>()> out{};
  std::size_t                                      n = 0;
  for (std::size_t i = 0; i < all.size(); ++i)
    if (label_gridded_of<LT>(all[i]) &&
        lg_label_carried<LevelsT>(all[i], stage_seq{}))
      out[n++] = all[i];
  return out;
}

template <typename LT, typename LevelsT>
using lg_grid_modes_t = array_to_seq_t<lg_grid_labels<LT, LevelsT>()>;

template <typename LT, typename LevelsT>
using lg_grid_tile_t =
    typename Impl::TileFromLabels<LT, lg_grid_modes_t<LT, LevelsT>>::type;

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

template <typename LT, typename GridModes, std::size_t N, typename LevelT,
          std::size_t... Ms>
void lg_note_level_extents(std::array<int, N>& ext, std::array<bool, N>& seen,
                           const LevelT& lv, std::index_sequence<Ms...>) {
  ((void)([&] {
     if constexpr (has_node_tag_v<StagedTag, tuple_element_t<Ms, LevelT>>)
       lg_note_extents<LT, GridModes, N>(ext, seen, lv.template get<Ms>());
   }()),
   ...);
}

template <typename LT, typename LevelsT, std::size_t... Ls>
Kokkos::Array<int, lg_grid_rank<LT, LevelsT>()> lg_grid_shape(
    const LevelsT& levels, std::index_sequence<Ls...>) {
  constexpr std::size_t N = lg_grid_rank<LT, LevelsT>();
  using GridModes         = lg_grid_modes_t<LT, LevelsT>;
  std::array<int, N>  ext{};
  std::array<bool, N> seen{};
  (lg_note_level_extents<LT, GridModes, N>(
       ext, seen, levels.template get<Ls>(),
       std::make_index_sequence<tuple_size_v<tuple_element_t<Ls, LevelsT>>>{}),
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

template <typename V, typename ES, typename LT, typename LevelsT,
          typename RootsSeq, typename... ViewTs>
int lg_execute(const LevelsT& levels, std::size_t bytes, int team_size,
               RootsSeq roots, const ViewTs&... views) {
  using member_t            = team_member_t<ES>;
  constexpr std::size_t NL  = tuple_size_v<LevelsT>;
  constexpr std::size_t NLS = lg_num_slots_v<LevelsT>;
  // The grid is the LABEL SET, not a designated node: every label the map
  // knows is a grid mode, and a label whose extent equals its tile just has one
  // tile. Nothing has to be staged last, and reordering the stages cannot
  // silently change which modes are iterated.
  using GridModes = lg_grid_modes_t<LT, LevelsT>;
  static_assert(
      GridModes::size() > 0,
      "level graph: at least one label must be blocked (LabelTile) -- "
      "with every label declared LabelWhole the whole problem is one "
      "team, which is a tile map that forgot to block an axis");
  using GridTile      = lg_grid_tile_t<LT, LevelsT>;
  constexpr int RootR = static_cast<int>(GridModes::size());

  using level_slot_seq = std::make_index_sequence<NLS>;
  using level_seq      = std::make_index_sequence<NL>;

  const LevelsT ld = levels;

  const auto grid_shape = lg_grid_shape<LT>(levels, level_seq{});
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

        auto store = lg_carve<V, ES, LevelsT, RootsSeq>(team, level_slot_seq{});

        lg_run_all_levels<V, ES, LevelsT, GridModes, RootR>(ld, store, grid_idx,
                                                            team, level_seq{});
        lg_store_roots<V, ES, LevelsT, GridModes, RootR>(ld, store, grid_idx,
                                                         team, varr, roots);
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
      Impl::lg_pool_count_v<typename Graph::levels_type, roots_seq>;

  template <typename ES, TensorLike... Ts>
  int execute(const TeamPolicyTag2<ES>&, const Ts&... views) const {
    return graph.template launch<ES, Roots...>(team, views...);
  }
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
    return Impl::lg_execute<ValueType, ExecSpace, LabelTilesT, LevelsT>(
        levels, scratch_bytes<std::index_sequence<Roots...>>(), team_size,
        std::index_sequence<Roots...>{}, views...);
  }

 private:
  template <typename Level, std::size_t... Ms>
  auto add_impl(const Level& level, std::index_sequence<Ms...>) const {
    using NewLevels         = decltype(tuple_append(levels, level));
    constexpr std::size_t L = num_levels;
    return std::tuple_cat(
        std::make_tuple(
            LevelGraph<ValueType, ExecSpace, LabelTilesT, NewLevels>{
                tuple_append(levels, level)}),
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
        Impl::dag_gather_seq_t<typename Member::modes_seq,
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

  template <std::size_t... Ss, std::size_t... Fs>
  bool index_consistent_impl(std::index_sequence<Ss...>,
                             std::index_sequence<Fs...>) const {
    (void)std::index_sequence<Ss...>{};
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
