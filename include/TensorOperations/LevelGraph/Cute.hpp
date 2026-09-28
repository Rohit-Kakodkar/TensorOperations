#pragma once
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph/Team.hpp>

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

namespace TensorOperations {
namespace Impl {

inline constexpr std::size_t lg_cute_default_smem_bytes = 48 * 1024;

struct LinearTileScheduler {
  static dim3 grid(int league) { return dim3(static_cast<unsigned>(league)); }

  template <std::size_t RootR, typename GridTile>
  __device__ static Kokkos::Array<int, RootR> tile_coord(
      const Kokkos::Array<int, RootR>& shape) {
    return decode_tile_index<static_cast<int>(RootR)>(
        static_cast<int>(blockIdx.x), shape, GridTile{});
  }
};

template <typename LevelT>
inline constexpr bool lg_cute_level_supported_v =
    lg_all_staged_v<LevelT> || lg_all_contraction_v<LevelT>;

template <typename LevelsT, std::size_t... Ls>
constexpr bool lg_cute_supported(std::index_sequence<Ls...>) {
  return (lg_cute_level_supported_v<tuple_element_t<Ls, LevelsT>> && ...);
}

template <typename Operand>
struct lg_cute_array_layout {
  static_assert(
      has_node_tag_v<InputTag, Operand>,
      "CuTe level graph: a staged operand must be a View-backed input node");
  using type = typename std::decay_t<
      decltype(std::declval<Operand>().handle)>::array_layout;
};

template <typename Node, int NumThreads>
struct lg_cute_stage {
  using tile_shape = cute_shape_of_t<member_out_tile_t<Node>>;
  using order      = view_contiguity_t<
      Node::Rank,
      typename lg_cute_array_layout<typename Node::operand_type>::type>;
  using thr_layout = cute_thr_layout_t<tile_shape, order, NumThreads>;
  using part       = CuteThreadPartitioner<thr_layout>;
};

template <typename Tile>
constexpr int lg_cute_mode_fit(int first, int last, int budget) {
  int best = 1, prefix = 1;
  for (int k = first; k < last && prefix <= budget; ++k) {
    const int e = Tile::extent(k);
    for (int d = 1; d <= e; ++d)
      if (e % d == 0 && prefix * d <= budget && prefix * d > best)
        best = prefix * d;
    prefix *= e;
  }
  return best;
}

template <typename Node, int NumThreads>
struct lg_cute_default_mma {
  using V                    = typename Node::value_type;
  using Tile                 = member_out_tile_t<Node>;
  static constexpr int FreeA = Node::node_a_type::Rank - Node::NumContracted;
  static constexpr int TM    = lg_cute_mode_fit<Tile>(0, FreeA, NumThreads);
  static constexpr int TN =
      lg_cute_mode_fit<Tile>(FreeA, Node::Rank, NumThreads / TM);
  using type = decltype(cute::make_tiled_mma(
      cute::UniversalFMA<V, V, V>{},
      cute::Layout<cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::_1>>{}));
};

template <typename Node, int NumThreads>
using lg_cute_mma_t = typename std::conditional_t<
    std::is_same_v<typename Node::mma_type, DefaultMma>,
    lg_cute_default_mma<Node, NumThreads>,
    std::type_identity<typename Node::mma_type>>::type;

template <int NumThreads, typename Node>
__device__ lg_cute_mma_t<Node, NumThreads> lg_cute_mma(const Node& n) {
  if constexpr (std::is_same_v<typename Node::mma_type, DefaultMma>)
    return {};
  else
    return n.mma;
}

template <typename Node, int NumThreads, typename Tag = typename Node::node_tag>
struct lg_cute_producer;

template <typename Node, int NumThreads>
struct lg_cute_producer<Node, NumThreads, StagedTag> {
  using S          = lg_cute_stage<Node, NumThreads>;
  using tile_shape = typename S::tile_shape;
  using part       = typename S::part;

  __device__ static part make(const Node&) {
    return {typename S::thr_layout{}, static_cast<int>(threadIdx.x)};
  }
};

template <typename Node, int NumThreads>
struct lg_cute_producer<Node, NumThreads, ContractionTag> {
  using mma                  = lg_cute_mma_t<Node, NumThreads>;
  static constexpr int FreeA = Node::node_a_type::Rank - Node::NumContracted;
  using tile_shape           = cute_shape_of_t<member_out_tile_t<Node>>;
  using part                 = CuteMmaPartitioner<mma, FreeA>;

  __device__ static part make(const Node& n) {
    return {lg_cute_mma<NumThreads>(n), static_cast<int>(threadIdx.x)};
  }
};

template <typename LevelsT, std::size_t S>
using lg_cute_slot_node_t = typename lg_slot_member_node<LevelsT, S>::type;

template <typename LevelsT, std::size_t S>
__device__ const auto& lg_cute_slot_node(const LevelsT& levels) {
  return levels.template get<lg_slot_level_v<LevelsT, S>>()
      .template get<lg_slot_member_v<LevelsT, S>>();
}

template <std::size_t NS>
struct LgCuteReadRange {
  std::array<std::size_t, NS> first{};
  std::array<std::size_t, NS> last{};
};

template <typename Node, std::size_t NS>
constexpr void lg_cute_note_reads(LgCuteReadRange<NS>& r, std::size_t l) {
  if constexpr (has_node_tag_v<ContractionTag, Node>) {
    const std::size_t reads[] = {Node::node_a_type::SlotIdx,
                                 Node::node_b_type::SlotIdx};
    for (const std::size_t s : reads) {
      if (l < r.first[s]) r.first[s] = l;
      r.last[s] = l;
    }
  }
}

template <typename LevelT, std::size_t NS, std::size_t... Ms>
constexpr void lg_cute_note_level_reads(LgCuteReadRange<NS>& r, std::size_t l,
                                        std::index_sequence<Ms...>) {
  (lg_cute_note_reads<tuple_element_t<Ms, LevelT>, NS>(r, l), ...);
}

template <typename LevelsT, std::size_t... Ls>
constexpr auto lg_cute_read_range(std::index_sequence<Ls...>) {
  constexpr std::size_t NS = lg_num_slots_v<LevelsT>;
  LgCuteReadRange<NS>   r{};
  for (std::size_t s = 0; s < NS; ++s) r.first[s] = sizeof...(Ls);
  (lg_cute_note_level_reads<tuple_element_t<Ls, LevelsT>, NS>(
       r, Ls,
       std::make_index_sequence<tuple_size_v<tuple_element_t<Ls, LevelsT>>>{}),
   ...);
  return r;
}

template <typename LevelsT>
inline constexpr auto lg_cute_read_range_v = lg_cute_read_range<LevelsT>(
    std::make_index_sequence<tuple_size_v<LevelsT>>{});

template <typename LevelsT, std::size_t S>
inline constexpr std::size_t lg_cute_first_reader_v =
    lg_cute_read_range_v<LevelsT>.first[S];

template <typename LevelsT, std::size_t S>
inline constexpr std::size_t lg_cute_last_reader_v =
    lg_cute_read_range_v<LevelsT>.last[S];

template <typename LevelsT, std::size_t S>
inline constexpr bool lg_cute_smem_slot_v =
    lg_cute_first_reader_v<LevelsT, S> < tuple_size_v<LevelsT>;

template <typename LevelsT>
constexpr auto lg_cute_pool_of_slot() {
  constexpr std::size_t       NS = lg_num_slots_v<LevelsT>;
  constexpr std::size_t       NL = tuple_size_v<LevelsT>;
  constexpr auto              r  = lg_cute_read_range_v<LevelsT>;
  std::array<std::size_t, NS> def{}, last{};
  for (std::size_t s = 0; s < NS; ++s) {
    const bool smem = r.first[s] < NL;
    def[s]          = smem ? r.first[s] : NL + 1 + s;
    last[s]         = smem ? r.last[s] : NL + 1 + s;
  }
  return left_edge_colour<NS>(def, last);
}

template <typename LevelsT, std::size_t S>
inline constexpr std::size_t lg_cute_slot_pool_v =
    lg_cute_pool_of_slot<LevelsT>()[S];

template <typename V, typename ES, typename LevelsT, std::size_t... Ss>
constexpr std::array<std::size_t, sizeof...(Ss)> lg_cute_smem_steps(
    std::index_sequence<Ss...>) {
  return {(lg_cute_smem_slot_v<LevelsT, Ss>
               ? slot_arena_step<V, ES>(
                     slot_tile_elems<lg_slot_tile_t<LevelsT, Ss>>())
               : std::size_t{0})...};
}

template <typename V, typename ES, typename LevelsT>
constexpr std::size_t lg_cute_smem_prefix(std::size_t i) {
  constexpr std::size_t NS = lg_num_slots_v<LevelsT>;
  const auto            steps =
      lg_cute_smem_steps<V, ES, LevelsT>(std::make_index_sequence<NS>{});
  const auto pools = lg_cute_pool_of_slot<LevelsT>();

  std::array<std::size_t, NS> pelems{};
  std::size_t                 np = 0;
  for (std::size_t k = 0; k < NS; ++k) {
    if (steps[k] > pelems[pools[k]]) pelems[pools[k]] = steps[k];
    if (steps[k] > 0 && pools[k] + 1 > np) np = pools[k] + 1;
  }
  const std::size_t upto = i >= NS ? np : pools[i];
  std::size_t       off  = 0;
  for (std::size_t p = 0; p < upto; ++p) off += pelems[p];
  return off;
}

template <typename V, typename ES, typename LevelsT>
constexpr std::size_t lg_cute_unpooled_smem_prefix(std::size_t n) {
  constexpr std::size_t NS = lg_num_slots_v<LevelsT>;
  const auto            steps =
      lg_cute_smem_steps<V, ES, LevelsT>(std::make_index_sequence<NS>{});
  std::size_t o = 0;
  for (std::size_t k = 0; k < n && k < NS; ++k) o += steps[k];
  return o;
}

template <typename V, typename ES, typename LevelsT, std::size_t S>
inline constexpr std::size_t lg_cute_smem_offset_v =
    lg_cute_smem_prefix<V, ES, LevelsT>(S);

template <typename V, typename ES, typename LevelsT>
inline constexpr std::size_t lg_cute_smem_elems_v =
    lg_cute_smem_prefix<V, ES, LevelsT>(lg_num_slots_v<LevelsT>);

template <typename V, typename ES, typename LevelsT>
inline constexpr std::size_t lg_cute_unpooled_smem_elems_v =
    lg_cute_unpooled_smem_prefix<V, ES, LevelsT>(lg_num_slots_v<LevelsT>);

template <typename LevelsT, std::size_t L, std::size_t S>
constexpr bool lg_cute_slot_reuses_at() {
  if constexpr (lg_cute_first_reader_v<LevelsT, S> != L) {
    return false;
  } else {
    constexpr std::size_t NS    = lg_num_slots_v<LevelsT>;
    constexpr auto        r     = lg_cute_read_range_v<LevelsT>;
    constexpr auto        pools = lg_cute_pool_of_slot<LevelsT>();
    for (std::size_t t = 0; t < NS; ++t)
      if (t != S && pools[t] == pools[S] && r.first[t] < L && r.last[t] < L)
        return true;
    return false;
  }
}

template <typename LevelsT, std::size_t L, typename Ss>
inline constexpr bool lg_cute_reuses_at_v = false;

template <typename LevelsT, std::size_t L, std::size_t... Ss>
inline constexpr bool
    lg_cute_reuses_at_v<LevelsT, L, std::index_sequence<Ss...>> =
        (lg_cute_slot_reuses_at<LevelsT, L, Ss>() || ...);

template <typename V, typename ES, int NumThreads, typename LevelsT,
          std::size_t S>
__device__ auto lg_cute_slot_smem(V* base) {
  using P = lg_cute_producer<lg_cute_slot_node_t<LevelsT, S>, NumThreads>;
  return cute::make_tensor(
      cute::make_smem_ptr(base + lg_cute_smem_offset_v<V, ES, LevelsT, S>),
      cute::make_layout(typename P::tile_shape{}, cute::LayoutRight{}));
}

template <typename ES, int NumThreads, typename LevelsT, typename GridModes,
          std::size_t RootR, std::size_t L, std::size_t M>
__device__ auto lg_cute_stage_member(
    const LevelsT& levels, const Kokkos::Array<int, RootR>& grid_idx) {
  using Node     = tuple_element_t<M, tuple_element_t<L, LevelsT>>;
  using S        = lg_cute_stage<Node, NumThreads>;
  using Gather   = gather_seq_t<typename Node::modes_seq, GridModes>;
  const auto idx = node_index<Node::Rank, RootR>(grid_idx, Gather{});
  auto       ev  = make_evaluator<CutePolicyTag<ES>>(
      levels.template get<L>().template get<M>(),
      CuteStagedTag<typename S::tile_shape, typename S::thr_layout>{
          {typename S::thr_layout{}, static_cast<int>(threadIdx.x)}});
  return ev(idx);
}

template <typename V, typename ES, int NumThreads, typename LevelsT,
          std::size_t L, std::size_t S, typename Acc>
__device__ void lg_cute_materialize(const LevelsT& levels, const Acc& acc,
                                    V* base) {
  if constexpr (lg_cute_first_reader_v<LevelsT, S> == L) {
    using P = lg_cute_producer<lg_cute_slot_node_t<LevelsT, S>, NumThreads>;
    make_evaluator<CutePolicyTag<ES>>(
        make_cute_interm_node<ES>(
            lg_cute_slot_smem<V, ES, NumThreads, LevelsT, S>(base)),
        CuteFragmentStoreTag<typename P::part>{P::make(
            lg_cute_slot_node<LevelsT, S>(levels))}) = acc.template get<S>();
  }
}

template <typename LevelsT, std::size_t L, typename Ss>
inline constexpr bool lg_cute_copies_at_v = false;

template <typename LevelsT, std::size_t L, std::size_t... Ss>
inline constexpr bool
    lg_cute_copies_at_v<LevelsT, L, std::index_sequence<Ss...>> =
        ((lg_cute_first_reader_v<LevelsT, Ss> == L) || ...);

template <typename V, typename ES, int NumThreads, typename LevelsT,
          typename Op>
__device__ auto lg_cute_operand(V* base) {
  constexpr std::size_t S = Op::SlotIdx;
  using OpModes           = typename Op::modes_seq;
  using Canon = typename lg_slot_canon_modes<LevelsT, S, OpModes>::type;
  static_assert(same_label_set_v<Canon, OpModes>,
                "CuTe level graph: an operand's labels must be a permutation "
                "of the labels it is read as");
  const auto t = lg_cute_slot_smem<V, ES, NumThreads, LevelsT, S>(base);
  return make_cute_value_evaluator<ES>(
      cute::make_tensor(
          t.data(), select_seq(t.layout(), label_perm_seq_t<OpModes, Canon>{})),
      NoHook{});
}

template <typename V, typename ES, int NumThreads, typename LevelsT,
          std::size_t L, std::size_t M>
__device__ auto lg_cute_contract_member(const LevelsT& levels, V* base) {
  using Node       = tuple_element_t<M, tuple_element_t<L, LevelsT>>;
  using Mma        = lg_cute_mma_t<Node, NumThreads>;
  const auto& node = levels.template get<L>().template get<M>();
  const auto  a =
      lg_cute_operand<V, ES, NumThreads, LevelsT, typename Node::node_a_type>(
          base);
  const auto b =
      lg_cute_operand<V, ES, NumThreads, LevelsT, typename Node::node_b_type>(
          base);
  static_assert(
      static_cast<int>(decltype(cute::size(std::declval<Mma>()))::value) <=
          NumThreads,
      "CuTe level graph: a contraction's TiledMMA needs more threads than "
      "the CutePolicyTag's NumThreads");
  return make_evaluator<CutePolicyTag<ES>>(
      node, CuteContractTag<decltype(a), decltype(b), Mma>{
                a, b, lg_cute_mma<NumThreads>(node),
                static_cast<int>(threadIdx.x)})();
}

template <typename V, typename ES, int NumThreads, typename LevelsT,
          typename GridModes, std::size_t RootR, std::size_t L, typename Acc,
          std::size_t... Ss, std::size_t... Ms>
__device__ auto lg_cute_run_level(const LevelsT&                   levels,
                                  const Kokkos::Array<int, RootR>& grid_idx,
                                  const Acc& acc, V* base,
                                  std::index_sequence<Ss...>,
                                  std::index_sequence<Ms...>) {
  if constexpr (lg_all_staged_v<tuple_element_t<L, LevelsT>>) {
    using R =
        DeviceTuple<decltype(lg_cute_stage_member<ES, NumThreads, LevelsT,
                                                  GridModes, RootR, L, Ms>(
            levels, grid_idx))...>;
    return R{
        lg_cute_stage_member<ES, NumThreads, LevelsT, GridModes, RootR, L, Ms>(
            levels, grid_idx)...};
  } else {
    if constexpr (lg_cute_reuses_at_v<LevelsT, L, std::index_sequence<Ss...>>)
      __syncthreads();
    (lg_cute_materialize<V, ES, NumThreads, LevelsT, L, Ss>(levels, acc, base),
     ...);
    if constexpr (lg_cute_copies_at_v<LevelsT, L, std::index_sequence<Ss...>>)
      __syncthreads();
    using R =
        DeviceTuple<decltype(lg_cute_contract_member<V, ES, NumThreads, LevelsT,
                                                     L, Ms>(levels, base))...>;
    return R{lg_cute_contract_member<V, ES, NumThreads, LevelsT, L, Ms>(
        levels, base)...};
  }
}

template <typename Acc, typename Level, std::size_t... As, std::size_t... Bs>
__device__ auto lg_cute_concat(const Acc& acc, const Level& lv,
                               std::index_sequence<As...>,
                               std::index_sequence<Bs...>) {
  using R = DeviceTuple<std::decay_t<decltype(acc.template get<As>())>...,
                        std::decay_t<decltype(lv.template get<Bs>())>...>;
  return R{acc.template get<As>()..., lv.template get<Bs>()...};
}

template <typename V, typename ES, int NumThreads, typename LevelsT,
          typename GridModes, std::size_t RootR, std::size_t L, typename Acc>
__device__ auto lg_cute_run_levels(const LevelsT&                   levels,
                                   const Kokkos::Array<int, RootR>& grid_idx,
                                   const Acc& acc, V* base) {
  if constexpr (L == tuple_size_v<LevelsT>) {
    return acc;
  } else {
    constexpr std::size_t NM = tuple_size_v<tuple_element_t<L, LevelsT>>;
    constexpr std::size_t NA = tuple_size_v<Acc>;
    static_assert(lg_member_base_v<LevelsT, L, 0> == NA,
                  "CuTe level graph: slots must be numbered in level order");
    const auto lv =
        lg_cute_run_level<V, ES, NumThreads, LevelsT, GridModes, RootR, L>(
            levels, grid_idx, acc, base, std::make_index_sequence<NA>{},
            std::make_index_sequence<NM>{});
    return lg_cute_run_levels<V, ES, NumThreads, LevelsT, GridModes, RootR,
                              L + 1>(
        levels, grid_idx,
        lg_cute_concat(acc, lv, std::make_index_sequence<NA>{},
                       std::make_index_sequence<NM>{}),
        base);
  }
}

template <typename ES, int NumThreads, typename LevelsT, typename GridModes,
          std::size_t RootR, std::size_t Rt, typename Slots, typename ViewT>
__device__ void lg_cute_store_root(const LevelsT& levels, const Slots& slots,
                                   const Kokkos::Array<int, RootR>& grid_idx,
                                   const ViewT&                     view) {
  using Node   = lg_cute_slot_node_t<LevelsT, Rt>;
  using P      = lg_cute_producer<Node, NumThreads>;
  using Gather = gather_seq_t<typename Node::modes_seq, GridModes>;

  const auto& f   = slots.template get<Rt>().node();
  const auto  idx = node_index<Node::Rank, RootR>(grid_idx, Gather{});
  const auto  node =
      make_cute_fragment_node<ES, Node::Rank, typename P::tile_shape>(
          f.frag_, f.coords_);
  const auto out =
      make_handle_seq(view, typename lg_member_decl_modes<Node>::type{});
  make_evaluator<CutePolicyTag<ES>>(
      node, CuteFragmentStoreTag<typename P::part>{P::make(
                lg_cute_slot_node<LevelsT, Rt>(levels))})(
      idx, out, output_perm_seq<Node>());
}

template <typename ES, int NumThreads, typename LevelsT, typename GridModes,
          std::size_t RootR, typename Slots, typename ViewArr,
          std::size_t... Rts>
__device__ void lg_cute_store_roots(const LevelsT& levels, const Slots& slots,
                                    const Kokkos::Array<int, RootR>& grid_idx,
                                    const ViewArr&                   views,
                                    std::index_sequence<Rts...>) {
  int i = 0;
  (lg_cute_store_root<ES, NumThreads, LevelsT, GridModes, RootR, Rts>(
       levels, slots, grid_idx, views[i++]),
   ...);
}

template <typename V, typename ES, int NumThreads, typename LevelsT,
          typename GridModes, typename GridTile, typename Scheduler,
          typename RootsSeq, std::size_t RootR, typename ViewArr>
__global__ void lg_cute_kernel(LevelsT levels, Kokkos::Array<int, RootR> shape,
                               ViewArr views) {
  extern __shared__ __align__(16) unsigned char lg_cute_smem[];
  V* const   base     = reinterpret_cast<V*>(lg_cute_smem);
  const auto grid_idx = Scheduler::template tile_coord<RootR, GridTile>(shape);
  const auto slots =
      lg_cute_run_levels<V, ES, NumThreads, LevelsT, GridModes, RootR, 0>(
          levels, grid_idx, DeviceTuple<>{}, base);
  lg_cute_store_roots<ES, NumThreads, LevelsT, GridModes, RootR>(
      levels, slots, grid_idx, views, RootsSeq{});
}

template <typename LT, typename GridModes, std::size_t N>
void lg_cute_check_divisible(const Kokkos::Array<int, N>& shape) {
  constexpr auto grid = seq_to_array(GridModes{});
  for (std::size_t d = 0; d < N; ++d)
    if (shape[d] % label_tile_of<LT>(grid[d]) != 0)
      Kokkos::abort(
          "level graph (CuTe): a gridded label's extent is not a multiple of "
          "its tile; the CuTe backend requires tiles to divide the extents");
}

template <typename LT, typename Node>
void lg_cute_check_whole(const Node& n) {
  constexpr auto modes = seq_to_array(typename Node::modes_seq{});
  const auto     shape = n.shape();
  for (std::size_t d = 0; d < modes.size(); ++d)
    if (!label_gridded_of<LT>(modes[d]) &&
        shape[d] != label_tile_of<LT>(modes[d]))
      Kokkos::abort(
          "level graph (CuTe): a LabelWhole label's extent differs from its "
          "tile; the CuTe backend requires tiles to divide the extents");
}

template <typename LT, typename LevelsT, std::size_t... Ls>
void lg_cute_check_wholes(const LevelsT& levels, std::index_sequence<Ls...>) {
  (
      [&]<std::size_t... Ms>(const auto& lv, std::index_sequence<Ms...>) {
        (lg_cute_check_whole<LT>(lv.template get<Ms>()), ...);
      }(levels.template get<Ls>(),
        std::make_index_sequence<tuple_size_v<tuple_element_t<Ls, LevelsT>>>{}),
      ...);
}

template <typename V, typename ES, typename LevelsT>
std::size_t lg_cute_smem_bytes() {
  return lg_cute_smem_elems_v<V, ES, LevelsT> * sizeof(V);
}

template <typename V, typename ES, typename LevelsT>
std::size_t lg_cute_unpooled_smem_bytes() {
  return lg_cute_unpooled_smem_elems_v<V, ES, LevelsT> * sizeof(V);
}

template <typename V, typename ES, typename LT, typename LevelsT,
          int NumThreads, typename RootsSeq, typename... ViewTs>
int lg_execute_cute(const LevelsT& levels, RootsSeq, const ViewTs&... views) {
  static_assert(lg_cute_supported<LevelsT>(
                    std::make_index_sequence<tuple_size_v<LevelsT>>{}),
                "level graph (CuTe): combine levels are not supported by the "
                "CuTe backend yet");
  using GridModes      = lg_grid_modes_t<LT, LevelsT>;
  using GridTile       = lg_grid_tile_t<LT, LevelsT>;
  using Scheduler      = LinearTileScheduler;
  constexpr auto RootR = GridModes::size();

  const auto grid_shape = lg_grid_shape<LT>(
      levels, std::make_index_sequence<tuple_size_v<LevelsT>>{});
  lg_cute_check_divisible<LT, GridModes>(grid_shape);
  lg_cute_check_wholes<LT>(levels,
                           std::make_index_sequence<tuple_size_v<LevelsT>>{});
  const int         wk    = lg_league_size<GridTile>(grid_shape);
  const auto        varr  = lg_view_array(views...);
  const std::size_t bytes = lg_cute_smem_bytes<V, ES, LevelsT>();

  const auto kernel =
      lg_cute_kernel<V, ES, NumThreads, LevelsT, GridModes, GridTile, Scheduler,
                     RootsSeq, RootR, std::decay_t<decltype(varr)>>;
  if (bytes > lg_cute_default_smem_bytes &&
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(bytes)) != cudaSuccess)
    Kokkos::abort(
        "level graph (CuTe): the graph's scratch exceeds the device's "
        "shared memory per block");

  kernel<<<Scheduler::grid(wk), NumThreads, bytes, ES{}.cuda_stream()>>>(
      levels, grid_shape, varr);
  if (cudaGetLastError() != cudaSuccess)
    Kokkos::abort("level graph (CuTe): kernel launch failed");
  return wk;
}

}  // namespace Impl
}  // namespace TensorOperations
