#pragma once
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph/Team.hpp>

#include <algorithm>
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
    lg_all_staged_v<LevelT> || lg_all_contraction_v<LevelT> ||
    lg_all_combine_v<LevelT>;

template <typename LevelsT, std::size_t... Ls>
constexpr bool lg_cute_supported(std::index_sequence<Ls...>) {
  return (lg_cute_level_supported_v<tuple_element_t<Ls, LevelsT>> && ...);
}

template <typename Operand, int R, typename Tag = typename Operand::node_tag>
struct lg_cute_stage_order {
  static_assert(has_node_tag_v<InputTag, Operand>,
                "CuTe level graph: a staged operand must be a View-backed or "
                "functional input node");
  using type = view_contiguity_t<
      R, typename std::decay_t<
             decltype(std::declval<Operand>().handle)>::array_layout>;
};

template <typename Operand, int R>
struct lg_cute_stage_order<Operand, R, FunctionalTag> {
  using type = order_contiguity_t<R, typename Operand::order_tag>;
};

template <typename Node, int NumThreads>
struct lg_cute_stage {
  using tile_shape = cute_shape_of_t<member_out_tile_t<Node>>;
  using order      = typename lg_cute_stage_order<typename Node::operand_type,
                                                  Node::Rank>::type;
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

template <typename Node>
struct lg_cute_default_mma {
  using V                     = typename Node::value_type;
  using Tile                  = member_out_tile_t<Node>;
  static constexpr int Budget = Node::mma_type::budget;
  static constexpr int FreeA  = Node::node_a_type::Rank - Node::NumContracted;
  static constexpr int TM     = lg_cute_mode_fit<Tile>(0, FreeA, Budget);
  static constexpr int TN =
      lg_cute_mode_fit<Tile>(FreeA, Node::Rank, Budget / TM);
  using type = decltype(cute::make_tiled_mma(
      cute::UniversalFMA<V, V, V>{},
      cute::Layout<cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::_1>>{}));
};

template <typename Node>
using lg_cute_mma_t = typename std::conditional_t<
    is_default_mma_v<typename Node::mma_type>, lg_cute_default_mma<Node>,
    std::type_identity<typename Node::mma_type>>::type;

template <typename Node>
__device__ lg_cute_mma_t<Node> lg_cute_mma(const Node& n) {
  if constexpr (is_default_mma_v<typename Node::mma_type>)
    return {};
  else
    return n.mma;
}

inline constexpr int lg_cute_no_mma_threads = 128;

template <typename Node>
constexpr int lg_cute_member_threads() {
  if constexpr (has_node_tag_v<ContractionTag, Node>)
    return static_cast<int>(
        decltype(cute::size(std::declval<lg_cute_mma_t<Node>>()))::value);
  else
    return 0;
}

template <typename LevelT, std::size_t... Ms>
constexpr int lg_cute_level_threads(std::index_sequence<Ms...>) {
  int n = 0;
  ((n = std::max(n, lg_cute_member_threads<tuple_element_t<Ms, LevelT>>())),
   ...);
  return n;
}

template <typename LevelsT, std::size_t... Ls>
constexpr int lg_cute_graph_threads(std::index_sequence<Ls...>) {
  int n = 0;
  ((n = std::max(n, lg_cute_level_threads<tuple_element_t<Ls, LevelsT>>(
                        std::make_index_sequence<
                            tuple_size_v<tuple_element_t<Ls, LevelsT>>>{}))),
   ...);
  return n > 0 ? n : lg_cute_no_mma_threads;
}

template <typename LevelsT>
inline constexpr int lg_cute_num_threads_v = lg_cute_graph_threads<LevelsT>(
    std::make_index_sequence<tuple_size_v<LevelsT>>{});

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
  using mma                  = lg_cute_mma_t<Node>;
  static constexpr int FreeA = Node::node_a_type::Rank - Node::NumContracted;
  using tile_shape           = cute_shape_of_t<member_out_tile_t<Node>>;
  using part                 = CuteMmaPartitioner<mma, FreeA>;

  __device__ static part make(const Node& n) {
    return {lg_cute_mma(n), static_cast<int>(threadIdx.x)};
  }
};

template <typename LevelsT, std::size_t S>
using lg_cute_slot_node_t = typename lg_slot_member_node<LevelsT, S>::type;

template <typename LevelsT, std::size_t S>
__device__ const auto& lg_cute_slot_node(const LevelsT& levels) {
  return levels.template get<lg_slot_level_v<LevelsT, S>>()
      .template get<lg_slot_member_v<LevelsT, S>>();
}

template <typename LevelsT, std::size_t L, std::size_t M, int N>
struct lg_cute_combine_plan;

template <typename LevelsT, std::size_t S, int N,
          typename Tag = typename lg_cute_slot_node_t<LevelsT, S>::node_tag>
struct lg_cute_slot_producer {
  using P          = lg_cute_producer<lg_cute_slot_node_t<LevelsT, S>, N>;
  using tile_shape = typename P::tile_shape;
  using part       = typename P::part;

  __device__ static part make(const LevelsT& levels) {
    return P::make(lg_cute_slot_node<LevelsT, S>(levels));
  }
};

template <typename LevelsT, std::size_t S, int N>
struct lg_cute_slot_producer<LevelsT, S, N, CombineTag>
    : lg_cute_combine_plan<LevelsT, lg_slot_level_v<LevelsT, S>,
                           lg_slot_member_v<LevelsT, S>, N> {};

template <typename ThrLayout>
struct lg_cute_thread_part {
  using part = CuteThreadPartitioner<ThrLayout>;
};

template <int R, std::size_t... Is>
auto lg_cute_right_order(std::index_sequence<Is...>)
    -> std::integer_sequence<int, (R - 1 - static_cast<int>(Is))...>;

template <typename LevelsT, std::size_t L, std::size_t M, int N>
struct lg_cute_combine_plan {
  using Node   = tuple_element_t<M, tuple_element_t<L, LevelsT>>;
  using CModes = typename Node::modes_seq;
  static constexpr std::size_t NumOps = static_cast<std::size_t>(Node::NumOps);
  using tile_shape = cute_shape_of_t<member_out_tile_t<Node>>;
  using thr_layout =
      cute_thr_layout_t<tile_shape,
                        decltype(lg_cute_right_order<Node::Rank>(
                            std::make_index_sequence<Node::Rank>{})),
                        N>;

  template <std::size_t K>
  using op_t = tuple_element_t<K, typename Node::ops_tuple_t>;
  template <std::size_t K>
  static constexpr std::size_t slot = op_t<K>::SlotIdx;
  template <std::size_t K>
  using canon_t =
      typename lg_slot_canon_modes<LevelsT, slot<K>,
                                   typename op_t<K>::modes_seq>::type;
  template <std::size_t K>
  using op_part_t = typename lg_cute_slot_producer<LevelsT, slot<K>, N>::part;

  template <std::size_t K>
  static constexpr bool eligible() {
    return std::is_same_v<canon_t<K>, CModes> &&
           std::is_same_v<typename op_t<K>::modes_seq, CModes> &&
           std::is_same_v<
               typename lg_cute_slot_producer<LevelsT, slot<K>, N>::tile_shape,
               tile_shape>;
  }

  template <std::size_t... Ks>
  static constexpr std::size_t first_eligible(std::index_sequence<Ks...>) {
    const bool e[] = {eligible<Ks>()..., false};
    for (std::size_t k = 0; k < sizeof...(Ks); ++k)
      if (e[k]) return k;
    return sizeof...(Ks);
  }

  static constexpr std::size_t D =
      first_eligible(std::make_index_sequence<NumOps>{});
  static constexpr bool register_driven = D < NumOps;

  template <std::size_t K>
  static constexpr bool in_register() {
    if constexpr (!register_driven)
      return false;
    else
      return eligible<K>() && std::is_same_v<op_part_t<K>, op_part_t<D>>;
  }

  template <std::size_t K>
  static constexpr bool in_register_v = in_register<K>();

  using part = typename std::conditional_t<
      register_driven,
      lg_cute_slot_producer<LevelsT, slot<(register_driven ? D : 0)>, N>,
      lg_cute_thread_part<thr_layout>>::part;

  __device__ static part make(const LevelsT& levels) {
    if constexpr (register_driven)
      return lg_cute_slot_producer<LevelsT, slot<D>, N>::make(levels);
    else
      return {thr_layout{}, static_cast<int>(threadIdx.x)};
  }
};

template <std::size_t NS>
struct LgCuteReadRange {
  std::array<std::size_t, NS> first{};
  std::array<std::size_t, NS> last{};
};

template <std::size_t NS>
constexpr void lg_cute_note(LgCuteReadRange<NS>& r, std::size_t s,
                            std::size_t l) {
  if (l < r.first[s]) r.first[s] = l;
  r.last[s] = l;
}

template <typename Plan, std::size_t NS, std::size_t... Ks>
constexpr void lg_cute_note_combine(LgCuteReadRange<NS>& r, std::size_t l,
                                    std::index_sequence<Ks...>) {
  ((Plan::template in_register<Ks>()
        ? void()
        : lg_cute_note<NS>(r, Plan::template slot<Ks>, l)),
   ...);
}

template <typename LevelsT, std::size_t L, std::size_t M, int N, std::size_t NS>
constexpr void lg_cute_note_reads(LgCuteReadRange<NS>& r) {
  using Node = tuple_element_t<M, tuple_element_t<L, LevelsT>>;
  if constexpr (has_node_tag_v<ContractionTag, Node>) {
    lg_cute_note<NS>(r, Node::node_a_type::SlotIdx, L);
    lg_cute_note<NS>(r, Node::node_b_type::SlotIdx, L);
  } else if constexpr (has_node_tag_v<CombineTag, Node>) {
    using Plan = lg_cute_combine_plan<LevelsT, L, M, N>;
    lg_cute_note_combine<Plan, NS>(r, L,
                                   std::make_index_sequence<Plan::NumOps>{});
  }
}

template <typename LevelsT, std::size_t L, int N, std::size_t NS,
          std::size_t... Ms>
constexpr void lg_cute_note_level_reads(LgCuteReadRange<NS>& r,
                                        std::index_sequence<Ms...>) {
  (lg_cute_note_reads<LevelsT, L, Ms, N, NS>(r), ...);
}

template <typename LevelsT, int N, std::size_t... Ls>
constexpr auto lg_cute_read_range(std::index_sequence<Ls...>) {
  constexpr std::size_t NS = lg_num_slots_v<LevelsT>;
  LgCuteReadRange<NS>   r{};
  for (std::size_t s = 0; s < NS; ++s) r.first[s] = sizeof...(Ls);
  (lg_cute_note_level_reads<LevelsT, Ls, N, NS>(
       r,
       std::make_index_sequence<tuple_size_v<tuple_element_t<Ls, LevelsT>>>{}),
   ...);
  return r;
}

template <typename LevelsT, int N>
inline constexpr auto lg_cute_read_range_v = lg_cute_read_range<LevelsT, N>(
    std::make_index_sequence<tuple_size_v<LevelsT>>{});

template <typename LevelsT, int N, std::size_t S>
inline constexpr std::size_t lg_cute_first_reader_v =
    lg_cute_read_range_v<LevelsT, N>.first[S];

template <typename LevelsT, int N, std::size_t S>
inline constexpr std::size_t lg_cute_last_reader_v =
    lg_cute_read_range_v<LevelsT, N>.last[S];

template <typename LevelsT, int N, std::size_t S>
inline constexpr bool lg_cute_smem_slot_v =
    lg_cute_first_reader_v<LevelsT, N, S> < tuple_size_v<LevelsT>;

template <typename LevelsT, int N>
constexpr auto lg_cute_pool_of_slot() {
  constexpr std::size_t       NS = lg_num_slots_v<LevelsT>;
  constexpr std::size_t       NL = tuple_size_v<LevelsT>;
  constexpr auto              r  = lg_cute_read_range_v<LevelsT, N>;
  std::array<std::size_t, NS> def{}, last{};
  for (std::size_t s = 0; s < NS; ++s) {
    const bool smem = r.first[s] < NL;
    def[s]          = smem ? r.first[s] : NL + 1 + s;
    last[s]         = smem ? r.last[s] : NL + 1 + s;
  }
  return left_edge_colour<NS>(def, last);
}

template <typename LevelsT, int N, std::size_t S>
inline constexpr std::size_t lg_cute_slot_pool_v =
    lg_cute_pool_of_slot<LevelsT, N>()[S];

template <typename V, typename ES, typename LevelsT, int N, std::size_t... Ss>
constexpr std::array<std::size_t, sizeof...(Ss)> lg_cute_smem_steps(
    std::index_sequence<Ss...>) {
  return {(lg_cute_smem_slot_v<LevelsT, N, Ss>
               ? slot_arena_step<V, ES>(
                     slot_tile_elems<lg_slot_tile_t<LevelsT, Ss>>())
               : std::size_t{0})...};
}

template <typename V, typename ES, typename LevelsT, int N>
constexpr std::size_t lg_cute_smem_prefix(std::size_t i) {
  constexpr std::size_t NS = lg_num_slots_v<LevelsT>;
  const auto            steps =
      lg_cute_smem_steps<V, ES, LevelsT, N>(std::make_index_sequence<NS>{});
  const auto pools = lg_cute_pool_of_slot<LevelsT, N>();

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

template <typename V, typename ES, typename LevelsT, int N>
constexpr std::size_t lg_cute_unpooled_smem_prefix(std::size_t n) {
  constexpr std::size_t NS = lg_num_slots_v<LevelsT>;
  const auto            steps =
      lg_cute_smem_steps<V, ES, LevelsT, N>(std::make_index_sequence<NS>{});
  std::size_t o = 0;
  for (std::size_t k = 0; k < n && k < NS; ++k) o += steps[k];
  return o;
}

template <typename V, typename ES, typename LevelsT, int N, std::size_t S>
inline constexpr std::size_t lg_cute_smem_offset_v =
    lg_cute_smem_prefix<V, ES, LevelsT, N>(S);

template <typename V, typename ES, typename LevelsT, int N>
inline constexpr std::size_t lg_cute_smem_elems_v =
    lg_cute_smem_prefix<V, ES, LevelsT, N>(lg_num_slots_v<LevelsT>);

template <typename V, typename ES, typename LevelsT, int N>
inline constexpr std::size_t lg_cute_unpooled_smem_elems_v =
    lg_cute_unpooled_smem_prefix<V, ES, LevelsT, N>(lg_num_slots_v<LevelsT>);

template <typename LevelsT, int N, std::size_t L, std::size_t S>
constexpr bool lg_cute_slot_reuses_at() {
  if constexpr (lg_cute_first_reader_v<LevelsT, N, S> != L) {
    return false;
  } else {
    constexpr std::size_t NS    = lg_num_slots_v<LevelsT>;
    constexpr auto        r     = lg_cute_read_range_v<LevelsT, N>;
    constexpr auto        pools = lg_cute_pool_of_slot<LevelsT, N>();
    for (std::size_t t = 0; t < NS; ++t)
      if (t != S && pools[t] == pools[S] && r.first[t] < L && r.last[t] < L)
        return true;
    return false;
  }
}

template <typename LevelsT, int N, std::size_t L, typename Ss>
inline constexpr bool lg_cute_reuses_at_v = false;

template <typename LevelsT, int N, std::size_t L, std::size_t... Ss>
inline constexpr bool
    lg_cute_reuses_at_v<LevelsT, N, L, std::index_sequence<Ss...>> =
        (lg_cute_slot_reuses_at<LevelsT, N, L, Ss>() || ...);

template <typename LevelsT, int N, std::size_t L, typename Ss>
inline constexpr bool lg_cute_copies_at_v = false;

template <typename LevelsT, int N, std::size_t L, std::size_t... Ss>
inline constexpr bool
    lg_cute_copies_at_v<LevelsT, N, L, std::index_sequence<Ss...>> =
        ((lg_cute_first_reader_v<LevelsT, N, Ss> == L) || ...);

template <typename V, typename ES, int NumThreads, typename LevelsT,
          std::size_t S>
__device__ auto lg_cute_slot_smem(V* base) {
  using P = lg_cute_slot_producer<LevelsT, S, NumThreads>;
  return cute::make_tensor(
      cute::make_smem_ptr(base +
                          lg_cute_smem_offset_v<V, ES, LevelsT, NumThreads, S>),
      cute::make_layout(typename P::tile_shape{}, cute::LayoutRight{}));
}

template <typename Outs, std::size_t... Os>
__device__ auto lg_cute_stage_outputs(const Outs& outs,
                                      std::index_sequence<Os...>) {
  return DeviceTuple<std::decay_t<decltype(outs[Os])>...>{outs[Os]...};
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
  if constexpr (Node::NumOut == 1) {
    using R = DeviceTuple<decltype(ev(idx))>;
    return R{ev(idx)};
  } else {
    const auto outs = ev(idx);
    return lg_cute_stage_outputs(
        outs,
        std::make_index_sequence<static_cast<std::size_t>(Node::NumOut)>{});
  }
}

template <typename V, typename ES, int NumThreads, typename LevelsT,
          std::size_t L, std::size_t S, typename Acc>
__device__ void lg_cute_materialize(const LevelsT& levels, const Acc& acc,
                                    V* base) {
  if constexpr (lg_cute_first_reader_v<LevelsT, NumThreads, S> == L) {
    using P = lg_cute_slot_producer<LevelsT, S, NumThreads>;
    make_evaluator<CutePolicyTag<ES>>(
        make_cute_interm_node<ES>(
            lg_cute_slot_smem<V, ES, NumThreads, LevelsT, S>(base)),
        CuteFragmentStoreTag<typename P::part>{P::make(levels)}) =
        acc.template get<S>();
  }
}

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
  using Mma        = lg_cute_mma_t<Node>;
  const auto& node = levels.template get<L>().template get<M>();
  const auto  a =
      lg_cute_operand<V, ES, NumThreads, LevelsT, typename Node::node_a_type>(
          base);
  const auto b =
      lg_cute_operand<V, ES, NumThreads, LevelsT, typename Node::node_b_type>(
          base);
  const auto c = make_evaluator<CutePolicyTag<ES>>(
      node, CuteContractTag<decltype(a), decltype(b), Mma>{
                a, b, lg_cute_mma(node), static_cast<int>(threadIdx.x)})();
  return DeviceTuple<decltype(c)>{c};
}

template <typename V, typename ES, int NumThreads, typename LevelsT,
          std::size_t L, std::size_t M, std::size_t K, typename Acc>
__device__ auto lg_cute_combine_operand(const Acc& acc, V* base) {
  using Plan = lg_cute_combine_plan<LevelsT, L, M, NumThreads>;
  if constexpr (Plan::template in_register_v<K>)
    return acc.template get<Plan::template slot<K>>();
  else
    return lg_cute_operand<V, ES, NumThreads, LevelsT,
                           typename Plan::template op_t<K>>(base);
}

template <typename V, typename ES, int NumThreads, typename LevelsT,
          typename GridModes, std::size_t RootR, std::size_t L, std::size_t M,
          typename Acc, std::size_t... Ks, std::size_t... Os>
__device__ auto lg_cute_combine_member(
    const LevelsT& levels, const Kokkos::Array<int, RootR>& grid_idx,
    const Acc& acc, V* base, std::index_sequence<Ks...>,
    std::index_sequence<Os...>) {
  using Node     = tuple_element_t<M, tuple_element_t<L, LevelsT>>;
  using Plan     = lg_cute_combine_plan<LevelsT, L, M, NumThreads>;
  using Gather   = gather_seq_t<typename Node::modes_seq, GridModes>;
  using OutTile  = member_out_tile_t<Node>;
  const auto idx = node_index<Node::Rank, RootR>(grid_idx, Gather{});
  Kokkos::Array<int, Node::Rank> origin{};
  for (int d = 0; d < Node::Rank; ++d) origin[d] = idx[d] * OutTile::extent(d);

  using Ops =
      DeviceTuple<decltype(lg_cute_combine_operand<V, ES, NumThreads, LevelsT,
                                                   L, M, Ks>(acc, base))...>;
  const Ops   ops{lg_cute_combine_operand<V, ES, NumThreads, LevelsT, L, M, Ks>(
      acc, base)...};
  const auto& node   = levels.template get<L>().template get<M>();
  const bool  active = Plan::make(levels).active();

  const auto run = [&] {
    if constexpr (Plan::register_driven)
      return make_evaluator<CutePolicyTag<ES>>(
          node,
          CuteCombineTag<std::decay_t<decltype(ops.template get<Ks>())>...>{
              ops, origin, active})();
    else
      return make_evaluator<CutePolicyTag<ES>>(
          node,
          CuteCombineThreadTag<
              typename Plan::thr_layout,
              std::decay_t<decltype(ops.template get<Ks>())>...>{
              {typename Plan::thr_layout{}, static_cast<int>(threadIdx.x)},
              ops,
              origin,
              active})();
  };
  if constexpr (Node::NumOut == 0) {
    run();
    return DeviceTuple<>{};
  } else {
    const auto outs = run();
    return DeviceTuple<std::decay_t<decltype(outs[Os])>...>{outs[Os]...};
  }
}

template <typename A, typename B, std::size_t... As, std::size_t... Bs>
__device__ auto lg_cute_concat(const A& a, const B& b,
                               std::index_sequence<As...>,
                               std::index_sequence<Bs...>) {
  using R = DeviceTuple<std::decay_t<decltype(a.template get<As>())>...,
                        std::decay_t<decltype(b.template get<Bs>())>...>;
  return R{a.template get<As>()..., b.template get<Bs>()...};
}

template <typename A, typename B>
__device__ auto lg_cute_cat(const A& a, const B& b) {
  return lg_cute_concat(a, b, std::make_index_sequence<tuple_size_v<A>>{},
                        std::make_index_sequence<tuple_size_v<B>>{});
}

template <typename A, typename B, typename... Rest>
__device__ auto lg_cute_cat(const A& a, const B& b, const Rest&... rest) {
  return lg_cute_cat(lg_cute_cat(a, b), rest...);
}

template <typename V, typename ES, int NumThreads, typename LevelsT,
          typename GridModes, std::size_t RootR, std::size_t L, typename Acc,
          std::size_t... Ss, std::size_t... Ms>
__device__ auto lg_cute_run_level(const LevelsT&                   levels,
                                  const Kokkos::Array<int, RootR>& grid_idx,
                                  const Acc& acc, V* base,
                                  std::index_sequence<Ss...>,
                                  std::index_sequence<Ms...>) {
  using LevelT = tuple_element_t<L, LevelsT>;
  if constexpr (lg_all_staged_v<LevelT>) {
    return lg_cute_cat(
        DeviceTuple<>{},
        lg_cute_stage_member<ES, NumThreads, LevelsT, GridModes, RootR, L, Ms>(
            levels, grid_idx)...);
  } else {
    if constexpr (lg_cute_reuses_at_v<LevelsT, NumThreads, L,
                                      std::index_sequence<Ss...>>)
      __syncthreads();
    (lg_cute_materialize<V, ES, NumThreads, LevelsT, L, Ss>(levels, acc, base),
     ...);
    if constexpr (lg_cute_copies_at_v<LevelsT, NumThreads, L,
                                      std::index_sequence<Ss...>>)
      __syncthreads();
    if constexpr (lg_all_contraction_v<LevelT>) {
      return lg_cute_cat(
          DeviceTuple<>{},
          lg_cute_contract_member<V, ES, NumThreads, LevelsT, L, Ms>(levels,
                                                                     base)...);
    } else {
      return lg_cute_cat(DeviceTuple<>{},
                         lg_cute_combine_member<V, ES, NumThreads, LevelsT,
                                                GridModes, RootR, L, Ms>(
                             levels, grid_idx, acc, base,
                             std::make_index_sequence<static_cast<std::size_t>(
                                 tuple_element_t<Ms, LevelT>::NumOps)>{},
                             std::make_index_sequence<static_cast<std::size_t>(
                                 tuple_element_t<Ms, LevelT>::NumOut)>{})...);
    }
  }
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
                              L + 1>(levels, grid_idx, lg_cute_cat(acc, lv),
                                     base);
  }
}

template <typename ES, int NumThreads, typename LevelsT, typename GridModes,
          std::size_t RootR, std::size_t Rt, typename Slots, typename ViewT>
__device__ void lg_cute_store_root(const LevelsT& levels, const Slots& slots,
                                   const Kokkos::Array<int, RootR>& grid_idx,
                                   const ViewT&                     view) {
  using Node   = lg_cute_slot_node_t<LevelsT, Rt>;
  using P      = lg_cute_slot_producer<LevelsT, Rt, NumThreads>;
  using Gather = gather_seq_t<typename Node::modes_seq, GridModes>;

  const auto& f   = slots.template get<Rt>().node();
  const auto  idx = node_index<Node::Rank, RootR>(grid_idx, Gather{});
  const auto  node =
      make_cute_fragment_node<ES, Node::Rank, typename P::tile_shape>(
          f.frag_, f.coords_);
  const auto out =
      make_handle_seq(view, typename lg_member_decl_modes<Node>::type{});
  make_evaluator<CutePolicyTag<ES>>(
      node, CuteFragmentStoreTag<typename P::part>{P::make(levels)})(
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
  return lg_cute_smem_elems_v<V, ES, LevelsT, lg_cute_num_threads_v<LevelsT>> *
         sizeof(V);
}

template <typename V, typename ES, typename LevelsT>
std::size_t lg_cute_unpooled_smem_bytes() {
  return lg_cute_unpooled_smem_elems_v<V, ES, LevelsT,
                                       lg_cute_num_threads_v<LevelsT>> *
         sizeof(V);
}

template <typename V, typename ES, typename LT, typename LevelsT,
          typename RootsSeq, typename... ViewTs>
int lg_execute_cute(const LevelsT& levels, RootsSeq, const ViewTs&... views) {
  static_assert(lg_cute_supported<LevelsT>(
                    std::make_index_sequence<tuple_size_v<LevelsT>>{}),
                "level graph (CuTe): every level must be all stage, all "
                "contraction or all combine members");
  constexpr int NumThreads = lg_cute_num_threads_v<LevelsT>;
  static_assert(NumThreads <= 1024,
                "level graph (CuTe): the largest contraction TiledMMA needs "
                "more than 1024 threads");
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
