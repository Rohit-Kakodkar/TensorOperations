#pragma once
#include <TensorOperations/DagGraph.hpp>
#include <TensorOperations/DeviceTuple.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/Graph.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/SlotStore.hpp>

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

namespace TensorOperations {
namespace Impl {

template <typename LevelT, std::size_t... Ms>
constexpr std::array<std::size_t, sizeof...(Ms)> lg_member_arities_impl(
    std::index_sequence<Ms...>) {
  return {static_cast<std::size_t>(
      output_arity<tuple_element_t<Ms, LevelT>>::value)...};
}
template <typename LevelT>
constexpr auto lg_member_arities() {
  return lg_member_arities_impl<LevelT>(
      std::make_index_sequence<tuple_size_v<LevelT>>{});
}

template <typename LevelT>
constexpr std::size_t lg_level_slots() {
  std::size_t n = 0;
  for (std::size_t a : lg_member_arities<LevelT>()) n += a;
  return n;
}

template <typename LevelsT, std::size_t... Ls>
constexpr std::array<std::size_t, sizeof...(Ls)> lg_level_slots_all_impl(
    std::index_sequence<Ls...>) {
  return {lg_level_slots<tuple_element_t<Ls, LevelsT>>()...};
}
template <typename LevelsT>
constexpr auto lg_level_slots_all() {
  return lg_level_slots_all_impl<LevelsT>(
      std::make_index_sequence<tuple_size_v<LevelsT>>{});
}

template <typename LevelsT, std::size_t... Ls>
constexpr std::array<std::size_t, sizeof...(Ls)> lg_level_member_counts_impl(
    std::index_sequence<Ls...>) {
  return {tuple_size_v<tuple_element_t<Ls, LevelsT>>...};
}
template <typename LevelsT>
constexpr auto lg_level_member_counts() {
  return lg_level_member_counts_impl<LevelsT>(
      std::make_index_sequence<tuple_size_v<LevelsT>>{});
}

template <typename LevelsT>
constexpr std::size_t lg_total_members() {
  std::size_t n = 0;
  for (std::size_t c : lg_level_member_counts<LevelsT>()) n += c;
  return n;
}

template <typename LevelT, std::size_t N>
constexpr void lg_append_level_arities(std::array<std::size_t, N>& out,
                                       std::size_t&                pos) {
  for (std::size_t a : lg_member_arities<LevelT>()) out[pos++] = a;
}

template <typename LevelsT, std::size_t N, std::size_t... Ls>
constexpr void lg_fill_flat_arities(std::array<std::size_t, N>& out,
                                    std::index_sequence<Ls...>) {
  std::size_t pos = 0;
  (lg_append_level_arities<tuple_element_t<Ls, LevelsT>, N>(out, pos), ...);
}

template <typename LevelsT>
constexpr auto lg_flat_arities() {
  std::array<std::size_t, lg_total_members<LevelsT>()> out{};
  lg_fill_flat_arities<LevelsT>(
      out, std::make_index_sequence<tuple_size_v<LevelsT>>{});
  return out;
}

template <typename LevelsT>
constexpr std::size_t lg_member_offset(std::size_t l) {
  const auto  c   = lg_level_member_counts<LevelsT>();
  std::size_t off = 0;
  for (std::size_t i = 0; i < l; ++i) off += c[i];
  return off;
}

template <typename LevelsT, std::size_t NumStages>
constexpr std::size_t lg_num_slots() {
  std::size_t n = NumStages;
  for (std::size_t s : lg_level_slots_all<LevelsT>()) n += s;
  return n;
}

template <typename LevelsT, std::size_t NumStages>
constexpr std::size_t lg_level_base(std::size_t l) {
  const auto  s   = lg_level_slots_all<LevelsT>();
  std::size_t off = NumStages;
  for (std::size_t i = 0; i < l; ++i) off += s[i];
  return off;
}

template <typename LevelsT, std::size_t NumStages>
constexpr std::size_t lg_member_base(std::size_t l, std::size_t m) {
  const auto  flat  = lg_flat_arities<LevelsT>();
  const auto  mbase = lg_member_offset<LevelsT>(l);
  std::size_t off   = lg_level_base<LevelsT, NumStages>(l);
  for (std::size_t i = 0; i < m; ++i) off += flat[mbase + i];
  return off;
}

template <std::size_t NumStages>
constexpr bool lg_is_stage_slot(std::size_t s) {
  return s < NumStages;
}

template <typename LevelsT, std::size_t NumStages>
constexpr std::size_t lg_slot_level(std::size_t s) {
  const auto sl = lg_level_slots_all<LevelsT>();
  if (s < NumStages) return sl.size();
  std::size_t end = NumStages;
  for (std::size_t l = 0; l < sl.size(); ++l) {
    end += sl[l];
    if (s < end) return l;
  }
  return sl.size();
}

template <typename LevelsT, std::size_t NumStages>
constexpr std::size_t lg_slot_member(std::size_t s) {
  const auto        c = lg_level_member_counts<LevelsT>();
  const std::size_t l = lg_slot_level<LevelsT, NumStages>(s);
  if (l >= c.size()) return 0;
  const auto  flat  = lg_flat_arities<LevelsT>();
  const auto  mbase = lg_member_offset<LevelsT>(l);
  std::size_t end   = lg_level_base<LevelsT, NumStages>(l);
  for (std::size_t m = 0; m < c[l]; ++m) {
    end += flat[mbase + m];
    if (s < end) return m;
  }
  return c[l];
}

template <typename LevelsT, std::size_t NumStages>
inline constexpr std::size_t lg_num_slots_v =
    lg_num_slots<LevelsT, NumStages>();
template <typename LevelsT, std::size_t NumStages, std::size_t L>
inline constexpr std::size_t lg_level_base_v =
    lg_level_base<LevelsT, NumStages>(L);
template <typename LevelsT, std::size_t NumStages, std::size_t L, std::size_t M>
inline constexpr std::size_t lg_member_base_v =
    lg_member_base<LevelsT, NumStages>(L, M);
template <typename LevelsT, std::size_t NumStages, std::size_t S>
inline constexpr std::size_t lg_slot_level_v =
    lg_slot_level<LevelsT, NumStages>(S);
template <typename LevelsT, std::size_t NumStages, std::size_t S>
inline constexpr std::size_t lg_slot_member_v =
    lg_slot_member<LevelsT, NumStages>(S);
template <typename LevelsT, std::size_t L>
inline constexpr std::size_t lg_level_slots_v =
    lg_level_slots<tuple_element_t<L, LevelsT>>();
template <typename LevelsT>
inline constexpr std::size_t lg_total_members_v = lg_total_members<LevelsT>();
template <std::size_t NumStages, std::size_t S>
inline constexpr bool lg_is_stage_slot_v = lg_is_stage_slot<NumStages>(S);

// --- liveness, on a LEVEL timeline -----------------------------------------
//
// Slots that cannot be alive at once share a buffer, which is what takes the
// SEM3D graph from 35 buffers to 19 and its team scratch from 16,848 to 9,584
// bytes at TE=1 -- occupancy 13 to 24 blocks/SM, and on that kernel that is
// worth about 12%.
//
// THE TIMELINE IS LEVELS, NOT MEMBERS, AND THAT IS A CORRECTNESS REQUIREMENT.
// A DagGraph node reads its operands and writes its outputs in ONE evaluation,
// so ranges closed at both ends make reuse safe at a node. A level does not
// work that way: barriers exist only at level ends, and every member of a level
// runs interleaved inside one TeamVectorRange, so every slot a level touches is
// live for the whole level. Ranking members within a level would produce a plan
// that is wrong exactly where nothing checks it.
//
//   t = 0            the stages
//   t = L + 1        level L
//   t = NumLevels+1  the root store, which runs after every level
//
// Ranges are CLOSED at both ends, so a slot defined at level L and a slot last
// read at level L overlap at L and can never pool together.
//
// The operand scan is DagGraph's: dag_note_node_reads dispatches on node tag,
// and a level's members are the same ContractionTag / CombineTag nodes with the
// same SlotTag operands. A stage node wraps an InputTag, so it reports no slot
// reads and the branch is a no-op. Only the iteration is new.

template <typename LevelT, std::size_t NS, std::size_t... Ms>
constexpr void lg_note_level_reads(std::array<std::size_t, NS>& last,
                                   std::size_t t, std::index_sequence<Ms...>) {
  (dag_note_node_reads<tuple_element_t<Ms, LevelT>, NS>(last, t), ...);
}

template <typename LevelsT, std::size_t NS, std::size_t... Ls>
constexpr void lg_note_all_reads(std::array<std::size_t, NS>& last,
                                 std::index_sequence<Ls...>) {
  (lg_note_level_reads<tuple_element_t<Ls, LevelsT>, NS>(
       last, Ls + 1,
       std::make_index_sequence<tuple_size_v<tuple_element_t<Ls, LevelsT>>>{}),
   ...);
}

template <typename LevelsT, std::size_t NumStages, std::size_t... Roots>
constexpr auto lg_pool_of_slot() {
  constexpr std::size_t NS = lg_num_slots<LevelsT, NumStages>();
  constexpr std::size_t NL = tuple_size_v<LevelsT>;

  std::array<std::size_t, NS> def{}, last{};
  for (std::size_t s = 0; s < NS; ++s) {
    // lg_slot_level returns NL as a SENTINEL for a stage slot, not a level, so
    // stages are mapped to t=0 explicitly rather than by trusting the return.
    def[s] = lg_is_stage_slot<NumStages>(s)
                 ? std::size_t{0}
                 : lg_slot_level<LevelsT, NumStages>(s) + 1;
    // A slot nobody reads is still live where it is written.
    last[s] = def[s];
  }
  lg_note_all_reads<LevelsT, NS>(last, std::make_index_sequence<NL>{});

  // A designated output is read by lg_store_roots AFTER every level has run.
  const std::array<std::size_t, sizeof...(Roots)> roots{Roots...};
  for (std::size_t i = 0; i < sizeof...(Roots); ++i) last[roots[i]] = NL + 1;

  return left_edge_colour<NS>(def, last);
}

template <typename LevelsT, std::size_t NumStages, std::size_t... Roots>
constexpr std::size_t lg_pool_count() {
  const auto  p = lg_pool_of_slot<LevelsT, NumStages, Roots...>();
  std::size_t n = 0;
  for (std::size_t s = 0; s < p.size(); ++s)
    if (p[s] + 1 > n) n = p[s] + 1;
  return n;
}

// Keyed on an index_sequence of the roots so the root set travels as one type,
// for the same reason dag_slot_pool_v is: the functions above are host-only
// constexpr (std::array is), so device code must name a constant.
template <typename LevelsT, std::size_t NumStages, typename RootsSeq>
struct lg_plan;
template <typename LevelsT, std::size_t NumStages, std::size_t... Roots>
struct lg_plan<LevelsT, NumStages, std::index_sequence<Roots...>> {
  static constexpr auto of_slot() {
    return lg_pool_of_slot<LevelsT, NumStages, Roots...>();
  }
  static constexpr std::size_t count() {
    return lg_pool_count<LevelsT, NumStages, Roots...>();
  }
};

template <typename LevelsT, std::size_t NumStages, typename RootsSeq,
          std::size_t S>
inline constexpr std::size_t lg_slot_pool_v =
    lg_plan<LevelsT, NumStages, RootsSeq>::of_slot()[S];
template <typename LevelsT, std::size_t NumStages, typename RootsSeq>
inline constexpr std::size_t lg_pool_count_v =
    lg_plan<LevelsT, NumStages, RootsSeq>::count();

template <typename Node>
using member_out_layout_t =
    typename SlotView<typename Node::value_type, typename Node::exec_space,
                      member_out_tile_t<Node>>::layout_t;

template <typename Node>
constexpr int lg_member_sa() {
  constexpr int FreeA = Node::node_a_type::Rank - Node::NumContracted;
  return extent_product<member_out_layout_t<Node>, 0, FreeA>();
}
template <typename Node>
constexpr int lg_member_sb() {
  constexpr int FreeA = Node::node_a_type::Rank - Node::NumContracted;
  return extent_product<member_out_layout_t<Node>, FreeA, Node::Rank>();
}
template <typename Node>
inline constexpr int lg_member_sa_v = lg_member_sa<Node>();
template <typename Node>
inline constexpr int lg_member_sb_v = lg_member_sb<Node>();

template <typename LevelT, std::size_t... Ms>
constexpr bool lg_all_contraction_impl(std::index_sequence<Ms...>) {
  return (has_node_tag_v<ContractionTag, tuple_element_t<Ms, LevelT>> && ...);
}
template <typename LevelT, std::size_t... Ms>
constexpr bool lg_all_combine_impl(std::index_sequence<Ms...>) {
  return (has_node_tag_v<CombineTag, tuple_element_t<Ms, LevelT>> && ...);
}
template <typename LevelT>
inline constexpr bool lg_all_contraction_v = lg_all_contraction_impl<LevelT>(
    std::make_index_sequence<tuple_size_v<LevelT>>{});
template <typename LevelT>
inline constexpr bool lg_all_combine_v = lg_all_combine_impl<LevelT>(
    std::make_index_sequence<tuple_size_v<LevelT>>{});

template <typename LevelT>
inline constexpr bool lg_level_nonempty_v = tuple_size_v<LevelT> > 0;

template <typename LevelT>
inline constexpr bool lg_level_homogeneous_v =
    lg_level_nonempty_v<LevelT> &&
    (lg_all_contraction_v<LevelT> || lg_all_combine_v<LevelT>);

template <typename LevelT, std::size_t... Ms>
constexpr bool lg_contraction_space_impl(std::index_sequence<Ms...>) {
  using M0 = tuple_element_t<0, LevelT>;
  return ((lg_member_sa_v<tuple_element_t<Ms, LevelT>> == lg_member_sa_v<M0> &&
           lg_member_sb_v<tuple_element_t<Ms, LevelT>> == lg_member_sb_v<M0>) &&
          ...);
}
template <typename LevelT, std::size_t... Ms>
constexpr bool lg_combine_space_impl(std::index_sequence<Ms...>) {
  using M0 = tuple_element_t<0, LevelT>;
  return (std::is_same_v<member_out_layout_t<tuple_element_t<Ms, LevelT>>,
                         member_out_layout_t<M0>> &&
          ...);
}

template <typename LevelT>
constexpr bool lg_level_space_agrees() {
  if constexpr (!lg_level_homogeneous_v<LevelT>) {
    return true;
  } else if constexpr (lg_all_contraction_v<LevelT>) {
    return lg_contraction_space_impl<LevelT>(
        std::make_index_sequence<tuple_size_v<LevelT>>{});
  } else {
    return lg_combine_space_impl<LevelT>(
        std::make_index_sequence<tuple_size_v<LevelT>>{});
  }
}
template <typename LevelT>
inline constexpr bool lg_level_space_agrees_v = lg_level_space_agrees<LevelT>();

template <typename Node, std::size_t... Is>
constexpr int lg_max_combine_slot(std::index_sequence<Is...>) {
  int m = -1;
  ((m = dag_operand_slot<tuple_element_t<Is, typename Node::ops_tuple_t>>() > m
            ? dag_operand_slot<
                  tuple_element_t<Is, typename Node::ops_tuple_t>>()
            : m),
   ...);
  return m;
}

template <typename Node>
constexpr int lg_max_operand_slot() {
  if constexpr (has_node_tag_v<ContractionTag, Node>) {
    const int a = dag_operand_slot<typename Node::node_a_type>();
    const int b = dag_operand_slot<typename Node::node_b_type>();
    return a > b ? a : b;
  } else if constexpr (has_node_tag_v<CombineTag, Node>) {
    return lg_max_combine_slot<Node>(
        std::make_index_sequence<static_cast<std::size_t>(Node::NumOps)>{});
  } else if constexpr (has_node_tag_v<StagedTag, Node>) {
    return dag_operand_slot<typename Node::operand_type>();
  } else {
    return -1;
  }
}
template <typename Node>
inline constexpr int lg_max_operand_slot_v = lg_max_operand_slot<Node>();

template <typename LevelsT, std::size_t NumStages, std::size_t L,
          std::size_t... Ms>
constexpr bool lg_level_reads_earlier_impl(std::index_sequence<Ms...>) {
  using LevelT = tuple_element_t<L, LevelsT>;
  return ((lg_max_operand_slot_v<tuple_element_t<Ms, LevelT>> <
           static_cast<int>(lg_level_base<LevelsT, NumStages>(L))) &&
          ...);
}
template <typename LevelsT, std::size_t NumStages, std::size_t L>
inline constexpr bool lg_level_reads_earlier_v =
    lg_level_reads_earlier_impl<LevelsT, NumStages, L>(
        std::make_index_sequence<tuple_size_v<tuple_element_t<L, LevelsT>>>{});

template <typename LevelsT, std::size_t... Ls>
constexpr bool lg_all_levels_homogeneous_impl(std::index_sequence<Ls...>) {
  return (lg_level_homogeneous_v<tuple_element_t<Ls, LevelsT>> && ...);
}
template <typename LevelsT, std::size_t... Ls>
constexpr bool lg_all_levels_space_impl(std::index_sequence<Ls...>) {
  return (lg_level_space_agrees_v<tuple_element_t<Ls, LevelsT>> && ...);
}
template <typename LevelsT, std::size_t NumStages, std::size_t... Ls>
constexpr bool lg_all_levels_reads_impl(std::index_sequence<Ls...>) {
  return (lg_level_reads_earlier_v<LevelsT, NumStages, Ls> && ...);
}

template <typename LevelsT>
inline constexpr bool lg_levels_homogeneous_v =
    lg_all_levels_homogeneous_impl<LevelsT>(
        std::make_index_sequence<tuple_size_v<LevelsT>>{});
template <typename LevelsT>
inline constexpr bool lg_levels_space_v = lg_all_levels_space_impl<LevelsT>(
    std::make_index_sequence<tuple_size_v<LevelsT>>{});
template <typename LevelsT, std::size_t NumStages>
inline constexpr bool lg_levels_reads_v =
    lg_all_levels_reads_impl<LevelsT, NumStages>(
        std::make_index_sequence<tuple_size_v<LevelsT>>{});

}  // namespace Impl

template <typename LevelsT, std::size_t NumStages>
struct LevelPlan {
  using levels_type = LevelsT;

  static constexpr std::size_t num_stages = NumStages;
  static constexpr std::size_t num_levels = tuple_size_v<LevelsT>;
  static constexpr std::size_t num_slots =
      Impl::lg_num_slots_v<LevelsT, NumStages>;
  static constexpr std::size_t num_members = Impl::lg_total_members_v<LevelsT>;

  static_assert(Impl::lg_levels_homogeneous_v<LevelsT>,
                "level graph: a level must be NON-EMPTY and HOMOGENEOUS -- "
                "every member a contraction, or every member a combine. The "
                "two decodes are different objects, so a mixed level computes "
                "both and banks neither");

  static_assert(Impl::lg_levels_space_v<LevelsT>,
                "level graph: every member of a level must share ONE iteration "
                "space -- contraction members must agree on SA and SB "
                "(separately, not on their product), combine members on their "
                "whole output tile layout");

  static_assert(
      Impl::lg_levels_reads_v<LevelsT, NumStages>,
      "level graph: a member may not read a slot ITS OWN LEVEL "
      "produces -- a level's members run concurrently, so a sibling's "
      "output is a cross-thread hazard, not an operand");
};

}  // namespace TensorOperations
