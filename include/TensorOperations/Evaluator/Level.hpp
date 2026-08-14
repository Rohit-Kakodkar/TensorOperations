#pragma once

struct AlignedOperands {};
struct PermutedOperands {};

using DefaultOperandPolicy = AlignedOperands;

namespace Impl {

template <typename PermSeq, typename Eval>
using reordered_layout_t = typename std::conditional_t<
    is_identity_v<PermSeq>,
    std::type_identity<typename Eval::storage_type::layout_t>,
    std::type_identity<typename decltype(reorder_view(
        std::declval<typename Eval::storage_type>(), PermSeq{}))::layout_t>>::
    type;

template <typename PermSeq, typename Eval, typename Team>
KOKKOS_FUNCTION auto reorder_operand(Eval e, const Team& team) {
  if constexpr (is_identity_v<PermSeq>) {
    return e;
  } else {
    return (make_evaluator<TeamPolicyTag2<typename Eval::exec_space>>(
                e, PermSeq{}, team) = e);
  }
}

template <typename Policy, typename PermSeq, typename Eval, typename Team>
KOKKOS_FUNCTION auto combine_operand(Eval e, const Team& team) {
  static_assert(std::is_same_v<Policy, AlignedOperands> ||
                    std::is_same_v<Policy, PermutedOperands>,
                "combine_operand: policy must be AlignedOperands or "
                "PermutedOperands");
  if constexpr (std::is_same_v<Policy, PermutedOperands>) {
    (void)team;
    return make_permuted_at<PermSeq>(e);
  } else {
    return reorder_operand<PermSeq>(e, team);
  }
}

template <typename AL, typename BL, int NumK, typename IA, typename IB>
struct CanonTileOf;

template <typename AL, typename BL, int NumK, std::size_t... IA,
          std::size_t... IB>
struct CanonTileOf<AL, BL, NumK, std::index_sequence<IA...>,
                   std::index_sequence<IB...>> {
  using type = StaticTile<AL::extent(static_cast<int>(IA))...,
                          BL::extent(NumK + static_cast<int>(IB))...>;
};

template <typename NodeT>
using node_permA_t = permA_seq_t<typename NodeT::node_a_type::modes_seq,
                                 typename NodeT::node_b_type::modes_seq>;
template <typename NodeT>
using node_permB_t = permB_seq_t<typename NodeT::node_a_type::modes_seq,
                                 typename NodeT::node_b_type::modes_seq>;

}  // namespace Impl

template <typename NodeT, typename AEval, typename BEval>
using contract_c_tile_t = typename Impl::CanonTileOf<
    Impl::reordered_layout_t<Impl::node_permA_t<NodeT>, AEval>,
    Impl::reordered_layout_t<Impl::node_permB_t<NodeT>, BEval>,
    NodeT::NumContracted,
    std::make_index_sequence<NodeT::node_a_type::Rank - NodeT::NumContracted>,
    std::make_index_sequence<NodeT::node_b_type::Rank - NodeT::NumContracted>>::
    type;

template <typename NodeT, typename AEval, typename BEval, typename CNode,
          typename Team>
KOKKOS_FUNCTION auto contract_into(NodeT node, AEval a, BEval b, CNode c,
                                   const Team& team) {
  using ES = typename NodeT::exec_space;

  static_assert(
      Impl::is_identity_v<typename NodeT::permC_seq>,
      "contract_into: Tag2 writes C in CANONICAL order (freeA in A's order, "
      "then freeB in B's order) and ignores permC, so the contraction node's "
      "output labels must already be canonical");

  return make_evaluator<TeamPolicyTag2<ES>>(
      node,
      ContractOperands{
          Impl::reorder_operand<Impl::node_permA_t<NodeT>>(a, team),
          Impl::reorder_operand<Impl::node_permB_t<NodeT>>(b, team), c},
      team);
}
