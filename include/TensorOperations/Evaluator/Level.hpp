#pragma once

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

template <typename Node>
struct SlotValueEval {
  static_assert(
      has_node_tag_v<SlotTag, Node> || has_node_tag_v<IntermTag, Node>,
      "level plan: a member's operand must NAME a buffer -- a stage slot or an "
      "earlier level's output slot. Nesting an operand's subtree by value is "
      "what a level graph exists to avoid");
  using type =
      value_evaluator_t<NodeHandle<IntermTag, typename Node::storage_type,
                                   std::integral_constant<int, Node::Rank>,
                                   typename Node::exec_space, NoHook>>;
};
template <typename Node>
using slot_value_eval_t = typename SlotValueEval<Node>::type;

template <typename Node>
using slot_layout_t = typename slot_value_eval_t<Node>::storage_type::layout_t;

template <typename SrcLayout, typename PermSeq, typename Idx>
struct TileFromPermImpl;
template <typename SrcLayout, typename PermSeq, std::size_t... Is>
struct TileFromPermImpl<SrcLayout, PermSeq, std::index_sequence<Is...>> {
  using type = StaticTile<SrcLayout::extent(
      static_cast<int>(seq_to_array(PermSeq{})[Is]))...>;
};
template <typename SrcLayout, typename PermSeq>
using TileFromPerm =
    typename TileFromPermImpl<SrcLayout, PermSeq,
                              std::make_index_sequence<PermSeq::size()>>::type;

template <typename Node>
struct CombineOutTile {
  using op0_type = tuple_element_t<0, typename Node::ops_tuple_t>;
  using perm_type =
      label_perm_seq_t<typename Node::modes_seq, typename op0_type::modes_seq>;
  using type = TileFromPerm<slot_layout_t<op0_type>, perm_type>;
};

}  // namespace Impl

template <typename NodeT, typename AEval, typename BEval>
using contract_c_tile_t = typename Impl::CanonTileOf<
    Impl::reordered_layout_t<Impl::node_permA_t<NodeT>, AEval>,
    Impl::reordered_layout_t<Impl::node_permB_t<NodeT>, BEval>,
    NodeT::NumContracted,
    std::make_index_sequence<NodeT::node_a_type::Rank - NodeT::NumContracted>,
    std::make_index_sequence<NodeT::node_b_type::Rank - NodeT::NumContracted>>::
    type;

template <typename NodeT>
using combine_out_tile_t = typename Impl::CombineOutTile<NodeT>::type;

namespace Impl {

template <typename Node, typename Tag = typename Node::node_tag>
struct MemberOutTile;
template <typename Node>
struct MemberOutTile<Node, ContractionTag> {
  using type =
      contract_c_tile_t<Node, slot_value_eval_t<typename Node::node_a_type>,
                        slot_value_eval_t<typename Node::node_b_type>>;
};
template <typename Node>
struct MemberOutTile<Node, CombineTag> {
  using type = combine_out_tile_t<Node>;
};

// A stage's tile cannot be derived -- it has no operands to derive from -- so
// it is carried on the node, resolved by LevelGraph::add from the graph's label
// map. `void` means the node was never added to a graph.
template <typename Node>
struct MemberOutTile<Node, StagedTag> {
  static_assert(
      !std::is_same_v<typename Node::tile_type, void>,
      "staged member: its tile is unresolved, which means this node "
      "was never handed to LevelGraph::add -- add() is what looks the "
      "tile up in the graph's label map");
  using type = typename Node::tile_type;
};

}  // namespace Impl

template <typename NodeT>
using member_out_tile_t = typename Impl::MemberOutTile<NodeT>::type;

template <typename NodeT, typename AEval, typename BEval, typename CNode,
          typename Team>
KOKKOS_FUNCTION auto contract_into(NodeT node, AEval a, BEval b, CNode c,
                                   const Team& team) {
  using ES = typename NodeT::exec_space;

  return make_evaluator<TeamPolicyTag2<ES>>(
      node,
      ContractOperands{
          Impl::reorder_operand<Impl::node_permA_t<NodeT>>(a, team),
          Impl::reorder_operand<Impl::node_permB_t<NodeT>>(b, team), c},
      team);
}
