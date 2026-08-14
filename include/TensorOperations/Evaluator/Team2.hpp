#pragma once

template <typename ES, typename Storage, typename IntRank, typename HookOp>
class Evaluator<TeamPolicyTag2<ES>,
                NodeHandle<IntermTag, Storage, IntRank, ES, HookOp>, void> {
 public:
  using node_type   = NodeHandle<IntermTag, Storage, IntRank, ES, HookOp>;
  using policy_tag  = TeamPolicyTag2<ES>;
  using tiling_type = void;
  static constexpr int Rank = node_type::Rank;
  using storage_type        = Storage;
  using value_type          = typename node_type::value_type;
  using exec_space          = ES;
  using team_member_t       = Impl::team_member_t<ES>;

  KOKKOS_FUNCTION Evaluator(node_type n, const team_member_t& team)
      : node_(n), team_(team) {}

  KOKKOS_FUNCTION const node_type&     node() const { return node_; }
  KOKKOS_FUNCTION const team_member_t& team() const { return team_; }

 private:
  node_type     node_;
  team_member_t team_;
};

template <typename Storage, typename IntRank, typename ES, typename HookOp,
          typename Team>
KOKKOS_FUNCTION auto make_value_evaluator(
    NodeHandle<IntermTag, Storage, IntRank, ES, HookOp> node, const Team& team)
    -> Evaluator<TeamPolicyTag2<ES>,
                 NodeHandle<IntermTag, Storage, IntRank, ES, HookOp>, void> {
  return {node, team};
}

namespace Impl {

template <typename Node>
using value_evaluator_t =
    Evaluator<TeamPolicyTag2<typename Node::exec_space>, Node, void>;

}

template <typename ES, TensorLike T, typename ModesSeq, typename HookOp,
          typename Tile_>
class Evaluator<TeamPolicyTag2<ES>, NodeHandle<InputTag, T, ModesSeq, HookOp>,
                Tile_> {
 public:
  using node_type     = NodeHandle<InputTag, T, ModesSeq, HookOp>;
  using policy_tag    = TeamPolicyTag2<ES>;
  using tiling_type   = Tile_;
  using exec_space    = ES;
  using team_member_t = Impl::team_member_t<ES>;

  KOKKOS_FUNCTION Evaluator(node_type n, Tile_ t, const team_member_t& team)
      : hook_(n.hook_op), tiled_input_(tile_view(n.handle, t)), team_(team) {}

  KOKKOS_FUNCTION auto operator()(
      Kokkos::Array<int, Tile_::rank> tile_idx) const {
    return make_value_evaluator(
        make_interm_node(subview_tile(tiled_input_, tile_idx), hook_), team_);
  }

 private:
  [[no_unique_address]] HookOp                hook_;
  TiledView<TensorHandle<T, ModesSeq>, Tile_> tiled_input_;
  team_member_t                               team_;
};

template <typename ES, typename BackingVT, typename Layout, typename IntRank,
          typename HookOp, int... Perm>
class Evaluator<TeamPolicyTag2<ES>,
                Evaluator<TeamPolicyTag2<ES>,
                          NodeHandle<IntermTag, View<BackingVT, Layout>,
                                     IntRank, ES, HookOp>,
                          void>,
                std::integer_sequence<int, Perm...>> {
  using source_node_type =
      NodeHandle<IntermTag, View<BackingVT, Layout>, IntRank, ES, HookOp>;

 public:
  using policy_tag    = TeamPolicyTag2<ES>;
  using source_type   = Impl::value_evaluator_t<source_node_type>;
  using perm_seq      = std::integer_sequence<int, Perm...>;
  using tiling_type   = perm_seq;
  using exec_space    = ES;
  using team_member_t = Impl::team_member_t<ES>;

  static_assert(Layout::rank == source_node_type::Rank,
                "view layout rank must equal node rank");
  static_assert(sizeof...(Perm) == source_node_type::Rank,
                "relabel permutation must have one entry per mode");
  static_assert(
      requires(typename source_node_type::storage_type v) {
        reorder_view(v, perm_seq{});
      },
      "Tag2 relabel: this storage layout has neither a reorder_layout "
      "(global/subview tiles) nor a reorder_tile (scratch tiles) overload");

  KOKKOS_FUNCTION Evaluator(source_type, perm_seq, const team_member_t& team)
      : team_(team) {}

  KOKKOS_FUNCTION auto operator=(const source_type& src) const {
    const auto& n = src.node();
    return make_value_evaluator(
        make_interm_node(reorder_view(n.storage_, perm_seq{}), n.hook_op),
        team_);
  }

 private:
  team_member_t team_;
};

struct StageTag {};

template <typename ES, typename ValueType, typename Layout, int Rank,
          typename HookOp>
class Evaluator<TeamPolicyTag2<ES>,
                NodeHandle<IntermTag, ScratchView<ValueType, ES, Layout>,
                           std::integral_constant<int, Rank>, ES, HookOp>,
                StageTag> {
 public:
  using node_type   = NodeHandle<IntermTag, ScratchView<ValueType, ES, Layout>,
                                 std::integral_constant<int, Rank>, ES, HookOp>;
  using policy_tag  = TeamPolicyTag2<ES>;
  using tiling_type = StageTag;
  using scratch_view_t = ScratchView<ValueType, ES, Layout>;
  using value_type     = ValueType;
  using exec_space     = ES;
  using team_member_t  = Impl::team_member_t<ES>;

  static_assert(Layout::rank == Rank,
                "destination scratch layout rank must equal node rank");

  KOKKOS_FUNCTION Evaluator(node_type n, StageTag, const team_member_t& team)
      : node_(n), team_(team) {}

  template <typename SrcEval>
  KOKKOS_FUNCTION auto operator=(const SrcEval& src) const {
    static_assert(SrcEval::Rank == Rank,
                  "staged source and destination must have equal rank");

    const auto sv  = src.node().storage_;
    const auto dst = node_.storage_;
    Impl::team_for_each_coord(team_, sv,
                              [=](auto coord) { dst[coord] = sv[coord]; });

    return make_value_evaluator(make_interm_node(dst, src.node().hook_op),
                                team_);
  }

 private:
  node_type     node_;
  team_member_t team_;
};

template <typename ES, typename Operand, typename ModesSeq, typename Tile_>
class Evaluator<TeamPolicyTag2<ES>, NodeHandle<StagedTag, Operand, ModesSeq>,
                Tile_> {
 public:
  using node_type     = NodeHandle<StagedTag, Operand, ModesSeq>;
  using policy_tag    = TeamPolicyTag2<ES>;
  using tiling_type   = Tile_;
  using value_type    = typename node_type::value_type;
  using exec_space    = ES;
  using modes_seq     = typename node_type::modes_seq;
  using team_member_t = Impl::team_member_t<ES>;

  using scratch_view_t = decltype(Impl::alloc_scratch_tile<value_type, ES>(
      std::declval<const team_member_t&>(), std::declval<const Tile_&>()));

  static_assert(static_cast<int>(Tile_::rank) == node_type::Rank,
                "staged tile rank must equal the operand's rank");

  KOKKOS_FUNCTION Evaluator(node_type n, Tile_ t, const team_member_t& team)
      : node_(n),
        tile_(t),
        dst_(Impl::alloc_scratch_tile<value_type, ES>(team, t)),
        team_(team) {}

  KOKKOS_FUNCTION Evaluator(node_type n, Tile_ t, scratch_view_t adopted,
                            const team_member_t& team)
      : node_(n), tile_(t), dst_(adopted), team_(team) {}

  KOKKOS_FUNCTION auto operator()(
      Kokkos::Array<int, Tile_::rank> tile_idx) const {
    auto src    = make_evaluator<TeamPolicyTag2<ES>>(node_.operand_, tile_,
                                                     team_)(tile_idx);
    auto stager = make_evaluator<TeamPolicyTag2<ES>>(make_interm_node(dst_),
                                                     StageTag{}, team_);
    return (stager = src);
  }

  KOKKOS_FUNCTION const scratch_view_t& storage() const { return dst_; }

 private:
  node_type      node_;
  Tile_          tile_;
  scratch_view_t dst_;
  team_member_t  team_;
};

template <typename AEval, typename BEval, typename CNode>
struct ContractOperands {
  AEval a;
  BEval b;
  CNode c;
};

namespace Impl {

template <typename Layout, int Begin, int End>
KOKKOS_FUNCTION constexpr int extent_product() noexcept {
  int p = 1;
  for (int d = Begin; d < End; ++d) p *= Layout::extent(d);
  return p;
}

template <typename L1, int O1, typename L2, int O2, int N>
KOKKOS_FUNCTION constexpr bool extents_agree() noexcept {
  for (int d = 0; d < N; ++d)
    if (L1::extent(O1 + d) != L2::extent(O2 + d)) return false;
  return true;
}

}  // namespace Impl

template <typename ES, typename NA, typename NB, typename IntCRank, typename S,
          typename HookOp, typename CModesSeq, typename PermCSeq,
          typename AEval, typename BEval, typename CNode>
class Evaluator<TeamPolicyTag2<ES>,
                NodeHandle<ContractionTag, NA, NB, IntCRank, S, ES, HookOp,
                           CModesSeq, PermCSeq>,
                ContractOperands<AEval, BEval, CNode>> {
 public:
  using node_type  = NodeHandle<ContractionTag, NA, NB, IntCRank, S, ES, HookOp,
                                CModesSeq, PermCSeq>;
  using policy_tag = TeamPolicyTag2<ES>;
  using tiling_type   = ContractOperands<AEval, BEval, CNode>;
  using value_type    = S;
  using exec_space    = ES;
  using team_member_t = Impl::team_member_t<ES>;

  static constexpr int RankC = node_type::Rank;
  static constexpr int NumK  = node_type::NumContracted;
  static constexpr int RankA = NA::Rank;
  static constexpr int RankB = NB::Rank;
  static constexpr int FreeA = RankA - NumK;
  static constexpr int FreeB = RankB - NumK;

 private:
  using a_layout_t = typename AEval::storage_type::layout_t;
  using b_layout_t = typename BEval::storage_type::layout_t;
  using c_layout_t = typename CNode::storage_type::layout_t;

 public:
  static constexpr int SA = Impl::extent_product<c_layout_t, 0, FreeA>();
  static constexpr int SB = Impl::extent_product<c_layout_t, FreeA, RankC>();
  static constexpr int SK = Impl::extent_product<a_layout_t, FreeA, RankA>();

  static_assert(FreeA + FreeB == RankC,
                "free-mode counts must sum to the output rank");
  static_assert(a_layout_t::rank == RankA,
                "staged A must carry one extent per A mode");
  static_assert(b_layout_t::rank == RankB,
                "staged B must carry one extent per B mode");
  static_assert(c_layout_t::rank == RankC,
                "destination C must carry one extent per output mode");
  static_assert((Impl::extents_agree<a_layout_t, 0, c_layout_t, 0, FreeA>()),
                "staged A's free extents must match C's leading extents");
  static_assert(
      (Impl::extents_agree<b_layout_t, NumK, c_layout_t, FreeA, FreeB>()),
      "staged B's free extents must match C's trailing extents");
  static_assert((Impl::extents_agree<a_layout_t, FreeA, b_layout_t, 0, NumK>()),
                "staged A's and B's contracted extents must match");

  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type ops,
                            const team_member_t& team)
      : node_(n),
        c_(ops.c),
        a2_(regroup_view(ops.a.node().storage_, Split<FreeA, RankA>{})),
        b2_(regroup_view(ops.b.node().storage_, Split<NumK, RankB>{})),
        c2_(regroup_view(ops.c.storage_, Split<FreeA, RankC>{})),
        team_(team) {}

  KOKKOS_FORCEINLINE_FUNCTION void operator()(int i, int j) const {
    value_type acc{};
    for (int k = 0; k < SK; ++k) acc += a2_(i, k) * b2_(k, j);
    c2_(i, j) = acc;
  }

  KOKKOS_FUNCTION auto operator()() const {
    const auto self = *this;
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team_, SA * SB),
                         [=](int t) { self(t / SB, t % SB); });
    return make_value_evaluator(make_interm_node(c_.storage_, node_.hook_op),
                                team_);
  }

 private:
  using a_matrix_t = decltype(regroup_view(
      std::declval<typename AEval::storage_type>(), Split<FreeA, RankA>{}));
  using b_matrix_t = decltype(regroup_view(
      std::declval<typename BEval::storage_type>(), Split<NumK, RankB>{}));
  using c_matrix_t = decltype(regroup_view(
      std::declval<typename CNode::storage_type>(), Split<FreeA, RankC>{}));

  node_type     node_;
  CNode         c_;
  a_matrix_t    a2_;
  b_matrix_t    b2_;
  c_matrix_t    c2_;
  team_member_t team_;
};

// ---------------------------------------------------------------------------
// Tag2 CombineTag — P{modes} = fn(A{modes}, B{modes}, ...), pointwise, N-ary,
// multi-output.
//
// The Tag2 counterpart of Evaluator/Team.hpp's Specialization 5, decomposed the
// same way the Tag2 contraction is: this evaluator neither carves scratch nor
// stages nor gathers axes. The caller hands it operands that are ALREADY
// aligned with the output -- staged, relabeled, or read straight off a subview
// -- plus the destination node(s) to write, and the evaluator only checks that
// every operand presents the output's extents. A permuted operand is composed
// upstream out of the relabel and StageTag evaluators above.
//
// fn is defined (NodeHandle.hpp) to see the GLOBAL output coordinate. Tag2
// consumes tile_idx at the InputTag step and does not retain it, so the tile's
// global offset is carried explicitly as CombineOperands::origin; the default
// all-zero origin makes fn see tile-local coordinates.
//
// Like the contraction, extents are read statically off each storage's layout,
// so operands must be statically tiled.
// ---------------------------------------------------------------------------
namespace Impl {

// The destination pack: one IntermTag node per output, all sharing one layout.
template <typename T>
struct out_array_info;
template <typename T, std::size_t M>
struct out_array_info<Kokkos::Array<T, M>> {
  using node_type          = T;
  static constexpr int num = static_cast<int>(M);
};

}  // namespace Impl

// The tiling type: one value evaluator per operand, one destination node per
// output, and the output tile's global origin. Operand storages need not share
// a layout -- a relabeled operand is a strided retype of another tile's bytes
// -- so they live in a DeviceTuple rather than a Kokkos::Array.
template <typename OutArray, typename... OpEvals>
struct CombineOperands {
  using out_node_t  = typename Impl::out_array_info<OutArray>::node_type;
  using ops_tuple_t = DeviceTuple<OpEvals...>;
  static constexpr int NumOut = Impl::out_array_info<OutArray>::num;
  static constexpr int Rank   = out_node_t::Rank;

  ops_tuple_t              ops;
  OutArray                 outs;
  Kokkos::Array<int, Rank> origin{};  // {} == tile-local coordinates

  // Chainable origin, so the factory below stays a plain operands-first /
  // destination-last call with no optional trailing argument to disambiguate.
  KOKKOS_FUNCTION CombineOperands at(Kokkos::Array<int, Rank> o) const {
    CombineOperands out = *this;
    out.origin          = o;
    return out;
  }
};

namespace Impl {

// Split the operands-first / destination-last argument list. Mirrors
// NodeHandle.hpp's combine_from_args, but tuple-based and KOKKOS_FUNCTION
// because this one runs inside the kernel.
template <typename Tup, std::size_t... Is>
KOKKOS_FUNCTION auto combine_operands_from(const Tup& t,
                                           std::index_sequence<Is...>) {
  constexpr std::size_t Last = sizeof...(Is);
  // as_result_array is exactly the bare-node-or-array rule: T -> Array<T,1>,
  // Array<T,M> passes through.
  const auto outs = as_result_array(t.template get<Last>());
  return CombineOperands<std::remove_const_t<decltype(outs)>,
                         tuple_element_t<Is, Tup>...>{
      DeviceTuple<tuple_element_t<Is, Tup>...>(t.template get<Is>()...),
      outs,
      {}};
}

}  // namespace Impl

// make_combine_operands(a, b, ..., dest) -- operands first, destination last,
// matching make_combine_node's operands-first/fn-last convention and reading
// like ContractOperands{A, B, C}. `dest` is either one IntermTag node
// (NumOut == 1) or a Kokkos::Array of them.
template <typename... Args>
KOKKOS_FUNCTION auto make_combine_operands(Args... args) {
  static_assert(sizeof...(Args) >= 2,
                "make_combine_operands needs at least one operand and a "
                "destination");
  return Impl::combine_operands_from(
      DeviceTuple<Args...>(args...),
      std::make_index_sequence<sizeof...(Args) - 1>{});
}

namespace Impl {

template <typename PermSeq, typename Eval>
struct PermutedAt {
  using perm_seq     = PermSeq;
  using eval_type    = Eval;
  using storage_type = typename Eval::storage_type;
  using value_type   = typename Eval::value_type;
  using exec_space   = typename Eval::exec_space;
  using inv_perm_seq = inverse_perm_seq_t<PermSeq>;

  static constexpr int Rank = static_cast<int>(storage_type::layout_t::rank);

  static_assert(PermSeq::size() == static_cast<std::size_t>(Rank),
                "PermutedAt: permutation must have one entry per operand mode");

  Eval e;

  template <typename Coord>
  KOKKOS_FORCEINLINE_FUNCTION value_type& at(const Coord& coord) const {
    return read_at(coord, inv_perm_seq{});
  }

 private:
  template <typename Coord, int... Inv>
  KOKKOS_FORCEINLINE_FUNCTION value_type& read_at(
      const Coord& coord, std::integer_sequence<int, Inv...>) const {
    return e.node().storage_[Index<Rank>{coord[Inv]...}];
  }
};

template <typename OpEval>
struct permuted_at_traits {
  static constexpr bool is_permuted = false;
  using perm_seq =
      iota_seq_t<0, static_cast<int>(OpEval::storage_type::layout_t::rank)>;
};

template <typename PermSeq, typename Eval>
struct permuted_at_traits<PermutedAt<PermSeq, Eval>> {
  static constexpr bool is_permuted = true;
  using perm_seq                    = PermSeq;
};

template <typename OpEval>
inline constexpr bool is_permuted_at_v =
    permuted_at_traits<OpEval>::is_permuted;

template <typename OpEval>
using read_perm_t = typename permuted_at_traits<OpEval>::perm_seq;

template <typename PermSeq, typename Eval>
KOKKOS_FUNCTION auto make_permuted_at(Eval e) {
  if constexpr (is_identity_v<PermSeq>)
    return e;
  else
    return PermutedAt<PermSeq, Eval>{e};
}

template <typename OpEval, typename Coord>
KOKKOS_FORCEINLINE_FUNCTION decltype(auto) combine_read(const OpEval& op,
                                                        const Coord&  coord) {
  if constexpr (is_permuted_at_v<OpEval>)
    return op.at(coord);
  else
    return op.node().storage_[coord];
}

template <typename OpLayout, typename OutLayout, int... P, std::size_t... I>
KOKKOS_FUNCTION constexpr bool perm_extents_agree(
    std::integer_sequence<int, P...>, std::index_sequence<I...>) noexcept {
  return ((OpLayout::extent(P) == OutLayout::extent(static_cast<int>(I))) &&
          ...);
}

// Does one operand present the output's shape? Stated at namespace scope so
// the evaluator can fold it over its operand pack in a single static_assert.
//
// An if constexpr chain rather than a `&&` one, for the same reason
// operand_stageable_v is spelled that way: extents_agree reads Rank extents off
// the operand layout, which is ill-formed -- not merely false -- when that
// layout has fewer dimensions, and `&&` would instantiate it anyway.
template <int Rank, typename OutLayout, typename OpEval>
inline constexpr bool combine_op_aligned_v = [] {
  if constexpr (OpEval::storage_type::layout_t::rank != Rank)
    return false;
  else
    return perm_extents_agree<typename OpEval::storage_type::layout_t,
                              OutLayout>(read_perm_t<OpEval>{},
                                         std::make_index_sequence<Rank>{});
}();

}  // namespace Impl

template <typename ES, typename CombineFn, typename IntCRank, typename S,
          typename CModesSeq, typename IntNumOut, typename... Ops,
          typename OutArray, typename... OpEvals>
class Evaluator<TeamPolicyTag2<ES>,
                NodeHandle<CombineTag, CombineFn, IntCRank, S, ES, CModesSeq,
                           IntNumOut, Ops...>,
                CombineOperands<OutArray, OpEvals...>> {
 public:
  using node_type     = NodeHandle<CombineTag, CombineFn, IntCRank, S, ES,
                                   CModesSeq, IntNumOut, Ops...>;
  using policy_tag    = TeamPolicyTag2<ES>;
  using tiling_type   = CombineOperands<OutArray, OpEvals...>;
  using value_type    = S;
  using exec_space    = ES;
  using team_member_t = Impl::team_member_t<ES>;

  static constexpr int Rank   = node_type::Rank;
  static constexpr int NumOps = node_type::NumOps;
  static constexpr int NumOut = node_type::NumOut;

  using out_node_t   = typename tiling_type::out_node_t;
  using out_layout_t = typename out_node_t::storage_type::layout_t;
  using result_type =
      Kokkos::Array<Impl::value_evaluator_t<out_node_t>, NumOut>;

 private:
  using ops_seq  = std::make_index_sequence<NumOps>;
  using outs_seq = std::make_index_sequence<NumOut>;

 public:
  static_assert(sizeof...(OpEvals) == NumOps,
                "combine operands must supply one value evaluator per node "
                "operand");
  static_assert(tiling_type::NumOut == NumOut,
                "combine operands must supply one destination node per output "
                "the combine fn emits");
  static_assert(Impl::has_node_tag_v<IntermTag, out_node_t>,
                "a combine destination must be an intermediate node wrapping "
                "the storage to write");
  static_assert(out_layout_t::rank == Rank,
                "combine destination must carry one extent per output mode");
  static_assert(
      (Impl::combine_op_aligned_v<Rank, out_layout_t, OpEvals> && ...),
      "Tag2 combine: every operand must present the output's extents on every "
      "mode, read through its own access permutation -- Tag2 gathers no axes, "
      "so a permuted operand must arrive either relabeled into the output "
      "order (AlignedOperands) or wrapped in PermutedAt (PermutedOperands)");

  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type t,
                            const team_member_t& team)
      : fn_(n.fn), ops_(t.ops), outs_(t.outs), origin_(t.origin), team_(team) {}

  // Pure per-element read: takes one index per output mode and writes nothing.
  // Distinct from the storing operator()(coord) below, which is the per-element
  // step a fused level driver calls.
  template <typename... Idx>
    requires(sizeof...(Idx) == Rank)
  KOKKOS_FUNCTION Kokkos::Array<value_type, NumOut> operator()(
      Idx... idx) const {
    return apply_at(Impl::Index<Rank>{static_cast<int>(idx)...});
  }

  template <typename Coord>
    requires(!std::is_integral_v<Coord>)
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Coord& coord) const {
    const Kokkos::Array<value_type, NumOut> r = apply_at(coord);
    TENSOR_PRAGMA_UNROLL
    for (int m = 0; m < NumOut; ++m) outs_[m].storage_[coord] = r[m];
  }

  // Team-parallel evaluation: apply fn over the whole output tile, scattering
  // component m into destination m, and hand back one value evaluator per
  // output. Like the contraction, no barrier of its own -- the caller fences
  // around it.
  KOKKOS_FUNCTION auto operator()() const {
    const auto self = *this;
    // Every output shares one layout, so the first drives the traversal.
    const auto out0 = outs_[0].storage_;
    Impl::team_for_each_coord(team_, out0, [=](auto coord) { self(coord); });
    return make_results(outs_seq{});
  }

 private:
  // Shared by both operators: gather every operand's value at `coord`, apply fn
  // at the GLOBAL coordinate, and normalize a scalar-or-array result to NumOut
  // components. Templated on the coordinate type so it takes both an
  // Impl::Index (what a layout's delinearize hands back) and a Kokkos::Array.
  template <typename Coord>
  KOKKOS_FUNCTION Kokkos::Array<value_type, NumOut> apply_at(
      const Coord& coord) const {
    Kokkos::Array<int, Rank> gidx{};
    TENSOR_PRAGMA_UNROLL
    for (int d = 0; d < Rank; ++d) gidx[d] = origin_[d] + coord[d];
    return Impl::as_output_array<value_type>(
        Impl::apply_combine(fn_, gidx, gather_vals(coord, ops_seq{})));
  }

  template <typename Coord, std::size_t... Ks>
  KOKKOS_FUNCTION Kokkos::Array<value_type, NumOps> gather_vals(
      const Coord& coord, std::index_sequence<Ks...>) const {
    return {static_cast<value_type>(
        Impl::combine_read(ops_.template get<Ks>(), coord))...};
  }

  template <std::size_t... Ms>
  KOKKOS_FUNCTION result_type make_results(std::index_sequence<Ms...>) const {
    return {make_value_evaluator(outs_[Ms], team_)...};
  }

  [[no_unique_address]] CombineFn fn_;
  DeviceTuple<OpEvals...>         ops_;
  OutArray                        outs_;
  Kokkos::Array<int, Rank>        origin_;
  team_member_t                   team_;
};
