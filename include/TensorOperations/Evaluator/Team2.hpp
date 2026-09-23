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

// Functional input: same contract as the InputTag evaluator above -- given a
// tile coordinate, hand back a value evaluator over that tile -- but the tile
// is a FunctionalView, which computes its elements instead of addressing them.
//
// TILING ACROSS TEAMS is the `origin` fold below: tile coordinate times tile
// extent, per axis, which is the coordinate-space twin of what
// subview_tile_params does in address space (`base += outer_idx[d] *
// stride(d)`). It is spelled here rather than obtained from tile_layout()
// because tile_layout builds a 2N-dimensional layout whose outer half resolves
// to a FLAT OFFSET -- exactly the thing a functional input does not have. The
// functor wants coordinates, so the tiling is done in coordinates.
//
// TRAVERSAL ORDER comes from the node, defaulting to LayoutRight. It is free
// to differ from the destination's, because the copy is coordinate-indexed on
// both sides; what it buys is control over which addresses consecutive lanes
// make the functor touch.
template <typename ES, typename Fn, typename ModesSeq, typename ValueType,
          typename Layout, typename HookOp, typename Tile_>
class Evaluator<
    TeamPolicyTag2<ES>,
    NodeHandle<FunctionalTag, Fn, ModesSeq, ValueType, ES, Layout, HookOp>,
    Tile_> {
 public:
  using node_type =
      NodeHandle<FunctionalTag, Fn, ModesSeq, ValueType, ES, Layout, HookOp>;
  // The tile inherits the declared tensor's order, exactly as tiling a real
  // view preserves that view's memory order.
  using order_tag     = typename node_type::order_tag;
  using policy_tag    = TeamPolicyTag2<ES>;
  using tiling_type   = Tile_;
  using exec_space    = ES;
  using team_member_t = Impl::team_member_t<ES>;

  static constexpr int Rank = node_type::Rank;
  static_assert(static_cast<int>(Tile_::rank) == Rank,
                "functional input: tile rank must equal the node's rank");

  using layout_t =
      decltype(make_tile_layout(std::declval<Tile_>(), order_tag{}));
  using view_t = FunctionalView<Fn, layout_t, ValueType, ES>;

  KOKKOS_FUNCTION Evaluator(node_type n, Tile_ t, const team_member_t& team)
      : fn_(n.fn_), hook_(n.hook_op), tile_(t), team_(team) {}

  KOKKOS_FUNCTION auto operator()(
      Kokkos::Array<int, Tile_::rank> tile_idx) const {
    // The functor is called at the GLOBAL coordinate, so the tile's element
    // origin is folded in here rather than at every read.
    Kokkos::Array<int, Rank> origin{};
    for (int d = 0; d < Rank; ++d) origin[d] = tile_idx[d] * tile_.extent(d);
    return make_value_evaluator(
        make_interm_node(
            view_t{fn_, make_tile_layout(tile_, order_tag{}), origin}, hook_),
        team_);
  }

 private:
  Fn                           fn_;
  [[no_unique_address]] HookOp hook_;
  Tile_                        tile_;
  team_member_t                team_;
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

template <typename ES, typename Operand, typename ModesSeq, typename NodeTile,
          typename Tile_>
class Evaluator<TeamPolicyTag2<ES>,
                NodeHandle<StagedTag, Operand, ModesSeq, NodeTile>, Tile_> {
 public:
  using node_type     = NodeHandle<StagedTag, Operand, ModesSeq, NodeTile>;
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

  // Split so a level driver can issue every member's LOADS before any member's
  // STORE. Fused, the members run load-compute-store in turn, and each store to
  // shared sits between the next member's loads -- nvcc cannot prove the store
  // misses the operator buffer, so it reloads. The nine members of a level all
  // take their A row at the same index from the SAME operator slot, so those
  // reloads are pure waste: measured 180 LDS per warp against 82 distinct
  // addresses, the operator elements fetched 9x each.
  //
  // Restrict is not the fix -- nvcc does not use it for shared-memory aliasing.
  // Hoisting the stores past the loads removes the ambiguity outright: with no
  // store in between, eliminating a duplicate load is valid whatever aliases
  // what.
  KOKKOS_FORCEINLINE_FUNCTION value_type compute(int i, int j) const {
    const auto a_row = slice(a2_, i, ALL);
    const auto b_col = slice(b2_, ALL, j);
    value_type acc{};
    for (int k = 0; k < SK; ++k) acc += a_row(k) * b_col(k);
    return acc;
  }

  KOKKOS_FORCEINLINE_FUNCTION void store(int i, int j, value_type acc) const {
    c2_(i, j) = acc;
  }

  KOKKOS_FORCEINLINE_FUNCTION void operator()(int i, int j) const {
    store(i, j, compute(i, j));
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
    return extents_agree<typename OpEval::storage_type::layout_t, 0, OutLayout,
                         0, Rank>();
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
      "Tag2 combine: every operand must already present the output's extents "
      "on every mode -- Tag2 gathers no axes, so a permuted operand must be "
      "relabeled and staged into the output order before it gets here");

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

  // Split for the same reason the contraction is: a level driver can then issue
  // every member's operand GATHER before any member's STORE. Both sides are
  // shared memory here, so a store between two members' gathers stops nvcc
  // eliminating a load the members share. This graph has one combine member per
  // level so there is nothing to collect today -- the split costs nothing and
  // makes the win available to graphs whose combine levels are wider.
  template <typename Coord>
    requires(!std::is_integral_v<Coord>)
  KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<value_type, NumOut> compute(
      const Coord& coord) const {
    return apply_at(coord);
  }

  template <typename Coord>
    requires(!std::is_integral_v<Coord>)
  KOKKOS_FORCEINLINE_FUNCTION void store(
      const Coord& coord, const Kokkos::Array<value_type, NumOut>& r) const {
    TENSOR_PRAGMA_UNROLL
    for (int m = 0; m < NumOut; ++m) outs_[m].storage_[coord] = r[m];
  }

  template <typename Coord>
    requires(!std::is_integral_v<Coord>)
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Coord& coord) const {
    store(coord, compute(coord));
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
        ops_.template get<Ks>().node().storage_[coord])...};
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

// The sink tiling: a combine whose fn returns void writes to global itself and
// keeps no destination. It carries only the operand value evaluators and the
// output tile's global origin -- there is no output node to derive Rank from,
// so Rank is a template parameter here (CombineOperands reads it off
// out_node_t).
template <int SinkRank, typename... OpEvals>
struct CombineSinkOperands {
  using ops_tuple_t           = DeviceTuple<OpEvals...>;
  static constexpr int NumOut = 0;
  static constexpr int Rank   = SinkRank;

  ops_tuple_t                  ops;
  Kokkos::Array<int, SinkRank> origin{};  // {} == tile-local coordinates

  KOKKOS_FUNCTION CombineSinkOperands at(Kokkos::Array<int, SinkRank> o) const {
    CombineSinkOperands out = *this;
    out.origin              = o;
    return out;
  }
};

// make_combine_sink_operands<Rank>(a, b, ...) -- operands only, no destination.
// The mirror of make_combine_operands for the NumOut == 0 case.
template <int SinkRank, typename... OpEvals>
KOKKOS_FUNCTION auto make_combine_sink_operands(OpEvals... ops) {
  static_assert(sizeof...(OpEvals) >= 1,
                "a sink combine still needs at least one operand to read");
  return CombineSinkOperands<SinkRank, OpEvals...>{
      DeviceTuple<OpEvals...>(ops...), {}};
}

// The sink evaluator: gather every operand at a coordinate, apply fn at the
// GLOBAL coordinate, and discard the void result. The functor performs the
// scatter (e.g. Kokkos::atomic_add into a global view) itself; the library
// neither allocates a destination nor learns the index map. Matched by the
// NumOut == 0 node (IntNumOut == integral_constant<int, 0>) paired with the
// sink tiling, so the NumOut >= 1 evaluator above is never perturbed.
template <typename ES, typename CombineFn, typename IntCRank, typename S,
          typename CModesSeq, typename... Ops, int SinkRank,
          typename... OpEvals>
class Evaluator<TeamPolicyTag2<ES>,
                NodeHandle<CombineTag, CombineFn, IntCRank, S, ES, CModesSeq,
                           std::integral_constant<int, 0>, Ops...>,
                CombineSinkOperands<SinkRank, OpEvals...>> {
 public:
  using node_type =
      NodeHandle<CombineTag, CombineFn, IntCRank, S, ES, CModesSeq,
                 std::integral_constant<int, 0>, Ops...>;
  using policy_tag    = TeamPolicyTag2<ES>;
  using tiling_type   = CombineSinkOperands<SinkRank, OpEvals...>;
  using value_type    = S;
  using exec_space    = ES;
  using team_member_t = Impl::team_member_t<ES>;

  static constexpr int Rank   = node_type::Rank;
  static constexpr int NumOps = node_type::NumOps;
  static constexpr int NumOut = 0;

 private:
  using ops_seq = std::make_index_sequence<NumOps>;
  using op0_t   = std::tuple_element_t<0, std::tuple<OpEvals...>>;

 public:
  static_assert(sizeof...(OpEvals) == NumOps,
                "combine operands must supply one value evaluator per node "
                "operand");
  static_assert(SinkRank == Rank,
                "sink tiling rank must match the combine node's output rank");
  static_assert(
      (Impl::combine_op_aligned_v<Rank, typename op0_t::storage_type::layout_t,
                                  OpEvals> &&
       ...),
      "Tag2 sink combine: every operand must already present the same extents "
      "on every mode -- a permuted operand must be relabeled and staged first");

  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type t,
                            const team_member_t& team)
      : fn_(n.fn), ops_(t.ops), origin_(t.origin), team_(team) {}

  // The per-element step a fused level driver calls, split the same way: the
  // operand GATHER is the compute, applying fn (which scatters) is the store.
  // A sink's fn writes GLOBAL while its operands are shared, so the two are
  // already reorderable across address spaces -- the split is for uniformity
  // with the multi-output combine, so one driver phases every member kind.
  template <typename Coord>
    requires(!std::is_integral_v<Coord>)
  KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<value_type, NumOps> compute(
      const Coord& coord) const {
    return gather_vals(coord, ops_seq{});
  }

  template <typename Coord>
    requires(!std::is_integral_v<Coord>)
  KOKKOS_FORCEINLINE_FUNCTION void store(
      const Coord& coord, const Kokkos::Array<value_type, NumOps>& v) const {
    Kokkos::Array<int, Rank> gidx{};
    TENSOR_PRAGMA_UNROLL
    for (int d = 0; d < Rank; ++d) gidx[d] = origin_[d] + coord[d];
    Impl::apply_combine(fn_, gidx, v);
  }

  template <typename Coord>
    requires(!std::is_integral_v<Coord>)
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Coord& coord) const {
    store(coord, compute(coord));
  }

  // A sink-only level has no output slot to drive its traversal, so it drives
  // from operand 0's storage tile instead: every operand presents the output
  // extents, so any of them enumerates the output coordinate set.
  KOKKOS_FUNCTION auto iter_view() const {
    return ops_.template get<0>().node().storage_;
  }

 private:
  template <typename Coord, std::size_t... Ks>
  KOKKOS_FUNCTION Kokkos::Array<value_type, NumOps> gather_vals(
      const Coord& coord, std::index_sequence<Ks...>) const {
    return {static_cast<value_type>(
        ops_.template get<Ks>().node().storage_[coord])...};
  }

  [[no_unique_address]] CombineFn fn_;
  DeviceTuple<OpEvals...>         ops_;
  Kokkos::Array<int, Rank>        origin_;
  team_member_t                   team_;
};

// ---------------------------------------------------------------------------
// Tag2 ReduceTag -- the parallel_reduce-shaped node (see NodeHandle.hpp).
//
// Per output coordinate: each operand's pointer is advanced ONCE by its
// output-bound axes; per reduction coordinate it is advanced by its
// reduction-bound axes (constant strides, unrolled constexpr loop); fn then
// reads through accessors whose free axes are constant-stride offsets.
// Nothing is loaded unless fn asks for it.
// ---------------------------------------------------------------------------

/// An operand as fn sees it: its bound axes are already folded into `ptr`,
/// and operator() takes the FREE axes in the operand's own order.
template <typename V, typename FreeStridesSeq>
struct ReduceAccessor;
template <typename V, int... S>
struct ReduceAccessor<V, std::integer_sequence<int, S...>> {
  using value_type          = V;
  static constexpr int rank = static_cast<int>(sizeof...(S));
  const V*             ptr;

  template <typename... I>
    requires(sizeof...(I) == sizeof...(S))
  KOKKOS_FORCEINLINE_FUNCTION const V& operator()(I... i) const {
    return ptr[(0 + ... + (S * static_cast<int>(i)))];
  }
};

template <typename OutArray, typename... OpEvals>
struct ReduceOperands {
  using out_node_t  = typename Impl::out_array_info<OutArray>::node_type;
  using ops_tuple_t = DeviceTuple<OpEvals...>;
  static constexpr int NumOut = Impl::out_array_info<OutArray>::num;
  static constexpr int Rank   = out_node_t::Rank;

  ops_tuple_t              ops;
  OutArray                 outs;
  Kokkos::Array<int, Rank> origin{};  // {} == tile-local coordinates

  KOKKOS_FUNCTION ReduceOperands at(Kokkos::Array<int, Rank> o) const {
    ReduceOperands out = *this;
    out.origin         = o;
    return out;
  }
};

namespace Impl {

template <typename Tup, std::size_t... Is>
KOKKOS_FUNCTION auto reduce_operands_from(const Tup& t,
                                          std::index_sequence<Is...>) {
  constexpr std::size_t Last = sizeof...(Is);
  const auto            outs = as_result_array(t.template get<Last>());
  return ReduceOperands<std::remove_const_t<decltype(outs)>,
                        tuple_element_t<Is, Tup>...>{
      DeviceTuple<tuple_element_t<Is, Tup>...>(t.template get<Is>()...), outs,
      {}};
}

// Axis classification of one operand against (Out, Red): for axis d, the
// position of its label in Out (or -1), in Red (or -1), and its stride.
template <typename OpModes, typename Seq>
constexpr auto reduce_axis_positions() {
  constexpr auto            m = seq_to_array(OpModes{});
  constexpr auto            f = seq_to_array(Seq{});
  std::array<int, m.size()> pos{};
  for (std::size_t d = 0; d < m.size(); ++d) {
    pos[d] = -1;
    for (std::size_t j = 0; j < f.size(); ++j)
      if (f[j] == m[d]) pos[d] = static_cast<int>(j);
  }
  return pos;
}
template <typename OpModes, typename Seq>
using reduce_axis_positions_seq_t =
    array_to_seq_t<reduce_axis_positions<OpModes, Seq>()>;

template <typename Layout, std::size_t... Ds>
constexpr std::array<int, sizeof...(Ds)> layout_strides(
    std::index_sequence<Ds...>) {
  return {Layout::stride(static_cast<int>(Ds))...};
}
template <typename Layout>
using layout_strides_seq_t = array_to_seq_t<layout_strides<Layout>(
    std::make_index_sequence<static_cast<std::size_t>(Layout::rank)>{})>;

template <typename OpModes, typename OutSeq, typename RedSeq, typename Layout>
constexpr std::size_t reduce_free_count() {
  constexpr auto po = reduce_axis_positions<OpModes, OutSeq>();
  constexpr auto pr = reduce_axis_positions<OpModes, RedSeq>();
  std::size_t    n  = 0;
  for (std::size_t d = 0; d < po.size(); ++d)
    if (po[d] < 0 && pr[d] < 0) ++n;
  return n;
}
template <typename OpModes, typename OutSeq, typename RedSeq, typename Layout>
constexpr auto reduce_free_strides() {
  constexpr auto po = reduce_axis_positions<OpModes, OutSeq>();
  constexpr auto pr = reduce_axis_positions<OpModes, RedSeq>();
  std::array<int, reduce_free_count<OpModes, OutSeq, RedSeq, Layout>()> fs{};
  std::size_t n = 0;
  for (std::size_t d = 0; d < po.size(); ++d)
    if (po[d] < 0 && pr[d] < 0) fs[n++] = Layout::stride(static_cast<int>(d));
  return fs;
}
template <typename OpModes, typename OutSeq, typename RedSeq, typename Layout>
using reduce_free_strides_seq_t =
    array_to_seq_t<reduce_free_strides<OpModes, OutSeq, RedSeq, Layout>()>;

// Every bound axis must present the extent of what it is bound to.
template <typename OpModes, typename OutSeq, typename RedSeq, typename Layout,
          typename OutLayout, typename RedExtSeq>
constexpr bool reduce_bound_extents_agree() {
  constexpr auto po  = reduce_axis_positions<OpModes, OutSeq>();
  constexpr auto pr  = reduce_axis_positions<OpModes, RedSeq>();
  constexpr auto rex = seq_to_array(RedExtSeq{});
  for (std::size_t d = 0; d < po.size(); ++d) {
    const int e = Layout::extent(static_cast<int>(d));
    if (po[d] >= 0 && e != OutLayout::extent(po[d])) return false;
    if (pr[d] >= 0 && e != rex[static_cast<std::size_t>(pr[d])]) return false;
  }
  return true;
}

template <typename Seq>
constexpr int seq_product() {
  constexpr auto a = seq_to_array(Seq{});
  int            p = 1;
  for (std::size_t i = 0; i < a.size(); ++i) p *= a[i];
  return p;
}

// Is fn callable as fn(int x FullRank, const Acc&..., AccT&)?
template <typename Fn, typename AccT, typename IdxSeq, typename... Accs>
struct reduce_fn_callable;
template <typename Fn, typename AccT, std::size_t... Is, typename... Accs>
struct reduce_fn_callable<Fn, AccT, std::index_sequence<Is...>, Accs...>
    : std::bool_constant<std::is_invocable_v<
          const Fn&, decltype((void(Is), 0))..., const Accs&..., AccT&>> {};

}  // namespace Impl

template <typename... Args>
KOKKOS_FUNCTION auto make_reduce_operands(Args... args) {
  static_assert(sizeof...(Args) >= 2,
                "make_reduce_operands needs at least one operand and a "
                "destination");
  return Impl::reduce_operands_from(
      DeviceTuple<Args...>(args...),
      std::make_index_sequence<sizeof...(Args) - 1>{});
}

template <typename ES, typename ReduceFn, typename IntCRank, typename S,
          typename RModesSeq, typename RedSeq, typename IntNumOut,
          typename Reducer, typename TileT, typename RedExtSeq,
          typename... Ops, typename OutArray, typename... OpEvals>
class Evaluator<TeamPolicyTag2<ES>,
                NodeHandle<ReduceTag, ReduceFn, IntCRank, S, ES, RModesSeq,
                           RedSeq, IntNumOut, Reducer, TileT, RedExtSeq,
                           Ops...>,
                ReduceOperands<OutArray, OpEvals...>> {
 public:
  using node_type =
      NodeHandle<ReduceTag, ReduceFn, IntCRank, S, ES, RModesSeq, RedSeq,
                 IntNumOut, Reducer, TileT, RedExtSeq, Ops...>;
  using policy_tag    = TeamPolicyTag2<ES>;
  using tiling_type   = ReduceOperands<OutArray, OpEvals...>;
  using value_type    = S;
  using exec_space    = ES;
  using team_member_t = Impl::team_member_t<ES>;

  static constexpr int Rank     = node_type::Rank;
  static constexpr int RedRank  = node_type::RedRank;
  static constexpr int FullRank = Rank + RedRank;
  static constexpr int NumOps   = node_type::NumOps;
  static constexpr int NumOut   = node_type::NumOut;

  using out_node_t   = typename tiling_type::out_node_t;
  using out_layout_t = typename out_node_t::storage_type::layout_t;
  using result_type =
      Kokkos::Array<Impl::value_evaluator_t<out_node_t>, NumOut>;
  using acc_type =
      std::conditional_t<NumOut == 1, value_type,
                         Kokkos::Array<value_type, NumOut>>;

  static_assert(!std::is_same_v<RedExtSeq, void>,
                "reduce member: its reduction extents are unresolved, which "
                "means this node was never handed to LevelGraph::add");

 private:
  using ops_seq = std::make_index_sequence<NumOps>;

  template <std::size_t K>
  using op_modes_t =
      typename tuple_element_t<K, typename node_type::ops_tuple_t>::modes_seq;
  template <std::size_t K>
  using op_layout_t =
      typename tuple_element_t<K, DeviceTuple<OpEvals...>>::storage_type::
          layout_t;
  template <std::size_t K>
  using accessor_t = ReduceAccessor<
      value_type, Impl::reduce_free_strides_seq_t<op_modes_t<K>, RModesSeq,
                                                  RedSeq, op_layout_t<K>>>;

  template <std::size_t... Ks>
  static constexpr bool extents_agree(std::index_sequence<Ks...>) {
    return (Impl::reduce_bound_extents_agree<op_modes_t<Ks>, RModesSeq,
                                             RedSeq, op_layout_t<Ks>,
                                             out_layout_t, RedExtSeq>() &&
            ...);
  }
  template <std::size_t... Ks>
  static constexpr bool fn_callable(std::index_sequence<Ks...>) {
    return Impl::reduce_fn_callable<
        ReduceFn, acc_type,
        std::make_index_sequence<static_cast<std::size_t>(FullRank)>,
        accessor_t<Ks>...>::value;
  }

 public:
  static_assert(sizeof...(OpEvals) == NumOps,
                "reduce operands must supply one evaluator per operand");
  static_assert(tiling_type::NumOut == NumOut,
                "reduce operands must supply one destination per output");
  static_assert(Impl::has_node_tag_v<IntermTag, out_node_t>,
                "a reduce destination must be an intermediate node");
  static_assert(out_layout_t::rank == Rank,
                "reduce destination must carry one extent per output mode");
  static_assert(extents_agree(ops_seq{}),
                "Tag2 reduce: an operand axis bound to an output or "
                "reduction label must have that label's extent");
  static_assert(fn_callable(ops_seq{}),
                "reduce fn must be const-callable as fn(o_0, ..., o_{R-1}, "
                "rho_0, ..., rho_{RR-1}, acc_0, ..., acc_{N-1}, acc&): the "
                "output coordinate, the reduction coordinate, one accessor "
                "per operand (called with that operand's FREE axes), and the "
                "accumulator (V&, or Kokkos::Array<V, M>& with outputs<M>)");

  static constexpr int red_count = Impl::seq_product<RedExtSeq>();

  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type t,
                            const team_member_t& team)
      : fn_(n.fn), ops_(t.ops), outs_(t.outs), origin_(t.origin), team_(team) {}

  template <typename Coord>
    requires(!std::is_integral_v<Coord>)
  KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<value_type, NumOut> compute(
      const Coord& coord) const {
    Kokkos::Array<int, Rank> gidx{};
    TENSOR_PRAGMA_UNROLL
    for (int d = 0; d < Rank; ++d) gidx[d] = origin_[d] + coord[d];

    // Hoisted: each operand's pointer advanced by its output-bound axes.
    const Kokkos::Array<const value_type*, NumOps> base =
        bases(coord, ops_seq{});

    acc_type acc = init_acc();
    constexpr auto red_ext = Impl::seq_to_karray(RedExtSeq{});
    TENSOR_PRAGMA_UNROLL
    for (int lin = 0; lin < red_count; ++lin) {
      // Reduction coordinate, last label fastest (LabelWhole: tile-local ==
      // global). Constant extents, so the unrolled decode folds away.
      Kokkos::Array<int, static_cast<std::size_t>(RedRank > 0 ? RedRank : 1)>
          rho{};
      int rem = lin;
      TENSOR_PRAGMA_UNROLL
      for (int j = RedRank - 1; j >= 0; --j) {
        rho[j] = rem % red_ext[j];
        rem /= red_ext[j];
      }
      call(gidx, rho, base, acc, ops_seq{},
           std::make_index_sequence<static_cast<std::size_t>(Rank)>{},
           std::make_index_sequence<static_cast<std::size_t>(RedRank)>{});
    }
    if constexpr (NumOut == 1)
      return {acc};
    else
      return acc;
  }

  template <typename Coord>
    requires(!std::is_integral_v<Coord>)
  KOKKOS_FORCEINLINE_FUNCTION void store(
      const Coord& coord, const Kokkos::Array<value_type, NumOut>& r) const {
    TENSOR_PRAGMA_UNROLL
    for (int m = 0; m < NumOut; ++m) outs_[m].storage_[coord] = r[m];
  }

  template <typename Coord>
    requires(!std::is_integral_v<Coord>)
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Coord& coord) const {
    store(coord, compute(coord));
  }

  KOKKOS_FUNCTION auto operator()() const {
    const auto self = *this;
    const auto out0 = outs_[0].storage_;
    Impl::team_for_each_coord(team_, out0, [=](auto coord) { self(coord); });
    return make_results(std::make_index_sequence<NumOut>{});
  }

 private:
  KOKKOS_FORCEINLINE_FUNCTION static acc_type init_acc() {
    if constexpr (NumOut == 1) {
      return Reducer::init();
    } else {
      acc_type a{};
      for (int m = 0; m < NumOut; ++m) a[m] = Reducer::init();
      return a;
    }
  }

  // Operand K's pointer at the output coordinate: data() plus the output-
  // bound axes' strides times their coordinates (tile-local).
  template <std::size_t K, typename Coord>
  KOKKOS_FORCEINLINE_FUNCTION const value_type* base_of(
      const Coord& coord) const {
    constexpr auto pos = Impl::seq_to_karray(
        Impl::reduce_axis_positions_seq_t<op_modes_t<K>, RModesSeq>{});
    constexpr auto str =
        Impl::seq_to_karray(Impl::layout_strides_seq_t<op_layout_t<K>>{});
    constexpr int OpRank = op_layout_t<K>::rank;
    int           off    = 0;
    TENSOR_PRAGMA_UNROLL
    for (int d = 0; d < OpRank; ++d)
      if (pos[d] >= 0) off += str[d] * coord[pos[d]];
    return ops_.template get<K>().node().storage_.data() + off;
  }

  template <typename Coord, std::size_t... Ks>
  KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<const value_type*, NumOps> bases(
      const Coord& coord, std::index_sequence<Ks...>) const {
    return {base_of<Ks>(coord)...};
  }

  // Operand K's accessor at a reduction coordinate.
  template <std::size_t K, typename RhoArr>
  KOKKOS_FORCEINLINE_FUNCTION accessor_t<K> accessor_at(
      const value_type* base, const RhoArr& rho) const {
    constexpr auto pos = Impl::seq_to_karray(
        Impl::reduce_axis_positions_seq_t<op_modes_t<K>, RedSeq>{});
    constexpr auto str =
        Impl::seq_to_karray(Impl::layout_strides_seq_t<op_layout_t<K>>{});
    constexpr int OpRank = op_layout_t<K>::rank;
    int           off    = 0;
    TENSOR_PRAGMA_UNROLL
    for (int d = 0; d < OpRank; ++d)
      if (pos[d] >= 0) off += str[d] * rho[pos[d]];
    return accessor_t<K>{base + off};
  }

  template <typename RhoArr, std::size_t... Ks, std::size_t... Is,
            std::size_t... Js>
  KOKKOS_FORCEINLINE_FUNCTION void call(
      const Kokkos::Array<int, Rank>& gidx, const RhoArr& rho,
      const Kokkos::Array<const value_type*, NumOps>& base, acc_type& acc,
      std::index_sequence<Ks...>, std::index_sequence<Is...>,
      std::index_sequence<Js...>) const {
    fn_(gidx[Is]..., rho[Js]..., accessor_at<Ks>(base[Ks], rho)..., acc);
  }

  template <std::size_t... Ms>
  KOKKOS_FUNCTION result_type make_results(std::index_sequence<Ms...>) const {
    return {make_value_evaluator(outs_[Ms], team_)...};
  }

  [[no_unique_address]] ReduceFn fn_;
  DeviceTuple<OpEvals...>        ops_;
  OutArray                       outs_;
  Kokkos::Array<int, Rank>       origin_;
  team_member_t                  team_;
};
