#pragma once
// ---------------------------------------------------------------------------
// Structured operands -- tensors built from dense ones by tensor algebra and
// never materialised. Each is an operand of make_contraction_node
// (GeneralContraction.hpp), as are dense nodes (slots, functional inputs), and
// they nest:
//
//   delta<'a','b'>()               identity delta_ab, zero storage
//   delta<'r'>(idx<t>)             unit vector e_t(r): 1 where r == t
//   outer(x, y, ...)               outer product; the operands' label sets
//                                  must be disjoint; labels = concatenation
//   stack<'r'>(b_0, ..., b_{n-1})  lazy np.stack: the value is b_t where
//                                  r == t; every b_t carries the same label
//                                  SET (any order); labels = ('r', then b_0's
//                                  labels in b_0's order)
//   x.as<New...>()                 positional relabel over x's label order,
//                                  zero-copy (like a slot's .as<>)
//
// Lowering. Every operand lowers at compile time to a TERM SET, a sum of
// terms, each a product of dense leaves, label equalities (a == b) and
// constant bindings (l == t):
//
//   dense leaf     { leaf }
//   delta<a,b>()   { a == b }
//   delta<r>(t)    { r == t }
//   outer          cartesian product of the operands' term sets
//   stack<r>       union over t of b_t's terms, each with r == t added
//   .as<>          the same terms, every label renamed
//
// make_contraction_node multiplies its operands' term sets (first operand
// outermost) and expands each term against the graph's label map: labels a
// delta joins are merged, a constant binds its class, the rest is summed.
//
// The SEM gradient, and the label order it gets:
//
//   B = stack<'r'>(outer(hx, delta<'y','j'>(), delta<'z','k'>()),
//                  outer(delta<'x','i'>(), hy, delta<'z','k'>()),
//                  outer(delta<'x','i'>(), delta<'y','j'>(), hz));
//
// with hx{x,i}, hy{y,j}, hz{z,k}: branch 0's labels are (x,i, y,j, z,k), so
// B's are (r, x, i, y, j, z, k), and B.as<'s','x','l','y','m','z','n'>() is
// the same operator with r->s, i->l, j->m, k->n. Then
//
//   K = make_contraction_node<'e','a','k','j','i','b','n','m','l'>(
//           B, M, B.as<'s','x','l','y','m','z','n'>());
//
// is K_e = B^T M B as nine sum-factored terms, never the O(N^9) sum.
// ---------------------------------------------------------------------------
#include <TensorOperations/DeviceTuple.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Permute.hpp>

#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

namespace TensorOperations {

// --- the operand types -----------------------------------------------------

/// A constant index t, as a type: the argument of delta<'r'>(idx<t>).
template <int T>
struct ConstIndex {
  static_assert(T >= 0, "idx<t>: the index must be non-negative");
  static constexpr int value = T;
};
template <int T>
inline constexpr ConstIndex<T> idx{};

template <typename Op, typename ModesSeq>
struct Relabeled;

/// delta_ab: the identity between two labels. No storage.
template <int32_t A, int32_t B>
struct Delta {
  static_assert(A != B, "delta<a, b>(): the two labels must differ");
  using structured_operand_tag = void;
  using modes_seq              = std::integer_sequence<int32_t, A, B>;
  static constexpr int Rank    = 2;

  template <int32_t... New>
  constexpr auto as() const {
    static_assert(sizeof...(New) == 2, "delta as(): one label per axis");
    constexpr int32_t l[] = {New...};
    return Delta<l[0], l[1]>{};
  }
};

/// e_T(R): 1 where R == T, else 0. No storage.
template <int32_t R, int T>
struct UnitVector {
  using structured_operand_tag = void;
  using modes_seq              = std::integer_sequence<int32_t, R>;
  static constexpr int Rank    = 1;
  static constexpr int index   = T;

  template <int32_t... New>
  constexpr auto as() const {
    static_assert(sizeof...(New) == 1, "delta<r>(idx<t>) as(): one label");
    constexpr int32_t l[] = {New...};
    return UnitVector<l[0], T>{};
  }
};

namespace Impl {

template <typename... Seqs>
struct gc_cat_seq;
template <typename T>
struct gc_cat_seq<std::integer_sequence<T>> {
  using type = std::integer_sequence<T>;
};
template <typename T, T... As>
struct gc_cat_seq<std::integer_sequence<T, As...>> {
  using type = std::integer_sequence<T, As...>;
};
template <typename T, T... As, T... Bs, typename... Rest>
struct gc_cat_seq<std::integer_sequence<T, As...>,
                  std::integer_sequence<T, Bs...>, Rest...> {
  using type = typename gc_cat_seq<std::integer_sequence<T, As..., Bs...>,
                                   Rest...>::type;
};
template <typename... Seqs>
using gc_cat_seq_t = typename gc_cat_seq<Seqs...>::type;

template <typename... Ts>
using gc_first_t = std::tuple_element_t<0, std::tuple<Ts...>>;

template <typename Op, typename New>
constexpr void check_relabel() {
  static_assert(Op::modes_seq::size() == New::size(),
                "as(): one label per axis");
  static_assert(labels_distinct_v<New>, "as(): labels must be distinct");
}

}  // namespace Impl

/// outer(ops...): the product of operands with pairwise-disjoint label sets.
template <typename... Ops>
struct Outer {
  using structured_operand_tag = void;
  using modes_seq           = Impl::gc_cat_seq_t<std::integer_sequence<int32_t>,
                                                 typename Ops::modes_seq...>;
  static constexpr int Rank = static_cast<int>(modes_seq::size());

  std::tuple<Ops...> ops;

  template <int32_t... New>
  auto as() const {
    using NewSeq = std::integer_sequence<int32_t, New...>;
    Impl::check_relabel<Outer, NewSeq>();
    return Relabeled<Outer, NewSeq>{*this};
  }
};

/// stack<R>(branches...): the value is branch t where R == t.
template <int32_t R, typename... Branches>
struct Stack {
  using structured_operand_tag = void;
  using modes_seq =
      Impl::gc_cat_seq_t<std::integer_sequence<int32_t, R>,
                         typename Impl::gc_first_t<Branches...>::modes_seq>;
  static constexpr int     Rank = static_cast<int>(modes_seq::size());
  static constexpr int     size = static_cast<int>(sizeof...(Branches));
  static constexpr int32_t axis = R;

  std::tuple<Branches...> branches;

  template <int32_t... New>
  auto as() const {
    using NewSeq = std::integer_sequence<int32_t, New...>;
    Impl::check_relabel<Stack, NewSeq>();
    return Relabeled<Stack, NewSeq>{*this};
  }
};

/// An outer or stack read under other labels: Op's labels, positionally
/// renamed to ModesSeq. Relabelling again renames from Op's labels directly.
template <typename Op, typename ModesSeq>
struct Relabeled {
  using structured_operand_tag = void;
  using modes_seq              = ModesSeq;
  using operand_type           = Op;
  static constexpr int Rank    = static_cast<int>(ModesSeq::size());

  Op op;

  template <int32_t... New>
  auto as() const {
    using NewSeq = std::integer_sequence<int32_t, New...>;
    Impl::check_relabel<Relabeled, NewSeq>();
    return Relabeled<Op, NewSeq>{op};
  }
};

// --- the factories ---------------------------------------------------------

/// delta<'a','b'>(): the identity delta_ab.
template <int32_t A, int32_t B>
constexpr Delta<A, B> delta() {
  return {};
}

/// delta<'r'>(idx<t>): the unit vector e_t(r).
template <int32_t R, int T>
constexpr UnitVector<R, T> delta(ConstIndex<T>) {
  return {};
}

/// outer(ops...): operands with pairwise-disjoint label sets, multiplied.
template <typename... Ops>
auto outer(Ops... ops) {
  static_assert(sizeof...(Ops) >= 1, "outer(): needs at least one operand");
  static_assert((Impl::is_contraction_operand_v<Ops> && ...),
                "outer(): every operand must be a slot, a functional input, "
                "or a structured operand (delta, outer, stack, .as<>)");
  static_assert(
      Impl::labels_distinct_v<Impl::gc_cat_seq_t<std::integer_sequence<int32_t>,
                                                 typename Ops::modes_seq...>>,
      "outer(): the operands' label sets must be pairwise disjoint -- a "
      "shared label is a contraction or a Hadamard product, which is "
      "make_contraction_node's job");
  return Outer<Ops...>{std::tuple<Ops...>(std::move(ops)...)};
}

/// stack<'r'>(b_0, ..., b_{n-1}): the value is b_t where r == t.
template <int32_t R, typename... Bs>
auto stack(Bs... bs) {
  static_assert(sizeof...(Bs) >= 1, "stack<r>(): needs at least one branch");
  static_assert((Impl::is_contraction_operand_v<Bs> && ...),
                "stack<r>(): every branch must be a slot, a functional "
                "input, or a structured operand (delta, outer, stack, .as<>)");
  using B0 = Impl::gc_first_t<Bs...>;
  static_assert(
      (Impl::same_label_set_v<typename Bs::modes_seq, typename B0::modes_seq> &&
       ...),
      "stack<r>(): every branch must carry the same label set (in any "
      "order)");
  static_assert(
      !Impl::arr_contains(Impl::seq_to_array(typename B0::modes_seq{}), R),
      "stack<r>(): the stacking label must not be one of the branches' "
      "labels");
  return Stack<R, Bs...>{std::tuple<Bs...>(std::move(bs)...)};
}

// --- lowering: every operand as a set of terms -----------------------------

namespace Impl {

template <typename... Ts>
struct GcList {};

template <typename... Ls>
struct gc_cat_list;
template <>
struct gc_cat_list<> {
  using type = GcList<>;
};
template <typename... As>
struct gc_cat_list<GcList<As...>> {
  using type = GcList<As...>;
};
template <typename... As, typename... Bs, typename... Rest>
struct gc_cat_list<GcList<As...>, GcList<Bs...>, Rest...> {
  using type = typename gc_cat_list<GcList<As..., Bs...>, Rest...>::type;
};
template <typename... Ls>
using gc_cat_list_t = typename gc_cat_list<Ls...>::type;

template <typename List>
struct gc_list_size;
template <typename... Ts>
struct gc_list_size<GcList<Ts...>>
    : std::integral_constant<int, static_cast<int>(sizeof...(Ts))> {};

template <std::size_t K, typename List>
struct gc_list_at;
template <std::size_t K, typename T, typename... Ts>
struct gc_list_at<K, GcList<T, Ts...>> : gc_list_at<K - 1, GcList<Ts...>> {};
template <typename T, typename... Ts>
struct gc_list_at<0, GcList<T, Ts...>> {
  using type = T;
};
template <std::size_t K, typename List>
using gc_list_at_t = typename gc_list_at<K, List>::type;

// Positional renaming: label l of From becomes the label at the same position
// of To; a label not in From is kept.
template <typename From, typename To>
constexpr int32_t map_label(int32_t l) {
  constexpr auto f = seq_to_array(From{});
  constexpr auto t = seq_to_array(To{});
  for (std::size_t i = 0; i < f.size(); ++i)
    if (f[i] == l) return t[i];
  return l;
}
template <typename Seq, typename From, typename To>
struct map_labels;
template <int32_t... Ls, typename From, typename To>
struct map_labels<std::integer_sequence<int32_t, Ls...>, From, To> {
  using type = std::integer_sequence<int32_t, map_label<From, To>(Ls)...>;
};
template <typename Seq, typename From, typename To>
using map_labels_t = typename map_labels<Seq, From, To>::type;

using gc_iseq0 = std::integer_sequence<int>;
using gc_lseq0 = std::integer_sequence<int32_t>;

/// One term: the leaves it multiplies (indices into its operand's flat leaf
/// list), the label pairs its deltas equate (a0, b0, a1, b1, ...), and its
/// constant bindings (label ConstLabs[c] == ConstVals[c]).
template <typename LeafSeq, typename PairSeq, typename ConstLabSeq,
          typename ConstValSeq>
struct GcTerm {};

/// An operand lowered: its flat dense leaves' labels (one sequence per leaf,
/// in the leaf's axis order), its terms, and every stack label inside it
/// with its number of branches.
template <typename LeafLabels, typename Terms, typename StackLabSeq,
          typename StackLenSeq>
struct GcLowered {};

/// The identity of the product: one empty term.
using GcUnit =
    GcLowered<GcList<>, GcList<GcTerm<gc_iseq0, gc_lseq0, gc_lseq0, gc_iseq0>>,
              gc_lseq0, gc_iseq0>;

// The product of two terms, the second's leaves shifted by Shift.
template <int Shift, typename TA, typename TB>
struct gc_term_join;
template <int Shift, int... La, int32_t... Pa, int32_t... Ca, int... Va,
          int... Lb, int32_t... Pb, int32_t... Cb, int... Vb>
struct gc_term_join<Shift,
                    GcTerm<std::integer_sequence<int, La...>,
                           std::integer_sequence<int32_t, Pa...>,
                           std::integer_sequence<int32_t, Ca...>,
                           std::integer_sequence<int, Va...>>,
                    GcTerm<std::integer_sequence<int, Lb...>,
                           std::integer_sequence<int32_t, Pb...>,
                           std::integer_sequence<int32_t, Cb...>,
                           std::integer_sequence<int, Vb...>>> {
  using type = GcTerm<std::integer_sequence<int, La..., (Lb + Shift)...>,
                      std::integer_sequence<int32_t, Pa..., Pb...>,
                      std::integer_sequence<int32_t, Ca..., Cb...>,
                      std::integer_sequence<int, Va..., Vb...>>;
};

template <int Shift, typename TA, typename TBs>
struct gc_row;
template <int Shift, typename TA, typename... TBs>
struct gc_row<Shift, TA, GcList<TBs...>> {
  using type = GcList<typename gc_term_join<Shift, TA, TBs>::type...>;
};

// Cartesian product of two term sets, A's terms outermost.
template <typename A, typename B>
struct gc_product;
template <typename... LA, typename... TA, int32_t... SA, int... NA,
          typename... LB, typename TB, int32_t... SB, int... NB>
struct gc_product<
    GcLowered<GcList<LA...>, GcList<TA...>,
              std::integer_sequence<int32_t, SA...>,
              std::integer_sequence<int, NA...>>,
    GcLowered<GcList<LB...>, TB, std::integer_sequence<int32_t, SB...>,
              std::integer_sequence<int, NB...>>> {
  using type =
      GcLowered<GcList<LA..., LB...>,
                gc_cat_list_t<typename gc_row<static_cast<int>(sizeof...(LA)),
                                              TA, TB>::type...>,
                std::integer_sequence<int32_t, SA..., SB...>,
                std::integer_sequence<int, NA..., NB...>>;
};

template <typename... Ls>
struct gc_product_all;
template <>
struct gc_product_all<> {
  using type = GcUnit;
};
template <typename L0, typename... Ls>
struct gc_product_all<L0, Ls...> {
  using type =
      typename gc_product<L0, typename gc_product_all<Ls...>::type>::type;
};

// Union of two term sets (leaves concatenated, B's already shifted).
template <typename A, typename B>
struct gc_union;
template <typename... LA, typename... TA, int32_t... SA, int... NA,
          typename... LB, typename... TB, int32_t... SB, int... NB>
struct gc_union<GcLowered<GcList<LA...>, GcList<TA...>,
                          std::integer_sequence<int32_t, SA...>,
                          std::integer_sequence<int, NA...>>,
                GcLowered<GcList<LB...>, GcList<TB...>,
                          std::integer_sequence<int32_t, SB...>,
                          std::integer_sequence<int, NB...>>> {
  using type = GcLowered<GcList<LA..., LB...>, GcList<TA..., TB...>,
                         std::integer_sequence<int32_t, SA..., SB...>,
                         std::integer_sequence<int, NA..., NB...>>;
};

// Branches T, T+1, ... of a stack on R, their leaves starting at Shift.
template <int32_t R, int T, int Shift, typename... Ls>
struct gc_stack_acc;
template <int32_t R, int T, int Shift>
struct gc_stack_acc<R, T, Shift> {
  using type = GcLowered<GcList<>, GcList<>, gc_lseq0, gc_iseq0>;
};
template <int32_t R, int T, int Shift, typename... LL, typename... TT,
          int32_t... SS, int... NN, typename... Rest>
struct gc_stack_acc<R, T, Shift,
                    GcLowered<GcList<LL...>, GcList<TT...>,
                              std::integer_sequence<int32_t, SS...>,
                              std::integer_sequence<int, NN...>>,
                    Rest...> {
  using at_t = GcTerm<gc_iseq0, gc_lseq0, std::integer_sequence<int32_t, R>,
                      std::integer_sequence<int, T>>;
  using here = GcLowered<
      GcList<LL...>, GcList<typename gc_term_join<Shift, at_t, TT>::type...>,
      std::integer_sequence<int32_t, SS...>, std::integer_sequence<int, NN...>>;
  using rest =
      typename gc_stack_acc<R, T + 1, Shift + static_cast<int>(sizeof...(LL)),
                            Rest...>::type;
  using type = typename gc_union<here, rest>::type;
};

template <int32_t R, int N, typename Lowered>
struct gc_add_stack;
template <int32_t R, int N, typename Ls, typename Ts, int32_t... S, int... Ns>
struct gc_add_stack<R, N,
                    GcLowered<Ls, Ts, std::integer_sequence<int32_t, S...>,
                              std::integer_sequence<int, Ns...>>> {
  using type = GcLowered<Ls, Ts, std::integer_sequence<int32_t, R, S...>,
                         std::integer_sequence<int, N, Ns...>>;
};

// Every label of a term set renamed positionally From -> To.
template <typename Term, typename From, typename To>
struct gc_relabel_term;
template <typename Ls, int32_t... P, int32_t... C, typename Vs, typename From,
          typename To>
struct gc_relabel_term<GcTerm<Ls, std::integer_sequence<int32_t, P...>,
                              std::integer_sequence<int32_t, C...>, Vs>,
                       From, To> {
  using type =
      GcTerm<Ls, std::integer_sequence<int32_t, map_label<From, To>(P)...>,
             std::integer_sequence<int32_t, map_label<From, To>(C)...>, Vs>;
};
template <typename Lowered, typename From, typename To>
struct gc_relabel;
template <typename... Ls, typename... Ts, int32_t... S, typename Ns,
          typename From, typename To>
struct gc_relabel<GcLowered<GcList<Ls...>, GcList<Ts...>,
                            std::integer_sequence<int32_t, S...>, Ns>,
                  From, To> {
  using type =
      GcLowered<GcList<map_labels_t<Ls, From, To>...>,
                GcList<typename gc_relabel_term<Ts, From, To>::type...>,
                std::integer_sequence<int32_t, map_label<From, To>(S)...>, Ns>;
};

/// An operand's term set. The primary is a dense leaf: one term, itself.
template <typename Op>
struct gc_lower {
  using type = GcLowered<GcList<typename Op::modes_seq>,
                         GcList<GcTerm<std::integer_sequence<int, 0>, gc_lseq0,
                                       gc_lseq0, gc_iseq0>>,
                         gc_lseq0, gc_iseq0>;
};
template <int32_t A, int32_t B>
struct gc_lower<Delta<A, B>> {
  using type =
      GcLowered<GcList<>,
                GcList<GcTerm<gc_iseq0, std::integer_sequence<int32_t, A, B>,
                              gc_lseq0, gc_iseq0>>,
                gc_lseq0, gc_iseq0>;
};
template <int32_t R, int T>
struct gc_lower<UnitVector<R, T>> {
  using type = GcLowered<
      GcList<>,
      GcList<GcTerm<gc_iseq0, gc_lseq0, std::integer_sequence<int32_t, R>,
                    std::integer_sequence<int, T>>>,
      gc_lseq0, gc_iseq0>;
};
template <typename... Ops>
struct gc_lower<Outer<Ops...>> {
  using type = typename gc_product_all<typename gc_lower<Ops>::type...>::type;
};
template <int32_t R, typename... Bs>
struct gc_lower<Stack<R, Bs...>> {
  using type = typename gc_add_stack<
      R, static_cast<int>(sizeof...(Bs)),
      typename gc_stack_acc<R, 0, 0,
                            typename gc_lower<Bs>::type...>::type>::type;
};
template <typename Op, typename M>
struct gc_lower<Relabeled<Op, M>> {
  using type = typename gc_relabel<typename gc_lower<Op>::type,
                                   typename Op::modes_seq, M>::type;
};
template <typename Op>
using gc_lower_t = typename gc_lower<Op>::type;

template <typename T>
struct is_outer : std::false_type {};
template <typename... Ops>
struct is_outer<Outer<Ops...>> : std::true_type {};
template <typename T>
struct is_stack : std::false_type {};
template <int32_t R, typename... Bs>
struct is_stack<Stack<R, Bs...>> : std::true_type {};
template <typename T>
struct is_relabeled : std::false_type {};
template <typename Op, typename M>
struct is_relabeled<Relabeled<Op, M>> : std::true_type {};

/// An operand's dense leaves, flattened in the order its lowering numbers
/// them (depth first, left to right). Deltas contribute none.
template <typename Op>
auto gc_leaves(const Op& op) {
  if constexpr (is_node_handle_v<Op>) {
    return std::tuple<Op>(op);
  } else if constexpr (is_outer<Op>::value) {
    return std::apply(
        [](const auto&... o) { return std::tuple_cat(gc_leaves(o)...); },
        op.ops);
  } else if constexpr (is_stack<Op>::value) {
    return std::apply(
        [](const auto&... b) { return std::tuple_cat(gc_leaves(b)...); },
        op.branches);
  } else if constexpr (is_relabeled<Op>::value) {
    return gc_leaves(op.op);
  } else {
    return std::tuple<>{};
  }
}

}  // namespace Impl

}  // namespace TensorOperations
