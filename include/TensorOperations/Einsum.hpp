#pragma once
// ---------------------------------------------------------------------------
// Einsum node -- C{Out} = sum over every label not in Out of the product of
// the operands, the labels deciding everything (numpy's einsum, implicit
// summation):
//
//   make_einsum_node<'i','k'>(a.as<'i','j'>(), b.as<'j','k'>())   // matmul
//
// * A label may appear on several operands and in Out (batch / Hadamard).
// * A label on an operand but not in Out is summed. Its extent comes from the
//   graph's label map (it must be LabelWhole there).
// * Operands are slots (level outputs) or functional inputs, read at the
//   coordinate the labels give; a functional input is evaluated on demand and
//   never staged.
//
// Delta-structured operands. A tensor that is mostly Kronecker deltas -- the
// GLL derivative operator D_r(q, i) = h(q_r, i_r) prod_{t != r} delta(q_t,
// i_t), say -- is declared as a sum of cases chosen by a SELECTOR label, each
// case a product of dense factors and deltas:
//
//   auto D = make_delta_operand<'r','z','y','x','k','j','i'>(select<'r'>(
//       kase(h.as<'x','i'>(), delta<'y','j'>{}, delta<'z','k'>{}),
//       kase(h.as<'y','j'>(), delta<'x','i'>{}, delta<'z','k'>{}),
//       kase(h.as<'z','k'>(), delta<'x','i'>{}, delta<'y','j'>{})));
//
// D is then an ordinary operand, relabelled with .as<>(). At compile time the
// node unrolls every selector into one TERM per combination of cases, merges
// the labels each delta joins, and so turns
//
//   K = einsum(D.as<r,..,k,j,i>, M, D.as<s,..,n,m,l>)      (sums r, s, z, y, x)
//
// into the sum-factored terms: a summed label merged with an output label is
// bound to it (no loop), two output labels merged become an equality test on
// the output coordinate, a selector label is a constant in its term, and what
// is left is summed. Written with a dense D the same expression is the plain
// O(N^9) sum -- the delta structure is a property of the operand, not of the
// expression.
// ---------------------------------------------------------------------------
#include <TensorOperations/DeviceTuple.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LabelTiles.hpp>
#include <TensorOperations/Macros.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Permute.hpp>

#include <array>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

namespace TensorOperations {

// --- delta-structured operands ---------------------------------------------

/// Kronecker delta between two labels of a delta operand's case.
template <int32_t A, int32_t B>
struct delta {
  static_assert(A != B, "delta<A, B>: the two labels must differ");
};

/// One case of a delta operand: dense factors times deltas. `pairs_seq` lists
/// the delta labels pairwise, (a0, b0, a1, b1, ...).
template <typename FactorsTuple, typename PairsSeq>
struct DeltaCase;
template <typename... Fs, int32_t... Ps>
struct DeltaCase<DeviceTuple<Fs...>, std::integer_sequence<int32_t, Ps...>> {
  static constexpr int NumFactors = static_cast<int>(sizeof...(Fs));
  using factors_tuple_t           = DeviceTuple<Fs...>;
  using pairs_seq                 = std::integer_sequence<int32_t, Ps...>;
  factors_tuple_t factors;
};

/// The cases of a delta operand, chosen by the selector label Sel: case c is
/// the operand's value where Sel == c.
template <int32_t Sel, typename... Cases>
struct Select {
  DeviceTuple<Cases...> cases;
};

/// A delta-structured operand. BaseSeq are the labels its cases are written
/// in; ModesSeq are the labels it is read as (positionally, after .as<>()).
template <typename BaseSeq, typename ModesSeq, int32_t Sel, typename... Cases>
struct DeltaOperand {
  using base_seq                      = BaseSeq;
  using modes_seq                     = ModesSeq;
  static constexpr int32_t sel_base   = Sel;
  static constexpr int     NumCases   = static_cast<int>(sizeof...(Cases));
  static constexpr int     Rank       = static_cast<int>(BaseSeq::size());
  using cases_tuple_t                 = DeviceTuple<Cases...>;
  using value_type =
      typename tuple_element_t<0, typename tuple_element_t<
                                      0, cases_tuple_t>::factors_tuple_t>::
          value_type;

  cases_tuple_t cases;

  template <int32_t... Modes>
  auto as() const {
    static_assert(static_cast<int>(sizeof...(Modes)) == Rank,
                  "delta operand as(): one label per axis");
    static_assert(
        Impl::labels_distinct_v<std::integer_sequence<int32_t, Modes...>>,
        "delta operand as(): labels must be distinct");
    return DeltaOperand<BaseSeq, std::integer_sequence<int32_t, Modes...>, Sel,
                        Cases...>{cases};
  }
};

namespace Impl {

template <typename T>
struct is_delta : std::false_type {};
template <int32_t A, int32_t B>
struct is_delta<delta<A, B>> : std::true_type {};

template <typename T>
struct is_delta_operand : std::false_type {};
template <typename B, typename M, int32_t S, typename... Cs>
struct is_delta_operand<DeltaOperand<B, M, S, Cs...>> : std::true_type {};

template <typename T>
struct delta_pair {
  using type = std::integer_sequence<int32_t>;
};
template <int32_t A, int32_t B>
struct delta_pair<delta<A, B>> {
  using type = std::integer_sequence<int32_t, A, B>;
};

template <typename... Seqs>
struct einsum_cat_seq;
template <typename T>
struct einsum_cat_seq<std::integer_sequence<T>> {
  using type = std::integer_sequence<T>;
};
template <typename T, T... As>
struct einsum_cat_seq<std::integer_sequence<T, As...>> {
  using type = std::integer_sequence<T, As...>;
};
template <typename T, T... As, T... Bs, typename... Rest>
struct einsum_cat_seq<std::integer_sequence<T, As...>,
                      std::integer_sequence<T, Bs...>, Rest...> {
  using type = typename einsum_cat_seq<std::integer_sequence<T, As..., Bs...>,
                                       Rest...>::type;
};
template <typename... Seqs>
using einsum_cat_seq_t = typename einsum_cat_seq<Seqs...>::type;

template <typename... Items>
constexpr bool factors_before_deltas() {
  constexpr bool d[] = {is_delta<Items>::value..., false};
  bool           seen = false;
  for (std::size_t i = 0; i < sizeof...(Items); ++i) {
    if (d[i])
      seen = true;
    else if (seen)
      return false;
  }
  return true;
}

template <typename Tuple, std::size_t... Is>
auto case_factors(const Tuple& t, std::index_sequence<Is...>) {
  return DeviceTuple<std::tuple_element_t<Is, Tuple>...>(std::get<Is>(t)...);
}

// Every label of Seq is in Base.
template <typename Seq, typename Base>
constexpr bool labels_within() {
  constexpr auto s = seq_to_array(Seq{});
  constexpr auto b = seq_to_array(Base{});
  for (std::size_t i = 0; i < s.size(); ++i)
    if (!arr_contains(b, s[i])) return false;
  return true;
}

template <typename Base, typename Case>
constexpr bool case_within() {
  return labels_within<typename Case::pairs_seq, Base>() &&
         []<std::size_t... Fs>(std::index_sequence<Fs...>) {
           return (labels_within<
                       typename tuple_element_t<
                           Fs, typename Case::factors_tuple_t>::modes_seq,
                       Base>() &&
                   ...);
         }(std::make_index_sequence<static_cast<std::size_t>(
               Case::NumFactors)>{});
}

}  // namespace Impl

/// kase(factors..., delta<A, B>{}...): dense factors first, then deltas.
template <typename... Items>
auto kase(Items... items) {
  constexpr std::size_t ND =
      (std::size_t{0} + ... + (Impl::is_delta<Items>::value ? 1 : 0));
  constexpr std::size_t NF = sizeof...(Items) - ND;
  static_assert(NF >= 1, "kase: a case needs at least one dense factor");
  static_assert(Impl::factors_before_deltas<Items...>(),
                "kase: list the dense factors first, then the deltas");
  using Pairs = Impl::einsum_cat_seq_t<std::integer_sequence<int32_t>,
                                       typename Impl::delta_pair<Items>::type...>;
  const auto t = std::tuple<Items...>(items...);
  auto       f = Impl::case_factors(t, std::make_index_sequence<NF>{});
  return DeltaCase<decltype(f), Pairs>{f};
}

/// select<'r'>(case_0, case_1, ...): case c is the value where 'r' == c.
template <int32_t Sel, typename... Cases>
auto select(Cases... cases) {
  static_assert(sizeof...(Cases) >= 1, "select: needs at least one case");
  return Select<Sel, Cases...>{DeviceTuple<Cases...>(cases...)};
}

/// make_delta_operand<Labels...>(select<Sel>(kase(...), ...))
template <int32_t... Labels, int32_t Sel, typename... Cases>
auto make_delta_operand(Select<Sel, Cases...> s) {
  using Base = std::integer_sequence<int32_t, Labels...>;
  static_assert(Impl::labels_distinct_v<Base>,
                "delta operand: labels must be distinct");
  static_assert(Impl::arr_contains(Impl::seq_to_array(Base{}), Sel),
                "delta operand: the selector label must be one of its labels");
  static_assert((Impl::case_within<Base, Cases>() && ...),
                "delta operand: every factor and delta label of a case must "
                "be one of the operand's labels");
  return DeltaOperand<Base, Base, Sel, Cases...>{s.cases};
}

// --- the node's compile-time structure -------------------------------------

namespace Impl {

template <typename... Ts>
struct EinsumTypeList {};

template <typename... Ls>
struct einsum_cat_list;
template <>
struct einsum_cat_list<> {
  using type = EinsumTypeList<>;
};
template <typename... As>
struct einsum_cat_list<EinsumTypeList<As...>> {
  using type = EinsumTypeList<As...>;
};
template <typename... As, typename... Bs, typename... Rest>
struct einsum_cat_list<EinsumTypeList<As...>, EinsumTypeList<Bs...>, Rest...> {
  using type =
      typename einsum_cat_list<EinsumTypeList<As..., Bs...>, Rest...>::type;
};
template <typename... Ls>
using einsum_cat_list_t = typename einsum_cat_list<Ls...>::type;

/// A case, in node-space labels: the leaves it multiplies and its delta pairs.
template <typename LeafIdxSeq, typename PairsSeq>
struct EinsumCaseDesc {};
/// A delta operand, in node-space labels: its selector and its cases.
template <int32_t Sel, typename... CaseDescs>
struct EinsumDeltaDesc {};

/// Everything the term expansion needs, as types: the output labels, each
/// leaf's axis labels in node space (in the leaf's own axis order), the
/// leaves that are plain operands, and the delta operands.
template <typename OutSeq, typename LeafLabels, typename PlainSeq,
          typename DeltaDescs>
struct EinsumStructure {};

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

template <int Start, typename Seq>
struct offset_seq;
template <int Start, std::size_t... Is>
struct offset_seq<Start, std::index_sequence<Is...>> {
  using type = std::integer_sequence<int, Start + static_cast<int>(Is)...>;
};
template <int Start, int N>
using offset_seq_t =
    typename offset_seq<Start, std::make_index_sequence<static_cast<std::size_t>(
                                   N)>>::type;

template <typename Case, typename Base, typename Modes>
struct case_leaf_labels;
template <typename... Fs, typename P, typename Base, typename Modes>
struct case_leaf_labels<DeltaCase<DeviceTuple<Fs...>, P>, Base, Modes> {
  using type =
      EinsumTypeList<map_labels_t<typename Fs::modes_seq, Base, Modes>...>;
};

template <typename Case, std::size_t... Fs>
auto case_leaves_tuple(const Case& c, std::index_sequence<Fs...>) {
  return std::make_tuple(c.factors.template get<Fs>()...);
}

// One operand's contribution to the flat leaf list, starting at leaf Offset.
template <typename Op, int Offset>
struct einsum_op_traits {
  static constexpr int num_leaves = 1;
  using leaf_labels               = EinsumTypeList<typename Op::modes_seq>;
  using plain                     = std::integer_sequence<int, Offset>;
  using deltas                    = EinsumTypeList<>;
  static auto leaves(const Op& op) { return std::make_tuple(op); }
};
template <typename Base, typename Modes, int32_t Sel, typename... Cases,
          int Offset>
struct einsum_op_traits<DeltaOperand<Base, Modes, Sel, Cases...>, Offset> {
  using Op                        = DeltaOperand<Base, Modes, Sel, Cases...>;
  static constexpr int num_leaves = (0 + ... + Cases::NumFactors);
  template <std::size_t C>
  static constexpr int first() {
    constexpr int counts[] = {Cases::NumFactors...};
    int           s        = Offset;
    for (std::size_t c = 0; c < C; ++c) s += counts[c];
    return s;
  }
  template <std::size_t... Cs>
  static auto deltas_impl(std::index_sequence<Cs...>) -> EinsumTypeList<
      EinsumDeltaDesc<map_label<Base, Modes>(Sel),
                      EinsumCaseDesc<offset_seq_t<first<Cs>(),
                                                  Cases::NumFactors>,
                                     map_labels_t<typename Cases::pairs_seq,
                                                  Base, Modes>>...>>;

  using leaf_labels = einsum_cat_list_t<
      typename case_leaf_labels<Cases, Base, Modes>::type...>;
  using plain  = std::integer_sequence<int>;
  using deltas = decltype(deltas_impl(std::index_sequence_for<Cases...>{}));

  template <std::size_t... Cs>
  static auto leaves_impl(const Op& op, std::index_sequence<Cs...>) {
    return std::tuple_cat(case_leaves_tuple(
        op.cases.template get<Cs>(),
        std::make_index_sequence<static_cast<std::size_t>(
            tuple_element_t<Cs, typename Op::cases_tuple_t>::NumFactors)>{})...);
  }
  static auto leaves(const Op& op) {
    return leaves_impl(op, std::index_sequence_for<Cases...>{});
  }
};

template <int Offset, typename... Ops>
struct einsum_collect;
template <int Offset>
struct einsum_collect<Offset> {
  using leaf_labels = EinsumTypeList<>;
  using plain       = std::integer_sequence<int>;
  using deltas      = EinsumTypeList<>;
};
template <int Offset, typename Op, typename... Rest>
struct einsum_collect<Offset, Op, Rest...> {
  using T = einsum_op_traits<Op, Offset>;
  using R = einsum_collect<Offset + T::num_leaves, Rest...>;
  using leaf_labels =
      einsum_cat_list_t<typename T::leaf_labels, typename R::leaf_labels>;
  using plain  = einsum_cat_seq_t<typename T::plain, typename R::plain>;
  using deltas = einsum_cat_list_t<typename T::deltas, typename R::deltas>;
};

// --- the structure as plain constexpr data ---------------------------------

inline constexpr int einsum_max_out    = 16;
inline constexpr int einsum_max_leaf   = 32;
inline constexpr int einsum_max_rank   = 16;
inline constexpr int einsum_max_dop    = 4;
inline constexpr int einsum_max_case   = 8;
inline constexpr int einsum_max_cleaf  = 8;
inline constexpr int einsum_max_cpair  = 8;
inline constexpr int einsum_max_labels = 64;
inline constexpr int einsum_max_chk    = 16;
inline constexpr int einsum_max_loop   = 16;

struct EinsumSpec {
  bool                                   overflow = false;
  int                                    nout     = 0;
  std::array<int32_t, einsum_max_out>    out{};
  int                                    nleaf = 0;
  std::array<int, einsum_max_leaf>       leaf_rank{};
  std::array<std::array<int32_t, einsum_max_rank>, einsum_max_leaf> leaf_lab{};
  int                                    nplain = 0;
  std::array<int, einsum_max_leaf>       plain{};
  int                                    ndop = 0;
  std::array<int32_t, einsum_max_dop>    sel{};
  std::array<int, einsum_max_dop>        ncase{};
  std::array<std::array<int, einsum_max_case>, einsum_max_dop> case_nleaf{};
  std::array<std::array<std::array<int, einsum_max_cleaf>, einsum_max_case>,
             einsum_max_dop>
      case_leaf{};
  std::array<std::array<int, einsum_max_case>, einsum_max_dop> case_npair{};
  std::array<
      std::array<std::array<int32_t, 2 * einsum_max_cpair>, einsum_max_case>,
      einsum_max_dop>
      case_pair{};
};

template <int32_t... Ls>
constexpr void spec_add_leaf(EinsumSpec& s,
                             std::integer_sequence<int32_t, Ls...>) {
  if (s.nleaf >= einsum_max_leaf ||
      static_cast<int>(sizeof...(Ls)) > einsum_max_rank) {
    s.overflow = true;
    return;
  }
  const int32_t l[] = {Ls..., 0};
  s.leaf_rank[s.nleaf] = static_cast<int>(sizeof...(Ls));
  for (std::size_t d = 0; d < sizeof...(Ls); ++d) s.leaf_lab[s.nleaf][d] = l[d];
  ++s.nleaf;
}
template <typename... Seqs>
constexpr void spec_add_leaves(EinsumSpec& s, EinsumTypeList<Seqs...>) {
  (spec_add_leaf(s, Seqs{}), ...);
}

template <int... Js, int32_t... Ps>
constexpr void spec_add_case(EinsumSpec& s, int d,
                             EinsumCaseDesc<std::integer_sequence<int, Js...>,
                                            std::integer_sequence<int32_t, Ps...>>) {
  const int c = s.ncase[d];
  if (c >= einsum_max_case ||
      static_cast<int>(sizeof...(Js)) > einsum_max_cleaf ||
      static_cast<int>(sizeof...(Ps)) > 2 * einsum_max_cpair) {
    s.overflow = true;
    return;
  }
  const int     j[] = {Js..., 0};
  const int32_t p[] = {Ps..., 0};
  s.case_nleaf[d][c] = static_cast<int>(sizeof...(Js));
  for (std::size_t i = 0; i < sizeof...(Js); ++i) s.case_leaf[d][c][i] = j[i];
  s.case_npair[d][c] = static_cast<int>(sizeof...(Ps)) / 2;
  for (std::size_t i = 0; i < sizeof...(Ps); ++i) s.case_pair[d][c][i] = p[i];
  ++s.ncase[d];
}
template <int32_t Sel, typename... Cs>
constexpr void spec_add_dop(EinsumSpec& s, EinsumDeltaDesc<Sel, Cs...>) {
  if (s.ndop >= einsum_max_dop) {
    s.overflow = true;
    return;
  }
  const int d = s.ndop++;
  s.sel[d]    = Sel;
  s.ncase[d]  = 0;
  (spec_add_case(s, d, Cs{}), ...);
}
template <typename... Ds>
constexpr void spec_add_dops(EinsumSpec& s, EinsumTypeList<Ds...>) {
  (spec_add_dop(s, Ds{}), ...);
}

template <typename Structure>
struct einsum_spec;
template <int32_t... Out, typename LeafLabels, int... Plain,
          typename DeltaDescs>
struct einsum_spec<EinsumStructure<std::integer_sequence<int32_t, Out...>,
                                   LeafLabels,
                                   std::integer_sequence<int, Plain...>,
                                   DeltaDescs>> {
  static constexpr EinsumSpec make() {
    EinsumSpec    s{};
    const int32_t o[] = {Out..., 0};
    const int     p[] = {Plain..., 0};
    if (static_cast<int>(sizeof...(Out)) > einsum_max_out) s.overflow = true;
    s.nout = static_cast<int>(sizeof...(Out));
    for (std::size_t i = 0; i < sizeof...(Out) && !s.overflow; ++i)
      s.out[i] = o[i];
    s.nplain = static_cast<int>(sizeof...(Plain));
    for (std::size_t i = 0; i < sizeof...(Plain); ++i) s.plain[i] = p[i];
    spec_add_leaves(s, LeafLabels{});
    spec_add_dops(s, DeltaDescs{});
    return s;
  }
  static constexpr EinsumSpec value = make();
};

// --- term expansion ---------------------------------------------------------

inline constexpr int einsum_bind_out   = 0;  // val = output position
inline constexpr int einsum_bind_const = 1;  // val = the constant
inline constexpr int einsum_bind_loop  = 2;  // val = loop index

// Error codes of a term (0 = fine).
inline constexpr int einsum_err_overflow       = 1;
inline constexpr int einsum_err_sum_not_in_map = 2;
inline constexpr int einsum_err_sum_gridded    = 3;
inline constexpr int einsum_err_sum_extent     = 4;
inline constexpr int einsum_err_gridded_merged = 5;
inline constexpr int einsum_err_selector_ext   = 6;

struct EinsumTerm {
  int  error = 0;
  bool dead  = false;  // two selectors forced to different constants

  int                              nleaf = 0;  // the leaves this term multiplies
  std::array<int, einsum_max_leaf> leaf{};

  int                                   nlab = 0;  // label universe + binding
  std::array<int32_t, einsum_max_labels> lab{};
  std::array<int, einsum_max_labels>     kind{};
  std::array<int, einsum_max_labels>     val{};

  // Output-coordinate tests: coord[chk_a] == coord[chk_b] (chk_b >= 0) or
  // coord[chk_a] == chk_c (chk_b < 0).
  int                             nchk = 0;
  std::array<int, einsum_max_chk> chk_a{};
  std::array<int, einsum_max_chk> chk_b{};
  std::array<int, einsum_max_chk> chk_c{};

  int                              nloop = 0;  // summed classes, and extents
  std::array<int, einsum_max_loop> loop_ext{};
};

constexpr int einsum_num_terms(const EinsumSpec& s) {
  int n = 1;
  for (int d = 0; d < s.ndop; ++d) n *= s.ncase[d];
  return n;
}

template <typename LT>
constexpr EinsumTerm einsum_term(const EinsumSpec& s, int t) {
  EinsumTerm r{};
  if (s.overflow) {
    r.error = einsum_err_overflow;
    return r;
  }

  // The case each delta operand takes in this term (mixed radix, last fastest).
  std::array<int, einsum_max_dop> cs{};
  {
    int rem = t;
    for (int d = s.ndop - 1; d >= 0; --d) {
      cs[d] = rem % s.ncase[d];
      rem /= s.ncase[d];
    }
  }

  auto add_leaf = [&](int j) {
    if (r.nleaf >= einsum_max_leaf) {
      r.error = einsum_err_overflow;
      return;
    }
    r.leaf[r.nleaf++] = j;
  };
  for (int i = 0; i < s.nplain; ++i) add_leaf(s.plain[i]);
  for (int d = 0; d < s.ndop; ++d)
    for (int i = 0; i < s.case_nleaf[d][cs[d]]; ++i)
      add_leaf(s.case_leaf[d][cs[d]][i]);

  // Label universe: the outputs first (so universe index == output position),
  // then every label the term's leaves, selectors and deltas name.
  auto add_label = [&](int32_t l) {
    for (int u = 0; u < r.nlab; ++u)
      if (r.lab[u] == l) return u;
    if (r.nlab >= einsum_max_labels) {
      r.error = einsum_err_overflow;
      return 0;
    }
    r.lab[r.nlab] = l;
    return r.nlab++;
  };
  for (int o = 0; o < s.nout; ++o) add_label(s.out[o]);
  for (int i = 0; i < r.nleaf; ++i) {
    const int j = r.leaf[i];
    for (int a = 0; a < s.leaf_rank[j]; ++a) add_label(s.leaf_lab[j][a]);
  }
  for (int d = 0; d < s.ndop; ++d) add_label(s.sel[d]);
  for (int d = 0; d < s.ndop; ++d)
    for (int p = 0; p < 2 * s.case_npair[d][cs[d]]; ++p)
      add_label(s.case_pair[d][cs[d]][p]);
  if (r.error) return r;

  // Union-find over the universe: each delta joins its two labels.
  std::array<int, einsum_max_labels> parent{};
  for (int u = 0; u < r.nlab; ++u) parent[u] = u;
  auto find = [&](int u) {
    while (parent[u] != u) u = parent[u];
    return u;
  };
  for (int d = 0; d < s.ndop; ++d)
    for (int p = 0; p < s.case_npair[d][cs[d]]; ++p) {
      const int a = find(add_label(s.case_pair[d][cs[d]][2 * p]));
      const int b = find(add_label(s.case_pair[d][cs[d]][2 * p + 1]));
      if (a != b) parent[a < b ? b : a] = a < b ? a : b;  // root = lowest
    }

  // Per class: its first output position, and its constant (selectors).
  std::array<int, einsum_max_labels> cls_out{};
  std::array<int, einsum_max_labels> cls_const{};
  std::array<int, einsum_max_labels> cls_loop{};
  for (int u = 0; u < r.nlab; ++u) {
    cls_out[u]   = -1;
    cls_const[u] = -1;
    cls_loop[u]  = -1;
  }
  auto add_check = [&](int a, int b, int c) {
    if (r.nchk >= einsum_max_chk) {
      r.error = einsum_err_overflow;
      return;
    }
    r.chk_a[r.nchk] = a;
    r.chk_b[r.nchk] = b;
    r.chk_c[r.nchk] = c;
    ++r.nchk;
  };
  for (int o = 0; o < s.nout; ++o) {
    const int root = find(o);
    if (cls_out[root] < 0)
      cls_out[root] = o;
    else
      add_check(cls_out[root], o, 0);
  }
  for (int d = 0; d < s.ndop; ++d) {
    const int root = find(add_label(s.sel[d]));
    if (cls_const[root] >= 0 && cls_const[root] != cs[d]) r.dead = true;
    cls_const[root] = cs[d];
    const int tile  = label_tile_of<LT>(s.sel[d]);
    if (label_index_of<LT>(s.sel[d]) >= 0 && tile != s.ncase[d])
      r.error = einsum_err_selector_ext;
  }
  for (int u = 0; u < r.nlab; ++u)
    if (find(u) == u && cls_out[u] >= 0 && cls_const[u] >= 0)
      add_check(cls_out[u], -1, cls_const[u]);

  // What is left is summed: one loop per class, its extent from the map.
  for (int u = 0; u < r.nlab; ++u) {
    if (find(u) != u || cls_out[u] >= 0 || cls_const[u] >= 0) continue;
    int ext = -1;
    for (int v = 0; v < r.nlab; ++v) {
      if (find(v) != u) continue;
      if (label_index_of<LT>(r.lab[v]) < 0) {
        r.error = einsum_err_sum_not_in_map;
        return r;
      }
      if (label_gridded_of<LT>(r.lab[v])) {
        r.error = einsum_err_sum_gridded;
        return r;
      }
      const int e = label_tile_of<LT>(r.lab[v]);
      if (ext >= 0 && e != ext) {
        r.error = einsum_err_sum_extent;
        return r;
      }
      ext = e;
    }
    if (r.nloop >= einsum_max_loop) {
      r.error = einsum_err_overflow;
      return r;
    }
    cls_loop[u]           = r.nloop;
    r.loop_ext[r.nloop++] = ext;
  }

  // A gridded label indexes a partial tile, so it may only be an output label
  // that nothing merges with (and never a summed one, rejected above).
  for (int u = 0; u < r.nlab; ++u) {
    if (!label_gridded_of<LT>(r.lab[u])) continue;
    const int root = find(u);
    int       size = 0;
    for (int v = 0; v < r.nlab; ++v) size += (find(v) == root) ? 1 : 0;
    if (size != 1 || cls_out[root] < 0 || cls_const[root] >= 0)
      r.error = einsum_err_gridded_merged;
  }

  for (int u = 0; u < r.nlab; ++u) {
    const int root = find(u);
    if (cls_out[root] >= 0) {
      r.kind[u] = einsum_bind_out;
      r.val[u]  = cls_out[root];
    } else if (cls_const[root] >= 0) {
      r.kind[u] = einsum_bind_const;
      r.val[u]  = cls_const[root];
    } else {
      r.kind[u] = einsum_bind_loop;
      r.val[u]  = cls_loop[root];
    }
  }
  return r;
}

/// A term's plan, as sequences the evaluator expands over.
template <typename Structure, typename LT, int T>
struct EinsumTermInfo {
  static constexpr EinsumTerm term =
      einsum_term<LT>(einsum_spec<Structure>::value, T);

  static constexpr int  error    = term.error;
  static constexpr bool dead     = term.dead;
  static constexpr int  num_loop = term.nloop;

  template <int N, typename F>
  static constexpr std::array<int, static_cast<std::size_t>(N)> take(F f) {
    std::array<int, static_cast<std::size_t>(N)> a{};
    for (int i = 0; i < N; ++i) a[static_cast<std::size_t>(i)] = f(i);
    return a;
  }
  static constexpr auto leaf_arr =
      take<term.nleaf>([](int i) { return term.leaf[i]; });
  static constexpr auto chk_a_arr =
      take<term.nchk>([](int i) { return term.chk_a[i]; });
  static constexpr auto chk_b_arr =
      take<term.nchk>([](int i) { return term.chk_b[i]; });
  static constexpr auto chk_c_arr =
      take<term.nchk>([](int i) { return term.chk_c[i]; });
  static constexpr auto loop_arr =
      take<term.nloop>([](int i) { return term.loop_ext[i]; });

  using leaves_seq   = array_to_seq_t<leaf_arr>;
  using chk_a_seq    = array_to_seq_t<chk_a_arr>;
  using chk_b_seq    = array_to_seq_t<chk_b_arr>;
  using chk_c_seq    = array_to_seq_t<chk_c_arr>;
  using loop_ext_seq = array_to_seq_t<loop_arr>;

  static constexpr int binding(int32_t l, bool want_kind) {
    for (int u = 0; u < term.nlab; ++u)
      if (term.lab[u] == l) return want_kind ? term.kind[u] : term.val[u];
    return -1;
  }
  // The binding of each axis of a leaf whose storage axes carry `Labels`.
  template <typename Labels>
  static constexpr auto kinds_arr() {
    constexpr auto l = seq_to_array(Labels{});
    std::array<int, l.size()> a{};
    for (std::size_t d = 0; d < l.size(); ++d) a[d] = binding(l[d], true);
    return a;
  }
  template <typename Labels>
  static constexpr auto vals_arr() {
    constexpr auto l = seq_to_array(Labels{});
    std::array<int, l.size()> a{};
    for (std::size_t d = 0; d < l.size(); ++d) a[d] = binding(l[d], false);
    return a;
  }
  template <typename Labels>
  using kinds_seq = array_to_seq_t<kinds_arr<Labels>()>;
  template <typename Labels>
  using vals_seq = array_to_seq_t<vals_arr<Labels>()>;
};

template <typename Structure, typename LT, std::size_t... Ts>
constexpr int einsum_first_error(std::index_sequence<Ts...>) {
  int e = 0;
  ((e = e ? e : EinsumTermInfo<Structure, LT, static_cast<int>(Ts)>::error),
   ...);
  return e;
}

template <typename LT, typename OutSeq>
struct einsum_tile {
  using type = tile_from_labels_t<LT, OutSeq>;
};
template <typename OutSeq>
struct einsum_tile<void, OutSeq> {
  using type = void;
};

}  // namespace Impl

// --- the node --------------------------------------------------------------
//
// LT (the graph's label map) is void until LevelGraph::add resolves it: the
// summed extents and the output tile both come from the map.
template <typename Scalar, typename ExecSpace, typename OutSeq,
          typename Structure, typename LT, typename... Leaves>
struct NodeHandle<EinsumTag, Scalar, ExecSpace, OutSeq, Structure, LT,
                  Leaves...> {
  using node_tag              = EinsumTag;
  static constexpr int Rank   = static_cast<int>(OutSeq::size());
  static constexpr int NumOps = static_cast<int>(sizeof...(Leaves));
  static constexpr int NumOut = 1;
  using value_type            = Scalar;
  using exec_space            = ExecSpace;
  using modes_seq             = OutSeq;
  using structure_type        = Structure;
  using label_tiles_type      = LT;
  using tile_type             = typename Impl::einsum_tile<LT, OutSeq>::type;
  using ops_tuple_t           = DeviceTuple<Leaves...>;  // the flat leaves

  static constexpr int NumTerms =
      Impl::einsum_num_terms(Impl::einsum_spec<Structure>::value);
  template <int T>
  using term_info = Impl::EinsumTermInfo<Structure, LT, T>;

  ops_tuple_t              operands;
  Kokkos::Array<int, Rank> shape_;  // -1 where no leaf carries the label,
                                    // until add() fills it in from the map

  KOKKOS_FUNCTION Kokkos::Array<int, Rank> shape() const { return shape_; }
};

namespace Impl {

// Extent of output label L read off the first plain leaf carrying it, or -1.
template <int32_t L, typename LeafLabels, std::size_t J, typename Leaf>
int einsum_extent_or_neg(const Leaf& leaf) {
  constexpr auto m = seq_to_array(LeafLabels{});
  for (std::size_t d = 0; d < m.size(); ++d)
    if (m[d] == L) return leaf.shape()[d];
  return -1;
}
template <int32_t L, typename... LeafLabels, typename... Leaves,
          std::size_t... Js>
int einsum_extent_of(EinsumTypeList<LeafLabels...>,
                     const DeviceTuple<Leaves...>& leaves,
                     std::index_sequence<Js...>) {
  int e = -1;
  ((void)([&] {
     const int f = einsum_extent_or_neg<L, LeafLabels, Js>(
         leaves.template get<Js>());
     assert((f < 0 || e < 0 || f == e) &&
            "einsum node: two operands disagree on a label's extent");
     if (e < 0) e = f;
   }()),
   ...);
  return e;
}

template <typename T>
inline constexpr bool einsum_leaf_ok_v =
    has_node_tag_v<SlotTag, T> || has_node_tag_v<FunctionalTag, T>;

template <typename Tuple, std::size_t... Is>
auto to_device_tuple(const Tuple& t, std::index_sequence<Is...>) {
  return DeviceTuple<std::tuple_element_t<Is, Tuple>...>(std::get<Is>(t)...);
}

template <typename Scalar, typename ExecSpace, int32_t... Out,
          typename... Ops>
auto make_einsum_node_impl(Ops... ops) {
  static_assert(sizeof...(Ops) >= 1, "einsum node needs at least one operand");
  using OutSeq = std::integer_sequence<int32_t, Out...>;
  static_assert(labels_distinct_v<OutSeq>,
                "einsum node: output labels must be pairwise distinct (a "
                "diagonal comes from a delta operand, not a repeated label)");
  static_assert(((labels_distinct_v<typename Ops::modes_seq>) && ...),
                "einsum node: an operand's labels must be pairwise distinct");

  using C = einsum_collect<0, Ops...>;
  using Structure =
      EinsumStructure<OutSeq, typename C::leaf_labels, typename C::plain,
                      typename C::deltas>;

  const auto leaves_std = std::tuple_cat(einsum_op_traits<Ops, 0>::leaves(ops)...);
  const auto leaves     = to_device_tuple(
      leaves_std,
      std::make_index_sequence<std::tuple_size_v<decltype(leaves_std)>>{});
  using LeavesT = std::remove_const_t<decltype(leaves)>;

  return [&]<typename... Leaves>(DeviceTuple<Leaves...>*) {
    static_assert((einsum_leaf_ok_v<Leaves> && ...),
                  "einsum node: every operand (and every delta-case factor) "
                  "must be a graph slot or a functional input");
    static_assert(
        (std::is_same_v<typename Leaves::value_type, Scalar> && ...),
        "einsum node: every operand must have the node's scalar type");
    Kokkos::Array<int, sizeof...(Out)> shape{einsum_extent_of<Out>(
        typename C::leaf_labels{}, leaves,
        std::index_sequence_for<Leaves...>{})...};
    return NodeHandle<EinsumTag, Scalar, ExecSpace, OutSeq, Structure, void,
                      Leaves...>{leaves, shape};
  }(static_cast<LeavesT*>(nullptr));
}

// Scalar and execution space of an operand; a delta operand answers for its
// first factor.
template <typename Op>
struct einsum_op_types {
  using value_type = typename Op::value_type;
  using exec_space = typename Op::exec_space;
};
template <typename B, typename M, int32_t S, typename... Cs>
struct einsum_op_types<DeltaOperand<B, M, S, Cs...>> {
  using factor0 = tuple_element_t<
      0, typename tuple_element_t<0, DeviceTuple<Cs...>>::factors_tuple_t>;
  using value_type = typename factor0::value_type;
  using exec_space = typename factor0::exec_space;
};

}  // namespace Impl

/// make_einsum_node<Out...>(ops...): C{Out} = sum_{labels not in Out} prod ops
template <int32_t... Out, typename Op0, typename... Ops>
auto make_einsum_node(Op0 op0, Ops... ops) {
  return Impl::make_einsum_node_impl<
      typename Impl::einsum_op_types<Op0>::value_type,
      typename Impl::einsum_op_types<Op0>::exec_space, Out...>(op0, ops...);
}

// --- evaluator ---------------------------------------------------------------

/// A slot leaf as the evaluator reads it: its scratch view and the node-space
/// label of each STORAGE axis.
template <typename StorageView, typename Labels>
struct EinsumSlotLeaf {
  using labels                     = Labels;
  using layout_t                   = typename StorageView::layout_t;
  static constexpr bool functional = false;
  StorageView           view;
};
/// A functional leaf: evaluated at the global coordinate, its axes in its own
/// declared order carrying these node-space labels.
template <typename FnNode, typename Labels>
struct EinsumFuncLeaf {
  using labels                     = Labels;
  static constexpr bool functional = true;
  FnNode                node;
};

template <typename Node, typename OutView, typename... Leaves>
class EinsumEvaluator {
 public:
  using value_type          = typename Node::value_type;
  static constexpr int Rank = Node::Rank;
  static constexpr int NT   = Node::NumTerms;
  using out_layout_t        = typename OutView::layout_t;

  KOKKOS_FUNCTION EinsumEvaluator(const DeviceTuple<Leaves...>& leaves,
                                  const OutView&                  out,
                                  const Kokkos::Array<int, Rank>& origin)
      : leaves_(leaves), out_(out), origin_(origin) {}

  template <typename Coord>
  KOKKOS_FORCEINLINE_FUNCTION value_type compute(const Coord& coord) const {
    Kokkos::Array<int, Rank> gidx{};
    TENSOR_PRAGMA_UNROLL
    for (int d = 0; d < Rank; ++d) gidx[d] = origin_[d] + coord[d];
    value_type acc = 0;
    terms(coord, gidx, acc, std::make_index_sequence<NT>{});
    return acc;
  }

  template <typename Coord>
  KOKKOS_FORCEINLINE_FUNCTION void store(const Coord& coord,
                                         const value_type v) const {
    out_[coord] = v;
  }

 private:
  template <typename Coord, std::size_t... Ts>
  KOKKOS_FORCEINLINE_FUNCTION void terms(const Coord&                    coord,
                                         const Kokkos::Array<int, Rank>& gidx,
                                         value_type&                     acc,
                                         std::index_sequence<Ts...>) const {
    (term<static_cast<int>(Ts)>(coord, gidx, acc), ...);
  }

  template <int A, int B, int Cst, typename Coord>
  KOKKOS_FORCEINLINE_FUNCTION static bool check_one(const Coord& coord) {
    if constexpr (B >= 0)
      return coord[A] == coord[B];
    else
      return coord[A] == Cst;
  }
  template <typename Coord, int... As, int... Bs, int... Cs>
  KOKKOS_FORCEINLINE_FUNCTION static bool checks(
      const Coord& coord, std::integer_sequence<int, As...>,
      std::integer_sequence<int, Bs...>, std::integer_sequence<int, Cs...>) {
    return (true && ... && check_one<As, Bs, Cs>(coord));
  }

  // A slot leaf's offset from its output-bound and constant axes: the part of
  // its address a term fixes before its loop.
  template <int K, int V, int S, typename Coord>
  KOKKOS_FORCEINLINE_FUNCTION static int base_term(const Coord& coord) {
    if constexpr (K == Impl::einsum_bind_out)
      return S * coord[V];
    else if constexpr (K == Impl::einsum_bind_const)
      return S * V;
    else
      return 0;
  }
  template <int K, int V, int S, typename LoopIdx>
  KOKKOS_FORCEINLINE_FUNCTION static int loop_term(const LoopIdx& li) {
    if constexpr (K == Impl::einsum_bind_loop)
      return S * li[V];
    else
      return 0;
  }
  template <int K, int V, typename LoopIdx>
  KOKKOS_FORCEINLINE_FUNCTION static int arg_of(
      const Kokkos::Array<int, Rank>& gidx, const LoopIdx& li) {
    if constexpr (K == Impl::einsum_bind_out)
      return gidx[V];
    else if constexpr (K == Impl::einsum_bind_const)
      return V;
    else
      return li[V];
  }

  template <typename Layout, std::size_t... Ds>
  static constexpr auto strides_of(std::index_sequence<Ds...>) {
    return std::integer_sequence<int, Layout::stride(static_cast<int>(Ds))...>{};
  }

  template <typename Coord, int... Ks, int... Vs, int... Ss>
  KOKKOS_FORCEINLINE_FUNCTION static int base_offset(
      const Coord& coord, std::integer_sequence<int, Ks...>,
      std::integer_sequence<int, Vs...>, std::integer_sequence<int, Ss...>) {
    return (0 + ... + base_term<Ks, Vs, Ss>(coord));
  }
  template <typename LoopIdx, int... Ks, int... Vs, int... Ss>
  KOKKOS_FORCEINLINE_FUNCTION static int loop_offset(
      const LoopIdx& li, std::integer_sequence<int, Ks...>,
      std::integer_sequence<int, Vs...>, std::integer_sequence<int, Ss...>) {
    return (0 + ... + loop_term<Ks, Vs, Ss>(li));
  }

  template <typename Info, int J>
  struct LeafPlan {
    using leaf      = tuple_element_t<static_cast<std::size_t>(J),
                                      DeviceTuple<Leaves...>>;
    using kinds     = typename Info::template kinds_seq<typename leaf::labels>;
    using vals      = typename Info::template vals_seq<typename leaf::labels>;
  };

  template <typename Info, int J, typename Coord>
  KOKKOS_FORCEINLINE_FUNCTION int leaf_base(const Coord& coord) const {
    using P = LeafPlan<Info, J>;
    if constexpr (P::leaf::functional) {
      return 0;
    } else {
      using Lay = typename P::leaf::layout_t;
      using Str = decltype(strides_of<Lay>(
          std::make_index_sequence<static_cast<std::size_t>(Lay::rank)>{}));
      return base_offset(coord, typename P::kinds{}, typename P::vals{},
                         Str{});
    }
  }

  template <typename FnNode, typename LoopIdx, int... Ks, int... Vs>
  KOKKOS_FORCEINLINE_FUNCTION static value_type call_fn(
      const FnNode& n, const Kokkos::Array<int, Rank>& gidx, const LoopIdx& li,
      std::integer_sequence<int, Ks...>, std::integer_sequence<int, Vs...>) {
    return n.fn_(arg_of<Ks, Vs>(gidx, li)...);
  }

  template <typename Info, int J, typename LoopIdx>
  KOKKOS_FORCEINLINE_FUNCTION value_type leaf_value(
      const Kokkos::Array<int, Rank>& gidx, const int base,
      const LoopIdx& li) const {
    using P           = LeafPlan<Info, J>;
    const auto& leaf  = leaves_.template get<static_cast<std::size_t>(J)>();
    if constexpr (P::leaf::functional) {
      return call_fn(leaf.node, gidx, li, typename P::kinds{},
                     typename P::vals{});
    } else {
      using Lay = typename P::leaf::layout_t;
      using Str = decltype(strides_of<Lay>(
          std::make_index_sequence<static_cast<std::size_t>(Lay::rank)>{}));
      return leaf.view.data()[base + loop_offset(li, typename P::kinds{},
                                                 typename P::vals{}, Str{})];
    }
  }

  template <typename Info, typename LoopIdx, int... Js, std::size_t... Us>
  KOKKOS_FORCEINLINE_FUNCTION value_type
  product(const Kokkos::Array<int, Rank>& gidx,
          const Kokkos::Array<int, sizeof...(Js)>& bases, const LoopIdx& li,
          std::integer_sequence<int, Js...>, std::index_sequence<Us...>) const {
    return (value_type(1) * ... * leaf_value<Info, Js>(gidx, bases[Us], li));
  }

  template <typename Info, typename Coord, int... Js>
  KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<int, sizeof...(Js)> bases_of(
      const Coord& coord, std::integer_sequence<int, Js...>) const {
    return {leaf_base<Info, Js>(coord)...};
  }

  template <int T, typename Coord>
  KOKKOS_FORCEINLINE_FUNCTION void term(const Coord&                    coord,
                                        const Kokkos::Array<int, Rank>& gidx,
                                        value_type& acc) const {
    using Info = typename Node::template term_info<T>;
    if constexpr (Info::dead) {
      return;
    } else {
      if (!checks(coord, typename Info::chk_a_seq{},
                  typename Info::chk_b_seq{}, typename Info::chk_c_seq{}))
        return;
      using Used            = typename Info::leaves_seq;
      constexpr int NU      = static_cast<int>(Used::size());
      constexpr int NL      = Info::num_loop;
      constexpr auto ext    = Impl::seq_to_karray(typename Info::loop_ext_seq{});
      constexpr int  total  = [] {
        int p = 1;
        for (int i = 0; i < NL; ++i) p *= Impl::seq_to_array(
                                          typename Info::loop_ext_seq{})[i];
        return p;
      }();
      const Kokkos::Array<int, NU> bases = bases_of<Info>(coord, Used{});
      TENSOR_PRAGMA_UNROLL
      for (int lin = 0; lin < total; ++lin) {
        Kokkos::Array<int, (NL > 0 ? NL : 1)> li{};
        int                                    rem = lin;
        TENSOR_PRAGMA_UNROLL
        for (int j = NL - 1; j >= 0; --j) {
          li[j] = rem % ext[j];
          rem /= ext[j];
        }
        acc += product<Info>(gidx, bases, li, Used{},
                             std::make_index_sequence<NU>{});
      }
    }
  }

  DeviceTuple<Leaves...>   leaves_;
  OutView                  out_;
  Kokkos::Array<int, Rank> origin_;
};

template <typename Node, typename OutView, typename... Leaves>
KOKKOS_FUNCTION auto make_einsum_evaluator(
    const DeviceTuple<Leaves...>& leaves, const OutView& out,
    const Kokkos::Array<int, Node::Rank>& origin) {
  return EinsumEvaluator<Node, OutView, Leaves...>(leaves, out, origin);
}

}  // namespace TensorOperations
