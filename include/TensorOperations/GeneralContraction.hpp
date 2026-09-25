#pragma once
// ---------------------------------------------------------------------------
// General contraction -- C{Out} = sum over every label not in Out of the
// product of the operands, the labels deciding everything:
//
//   make_contraction_node<'e','i'>(a.as<'e','i','j'>(), w.as<'e','j'>())
//   make_contraction_node<Scalar, ExecSpace, 'e','i'>(...)
//
// * A label may appear on several operands and in Out (batch / Hadamard).
// * A label on an operand but not in Out is summed. Its extent comes from the
//   graph's label map (it must be LabelWhole there).
// * Dense operands are slots (level outputs) or functional inputs, read at the
//   coordinate the labels give; a functional input is evaluated on demand and
//   never staged.
// * Structured operands -- delta<'a','b'>(), delta<'r'>(idx<t>), outer(...),
//   stack<'r'>(...), and .as<>() of those (Structured.hpp) -- carry no data.
//   Each lowers to a set of terms; the node multiplies its operands' term
//   sets, first operand outermost.
//
// Dispatch. make_contraction_node is the binary GEMM (ContractionTag,
// NodeHandle.hpp) exactly when it is given two node operands whose labels make
// a valid A x B -> C{Out} -- each Out label on exactly one side -- optionally
// followed by a hook. Every other call is this node (GeneralContractionTag).
//
// Expansion. LevelGraph::add, which knows the label map, expands every term:
// the labels its deltas join are merged; a class holding an output label is
// bound to it (two output labels merged become an equality test on the output
// coordinate), a class holding a constant is bound to it (two different
// constants make the term vanish), and every other class is a summed loop. A
// delta joining two axes of one dense operand is a diagonal read. Written with
// a dense D the stiffness B^T M B is the plain O(N^9) sum; written with the
// stacked-delta B of Structured.hpp it is the nine sum-factored terms -- the
// structure is a property of the operand, not of the expression.
// ---------------------------------------------------------------------------
#include <TensorOperations/DeviceTuple.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LabelTiles.hpp>
#include <TensorOperations/Macros.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Permute.hpp>
#include <TensorOperations/Structured.hpp>

#include <array>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

namespace TensorOperations {

// --- the node's compile-time structure -------------------------------------

namespace Impl {

/// The node's structure, as types: its output labels and the product of its
/// operands' term sets (a GcLowered, Structured.hpp).
template <typename OutSeq, typename Lowered>
struct GcStructure {};

template <typename Structure>
struct gc_structure_traits;
template <typename OutSeq, typename LeafLabels, typename Terms,
          typename StackLabs, typename StackLens>
struct gc_structure_traits<
    GcStructure<OutSeq, GcLowered<LeafLabels, Terms, StackLabs, StackLens>>> {
  using out_seq                   = OutSeq;
  using leaf_labels               = LeafLabels;
  using terms                     = Terms;
  static constexpr int num_terms  = gc_list_size<Terms>::value;
  static constexpr int num_leaves = gc_list_size<LeafLabels>::value;
};

/// Node-space labels of every leaf, in each leaf's own axis order.
template <typename Structure>
using gc_leaf_labels_t = typename gc_structure_traits<Structure>::leaf_labels;
/// Term T of a structure.
template <typename Structure, int T>
using gc_term_t = gc_list_at_t<static_cast<std::size_t>(T),
                               typename gc_structure_traits<Structure>::terms>;

// --- the structure as plain constexpr data ---------------------------------

inline constexpr int gc_max_out    = 16;  // output labels
inline constexpr int gc_max_leaf   = 32;  // dense leaves per node
inline constexpr int gc_max_rank   = 16;  // axes per leaf
inline constexpr int gc_max_pair   = 32;  // deltas per term
inline constexpr int gc_max_const  = 16;  // constant bindings per term
inline constexpr int gc_max_stack  = 16;  // stack labels per node
inline constexpr int gc_max_labels = 64;  // distinct labels per term
inline constexpr int gc_max_chk    = 16;  // output-coordinate tests per term
inline constexpr int gc_max_loop   = 16;  // summed classes per term

/// Node-wide data: the outputs, every leaf's node-space labels, and every
/// stack label with its branch count.
struct GcSpec {
  bool                                                      overflow = false;
  int                                                       nout     = 0;
  std::array<int32_t, gc_max_out>                           out{};
  int                                                       nleaf = 0;
  std::array<int, gc_max_leaf>                              leaf_rank{};
  std::array<std::array<int32_t, gc_max_rank>, gc_max_leaf> leaf_lab{};
  int                                                       nstack = 0;
  std::array<int32_t, gc_max_stack>                         stack_lab{};
  std::array<int, gc_max_stack>                             stack_len{};
};

/// One term: the leaves it multiplies, the pairs its deltas equate, and its
/// constant bindings.
struct GcTermDesc {
  bool                                 overflow = false;
  int                                  nleaf    = 0;
  std::array<int, gc_max_leaf>         leaf{};
  int                                  npair = 0;
  std::array<int32_t, 2 * gc_max_pair> pair{};
  int                                  nconst = 0;
  std::array<int32_t, gc_max_const>    const_lab{};
  std::array<int, gc_max_const>        const_val{};
};

template <int32_t... Ls>
constexpr void gc_spec_add_leaf(GcSpec& s,
                                std::integer_sequence<int32_t, Ls...>) {
  if (s.nleaf >= gc_max_leaf || static_cast<int>(sizeof...(Ls)) > gc_max_rank) {
    s.overflow = true;
    return;
  }
  const int32_t l[]    = {Ls..., 0};
  s.leaf_rank[s.nleaf] = static_cast<int>(sizeof...(Ls));
  for (std::size_t d = 0; d < sizeof...(Ls); ++d) s.leaf_lab[s.nleaf][d] = l[d];
  ++s.nleaf;
}

template <typename Structure>
struct gc_spec;
template <int32_t... Out, typename... LeafLabels, typename Terms, int32_t... SL,
          int... SN>
struct gc_spec<GcStructure<std::integer_sequence<int32_t, Out...>,
                           GcLowered<GcList<LeafLabels...>, Terms,
                                     std::integer_sequence<int32_t, SL...>,
                                     std::integer_sequence<int, SN...>>>> {
  static constexpr GcSpec make() {
    GcSpec s{};
    if (static_cast<int>(sizeof...(Out)) > gc_max_out ||
        static_cast<int>(sizeof...(SL)) > gc_max_stack) {
      s.overflow = true;
      return s;
    }
    const int32_t o[] = {Out..., 0};
    s.nout            = static_cast<int>(sizeof...(Out));
    for (std::size_t i = 0; i < sizeof...(Out); ++i) s.out[i] = o[i];
    (gc_spec_add_leaf(s, LeafLabels{}), ...);
    const int32_t sl[] = {SL..., 0};
    const int     sn[] = {SN..., 0};
    s.nstack           = static_cast<int>(sizeof...(SL));
    for (std::size_t i = 0; i < sizeof...(SL); ++i) {
      s.stack_lab[i] = sl[i];
      s.stack_len[i] = sn[i];
    }
    return s;
  }
  static constexpr GcSpec value = make();
};

template <typename Term>
struct gc_term_desc;
template <int... L, int32_t... P, int32_t... C, int... V>
struct gc_term_desc<GcTerm<
    std::integer_sequence<int, L...>, std::integer_sequence<int32_t, P...>,
    std::integer_sequence<int32_t, C...>, std::integer_sequence<int, V...>>> {
  static constexpr GcTermDesc make() {
    GcTermDesc d{};
    if (static_cast<int>(sizeof...(L)) > gc_max_leaf ||
        static_cast<int>(sizeof...(P)) > 2 * gc_max_pair ||
        static_cast<int>(sizeof...(C)) > gc_max_const) {
      d.overflow = true;
      return d;
    }
    const int     l[] = {L..., 0};
    const int32_t p[] = {P..., 0};
    const int32_t c[] = {C..., 0};
    const int     v[] = {V..., 0};
    d.nleaf           = static_cast<int>(sizeof...(L));
    for (std::size_t i = 0; i < sizeof...(L); ++i) d.leaf[i] = l[i];
    d.npair = static_cast<int>(sizeof...(P)) / 2;
    for (std::size_t i = 0; i < sizeof...(P); ++i) d.pair[i] = p[i];
    d.nconst = static_cast<int>(sizeof...(C));
    for (std::size_t i = 0; i < sizeof...(C); ++i) {
      d.const_lab[i] = c[i];
      d.const_val[i] = v[i];
    }
    return d;
  }
  static constexpr GcTermDesc value = make();
};

// Whether the structure exceeds a capacity before any label map is involved.
template <typename Structure,
          typename Terms = typename gc_structure_traits<Structure>::terms>
struct gc_overflow;
template <typename Structure, typename... Ts>
struct gc_overflow<Structure, GcList<Ts...>> {
  static constexpr bool value =
      gc_spec<Structure>::value.overflow ||
      (false || ... || gc_term_desc<Ts>::value.overflow);
};

// --- term expansion ---------------------------------------------------------

inline constexpr int gc_bind_out   = 0;  // val = output position
inline constexpr int gc_bind_const = 1;  // val = the constant
inline constexpr int gc_bind_loop  = 2;  // val = loop index

// Error codes of a term (0 = fine), reported at LevelGraph::add.
inline constexpr int gc_err_overflow       = 1;
inline constexpr int gc_err_sum_not_in_map = 2;
inline constexpr int gc_err_sum_gridded    = 3;
inline constexpr int gc_err_extent         = 4;
inline constexpr int gc_err_gridded_merged = 5;
inline constexpr int gc_err_stack_extent   = 6;
inline constexpr int gc_err_const_range    = 7;

struct GcTermPlan {
  int  error = 0;
  bool dead  = false;  // two constants bound to one class: the term is zero

  int                          nleaf = 0;  // the leaves this term multiplies
  std::array<int, gc_max_leaf> leaf{};

  int                                nlab = 0;  // label universe + binding
  std::array<int32_t, gc_max_labels> lab{};
  std::array<int, gc_max_labels>     kind{};
  std::array<int, gc_max_labels>     val{};

  // Output-coordinate tests: coord[chk_a] == coord[chk_b] (chk_b >= 0) or
  // coord[chk_a] == chk_c (chk_b < 0).
  int                         nchk = 0;
  std::array<int, gc_max_chk> chk_a{};
  std::array<int, gc_max_chk> chk_b{};
  std::array<int, gc_max_chk> chk_c{};

  int                          nloop = 0;  // summed classes, and extents
  std::array<int, gc_max_loop> loop_ext{};

  // Per output position: its class is that label alone -- no delta, stack
  // or constant touches it. What a value label (below) needs of every term.
  std::array<bool, gc_max_out> out_alone{};
};

template <typename LT>
constexpr bool gc_in_map(int32_t l) {
  return label_index_of<LT>(l) >= 0;
}

template <typename LT>
constexpr GcTermPlan gc_term(const GcSpec& s, const GcTermDesc& d) {
  GcTermPlan r{};
  if (s.overflow || d.overflow) {
    r.error = gc_err_overflow;
    return r;
  }

  for (int i = 0; i < d.nleaf; ++i) r.leaf[r.nleaf++] = d.leaf[i];

  // Label universe: the outputs first (so universe index == output position),
  // then every label the term's leaves, constants and deltas name.
  auto add_label = [&](int32_t l) {
    for (int u = 0; u < r.nlab; ++u)
      if (r.lab[u] == l) return u;
    if (r.nlab >= gc_max_labels) {
      r.error = gc_err_overflow;
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
  for (int c = 0; c < d.nconst; ++c) add_label(d.const_lab[c]);
  for (int p = 0; p < 2 * d.npair; ++p) add_label(d.pair[p]);
  if (r.error) return r;

  // A stack label the map knows must have one index per branch.
  for (int k = 0; k < s.nstack; ++k)
    if (gc_in_map<LT>(s.stack_lab[k]) &&
        label_tile_of<LT>(s.stack_lab[k]) != s.stack_len[k]) {
      r.error = gc_err_stack_extent;
      return r;
    }

  // Union-find over the universe: each delta joins its two labels.
  std::array<int, gc_max_labels> parent{};
  for (int u = 0; u < r.nlab; ++u) parent[u] = u;
  auto find = [&](int u) {
    while (parent[u] != u) u = parent[u];
    return u;
  };
  for (int p = 0; p < d.npair; ++p) {
    const int a = find(add_label(d.pair[2 * p]));
    const int b = find(add_label(d.pair[2 * p + 1]));
    if (a != b) parent[a < b ? b : a] = a < b ? a : b;  // root = lowest
  }

  // Per class: its first output position, and its constant.
  std::array<int, gc_max_labels> cls_out{};
  std::array<int, gc_max_labels> cls_const{};
  std::array<int, gc_max_labels> cls_loop{};
  for (int u = 0; u < r.nlab; ++u) {
    cls_out[u]   = -1;
    cls_const[u] = -1;
    cls_loop[u]  = -1;
  }
  for (int c = 0; c < d.nconst; ++c) {
    const int root = find(add_label(d.const_lab[c]));
    if (cls_const[root] >= 0 && cls_const[root] != d.const_val[c])
      r.dead = true;
    cls_const[root] = d.const_val[c];
  }
  if (r.dead) return r;  // identically zero: nothing to bind or check

  auto add_check = [&](int a, int b, int c) {
    if (r.nchk >= gc_max_chk) {
      r.error = gc_err_overflow;
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
  for (int u = 0; u < r.nlab; ++u)
    if (find(u) == u && cls_out[u] >= 0 && cls_const[u] >= 0)
      add_check(cls_out[u], -1, cls_const[u]);

  // What is left is summed: one loop per class, its extent from the map.
  for (int u = 0; u < r.nlab; ++u) {
    if (find(u) != u || cls_out[u] >= 0 || cls_const[u] >= 0) continue;
    int ext = -1;
    for (int v = 0; v < r.nlab; ++v) {
      if (find(v) != u) continue;
      if (!gc_in_map<LT>(r.lab[v])) {
        r.error = gc_err_sum_not_in_map;
        return r;
      }
      if (label_gridded_of<LT>(r.lab[v])) {
        r.error = gc_err_sum_gridded;
        return r;
      }
      const int e = label_tile_of<LT>(r.lab[v]);
      if (ext >= 0 && e != ext) {
        r.error = gc_err_extent;
        return r;
      }
      ext = e;
    }
    if (r.nloop >= gc_max_loop) {
      r.error = gc_err_overflow;
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
    if (size != 1 || cls_out[root] < 0 || cls_const[root] >= 0) {
      r.error = gc_err_gridded_merged;
      return r;
    }
  }

  // Labels a delta joins are one index, so the map must give them one extent,
  // and a constant bound to them must lie inside it.
  for (int u = 0; u < r.nlab; ++u) {
    if (find(u) != u) continue;
    int ext = -1;
    for (int v = 0; v < r.nlab; ++v) {
      if (find(v) != u || !gc_in_map<LT>(r.lab[v])) continue;
      const int e = label_tile_of<LT>(r.lab[v]);
      if (ext >= 0 && e != ext) {
        r.error = gc_err_extent;
        return r;
      }
      ext = e;
    }
    if (cls_const[u] >= 0 && ext >= 0 && cls_const[u] >= ext) {
      r.error = gc_err_const_range;
      return r;
    }
  }

  for (int u = 0; u < r.nlab; ++u) {
    const int root = find(u);
    if (cls_out[root] >= 0) {
      r.kind[u] = gc_bind_out;
      r.val[u]  = cls_out[root];
    } else if (cls_const[root] >= 0) {
      r.kind[u] = gc_bind_const;
      r.val[u]  = cls_const[root];
    } else {
      r.kind[u] = gc_bind_loop;
      r.val[u]  = cls_loop[root];
    }
  }

  for (int o = 0; o < s.nout; ++o) {
    const int root = find(o);
    int       size = 0;
    for (int v = 0; v < r.nlab; ++v) size += (find(v) == root) ? 1 : 0;
    r.out_alone[o] = size == 1 && cls_const[root] < 0;
  }
  return r;
}

/// A term's plan, as sequences the evaluator expands over.
template <typename Structure, typename LT, int T>
struct GcTermInfo {
  static constexpr GcTermPlan term = gc_term<LT>(
      gc_spec<Structure>::value, gc_term_desc<gc_term_t<Structure, T>>::value);

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
    constexpr auto            l = seq_to_array(Labels{});
    std::array<int, l.size()> a{};
    for (std::size_t d = 0; d < l.size(); ++d) a[d] = binding(l[d], true);
    return a;
  }
  template <typename Labels>
  static constexpr auto vals_arr() {
    constexpr auto            l = seq_to_array(Labels{});
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
constexpr int gc_first_error(std::index_sequence<Ts...>) {
  int e = 0;
  ((e = e ? e : GcTermInfo<Structure, LT, static_cast<int>(Ts)>::error), ...);
  return e;
}

// --- value labels -----------------------------------------------------------
//
// A VALUE LABEL is an output label whose coordinate only ever selects an
// element of ONE leaf per term. In every live term:
//
//   * its class is itself alone -- no delta, stack or constant touches it;
//   * exactly one of the term's leaves carries it, and that leaf carries
//     every value label of the term (it may be a different leaf in different
//     terms);
//
// and it is LabelWhole, so every team holds its whole extent. Such a label
// takes no part in a term's equality checks, its loop, or its other leaves:
// at a fixed coordinate of the remaining (STRUCTURAL) output labels, the
// entries along the value labels share all of that work and differ only in
// the element of the value leaf they read. The evaluator therefore iterates
// the structural space and carries the whole value block in registers.
//
// Selection is GREEDY IN OUTPUT ORDER. A label that qualifies on its own is
// accepted if, in every live term, the leaf carrying it is the leaf carrying
// the labels accepted before it, and the block -- the product of the accepted
// extents -- stays within gc_max_value entries; otherwise it stays
// structural. A node with a term error or no live term has no value labels.
inline constexpr int gc_max_value = 32;  // entries of the in-register block

struct GcValuePlan {
  int                             n = 0;  // value labels, in output order
  std::array<int, gc_max_out>     dim{};  // their output positions
  std::array<int32_t, gc_max_out> lab{};  // and their labels
};

constexpr bool gc_leaf_carries(const GcSpec& s, int j, int32_t l) {
  for (int a = 0; a < s.leaf_rank[j]; ++a)
    if (s.leaf_lab[j][a] == l) return true;
  return false;
}

template <typename Structure, typename LT, std::size_t... Ts>
constexpr GcValuePlan gc_value_plan(std::index_sequence<Ts...>) {
  constexpr std::size_t            NT = sizeof...(Ts);
  GcValuePlan                      p{};
  const GcSpec&                    s = gc_spec<Structure>::value;
  const std::array<GcTermPlan, NT> terms{
      GcTermInfo<Structure, LT, static_cast<int>(Ts)>::term...};
  int live = 0;
  for (std::size_t t = 0; t < NT; ++t) {
    if (terms[t].error) return p;
    live += terms[t].dead ? 0 : 1;
  }
  if (live == 0) return p;

  std::array<int, NT> carrier{};  // per term, the leaf holding the block
  for (std::size_t t = 0; t < NT; ++t) carrier[t] = -1;
  int block = 1;
  for (int o = 0; o < s.nout; ++o) {
    const int32_t l = s.out[o];
    if (!gc_in_map<LT>(l) || label_gridded_of<LT>(l)) continue;
    const int e = label_tile_of<LT>(l);
    if (block * e > gc_max_value) continue;
    std::array<int, NT> leaf_of{};
    bool                ok = true;
    for (std::size_t t = 0; t < NT && ok; ++t) {
      const GcTermPlan& r = terms[t];
      leaf_of[t]          = -1;
      if (r.dead) continue;
      int count = 0;
      for (int i = 0; i < r.nleaf; ++i)
        if (gc_leaf_carries(s, r.leaf[i], l)) {
          leaf_of[t] = r.leaf[i];
          ++count;
        }
      ok = r.out_alone[o] && count == 1 &&
           (carrier[t] < 0 || carrier[t] == leaf_of[t]);
    }
    if (!ok) continue;
    for (std::size_t t = 0; t < NT; ++t)
      if (!terms[t].dead) carrier[t] = leaf_of[t];
    p.dim[p.n] = o;
    p.lab[p.n] = l;
    ++p.n;
    block *= e;
  }
  return p;
}

/// A node's value labels, as sequences: their output positions (dims_seq)
/// and their labels (labels_seq), both in output order. Empty until the label
/// map is known.
template <typename Structure, typename LT>
struct gc_value {
  static constexpr GcValuePlan plan = gc_value_plan<Structure, LT>(
      std::make_index_sequence<static_cast<std::size_t>(
          gc_structure_traits<Structure>::num_terms)>{});
  static constexpr auto dims_arr = [] {
    std::array<int, static_cast<std::size_t>(plan.n)> a{};
    for (int i = 0; i < plan.n; ++i)
      a[static_cast<std::size_t>(i)] = plan.dim[i];
    return a;
  }();
  static constexpr auto labs_arr = [] {
    std::array<int32_t, static_cast<std::size_t>(plan.n)> a{};
    for (int i = 0; i < plan.n; ++i)
      a[static_cast<std::size_t>(i)] = plan.lab[i];
    return a;
  }();
  using dims_seq   = array_to_seq_t<dims_arr>;
  using labels_seq = array_to_seq_t<labs_arr>;
};
template <typename Structure>
struct gc_value<Structure, void> {
  static constexpr GcValuePlan plan{};
  using dims_seq   = std::integer_sequence<int>;
  using labels_seq = std::integer_sequence<int32_t>;
};

/// Position, in term T's leaf list, of the leaf carrying the node's value
/// labels; -1 for a node without value labels and for a dead term.
template <typename Structure, typename LT, int T>
constexpr int gc_value_leaf_pos() {
  const GcValuePlan& p = gc_value<Structure, LT>::plan;
  const GcTermPlan&  r = GcTermInfo<Structure, LT, T>::term;
  if (p.n == 0 || r.dead) return -1;
  const GcSpec& s = gc_spec<Structure>::value;
  for (int i = 0; i < r.nleaf; ++i)
    if (gc_leaf_carries(s, r.leaf[i], p.lab[0])) return i;
  return -1;
}
template <typename Structure, typename LT, int T>
inline constexpr int gc_value_leaf_pos_v =
    gc_value_leaf_pos<Structure, LT, T>();

template <typename LT, typename OutSeq>
struct gc_tile {
  using type = tile_from_labels_t<LT, OutSeq>;
};
template <typename OutSeq>
struct gc_tile<void, OutSeq> {
  using type = void;
};

}  // namespace Impl

// --- the node --------------------------------------------------------------
//
// LT (the graph's label map) is void until LevelGraph::add resolves it: the
// summed extents and the output tile both come from the map.
template <typename Scalar, typename ExecSpace, typename OutSeq,
          typename Structure, typename LT, typename... Leaves>
struct NodeHandle<GeneralContractionTag, Scalar, ExecSpace, OutSeq, Structure,
                  LT, Leaves...> {
  using node_tag              = GeneralContractionTag;
  static constexpr int Rank   = static_cast<int>(OutSeq::size());
  static constexpr int NumOps = static_cast<int>(sizeof...(Leaves));
  static constexpr int NumOut = 1;
  using value_type            = Scalar;
  using exec_space            = ExecSpace;
  using modes_seq             = OutSeq;
  using structure_type        = Structure;
  using label_tiles_type      = LT;
  using tile_type             = typename Impl::gc_tile<LT, OutSeq>::type;
  using ops_tuple_t           = DeviceTuple<Leaves...>;  // the flat leaves

  static constexpr int NumTerms =
      Impl::gc_structure_traits<Structure>::num_terms;
  template <int T>
  using term_info = Impl::GcTermInfo<Structure, LT, T>;

  // The value labels (Impl::gc_value): output positions and labels, in
  // output order. Empty until LevelGraph::add resolves the label map.
  using value_dims_seq   = typename Impl::gc_value<Structure, LT>::dims_seq;
  using value_labels_seq = typename Impl::gc_value<Structure, LT>::labels_seq;

  ops_tuple_t              operands;
  Kokkos::Array<int, Rank> shape_;  // -1 where no leaf carries the label,
                                    // until add() fills it in from the map

  KOKKOS_FUNCTION Kokkos::Array<int, Rank> shape() const { return shape_; }
};

namespace Impl {

// Extent of output label L read off the first leaf carrying it, or -1.
template <int32_t L, typename LeafLabels, typename Leaf>
int gc_extent_or_neg(const Leaf& leaf) {
  constexpr auto m = seq_to_array(LeafLabels{});
  for (std::size_t d = 0; d < m.size(); ++d)
    if (m[d] == L) return leaf.shape()[d];
  return -1;
}
template <int32_t L, typename... LeafLabels, typename... Leaves,
          std::size_t... Js>
int gc_extent_of(GcList<LeafLabels...>, const DeviceTuple<Leaves...>& leaves,
                 std::index_sequence<Js...>) {
  int e = -1;
  ((void)([&] {
     const int f = gc_extent_or_neg<L, LeafLabels>(leaves.template get<Js>());
     assert((f < 0 || e < 0 || f == e) &&
            "make_contraction_node: two operands disagree on a label's "
            "extent");
     if (e < 0) e = f;
   }()),
   ...);
  return e;
}

template <typename T>
inline constexpr bool gc_leaf_ok_v =
    has_node_tag_v<SlotTag, T> || has_node_tag_v<FunctionalTag, T>;

template <typename Tuple, std::size_t... Is>
auto to_device_tuple(const Tuple& t, std::index_sequence<Is...>) {
  return DeviceTuple<std::tuple_element_t<Is, Tuple>...>(std::get<Is>(t)...);
}

// The arguments of a general make_contraction_node call, checked before
// anything is derived from them.
template <typename... Ops>
constexpr bool gc_check_operands() {
  static_assert(sizeof...(Ops) >= 1,
                "make_contraction_node: needs at least one operand");
  static_assert((is_contraction_operand_v<Ops> && ...),
                "make_contraction_node: every argument must be an operand -- "
                "a slot, a functional input, or a structured operand (delta, "
                "outer, stack, .as<>). A trailing hook is accepted only by "
                "the binary GEMM form: two node operands, each output label "
                "on exactly one of them");
  return sizeof...(Ops) >= 1 && (is_contraction_operand_v<Ops> && ...);
}

template <typename Scalar, typename ExecSpace, int32_t... Out, typename... Ops>
auto make_general_contraction_node_impl(const Ops&... ops) {
  using OutSeq = std::integer_sequence<int32_t, Out...>;
  static_assert(labels_distinct_v<OutSeq>,
                "make_contraction_node: output labels must be pairwise "
                "distinct (a diagonal is a delta<'a','b'>() operand, not a "
                "repeated label)");
  static_assert(((labels_distinct_v<typename Ops::modes_seq>) && ...),
                "make_contraction_node: an operand's labels must be pairwise "
                "distinct (read a diagonal by joining two distinct labels "
                "with a delta<'a','b'>() operand)");

  using Lowered   = typename gc_product_all<gc_lower_t<Ops>...>::type;
  using Structure = GcStructure<OutSeq, Lowered>;
  static_assert(!gc_overflow<Structure>::value,
                "make_contraction_node: the operands exceed a fixed capacity "
                "(output labels, dense leaves, axes per leaf, deltas or "
                "constants per term, stack labels); raise the gc_max_* "
                "constants in GeneralContraction.hpp");

  const auto leaves_std = std::tuple_cat(gc_leaves(ops)...);
  const auto leaves     = to_device_tuple(
      leaves_std,
      std::make_index_sequence<std::tuple_size_v<decltype(leaves_std)>>{});
  using LeavesT = std::remove_const_t<decltype(leaves)>;

  // Which KIND of dense leaf is read is checked at LevelGraph::add, where the
  // leaves are read -- so the node, and its dispatch, can be built from any
  // labelled node.
  return [&]<typename... Leaves>(DeviceTuple<Leaves...>*) {
    static_assert(
        (std::is_same_v<typename Leaves::value_type, Scalar> && ...),
        "make_contraction_node: every dense operand must have the node's "
        "scalar type");
    Kokkos::Array<int, sizeof...(Out)> shape{
        gc_extent_of<Out>(gc_leaf_labels_t<Structure>{}, leaves,
                          std::index_sequence_for<Leaves...>{})...};
    return NodeHandle<GeneralContractionTag, Scalar, ExecSpace, OutSeq,
                      Structure, void, Leaves...>{leaves, shape};
  }(static_cast<LeavesT*>(nullptr));
}

// The first dense leaf of the operands (depth first), or void if none.
template <typename... Ops>
struct gc_first_leaf {
  using leaves_t =
      decltype(std::tuple_cat(gc_leaves(std::declval<const Ops&>())...));
  static auto pick() {
    if constexpr (std::tuple_size_v<leaves_t> == 0)
      return static_cast<void*>(nullptr);
    else
      return static_cast<std::tuple_element_t<0, leaves_t>*>(nullptr);
  }
  using type = std::remove_pointer_t<decltype(pick())>;
};

// Whether make_contraction_node(args...) is the binary GEMM of NodeHandle.hpp;
// the general overloads below take exactly the calls it does not.
template <typename OutSeq, typename... Args>
constexpr bool gc_gemm_call() {
  if constexpr (sizeof...(Args) == 2 || sizeof...(Args) == 3) {
    using T = std::tuple<Args...>;
    if constexpr (sizeof...(Args) == 2)
      return gemm_call_v<OutSeq, std::tuple_element_t<0, T>,
                         std::tuple_element_t<1, T>>;
    else
      return gemm_call_v<OutSeq, std::tuple_element_t<0, T>,
                         std::tuple_element_t<1, T>,
                         std::tuple_element_t<2, T>>;
  } else {
    return false;
  }
}
template <typename OutSeq, typename... Args>
inline constexpr bool gc_gemm_call_v = gc_gemm_call<OutSeq, Args...>();

template <typename>
inline constexpr bool gc_dependent_false_v = false;

}  // namespace Impl

/// make_contraction_node<Out...>(ops...): C{Out} = sum_{labels not in Out}
/// prod ops. The scalar and execution space are the first dense operand's.
template <int32_t... Out, typename... Ops>
  requires(
      !Impl::gc_gemm_call_v<std::integer_sequence<int32_t, Out...>, Ops...>)
auto make_contraction_node(Ops... ops) {
  constexpr bool ok = Impl::gc_check_operands<Ops...>();
  using Leaf0       = typename Impl::gc_first_leaf<Ops...>::type;
  if constexpr (!ok) {
    return;  // diagnosed by gc_check_operands
  } else if constexpr (std::is_void_v<Leaf0>) {
    static_assert(Impl::gc_dependent_false_v<Leaf0>,
                  "make_contraction_node<Out...>: no operand is dense, so "
                  "there is no scalar type to deduce -- name it: "
                  "make_contraction_node<Scalar, ExecSpace, Out...>(...)");
  } else {
    return Impl::make_general_contraction_node_impl<
        typename Leaf0::value_type, typename Leaf0::exec_space, Out...>(ops...);
  }
}

/// make_contraction_node<Scalar, ExecSpace, Out...>(ops...)
template <typename Scalar, typename ExecSpace = Kokkos::DefaultExecutionSpace,
          int32_t... Out, typename... Ops>
  requires(
      !Impl::gc_gemm_call_v<std::integer_sequence<int32_t, Out...>, Ops...>)
auto make_contraction_node(Ops... ops) {
  if constexpr (!Impl::gc_check_operands<Ops...>()) {
    return;  // diagnosed by gc_check_operands
  } else {
    return Impl::make_general_contraction_node_impl<Scalar, ExecSpace, Out...>(
        ops...);
  }
}

// --- evaluator ---------------------------------------------------------------

/// A slot leaf as the evaluator reads it: its scratch view and the node-space
/// label of each STORAGE axis.
template <typename StorageView, typename Labels>
struct GcSlotLeaf {
  using labels                     = Labels;
  using layout_t                   = typename StorageView::layout_t;
  static constexpr bool functional = false;
  StorageView           view;
};
/// A functional leaf: evaluated at the global coordinate, its axes in its own
/// declared order carrying these node-space labels.
template <typename FnNode, typename Labels>
struct GcFuncLeaf {
  using labels                     = Labels;
  static constexpr bool functional = true;
  FnNode                node;
};

namespace Impl {

/// Element I of an integer sequence.
template <std::size_t I, typename T, T... Vs>
KOKKOS_FUNCTION constexpr T gc_seq_at(std::integer_sequence<T, Vs...>) {
  constexpr T a[] = {Vs..., T{}};
  return a[I];
}

/// The value block of an output tile: the value dims Ds (output positions, in
/// output order) and their tile extents. Value index v runs row-major over
/// them -- the last value dim fastest, the order the tile itself uses.
template <typename Tile, typename DimsSeq>
struct GcBlock;
template <typename Tile, int... Ds>
struct GcBlock<Tile, std::integer_sequence<int, Ds...>> {
  static constexpr int nv   = static_cast<int>(sizeof...(Ds));
  static constexpr int size = (1 * ... * Tile::extent(Ds));
  using dims_seq            = std::integer_sequence<int, Ds...>;

  /// Which value dim output dim d is, or -1 for a structural dim.
  KOKKOS_FUNCTION static constexpr int value_pos(int d) {
    constexpr int dims[] = {Ds..., -1};
    for (int k = 0; k < nv; ++k)
      if (dims[k] == d) return k;
    return -1;
  }
  /// The k-th value dim's coordinate within value index v.
  KOKKOS_FUNCTION static constexpr int digit(int v, int k) {
    constexpr int ext[] = {Tile::extent(Ds)..., 1};
    int           inner = 1;
    for (int q = nv - 1; q > k; --q) inner *= ext[q];
    return (v / inner) % ext[k];
  }
  /// Output dim d's coordinate within value index v (0 for a structural dim).
  KOKKOS_FUNCTION static constexpr int digit_of_dim(int v, int d) {
    const int k = value_pos(d);
    return k < 0 ? 0 : digit(v, k);
  }
};

}  // namespace Impl

// --- where a member's values go ---------------------------------------------
//
// put(coord, origin, v) stores the entry at LOCAL tile coordinate `coord`;
// put_block<Blk>(coord0, origin, acc, seq) stores a whole value block, coord0
// being the structural coordinate (value dims at 0) and acc[v] the entry at
// value index v of Blk.

/// The member's own scratch tile: where later members, and the root store,
/// read it.
template <typename TileView>
struct GcScratchOut {
  using layout_t = typename TileView::layout_t;
  TileView view;

  template <typename Coord, std::size_t R, typename V>
  KOKKOS_FORCEINLINE_FUNCTION void put(const Coord& coord,
                                       const Kokkos::Array<int, R>&,
                                       const V v) const {
    view[coord] = v;
  }

  template <typename Blk, int V>
  KOKKOS_FUNCTION static constexpr int block_offset() {
    int off = 0;
    for (int d = 0; d < layout_t::rank; ++d)
      off += layout_t::stride(d) * Blk::digit_of_dim(V, d);
    return off;
  }
  template <typename Blk, int V, typename T>
  KOKKOS_FORCEINLINE_FUNCTION static void put_one(T* const p, const T v) {
    constexpr int off = block_offset<Blk, V>();
    p[off]            = v;
  }
  template <typename Blk, typename Coord, std::size_t R, typename Acc,
            std::size_t... Vs>
  KOKKOS_FORCEINLINE_FUNCTION void put_block(const Coord& coord0,
                                             const Kokkos::Array<int, R>&,
                                             const Acc& acc,
                                             std::index_sequence<Vs...>) const {
    auto* const p = &view[coord0];
    (put_one<Blk, static_cast<int>(Vs)>(p, acc[Vs]), ...);
  }
};

/// A STREAMED root: the value goes straight to the designated output view at
/// the GLOBAL coordinate (tile origin + local), in the node's declared output
/// order -- the view's axis order; a general contraction stores unpermuted.
/// No scratch tile backs it (LevelPlan.hpp). `vstride` holds the view's
/// strides along the value dims Ds, read once, so a value block is one
/// address computation plus a constant-digit offset per entry.
template <typename RootView, int Rank, typename DimsSeq>
struct GcRootOut;
template <typename RootView, int Rank, int... Ds>
struct GcRootOut<RootView, Rank, std::integer_sequence<int, Ds...>> {
  RootView                                     root;
  Kokkos::Array<std::ptrdiff_t, sizeof...(Ds)> vstride;

  KOKKOS_FUNCTION explicit GcRootOut(const RootView& r)
      : root(r), vstride{static_cast<std::ptrdiff_t>(r.stride(Ds))...} {}

  template <typename Coord, std::size_t... Is>
  KOKKOS_FORCEINLINE_FUNCTION decltype(auto) at(
      const Coord& coord, const Kokkos::Array<int, Rank>& origin,
      std::index_sequence<Is...>) const {
    return root((origin[Is] + coord[Is])...);
  }

  template <typename Coord, typename V>
  KOKKOS_FORCEINLINE_FUNCTION void put(const Coord&                    coord,
                                       const Kokkos::Array<int, Rank>& origin,
                                       const V v) const {
    at(coord, origin, std::make_index_sequence<Rank>{}) = v;
  }

  template <typename Blk, int V, std::size_t... Ks>
  KOKKOS_FORCEINLINE_FUNCTION std::ptrdiff_t block_offset(
      std::index_sequence<Ks...>) const {
    return (std::ptrdiff_t(0) + ... +
            (vstride[Ks] *
             std::integral_constant<int, Blk::digit(
                                             V, static_cast<int>(Ks))>::value));
  }
  template <typename Blk, typename Coord, typename Acc, std::size_t... Vs>
  KOKKOS_FORCEINLINE_FUNCTION void put_block(
      const Coord& coord0, const Kokkos::Array<int, Rank>& origin,
      const Acc& acc, std::index_sequence<Vs...>) const {
    static_assert(std::is_same_v<typename Blk::dims_seq,
                                 std::integer_sequence<int, Ds...>>,
                  "streamed root: built for other value dims than the block");
    auto* const p = &at(coord0, origin, std::make_index_sequence<Rank>{});
    ((p[block_offset<Blk, static_cast<int>(Vs)>(
          std::make_index_sequence<sizeof...(Ds)>{})] = acc[Vs]),
     ...);
  }
};

/// Per output coordinate (or per structural coordinate and value block, when
/// Blocked), the node's terms unrolled: each term's equality checks, its
/// leaves' base offsets, then its compile-time loop over the product of its
/// leaves.
///
/// BLOCKED (the node's value labels, Impl::gc_value). The output tile is
/// iterated with the value dims removed, and one call computes the whole value
/// block: a term's checks, its non-value leaves' bases and its loop run once
/// for all NB entries. The product keeps the unblocked evaluation's shape
/// EXACTLY -- ((1 * l_0) * ... * l_{P-1}) is formed once per loop step (the
/// leaves before the value leaf P), then for each entry v
///
///   acc[v] += ((pre * value[v]) * l_{P+1}) * ... * l_{NU-1}
///
/// with the leaves after P loaded once and only multiplied per entry. Terms
/// and loop steps are visited in the same order as unblocked, so every entry
/// is BITWISE the unblocked result. Floating-point products are not
/// associative, which is why the leaves after P are not folded into `pre`:
/// with the value leaf last the per-entry work is a single `pre * value[v]`.
template <typename Node, typename Out, bool Blocked, typename... Leaves>
class GeneralContractionEvaluator {
 public:
  using value_type          = typename Node::value_type;
  static constexpr int Rank = Node::Rank;
  static constexpr int NT   = Node::NumTerms;
  using tile_type           = typename Node::tile_type;
  using block_type =
      Impl::GcBlock<tile_type,
                    std::conditional_t<Blocked, typename Node::value_dims_seq,
                                       std::integer_sequence<int>>>;
  static constexpr bool blocked = Blocked;
  static constexpr int  NB      = block_type::size;  // entries per block
  static_assert(!Blocked || block_type::nv > 0,
                "general contraction: blocking needs value labels");

  KOKKOS_FUNCTION GeneralContractionEvaluator(
      const DeviceTuple<Leaves...>& leaves, const Out& out,
      const Kokkos::Array<int, Rank>& origin)
      : leaves_(leaves), out_(out), origin_(origin) {}

  // --- one entry per call --------------------------------------------------

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
  KOKKOS_FORCEINLINE_FUNCTION void store(const Coord&     coord,
                                         const value_type v) const {
    out_.put(coord, origin_, v);
  }

  // --- one value block per call --------------------------------------------

  /// The number of structural coordinates: the tile's extents with the value
  /// dims left out.
  static constexpr int structural_size = [] {
    int n = 1;
    for (int d = 0; d < Rank; ++d)
      if (block_type::value_pos(d) < 0) n *= tile_type::extent(d);
    return n;
  }();

  /// Structural coordinate `lin` (row-major over the structural dims, the last
  /// fastest) as a full tile coordinate with every value dim at 0.
  KOKKOS_FORCEINLINE_FUNCTION static Kokkos::Array<int, Rank> structural_coord(
      int lin) {
    return decode(lin,
                  std::make_index_sequence<static_cast<std::size_t>(Rank)>{});
  }

  template <typename Coord>
  KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<value_type, NB> compute_block(
      const Coord& coord0) const {
    Kokkos::Array<int, Rank> gidx{};
    TENSOR_PRAGMA_UNROLL
    for (int d = 0; d < Rank; ++d) gidx[d] = origin_[d] + coord0[d];
    Kokkos::Array<value_type, NB> acc{};
    terms_block(coord0, gidx, acc, std::make_index_sequence<NT>{});
    return acc;
  }

  template <typename Coord>
  KOKKOS_FORCEINLINE_FUNCTION void store_block(
      const Coord& coord0, const Kokkos::Array<value_type, NB>& acc) const {
    out_.template put_block<block_type>(coord0, origin_, acc,
                                        std::make_index_sequence<NB>{});
  }

 private:
  // --- the structural decode -------------------------------------------------

  // The outermost structural dim with more than one entry: the one left
  // holding the whole quotient, so it needs no division.
  static constexpr int outer_dim = [] {
    for (int d = 0; d < Rank; ++d)
      if (block_type::value_pos(d) < 0 && tile_type::extent(d) > 1) return d;
    return -1;
  }();

  template <int D>
  KOKKOS_FORCEINLINE_FUNCTION static int decode_dim(int& lin) {
    constexpr int e = tile_type::extent(D);
    if constexpr (block_type::value_pos(D) >= 0 || e == 1) {
      return 0;
    } else if constexpr (D == outer_dim) {
      return lin;
    } else {
      const int c = lin % e;
      lin /= e;
      return c;
    }
  }
  template <std::size_t... Ds>
  KOKKOS_FORCEINLINE_FUNCTION static Kokkos::Array<int, Rank> decode(
      int lin, std::index_sequence<Ds...>) {
    Kokkos::Array<int, Rank> c{};
    ((c[Rank - 1 - static_cast<int>(Ds)] =
          decode_dim<Rank - 1 - static_cast<int>(Ds)>(lin)),
     ...);
    return c;
  }

  // --- shared by both forms --------------------------------------------------

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
  // its address a term fixes before its loop. Two axes with one binding (a
  // diagonal read) simply add their strides.
  template <int K, int V, int S, typename Coord>
  KOKKOS_FORCEINLINE_FUNCTION static int base_term(const Coord& coord) {
    if constexpr (K == Impl::gc_bind_out)
      return S * coord[V];
    else if constexpr (K == Impl::gc_bind_const)
      return S * V;
    else
      return 0;
  }
  template <int K, int V, int S, typename LoopIdx>
  KOKKOS_FORCEINLINE_FUNCTION static int loop_term(const LoopIdx& li) {
    if constexpr (K == Impl::gc_bind_loop)
      return S * li[V];
    else
      return 0;
  }
  template <int K, int V, typename LoopIdx>
  KOKKOS_FORCEINLINE_FUNCTION static int arg_of(
      const Kokkos::Array<int, Rank>& gidx, const LoopIdx& li) {
    if constexpr (K == Impl::gc_bind_out)
      return gidx[V];
    else if constexpr (K == Impl::gc_bind_const)
      return V;
    else
      return li[V];
  }

  template <typename Layout, std::size_t... Ds>
  static constexpr auto strides_of(std::index_sequence<Ds...>) {
    return std::integer_sequence<int,
                                 Layout::stride(static_cast<int>(Ds))...>{};
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
    using leaf =
        tuple_element_t<static_cast<std::size_t>(J), DeviceTuple<Leaves...>>;
    using kinds = typename Info::template kinds_seq<typename leaf::labels>;
    using vals  = typename Info::template vals_seq<typename leaf::labels>;
  };
  template <typename Info, int J>
  struct SlotStrides {
    using Lay  = typename LeafPlan<Info, J>::leaf::layout_t;
    using type = decltype(strides_of<Lay>(
        std::make_index_sequence<static_cast<std::size_t>(Lay::rank)>{}));
  };

  template <typename Info, int J, typename Coord>
  KOKKOS_FORCEINLINE_FUNCTION int leaf_base(const Coord& coord) const {
    using P = LeafPlan<Info, J>;
    if constexpr (P::leaf::functional) {
      return 0;
    } else {
      return base_offset(coord, typename P::kinds{}, typename P::vals{},
                         typename SlotStrides<Info, J>::type{});
    }
  }

  template <typename FnNode, typename LoopIdx, int... Ks, int... Vs>
  KOKKOS_FORCEINLINE_FUNCTION static value_type call_fn(
      const FnNode& n, const Kokkos::Array<int, Rank>& gidx, const LoopIdx& li,
      std::integer_sequence<int, Ks...>, std::integer_sequence<int, Vs...>) {
    return n.fn_(arg_of<Ks, Vs>(gidx, li)...);
  }

  template <typename Info, int J, typename LoopIdx>
  KOKKOS_FORCEINLINE_FUNCTION value_type
  leaf_value(const Kokkos::Array<int, Rank>& gidx, const int base,
             const LoopIdx& li) const {
    using P          = LeafPlan<Info, J>;
    const auto& leaf = leaves_.template get<static_cast<std::size_t>(J)>();
    if constexpr (P::leaf::functional) {
      return call_fn(leaf.node, gidx, li, typename P::kinds{},
                     typename P::vals{});
    } else {
      return leaf.view
          .data()[base + loop_offset(li, typename P::kinds{},
                                     typename P::vals{},
                                     typename SlotStrides<Info, J>::type{})];
    }
  }

  template <typename Info, typename Coord, int... Js>
  KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<int, sizeof...(Js)> bases_of(
      const Coord& coord, std::integer_sequence<int, Js...>) const {
    return {leaf_base<Info, Js>(coord)...};
  }

  // A term's loop trip count: the product of its summed extents.
  template <int... Es>
  KOKKOS_FUNCTION static constexpr int ext_product(
      std::integer_sequence<int, Es...>) {
    return (1 * ... * Es);
  }

  // --- one entry -------------------------------------------------------------

  template <typename Coord, std::size_t... Ts>
  KOKKOS_FORCEINLINE_FUNCTION void terms(const Coord&                    coord,
                                         const Kokkos::Array<int, Rank>& gidx,
                                         value_type&                     acc,
                                         std::index_sequence<Ts...>) const {
    (term<static_cast<int>(Ts)>(coord, gidx, acc), ...);
  }

  template <typename Info, typename LoopIdx, int... Js, std::size_t... Us>
  KOKKOS_FORCEINLINE_FUNCTION value_type
  product(const Kokkos::Array<int, Rank>&          gidx,
          const Kokkos::Array<int, sizeof...(Js)>& bases, const LoopIdx& li,
          std::integer_sequence<int, Js...>, std::index_sequence<Us...>) const {
    return (value_type(1) * ... * leaf_value<Info, Js>(gidx, bases[Us], li));
  }

  template <int T, typename Coord>
  KOKKOS_FORCEINLINE_FUNCTION void term(const Coord&                    coord,
                                        const Kokkos::Array<int, Rank>& gidx,
                                        value_type& acc) const {
    using Info = typename Node::template term_info<T>;
    if constexpr (Info::dead) {
      return;
    } else {
      if (!checks(coord, typename Info::chk_a_seq{}, typename Info::chk_b_seq{},
                  typename Info::chk_c_seq{}))
        return;
      using Used           = typename Info::leaves_seq;
      constexpr int  NU    = static_cast<int>(Used::size());
      constexpr int  NL    = Info::num_loop;
      constexpr auto ext   = Impl::seq_to_karray(typename Info::loop_ext_seq{});
      constexpr int  total = ext_product(typename Info::loop_ext_seq{});
      const Kokkos::Array<int, NU> bases = bases_of<Info>(coord, Used{});
      TENSOR_PRAGMA_UNROLL
      for (int lin = 0; lin < total; ++lin) {
        Kokkos::Array<int, (NL > 0 ? NL : 1)> li{};
        int                                   rem = lin;
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

  // --- one value block -------------------------------------------------------

  template <typename Coord, std::size_t... Ts>
  KOKKOS_FORCEINLINE_FUNCTION void terms_block(
      const Coord& coord0, const Kokkos::Array<int, Rank>& gidx,
      Kokkos::Array<value_type, NB>& acc, std::index_sequence<Ts...>) const {
    (term_block<static_cast<int>(Ts)>(coord0, gidx, acc), ...);
  }

  // (1 * l_0) * ... * l_{P-1}: the leaves before the value leaf.
  template <typename Info, typename Bases, typename LoopIdx, std::size_t... Us>
  KOKKOS_FORCEINLINE_FUNCTION value_type
  prefix(const Kokkos::Array<int, Rank>& gidx, const Bases& bases,
         const LoopIdx& li, std::index_sequence<Us...>) const {
    return (value_type(1) * ... *
            leaf_value<Info, Impl::gc_seq_at<Us>(typename Info::leaves_seq{})>(
                gidx, bases[Us], li));
  }
  // l_{P+1}, ..., l_{NU-1}: the leaves after it, loaded once per loop step.
  template <typename Info, std::size_t P, typename Bases, typename LoopIdx,
            std::size_t... Ks>
  KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<value_type, sizeof...(Ks)> suffix(
      const Kokkos::Array<int, Rank>& gidx, const Bases& bases,
      const LoopIdx& li, std::index_sequence<Ks...>) const {
    return {leaf_value<Info, Impl::gc_seq_at<P + 1 + Ks>(
                                 typename Info::leaves_seq{})>(
        gidx, bases[P + 1 + Ks], li)...};
  }

  // A slot value leaf's offset for value index V: its value-bound axes'
  // strides times V's digits. Compile-time.
  template <int V, int... Ks, int... Vs, int... Ss>
  KOKKOS_FUNCTION static constexpr int value_offset(
      std::integer_sequence<int, Ks...>, std::integer_sequence<int, Vs...>,
      std::integer_sequence<int, Ss...>) {
    return (
        0 + ... +
        (Ks == Impl::gc_bind_out ? Ss * block_type::digit_of_dim(V, Vs) : 0));
  }
  template <typename Info, int J, int V>
  KOKKOS_FORCEINLINE_FUNCTION static value_type slot_value(
      const value_type* const vp) {
    using P           = LeafPlan<Info, J>;
    constexpr int off = value_offset<V>(typename P::kinds{}, typename P::vals{},
                                        typename SlotStrides<Info, J>::type{});
    return vp[off];
  }
  // A functional value leaf at value index V: the global coordinate with the
  // value dims advanced by V's digits.
  template <int V, std::size_t... Ks>
  KOKKOS_FORCEINLINE_FUNCTION static void add_digits(
      Kokkos::Array<int, Rank>& g, std::index_sequence<Ks...>) {
    ((g[Impl::gc_seq_at<Ks>(typename block_type::dims_seq{})] +=
      std::integral_constant<int, block_type::digit(
                                      V, static_cast<int>(Ks))>::value),
     ...);
  }
  template <typename Info, int J, int V, typename LoopIdx>
  KOKKOS_FORCEINLINE_FUNCTION value_type
  fn_value(const Kokkos::Array<int, Rank>& gidx, const LoopIdx& li) const {
    using P                    = LeafPlan<Info, J>;
    Kokkos::Array<int, Rank> g = gidx;
    add_digits<V>(
        g,
        std::make_index_sequence<static_cast<std::size_t>(block_type::nv)>{});
    return call_fn(leaves_.template get<static_cast<std::size_t>(J)>().node, g,
                   li, typename P::kinds{}, typename P::vals{});
  }

  template <typename Suf, std::size_t... Ks>
  KOKKOS_FORCEINLINE_FUNCTION static void accumulate(
      value_type& a, const value_type t, const Suf& suf,
      std::index_sequence<Ks...>) {
    a += (t * ... * suf[Ks]);
  }

  template <typename Info, int J, typename Suf, typename LoopIdx, typename S,
            std::size_t... Vs>
  KOKKOS_FORCEINLINE_FUNCTION void block_accumulate(
      Kokkos::Array<value_type, NB>& acc, const value_type pre, const Suf& suf,
      const Kokkos::Array<int, Rank>& gidx, const int vbase, const LoopIdx& li,
      std::index_sequence<Vs...>, S) const {
    using P = LeafPlan<Info, J>;
    if constexpr (P::leaf::functional) {
      (accumulate(acc[Vs],
                  pre * fn_value<Info, J, static_cast<int>(Vs)>(gidx, li), suf,
                  S{}),
       ...);
    } else {
      const value_type* const vp =
          leaves_.template get<static_cast<std::size_t>(J)>().view.data() +
          vbase +
          loop_offset(li, typename P::kinds{}, typename P::vals{},
                      typename SlotStrides<Info, J>::type{});
      (accumulate(acc[Vs], pre * slot_value<Info, J, static_cast<int>(Vs)>(vp),
                  suf, S{}),
       ...);
    }
  }

  template <int T, typename Coord>
  KOKKOS_FORCEINLINE_FUNCTION void term_block(
      const Coord& coord0, const Kokkos::Array<int, Rank>& gidx,
      Kokkos::Array<value_type, NB>& acc) const {
    using Info = typename Node::template term_info<T>;
    if constexpr (Info::dead) {
      return;
    } else {
      if (!checks(coord0, typename Info::chk_a_seq{},
                  typename Info::chk_b_seq{}, typename Info::chk_c_seq{}))
        return;
      using Used       = typename Info::leaves_seq;
      constexpr int NU = static_cast<int>(Used::size());
      constexpr int Pos =
          Impl::gc_value_leaf_pos_v<typename Node::structure_type,
                                    typename Node::label_tiles_type, T>;
      static_assert(Pos >= 0 && Pos < NU,
                    "general contraction: a live term of a blocked node has "
                    "no value leaf");
      constexpr std::size_t P  = static_cast<std::size_t>(Pos);
      constexpr int         JV = Impl::gc_seq_at<P>(Used{});
      constexpr int         NL = Info::num_loop;
      constexpr auto ext   = Impl::seq_to_karray(typename Info::loop_ext_seq{});
      constexpr int  total = ext_product(typename Info::loop_ext_seq{});
      // The value leaf's base is taken at coord0, whose value dims are 0: its
      // value part is added per entry, as a constant.
      const Kokkos::Array<int, NU> bases = bases_of<Info>(coord0, Used{});
      TENSOR_PRAGMA_UNROLL
      for (int lin = 0; lin < total; ++lin) {
        Kokkos::Array<int, (NL > 0 ? NL : 1)> li{};
        int                                   rem = lin;
        TENSOR_PRAGMA_UNROLL
        for (int j = NL - 1; j >= 0; --j) {
          li[j] = rem % ext[j];
          rem /= ext[j];
        }
        const value_type pre =
            prefix<Info>(gidx, bases, li, std::make_index_sequence<P>{});
        using SufSeq =
            std::make_index_sequence<static_cast<std::size_t>(NU) - P - 1>;
        const auto suf = suffix<Info, P>(gidx, bases, li, SufSeq{});
        block_accumulate<Info, JV>(acc, pre, suf, gidx, bases[P], li,
                                   std::make_index_sequence<NB>{}, SufSeq{});
      }
    }
  }

  DeviceTuple<Leaves...>   leaves_;
  Out                      out_;
  Kokkos::Array<int, Rank> origin_;
};

/// A member's evaluator: `out` is where its values go (GcScratchOut or a
/// streamed GcRootOut); Blocked iterates value blocks (the node's value
/// labels), else single entries.
template <typename Node, bool Blocked, typename Out, typename... Leaves>
KOKKOS_FUNCTION auto make_general_contraction_evaluator(
    const DeviceTuple<Leaves...>& leaves, const Out& out,
    const Kokkos::Array<int, Node::Rank>& origin) {
  return GeneralContractionEvaluator<Node, Out, Blocked, Leaves...>(leaves, out,
                                                                    origin);
}

}  // namespace TensorOperations
