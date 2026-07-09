#pragma once
/**
 * @file Permute.hpp
 * @brief Compile-time einsum-mode planning plus a zero-copy permuted tensor
 * view.
 *
 * A binary contraction is specified by einsum-style *labels* on each operand
 * and the output (e.g. `A[j,i,k] * B[j,k,l] -> C[l,i]`). The team GEMM kernel,
 * however, requires the *canonical* positional layout
 * @code
 *     A = [freeA.., contracted..]   B = [contracted.., freeB..]   C = [freeA..,
 * freeB..]
 * @endcode
 * with A's and B's contracted axes in the same order. This header derives, at
 * compile time from the labels, the per-operand gather permutations that map an
 * operand's user axis order into that canonical order, and provides a zero-copy
 * strided view (@ref TensorOperations::Impl::PermutedView) that presents an
 * operand's data already permuted.
 *
 * @par Gather convention
 * Matches `reorder_layout` / `reorder_view` in `TiledLayout.hpp`:
 * `Perm[i]` is the SOURCE (user) axis that becomes canonical axis `i`, i.e.
 * `canonical.extent(i) == user.extent(Perm[i])`.
 *
 * @par Canonical order
 * `freeA` in A's order, then `contracted` in A's order (the "Korder"); B's
 * contracted axes are mapped to that same Korder; the output's canonical order
 * is `freeA` (A order) followed by `freeB` (B order).
 *
 * @note Labels are carried as `std::integer_sequence` template parameters, so
 * the entire plan is resolved at compile time and leaves no runtime cost.
 */
#include <array>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

namespace TensorOperations {
namespace Impl {

// --- sequence <-> array helpers -------------------------------------------

/**
 * @brief Materialize an `integer_sequence` as a `std::array` (host constexpr).
 *
 * Works for any element type (`int` permutations, `int32_t` labels, ...).
 * Host-only: use @ref seq_to_karray for arrays consumed in device code.
 */
template <typename T, T... V>
constexpr std::array<T, sizeof...(V)> seq_to_array(
    std::integer_sequence<T, V...>) {
  return {V...};
}

/// Materialize an `integer_sequence` as a device-usable `Kokkos::Array`
/// (pack-expanded, so no host-only `constexpr` call is emitted into device
/// code).
template <typename T, T... V>
KOKKOS_FUNCTION constexpr Kokkos::Array<T, sizeof...(V)> seq_to_karray(
    std::integer_sequence<T, V...>) noexcept {
  return {V...};
}

/**
 * @brief Rebuild a constexpr `std::array` as an `integer_sequence`.
 *
 * The single converter behind every `*_seq_t` alias below: a compile-time plan
 * array (structural-type NTTP) becomes `integer_sequence<value_type, Arr[0],
 * Arr[1], ...>`, so plans computed as plain constexpr loops can be carried as
 * template parameters.
 */
template <auto Arr, std::size_t... I>
auto array_to_seq_impl(std::index_sequence<I...>)
    -> std::integer_sequence<typename decltype(Arr)::value_type, Arr[I]...>;

template <auto Arr>
using array_to_seq_t =
    decltype(array_to_seq_impl<Arr>(std::make_index_sequence<Arr.size()>{}));

/**
 * @brief Test whether a permutation sequence is the identity (`0,1,2,...`).
 * @return `true` iff `P[i] == i` for every position `i`.
 */
template <int... P>
constexpr bool is_identity_seq(std::integer_sequence<int, P...>) {
  int  i  = 0;
  bool id = true;
  ((id = id && (P == i++)), ...);
  return id;
}

/// @return `true` iff @p v occurs in @p a.
template <std::size_t N>
constexpr bool arr_contains(const std::array<int32_t, N>& a, int32_t v) {
  for (std::size_t i = 0; i < N; ++i)
    if (a[i] == v) return true;
  return false;
}

/// @return `true` iff the entries of @p a are pairwise distinct.
template <typename T, std::size_t N>
constexpr bool all_distinct(const std::array<T, N>& a) {
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = i + 1; j < N; ++j)
      if (a[i] == a[j]) return false;
  return true;
}

// --- contraction plan (all constexpr, from the operand label sequences) ----

/**
 * @brief Gather permutation reordering A's user axes to canonical order.
 *
 * Produces the permutation that maps A into `[freeA.., contracted..]`, with the
 * free axes and contracted axes each kept in A's own order (the latter defining
 * the Korder used by @ref compute_permB).
 *
 * @tparam AModesSeq Label sequence of operand A (user order).
 * @tparam BModesSeq Label sequence of operand B (used to identify contracted
 * labels).
 * @return `perm` of size `rank(A)` where `perm[i]` is the user axis of A that
 *         becomes canonical axis `i`.
 */
template <typename AModesSeq, typename BModesSeq>
constexpr auto compute_permA() {
  constexpr auto      a  = seq_to_array(AModesSeq{});
  constexpr auto      b  = seq_to_array(BModesSeq{});
  constexpr int       RA = static_cast<int>(a.size());
  std::array<int, RA> perm{};
  int                 pos = 0;
  for (int i = 0; i < RA; ++i)
    if (!arr_contains(b, a[i])) perm[pos++] = i;  // freeA (A order)
  for (int i = 0; i < RA; ++i)
    if (arr_contains(b, a[i]))
      perm[pos++] = i;  // contracted (A order = Korder)
  return perm;
}

/**
 * @brief Gather permutation reordering B's user axes to canonical order.
 *
 * Produces the permutation that maps B into `[contracted.., freeB..]`, where
 * the contracted axes are emitted in A's Korder (so A's and B's contracted axes
 * align positionally) and the free axes follow in B's own order.
 *
 * @tparam AModesSeq Label sequence of operand A (defines the Korder).
 * @tparam BModesSeq Label sequence of operand B (user order).
 * @return `perm` of size `rank(B)` where `perm[i]` is the user axis of B that
 *         becomes canonical axis `i`.
 */
template <typename AModesSeq, typename BModesSeq>
constexpr auto compute_permB() {
  constexpr auto      a  = seq_to_array(AModesSeq{});
  constexpr auto      b  = seq_to_array(BModesSeq{});
  constexpr int       RA = static_cast<int>(a.size());
  constexpr int       RB = static_cast<int>(b.size());
  std::array<int, RB> perm{};
  int                 pos = 0;
  for (int i = 0; i < RA; ++i)  // walk A to fix Korder
    if (arr_contains(b, a[i]))
      for (int j = 0; j < RB; ++j)
        if (b[j] == a[i]) {
          perm[pos++] = j;
          break;
        }
  for (int j = 0; j < RB; ++j)
    if (!arr_contains(a, b[j])) perm[pos++] = j;  // freeB (B order)
  return perm;
}

/**
 * @brief Count the contracted modes of the contraction, i.e. `|A ∩ B|`.
 */
template <typename AModesSeq, typename BModesSeq>
constexpr int num_contracted() {
  constexpr auto a = seq_to_array(AModesSeq{});
  constexpr auto b = seq_to_array(BModesSeq{});
  int            n = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (arr_contains(b, a[i])) ++n;
  return n;
}

/**
 * @brief Compute the canonical output *labels*, `freeA (A order) ++ freeB (B
 * order)`.
 * @tparam RC Output rank (must equal the total number of free modes).
 * @return An array of `RC` labels in canonical output order.
 */
template <int RC, typename AModesSeq, typename BModesSeq>
constexpr std::array<int32_t, RC> compute_canonC_modes() {
  constexpr auto          a = seq_to_array(AModesSeq{});
  constexpr auto          b = seq_to_array(BModesSeq{});
  std::array<int32_t, RC> c{};
  int                     pos = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (!arr_contains(b, a[i])) c[pos++] = a[i];  // freeA
  for (std::size_t j = 0; j < b.size(); ++j)
    if (!arr_contains(a, b[j])) c[pos++] = b[j];  // freeB
  return c;
}

/**
 * @brief Gather permutation from canonical output order to the user output
 * order.
 *
 * `permC[i]` is the position, in the *user* output order, of the canonical
 * output mode `canonC[i]`. It is used to build the canonical output tile/view
 * as a gather of the user-order output.
 *
 * @return `perm` of size `RC` mapping canonical output axis `i` to its user
 * axis.
 */
template <int RC, typename AModesSeq, typename BModesSeq, typename OutSeq>
constexpr std::array<int, RC> compute_permC() {
  constexpr auto      canonC = compute_canonC_modes<RC, AModesSeq, BModesSeq>();
  constexpr auto      out    = seq_to_array(OutSeq{});
  std::array<int, RC> perm{};
  for (int i = 0; i < RC; ++i)
    for (int j = 0; j < RC; ++j)
      if (out[j] == canonC[i]) {
        perm[i] = j;
        break;
      }
  return perm;
}

/**
 * @brief Validate that the label sets describe a well-formed contraction.
 *
 * Checks that the output labels are pairwise distinct, that every output label
 * appears in exactly one input, and that the output rank matches the number of
 * free modes derived from the inputs. (Distinctness *within* each operand is
 * enforced where the labels are attached, in `TensorHandle`.)
 *
 * @return `true` iff the contraction is consistent; suitable for
 * `static_assert`.
 */
template <int RC, typename AModesSeq, typename BModesSeq, typename OutSeq>
constexpr bool valid_contraction() {
  constexpr auto a   = seq_to_array(AModesSeq{});
  constexpr auto b   = seq_to_array(BModesSeq{});
  constexpr auto out = seq_to_array(OutSeq{});
  if (!all_distinct(out)) return false;
  for (int i = 0; i < RC; ++i) {
    const bool in_a = arr_contains(a, out[i]);
    const bool in_b = arr_contains(b, out[i]);
    if (in_a == in_b) return false;  // must be in exactly one input
  }
  constexpr int nc = num_contracted<AModesSeq, BModesSeq>();
  const int     free_count =
      (static_cast<int>(a.size()) - nc) + (static_cast<int>(b.size()) - nc);
  return free_count == RC;
}

// --- derived plans as plain constexpr arrays --------------------------------

/// The inverse of a gather permutation: `inverse[Perm[i]] == i`. Maps a source
/// (native) axis to its position in the reordered layout.
template <typename PermSeq>
constexpr auto inverse_perm_array() {
  constexpr auto            p = seq_to_array(PermSeq{});
  std::array<int, p.size()> inv{};
  for (int i = 0; i < static_cast<int>(p.size()); ++i) inv[p[i]] = i;
  return inv;
}

/// Labels of @p ModesSeq gathered by @p PermSeq: `out[i] = modes[perm[i]]`.
template <typename ModesSeq, typename PermSeq>
constexpr auto gather_modes_array() {
  constexpr auto m = seq_to_array(ModesSeq{});
  constexpr auto p = seq_to_array(PermSeq{});
  static_assert(m.size() == p.size(),
                "gather permutation rank must match label rank");
  std::array<int32_t, m.size()> out{};
  for (std::size_t i = 0; i < m.size(); ++i) out[i] = m[p[i]];
  return out;
}

/**
 * @name Memoized plan arrays
 * @brief One-shot compile-time cache for each `compute_*` result.
 *
 * Each plan array is stored in a `constexpr` *variable template*: its
 * initializer is evaluated exactly once per template-argument set, cutting the
 * amount of constant-folding the compiler must do when a plan is used several
 * times.
 * @{
 */
template <typename AModesSeq, typename BModesSeq>
inline constexpr auto permA_v = compute_permA<AModesSeq, BModesSeq>();

template <typename AModesSeq, typename BModesSeq>
inline constexpr auto permB_v = compute_permB<AModesSeq, BModesSeq>();

template <int RC, typename AModesSeq, typename BModesSeq, typename OutSeq>
inline constexpr auto permC_v =
    compute_permC<RC, AModesSeq, BModesSeq, OutSeq>();

template <int RC, typename AModesSeq, typename BModesSeq>
inline constexpr auto canonC_v =
    compute_canonC_modes<RC, AModesSeq, BModesSeq>();
/** @} */

/**
 * @name Resolved sequence aliases
 * @brief The plan arrays lifted to `integer_sequence` types via @ref
 * array_to_seq_t.
 * @{
 */
/// A's canonicalizing gather permutation as an `integer_sequence`.
template <typename AModesSeq, typename BModesSeq>
using permA_seq_t = array_to_seq_t<permA_v<AModesSeq, BModesSeq>>;
/// B's canonicalizing gather permutation as an `integer_sequence`.
template <typename AModesSeq, typename BModesSeq>
using permB_seq_t = array_to_seq_t<permB_v<AModesSeq, BModesSeq>>;
/// Canonical-to-user output gather permutation as an `integer_sequence`.
template <int RC, typename AModesSeq, typename BModesSeq, typename OutSeq>
using permC_seq_t = array_to_seq_t<permC_v<RC, AModesSeq, BModesSeq, OutSeq>>;
/// Canonical output labels as an `integer_sequence`.
template <int RC, typename AModesSeq, typename BModesSeq>
using canonC_modes_seq_t = array_to_seq_t<canonC_v<RC, AModesSeq, BModesSeq>>;
/// The inverse of a gather permutation as an `integer_sequence`. Used to turn a
/// canonical-order tile back into the operand's native axis order
/// (`native_tile = reorder(canon_tile, inverse)`).
template <typename PermSeq>
using inverse_perm_seq_t = array_to_seq_t<inverse_perm_array<PermSeq>()>;
/// Labels of @p ModesSeq gathered by @p PermSeq as an `integer_sequence`.
template <typename ModesSeq, typename PermSeq>
using gather_modes_seq_t =
    array_to_seq_t<gather_modes_array<ModesSeq, PermSeq>()>;
/** @} */

// --- device-side permutation application ------------------------------------

/// Canonical -> native index scatter: `native[Perm[i]] = canonical[i]`.
/// Pack-expanded so it fully unrolls and stays device-safe; the identity
/// permutation reduces to a copy.
template <int... P>
KOKKOS_FUNCTION constexpr Kokkos::Array<int, sizeof...(P)> scatter_index(
    const Kokkos::Array<int, sizeof...(P)>& c,
    std::integer_sequence<int, P...>) noexcept {
  Kokkos::Array<int, sizeof...(P)> n{};
  int                              i = 0;
  ((n[P] = c[i++]), ...);
  return n;
}

// --- PermutedView: a zero-copy, strided relabel of a tensor's axes ---------

/**
 * @brief Zero-copy, strided view that relabels a tensor's axes by a
 * permutation.
 *
 * Presents the wrapped view @ref v with its axes gathered by @p Perm, so that
 * `extent(d) == v.extent(Perm[d])` and `stride(d) == v.stride(Perm[d])`, while
 * sharing the same underlying data pointer. It reports `Kokkos::LayoutStride`,
 * which routes callers through the generic (runtime) `subview_tile` path. The
 * type satisfies the project's `TensorLike` interface.
 *
 * @tparam V The wrapped Kokkos-view-like tensor type.
 * @tparam Perm The gather permutation: source axis for each presented axis.
 */
template <typename V, int... Perm>
struct PermutedView {
  V v;  ///< The wrapped tensor whose axes are relabeled (data is shared, not
        ///< copied).

  using value_type      = typename V::value_type;
  using array_layout    = Kokkos::LayoutStride;
  using execution_space = typename V::execution_space;
  using memory_space    = typename V::memory_space;

  /// The wrapped native view type (before permutation). Consumers that want the
  /// ordered (register-resident) subview path tile @ref v directly and reorder
  /// the resulting subview by @ref perm_seq instead of tiling this strided
  /// view.
  using inner_view_t = V;
  /// The gather permutation as a sequence: presented axis d <- native axis
  /// Perm[d].
  using perm_seq = std::integer_sequence<int, Perm...>;

  static constexpr int rank = sizeof...(Perm);  ///< Rank of the permuted view.

  /// Extent along presented axis @p d: `v.extent(src(d))`.
  KOKKOS_FUNCTION int extent(int d) const noexcept {
    return static_cast<int>(v.extent(src(d)));
  }
  /// Stride along presented axis @p d: `v.stride(src(d))`.
  KOKKOS_FUNCTION std::ptrdiff_t stride(int d) const noexcept {
    return static_cast<std::ptrdiff_t>(v.stride(src(d)));
  }
  /// @return Pointer to the shared underlying data (identical to `v.data()`).
  KOKKOS_FUNCTION value_type* data() const noexcept { return v.data(); }

  /**
   * @brief Element access in the permuted axis order.
   *
   * Applies the wrapped view's strides through the permutation so that
   * `(*this)(i0, i1, ...)` addresses `v` at the corresponding permuted axes.
   */
  template <typename... Idx>
    requires(sizeof...(Idx) == rank)
  KOKKOS_FUNCTION value_type& operator()(Idx... idx) const {
    const int      c[rank] = {static_cast<int>(idx)...};
    std::ptrdiff_t off     = 0;
    for (int d = 0; d < rank; ++d)
      off += static_cast<std::ptrdiff_t>(c[d]) * v.stride(src(d));
    return v.data()[off];
  }

 private:
  /// Source (native) axis of presented axis @p d.
  KOKKOS_FUNCTION static constexpr int src(int d) noexcept {
    constexpr int p[rank] = {Perm...};
    return p[d];
  }
};

/// Trait: detect a @ref PermutedView (and recover its native view +
/// permutation).
template <typename>
struct is_permuted_view : std::false_type {};
template <typename V, int... Perm>
struct is_permuted_view<PermutedView<V, Perm...>> : std::true_type {};
template <typename T>
inline constexpr bool is_permuted_view_v =
    is_permuted_view<std::decay_t<T>>::value;

/**
 * @brief Return @p v relabeled by @p perm, avoiding a wrapper when possible.
 *
 * If @p perm is the identity the original view is returned unchanged —
 * preserving its native layout and the fast compile-time `subview_tile` path.
 * Any other permutation yields a zero-copy @ref PermutedView.
 */
template <typename V, int... Perm>
KOKKOS_FUNCTION auto permuted_alias(const V&                            v,
                                    std::integer_sequence<int, Perm...> perm) {
  if constexpr (is_identity_seq(perm))
    return v;
  else
    return PermutedView<V, Perm...>{v};
}

}  // namespace Impl
}  // namespace TensorOperations
