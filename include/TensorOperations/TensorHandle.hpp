#pragma once
#include <TensorOperations/Concept.hpp>
#include <TensorOperations/Permute.hpp>
#include <Kokkos_Core.hpp>
#include <array>
#include <cstdint>
#include <utility>

namespace TensorOperations {

// A tensor plus its einsum-style mode labels, carried in the type as a
// compile-time std::integer_sequence<int32_t, ...> so contraction permutations
// can be derived at compile time (zero storage per handle).
template <TensorLike T, typename ModesSeq>
struct TensorHandle : T {
  static constexpr int Rank = static_cast<int>(T::rank);
  static_assert(static_cast<int>(ModesSeq::size()) == Rank,
                "TensorHandle: number of mode labels must equal tensor rank");
  static_assert(Impl::all_distinct(Impl::seq_to_array(ModesSeq{})),
                "TensorHandle: mode labels must be pairwise distinct");

  using modes_seq = ModesSeq;

  KOKKOS_FUNCTION explicit TensorHandle(T base) : T(std::move(base)) {}
};

// make_handle<'i','j','k'>(view): mode labels are template arguments.
template <int32_t... Modes, TensorLike T>
auto make_handle(T base) {
  return TensorHandle<T, std::integer_sequence<int32_t, Modes...>>(
      std::move(base));
}

// Build a handle from a computed label sequence (used when canonicalizing an
// operand into the GEMM's canonical axis order).
template <typename T, int32_t... Modes>
KOKKOS_FUNCTION auto make_handle_seq(T base,
                                     std::integer_sequence<int32_t, Modes...>) {
  return TensorHandle<T, std::integer_sequence<int32_t, Modes...>>(
      std::move(base));
}

}  // namespace TensorOperations
