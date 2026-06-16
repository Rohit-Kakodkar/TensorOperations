#pragma once
#include <TensorOperations/Concept.hpp>
#include <Kokkos_Core.hpp>
#include <array>
#include <cstdint>

namespace TensorOperations {

template <TensorLike T>
struct TensorHandle : T {
  static constexpr int Rank = static_cast<int>(T::rank);

  Kokkos::Array<int32_t, Rank> modes;

  explicit TensorHandle(T base, std::array<int32_t, Rank> modes_)
      : T(std::move(base)) {
    for (int i = 0; i < Rank; ++i) modes[i] = modes_[i];
  }
};

template <TensorLike T>
TensorHandle<T> make_handle(
    T base, std::array<int32_t, static_cast<int>(T::rank)> modes) {
  return TensorHandle<T>(std::move(base), modes);
}

}  // namespace TensorOperations
