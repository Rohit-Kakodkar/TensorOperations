#pragma once
#include <TensorOperations/Concept.hpp>
#include <array>
#include <cstdint>

namespace TensorOperations {

template <TensorLike T>
struct TensorHandle : T {
    static constexpr int Rank = T::rank();

    std::array<int32_t, Rank> modes;

    explicit TensorHandle(T base, std::array<int32_t, Rank> modes_)
        : T(std::move(base)), modes(modes_) {}
};

template <TensorLike T>
TensorHandle<T> make_handle(T base, std::array<int32_t, T::rank()> modes) {
    return TensorHandle<T>(std::move(base), modes);
}

} // namespace TensorOperations
