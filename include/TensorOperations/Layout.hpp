#pragma once
#include <array>

namespace TensorOperations {

namespace Impl {

// constexpr std::array filled with a single value (no <algorithm> on device).
template <int Rank>
constexpr std::array<int, Rank> filled_array(int v) {
  std::array<int, Rank> a{};
  for (int i = 0; i < Rank; ++i) a[i] = v;
  return a;
}

}  // namespace Impl

// ---------------------------------------------------------------------------
// StridedLayout — the slot -> global-address rule for register-tier staging.
//
// A staging evaluator maps each compile-time register slot's local multi-index
// `local[d]` to a global element of the input tensor as:
//
//     global[d] = tile_offset[d] + local[d] * stride[d]
//
// This decouples the *block shape* (the Tiling) from the *access pattern* (this
// layout). The strides are runtime, which is residency-safe: they only enter
// the runtime global read index, never the compile-time register write index.
//
//   stride[d] == 1            -> contiguous sub-block load (default).
//   stride[d] == #threads(d)  -> staggered / coalesced load: across a thread
//                                group, each staging step reads consecutive
//                                global addresses.
//
// Thread identity is folded into `tile_offset` by the launcher, so the layout
// only needs to carry the per-mode strides.
// ---------------------------------------------------------------------------
template <int Rank>
struct StridedLayout {
  static constexpr int  rank   = Rank;
  std::array<int, Rank> stride = Impl::filled_array<Rank>(1);

  constexpr int operator[](int d) const { return stride[d]; }
};

}  // namespace TensorOperations
