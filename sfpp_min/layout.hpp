#pragma once

#include <config.hpp>

#include <Kokkos_Core.hpp>

#include <cstddef>

namespace sfpp_min {

struct ChunkTiledOffset {
  static constexpr std::size_t kPointsPerElement = NGLL * NGLL * NGLL;
  static constexpr std::size_t kTileSpan =
      static_cast<std::size_t>(kStorageChunk) * kPointsPerElement;

  KOKKOS_INLINE_FUNCTION static std::size_t at(int ispec, int iz, int iy,
                                               int ix) {
    const std::size_t tile =
        static_cast<std::size_t>(ispec / kStorageChunk) * kTileSpan;
    const std::size_t within =
        static_cast<std::size_t>(ispec % kStorageChunk) +
        static_cast<std::size_t>(iz) * kStorageChunk +
        static_cast<std::size_t>(iy) * kStorageChunk * NGLL +
        static_cast<std::size_t>(ix) * kStorageChunk * NGLL * NGLL;
    return tile + within;
  }

  KOKKOS_INLINE_FUNCTION static std::size_t span(int nspec) {
    const std::size_t tiles =
        static_cast<std::size_t>((nspec + kStorageChunk - 1) / kStorageChunk);
    return tiles * kTileSpan;
  }
};

struct PlainOffset {
  static constexpr std::size_t kPointsPerElement = NGLL * NGLL * NGLL;

  KOKKOS_INLINE_FUNCTION static std::size_t at(int ispec, int iz, int iy,
                                               int ix) {
    return ((static_cast<std::size_t>(ispec) * NGLL + iz) * NGLL + iy) * NGLL +
           ix;
  }

  KOKKOS_INLINE_FUNCTION static std::size_t span(int nspec) {
    return static_cast<std::size_t>(nspec) * kPointsPerElement;
  }
};

}  // namespace sfpp_min
