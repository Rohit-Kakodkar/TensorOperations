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

  ChunkTiledOffset() = default;
  KOKKOS_INLINE_FUNCTION explicit ChunkTiledOffset(int nspec) : nspec_(nspec) {}

  KOKKOS_INLINE_FUNCTION std::size_t operator()(int ispec, int iz, int iy,
                                                int ix) const {
    return at(ispec, iz, iy, ix);
  }

  KOKKOS_INLINE_FUNCTION std::size_t span() const { return span(nspec_); }

 private:
  int nspec_ = 0;
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

  PlainOffset() = default;
  KOKKOS_INLINE_FUNCTION explicit PlainOffset(int nspec) : nspec_(nspec) {}

  KOKKOS_INLINE_FUNCTION std::size_t operator()(int ispec, int iz, int iy,
                                                int ix) const {
    return at(ispec, iz, iy, ix);
  }

  KOKKOS_INLINE_FUNCTION std::size_t span() const { return span(nspec_); }

 private:
  int nspec_ = 0;
};

// Transcribed from SPECFEMPP
// core/specfem/datatype/domain_view.hpp:43-61,:144-168. The extents are runtime
// values (Kokkos::dextents) and the stride product is REBUILT on every access
// instead of being cached in a stride table. That is the address math the ncu
// ground truth measured, so collapsing it into precomputed strides -- the
// obvious optimisation -- silently deletes the thing under study and makes the
// dummy kernel unrepresentatively fast. Do not "fix" this.
struct ChunkTiledDynamicOffset {
  static constexpr std::size_t kRank = 4;

  ChunkTiledDynamicOffset() = default;

  KOKKOS_INLINE_FUNCTION explicit ChunkTiledDynamicOffset(int nspec)
      : nspec_(nspec), nz_(NGLL), ny_(NGLL), nx_(NGLL) {
    chunk_span_ = static_cast<std::size_t>(kStorageChunk) * nz_ * ny_ * nx_;
  }

  KOKKOS_INLINE_FUNCTION std::size_t extent(std::size_t dim) const {
    return dim == 1 ? nz_ : (dim == 2 ? ny_ : nx_);
  }

  KOKKOS_INLINE_FUNCTION std::size_t tile_size(std::size_t dim) const {
    if (dim == 0) return static_cast<std::size_t>(kStorageChunk);
    return extent(dim);
  }

  KOKKOS_INLINE_FUNCTION std::size_t fwd_prod_of_tile_size(
      std::size_t idim) const {
    std::size_t prod = 1;
    for (std::size_t i = 0; i < idim; ++i) prod *= tile_size(i);
    return prod;
  }

  KOKKOS_INLINE_FUNCTION std::size_t operator()(int ispec, int iz, int iy,
                                                int ix) const {
    const std::size_t index_array[kRank] = {
        static_cast<std::size_t>(ispec), static_cast<std::size_t>(iz),
        static_cast<std::size_t>(iy), static_cast<std::size_t>(ix)};

    std::size_t offset = index_array[0] % kStorageChunk;
    for (std::size_t i = 1; i < kRank; ++i) {
      offset += index_array[i] * fwd_prod_of_tile_size(i);
    }

    return (index_array[0] / kStorageChunk) * chunk_span_ + offset;
  }

  KOKKOS_INLINE_FUNCTION std::size_t span() const {
    const std::size_t tiles =
        static_cast<std::size_t>((nspec_ + kStorageChunk - 1) / kStorageChunk);
    return tiles * chunk_span_;
  }

 private:
  int         nspec_      = 0;
  std::size_t nz_         = 0;
  std::size_t ny_         = 0;
  std::size_t nx_         = 0;
  std::size_t chunk_span_ = 0;
};

}  // namespace sfpp_min
