#include <config.hpp>
#include <data.hpp>
#include <layout.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

using namespace sfpp_min;

namespace {

template <typename Offset>
std::vector<std::size_t> all_offsets(int nspec) {
  std::vector<std::size_t> v;
  v.reserve(static_cast<std::size_t>(nspec) * NGLL * NGLL * NGLL);
  for (int ispec = 0; ispec < nspec; ++ispec)
    for (int iz = 0; iz < NGLL; ++iz)
      for (int iy = 0; iy < NGLL; ++iy)
        for (int ix = 0; ix < NGLL; ++ix)
          v.push_back(Offset::at(ispec, iz, iy, ix));
  return v;
}

template <typename Offset>
void expect_bijection(int nspec) {
  const auto        v    = all_offsets<Offset>(nspec);
  const auto        span = Offset::span(nspec);
  std::vector<char> seen(span, 0);
  for (std::size_t o : v) {
    ASSERT_LT(o, span) << "offset escapes the allocation, nspec=" << nspec;
    ASSERT_EQ(seen[o], 0) << "offset " << o << " aliases, nspec=" << nspec;
    seen[o] = 1;
  }
}

template <typename Offset>
void expect_bijection_instance(int nspec) {
  const Offset      off(nspec);
  const std::size_t span = off.span();
  std::vector<char> seen(span, 0);
  for (int ispec = 0; ispec < nspec; ++ispec)
    for (int iz = 0; iz < NGLL; ++iz)
      for (int iy = 0; iy < NGLL; ++iy)
        for (int ix = 0; ix < NGLL; ++ix) {
          const std::size_t o = off(ispec, iz, iy, ix);
          ASSERT_LT(o, span) << "escapes the allocation, nspec=" << nspec;
          ASSERT_EQ(seen[o], 0)
              << "offset " << o << " aliases, nspec=" << nspec;
          seen[o] = 1;
        }
}

std::size_t constructive_offset(int ispec, int iz, int iy, int ix) {
  const int   tile = ispec / kStorageChunk;
  std::size_t o =
      static_cast<std::size_t>(tile) * kStorageChunk * NGLL * NGLL * NGLL;
  for (int px = 0; px < NGLL; ++px)
    for (int py = 0; py < NGLL; ++py)
      for (int pz = 0; pz < NGLL; ++pz)
        for (int e = 0; e < kStorageChunk; ++e) {
          if (px == ix && py == iy && pz == iz && e == ispec % kStorageChunk)
            return o;
          ++o;
        }
  return o;
}

double round_trip_probe(int nspec) {
  DomainArray<ChunkTiledOffset> a("probe", nspec);
  for (int ispec = 0; ispec < nspec; ++ispec)
    for (int iz = 0; iz < NGLL; ++iz)
      for (int iy = 0; iy < NGLL; ++iy)
        for (int ix = 0; ix < NGLL; ++ix)
          a.host(ispec, iz, iy, ix) =
              static_cast<real_t>(ispec * 1000 + iz * 100 + iy * 10 + ix);
  a.to_device();
  auto                  acc = a.accessor();
  Kokkos::View<double*> bad("bad", 1);
  Kokkos::parallel_for(
      "layout_round_trip", Kokkos::RangePolicy<>(0, nspec),
      KOKKOS_LAMBDA(const int ispec) {
        for (int iz = 0; iz < NGLL; ++iz)
          for (int iy = 0; iy < NGLL; ++iy)
            for (int ix = 0; ix < NGLL; ++ix) {
              const double want =
                  static_cast<double>(ispec * 1000 + iz * 100 + iy * 10 + ix);
              const double got = acc(ispec, iz, iy, ix);
              Kokkos::atomic_max(&bad(0), Kokkos::abs(got - want));
            }
      });
  Kokkos::fence();
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bad);
  return h(0);
}

double round_trip_probe_dynamic(int nspec) {
  DomainArray<ChunkTiledDynamicOffset> a("probe_dyn", nspec);
  for (int ispec = 0; ispec < nspec; ++ispec)
    for (int iz = 0; iz < NGLL; ++iz)
      for (int iy = 0; iy < NGLL; ++iy)
        for (int ix = 0; ix < NGLL; ++ix)
          a.host(ispec, iz, iy, ix) =
              static_cast<real_t>(ispec * 1000 + iz * 100 + iy * 10 + ix);
  a.to_device();
  auto                  acc = a.accessor();
  Kokkos::View<double*> bad("bad_dyn", 1);
  Kokkos::parallel_for(
      "layout_round_trip_dynamic", Kokkos::RangePolicy<>(0, nspec),
      KOKKOS_LAMBDA(const int ispec) {
        for (int iz = 0; iz < NGLL; ++iz)
          for (int iy = 0; iy < NGLL; ++iy)
            for (int ix = 0; ix < NGLL; ++ix) {
              const double want =
                  static_cast<double>(ispec * 1000 + iz * 100 + iy * 10 + ix);
              const double got = acc(ispec, iz, iy, ix);
              Kokkos::atomic_max(&bad(0), Kokkos::abs(got - want));
            }
      });
  Kokkos::fence();
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bad);
  return h(0);
}

template <typename Offset>
void expect_aos_injective(int nspec, int narrays) {
  const Offset      off(nspec);
  const std::size_t span = off.span() * static_cast<std::size_t>(narrays);
  std::vector<char> seen(span, 0);
  for (int ispec = 0; ispec < nspec; ++ispec)
    for (int iz = 0; iz < NGLL; ++iz)
      for (int iy = 0; iy < NGLL; ++iy)
        for (int ix = 0; ix < NGLL; ++ix)
          for (int c = 0; c < narrays; ++c) {
            const std::size_t o =
                off(ispec, iz, iy, ix) * static_cast<std::size_t>(narrays) + c;
            ASSERT_LT(o, span) << "escapes the allocation, nspec=" << nspec;
            ASSERT_EQ(seen[o], 0) << "interleaved index " << o << " aliases";
            seen[o] = 1;
          }
}

double round_trip_probe_aos(int nspec) {
  constexpr int                kArrays = MetricsAoS<ChunkTiledOffset>::kArrays;
  MetricsAoS<ChunkTiledOffset> m(nspec);
  const auto encode = [](int ispec, int iz, int iy, int ix, int c) {
    return static_cast<real_t>(ispec * 100000 +
                               (iz * 100 + iy * 10 + ix) * 100 + c);
  };
  typename MetricsAoS<ChunkTiledOffset>::component_type comp[kArrays] = {
      m.xix,  m.xiy,    m.xiz,    m.etax,   m.etay,
      m.etaz, m.gammax, m.gammay, m.gammaz, m.jacobian};
  for (int ispec = 0; ispec < nspec; ++ispec)
    for (int iz = 0; iz < NGLL; ++iz)
      for (int iy = 0; iy < NGLL; ++iy)
        for (int ix = 0; ix < NGLL; ++ix)
          for (int c = 0; c < kArrays; ++c)
            comp[c].host(ispec, iz, iy, ix) = encode(ispec, iz, iy, ix, c);
  m.to_device();

  const ChunkTiledOffset off(nspec);
  auto                   buf = m.device_view();
  Kokkos::View<double*>  bad("bad_aos", 1);
  Kokkos::parallel_for(
      "aos_round_trip", Kokkos::RangePolicy<>(0, nspec),
      KOKKOS_LAMBDA(const int ispec) {
        for (int iz = 0; iz < NGLL; ++iz)
          for (int iy = 0; iy < NGLL; ++iy)
            for (int ix = 0; ix < NGLL; ++ix) {
              const std::size_t base = off(ispec, iz, iy, ix) * kArrays;
              for (int c = 0; c < kArrays; ++c) {
                const double want = static_cast<double>(
                    ispec * 100000 + (iz * 100 + iy * 10 + ix) * 100 + c);
                const double got = buf(base + c);
                Kokkos::atomic_max(&bad(0), Kokkos::abs(got - want));
              }
            }
      });
  Kokkos::fence();
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bad);
  return h(0);
}

}  // namespace

TEST(SfppMinLayout, ChunkTiledIsBijective) {
  for (int nspec : {1, 31, 32, 33, 64, 70})
    expect_bijection<ChunkTiledOffset>(nspec);
}

TEST(SfppMinLayout, PlainIsBijective) {
  for (int nspec : {1, 33, 64}) expect_bijection<PlainOffset>(nspec);
}

TEST(SfppMinLayout, LayoutLeftIsBijective) {
  for (int nspec : {1, 4, 33, 64})
    expect_bijection_instance<LayoutLeftOffset>(nspec);
}

TEST(SfppMinLayout, LayoutLeftDynamicIsBijective) {
  for (int nspec : {1, 4, 33, 64})
    expect_bijection_instance<LayoutLeftDynamicOffset>(nspec);
}

TEST(SfppMinLayout, LayoutRightDynamicIsBijective) {
  for (int nspec : {1, 33, 64})
    expect_bijection_instance<LayoutRightDynamicOffset>(nspec);
}

TEST(SfppMinLayout, LayoutRightDynamicMatchesPlainBitForBit) {
  for (int nspec : {1, 4, 33, 64, 100}) {
    const LayoutRightDynamicOffset dyn(nspec);
    EXPECT_EQ(dyn.span(), PlainOffset::span(nspec)) << "nspec=" << nspec;
    for (int ispec = 0; ispec < nspec; ++ispec)
      for (int iz = 0; iz < NGLL; ++iz)
        for (int iy = 0; iy < NGLL; ++iy)
          for (int ix = 0; ix < NGLL; ++ix)
            ASSERT_EQ(dyn(ispec, iz, iy, ix),
                      PlainOffset::at(ispec, iz, iy, ix))
                << "nspec=" << nspec << " ispec=" << ispec << " iz=" << iz
                << " iy=" << iy << " ix=" << ix;
  }
}

TEST(SfppMinLayout, LayoutLeftDynamicMatchesStaticBitForBit) {
  for (int nspec : {1, 4, 33, 64, 100}) {
    const LayoutLeftOffset        stat(nspec);
    const LayoutLeftDynamicOffset dyn(nspec);
    EXPECT_EQ(dyn.span(), stat.span()) << "nspec=" << nspec;
    for (int ispec = 0; ispec < nspec; ++ispec)
      for (int iz = 0; iz < NGLL; ++iz)
        for (int iy = 0; iy < NGLL; ++iy)
          for (int ix = 0; ix < NGLL; ++ix)
            ASSERT_EQ(dyn(ispec, iz, iy, ix), stat(ispec, iz, iy, ix))
                << "nspec=" << nspec << " ispec=" << ispec << " iz=" << iz
                << " iy=" << iy << " ix=" << ix;
  }
}

TEST(SfppMinLayout, LayoutLeftHasElementStrideOne) {
  const int              nspec = 64;
  const LayoutLeftOffset off(nspec);
  for (int iz = 0; iz < NGLL; ++iz)
    for (int iy = 0; iy < NGLL; ++iy)
      for (int ix = 0; ix < NGLL; ++ix)
        for (int e = 0; e + 1 < nspec; ++e)
          ASSERT_EQ(off(e + 1, iz, iy, ix) - off(e, iz, iy, ix), 1u)
              << "e=" << e;
}

TEST(SfppMinLayout, SpanRoundsUpToWholeTiles) {
  const std::size_t pts = NGLL * NGLL * NGLL;
  EXPECT_EQ(ChunkTiledOffset::span(1), 1u * kStorageChunk * pts);
  EXPECT_EQ(ChunkTiledOffset::span(32), 1u * kStorageChunk * pts);
  EXPECT_EQ(ChunkTiledOffset::span(33), 2u * kStorageChunk * pts);
  EXPECT_EQ(ChunkTiledOffset::span(64), 2u * kStorageChunk * pts);
  EXPECT_EQ(ChunkTiledOffset::span(25920), 810u * kStorageChunk * pts);
  EXPECT_EQ(PlainOffset::span(33), 33u * pts);
}

TEST(SfppMinLayout, PartialChunkDoesNotCollideWithTheNext) {
  const int nspec = 33;
  for (int iz = 0; iz < NGLL; ++iz)
    for (int iy = 0; iy < NGLL; ++iy)
      for (int ix = 0; ix < NGLL; ++ix) {
        const auto last_of_tile0 =
            ChunkTiledOffset::at(kStorageChunk - 1, iz, iy, ix);
        const auto first_of_tile1 =
            ChunkTiledOffset::at(kStorageChunk, iz, iy, ix);
        EXPECT_LT(last_of_tile0, ChunkTiledOffset::kTileSpan);
        EXPECT_GE(first_of_tile1, ChunkTiledOffset::kTileSpan);
      }
  EXPECT_GT(ChunkTiledOffset::span(nspec),
            2u * 0 + ChunkTiledOffset::kTileSpan);
}

TEST(SfppMinLayout, ElementIsTheFastestIndex) {
  for (int iz = 0; iz < NGLL; ++iz)
    for (int iy = 0; iy < NGLL; ++iy)
      for (int ix = 0; ix < NGLL; ++ix)
        for (int e = 0; e + 1 < kStorageChunk; ++e) {
          const auto a = ChunkTiledOffset::at(e, iz, iy, ix);
          const auto b = ChunkTiledOffset::at(e + 1, iz, iy, ix);
          ASSERT_EQ(b - a, 1u) << "e=" << e;
        }
}

TEST(SfppMinLayout, MatchesAnIndependentlyConstructedOrdering) {
  const int nspec = 33;
  for (int ispec = 0; ispec < nspec; ++ispec)
    for (int iz = 0; iz < NGLL; ++iz)
      for (int iy = 0; iy < NGLL; ++iy)
        for (int ix = 0; ix < NGLL; ++ix)
          ASSERT_EQ(ChunkTiledOffset::at(ispec, iz, iy, ix),
                    constructive_offset(ispec, iz, iy, ix))
              << "ispec=" << ispec << " iz=" << iz << " iy=" << iy
              << " ix=" << ix;
}

TEST(SfppMinLayout, DomainArrayRoundTripsThroughDevice) {
  EXPECT_EQ(round_trip_probe(33), 0.0);
}

TEST(SfppMinLayout, DynamicExtentOffsetMatchesStaticBitForBit) {
  for (int nspec : {1, 4, 31, 32, 33, 64, 100}) {
    const ChunkTiledDynamicOffset dyn(nspec);
    EXPECT_EQ(dyn.span(), ChunkTiledOffset::span(nspec)) << "nspec=" << nspec;
    for (int ispec = 0; ispec < nspec; ++ispec)
      for (int iz = 0; iz < NGLL; ++iz)
        for (int iy = 0; iy < NGLL; ++iy)
          for (int ix = 0; ix < NGLL; ++ix)
            ASSERT_EQ(dyn(ispec, iz, iy, ix),
                      ChunkTiledOffset::at(ispec, iz, iy, ix))
                << "nspec=" << nspec << " ispec=" << ispec << " iz=" << iz
                << " iy=" << iy << " ix=" << ix;
  }
}

TEST(SfppMinLayout, DynamicExtentOffsetIsABijection) {
  for (int nspec : {1, 33, 64}) {
    const ChunkTiledDynamicOffset dyn(nspec);
    const std::size_t             span = dyn.span();
    std::vector<char>             seen(span, 0);
    for (int ispec = 0; ispec < nspec; ++ispec)
      for (int iz = 0; iz < NGLL; ++iz)
        for (int iy = 0; iy < NGLL; ++iy)
          for (int ix = 0; ix < NGLL; ++ix) {
            const std::size_t o = dyn(ispec, iz, iy, ix);
            ASSERT_LT(o, span) << "escapes the allocation, nspec=" << nspec;
            ASSERT_EQ(seen[o], 0) << "offset " << o << " aliases";
            seen[o] = 1;
          }
  }
}

TEST(SfppMinLayout, DynamicExtentDomainArrayRoundTripsThroughDevice) {
  EXPECT_EQ(round_trip_probe_dynamic(33), 0.0);
}

TEST(SfppMinLayout, AoSInterleaveIsInjective) {
  for (int nspec : {1, 31, 32, 33, 64}) {
    expect_aos_injective<ChunkTiledOffset>(nspec, 10);
    expect_aos_injective<ChunkTiledDynamicOffset>(nspec, 3);
  }
}

TEST(SfppMinLayout, AoSContainerRoundTripsThroughDevice) {
  EXPECT_EQ(round_trip_probe_aos(33), 0.0);
}

TEST(SfppMinLayout, ContainersAllocateTheExpectedFootprint) {
  const int                    nspec = 64;
  Metrics<ChunkTiledOffset>    m(nspec);
  Properties<ChunkTiledOffset> p(nspec);
  IglobMap                     g(nspec);

  const std::size_t per_array = ChunkTiledOffset::span(nspec) * sizeof(real_t);
  EXPECT_EQ(m.bytes(), 10u * per_array);
  EXPECT_EQ(p.bytes(), 3u * per_array);
  EXPECT_EQ(g.bytes(),
            static_cast<std::size_t>(nspec) * NGLL * NGLL * NGLL * sizeof(int));

  const std::size_t perf_domain =
      13u * ChunkTiledOffset::span(25920) * sizeof(real_t);
  EXPECT_EQ(perf_domain, 13u * 25920u * 125u * sizeof(real_t));
  EXPECT_NEAR(perf_domain / (1024.0 * 1024.0), 160.7, 0.5);
}

TEST(SfppMinLayout, TilePaddingIsZeroInitialised) {
  const int                     nspec = 33;
  DomainArray<ChunkTiledOffset> a("pad", nspec);
  const auto&                   h = a.host_view();
  for (std::size_t o = 0; o < a.span(); ++o) ASSERT_EQ(h(o), real_t(0)) << o;
  EXPECT_GT(a.span(), static_cast<std::size_t>(nspec) * NGLL * NGLL * NGLL);
}

TEST(SfppMinLayout, FieldsAreIglobFastest) {
  Fields f(1000);
  EXPECT_EQ(f.nglob(), 1000);
  EXPECT_EQ(f.displacement.stride(0), 1u);
  EXPECT_EQ(f.displacement.stride(1), 1000u);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
