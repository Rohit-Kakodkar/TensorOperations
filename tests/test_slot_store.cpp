// ===========================================================================
// test_slot_store.cpp — one team-scratch buffer per node, carved in one pass.
//
// Stage 4 of the DAG evaluator. The driver carves every node's output up front
// so a result can outlive the evaluator that produced it and other nodes can
// name it. The property that matters is DISJOINTNESS: an offset bug that
// overlapped two slots would not crash, it would silently feed one node another
// node's data, and every downstream number would be wrong with nothing to point
// at.
//
// So it is checked two independent ways:
//   1. ADDRESSES  — the used regions [data, data + size) are pairwise disjoint.
//   2. CONTENTS   — every slot is filled with its own distinct pattern, and all
//                   patterns survive. This catches an overlap that address
//                   arithmetic alone might miss (a wrong element size, say),
//                   and it is the property a caller actually depends on.
//
// The store is carved as ONE bump allocation with compile-time per-slot
// offsets, so a third property is now the STORE's to keep rather than Kokkos's:
//   3. ALIGNMENT  — every slot base sits on the scratch alignment. Kokkos used
//                   to guarantee this per allocation; a packed arena does not
//                   get it for free.
//
// Tiles are deliberately heterogeneous in both rank and extent: equal-sized
// slots would let a stride bug cancel out and still look disjoint. T1 is there
// specifically to break the alignment: 25 floats is 100 bytes, which is NOT a
// multiple of the 8-byte scratch alignment, so every slot after it lands wrong
// if slot_arena_step's round-up goes missing. The other four tiles are all
// already 8-aligned and cannot catch that on their own.
// ===========================================================================
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/SlotStore.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

using namespace TensorOperations;
using ES     = Kokkos::DefaultExecutionSpace;
using team_t = typename Kokkos::TeamPolicy<ES>::member_type;

namespace {

// Five nodes' worth of output tiles, no two the same shape and not all the same
// size: 32, 25, 64, 32 and 96 floats. T1's 25 is odd on purpose -- see above.
using T0 = StaticTile<4, 8>;
using T1 = StaticTile<5, 5>;
using T2 = StaticTile<8, 8>;
using T3 = StaticTile<2, 16>;
using T4 = StaticTile<3, 4, 8>;

constexpr std::size_t kSlots = 5;

using Store = decltype(carve_arena_slot_store<float, ES>(
    std::declval<team_t>(), std::declval<T0>(), std::declval<T1>(),
    std::declval<T2>(), std::declval<T3>(), std::declval<T4>()));

// Distinct per slot AND per element, so a partial overlap is as visible as a
// total one.
KOKKOS_INLINE_FUNCTION float pattern(std::size_t k, int i) {
  return 1000.0f * static_cast<float>(k + 1) + static_cast<float>(i);
}

template <std::size_t K, typename Team>
KOKKOS_FUNCTION void fill_slot(const Team& team, const Store& s) {
  const auto v = s.template get<K>();
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, v.size()),
                       [=](int i) { v.data()[i] = pattern(K, i); });
}

template <typename Team, std::size_t... Ks>
KOKKOS_FUNCTION void fill_all(const Team& team, const Store& s,
                              std::index_sequence<Ks...>) {
  (fill_slot<Ks>(team, s), ...);
}

// Record each slot's base address and used byte count, and verify its contents
// survived every other slot's write. Plain function templates rather than
// lambdas with `if constexpr` inside: nvcc is unreliable with the latter.
template <std::size_t K, typename PtrV, typename LenV, typename OkV>
KOKKOS_FUNCTION void inspect_slot(const Store& s, PtrV ptrs, LenV lens,
                                  OkV ok) {
  const auto v = s.template get<K>();
  ptrs(K)      = reinterpret_cast<std::uintptr_t>(v.data());
  lens(K)      = static_cast<std::size_t>(v.size()) * sizeof(float);
  for (int i = 0; i < v.size(); ++i)
    if (v.data()[i] != pattern(K, i)) ok() = 0;
}

template <typename PtrV, typename LenV, typename OkV, std::size_t... Ks>
KOKKOS_FUNCTION void inspect_all(const Store& s, PtrV ptrs, LenV lens, OkV ok,
                                 std::index_sequence<Ks...>) {
  (inspect_slot<Ks>(s, ptrs, lens, ok), ...);
}

struct Region {
  std::uintptr_t begin;
  std::size_t    len;
};

// Two stores carved back to back in one kernel; returns the four base
// addresses. Also a free function, for the same nvcc reason as below.
std::vector<std::uintptr_t> carve_twice() {
  const std::size_t one   = arena_slot_store_bytes<float, ES>(T0{}, T1{});
  const std::size_t bytes = 2 * one;

  Kokkos::View<std::uintptr_t*, ES> ptrs("ptrs", 4);
  Kokkos::parallel_for(
      "carve_twice",
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(static_cast<int>(bytes))),
      KOKKOS_LAMBDA(const team_t& team) {
        const auto a = carve_arena_slot_store<float, ES>(team, T0{}, T1{});
        const auto b = carve_arena_slot_store<float, ES>(team, T0{}, T1{});
        Kokkos::single(Kokkos::PerTeam(team), [&] {
          ptrs(0) =
              reinterpret_cast<std::uintptr_t>(a.template get<0>().data());
          ptrs(1) =
              reinterpret_cast<std::uintptr_t>(a.template get<1>().data());
          ptrs(2) =
              reinterpret_cast<std::uintptr_t>(b.template get<0>().data());
          ptrs(3) =
              reinterpret_cast<std::uintptr_t>(b.template get<1>().data());
        });
      });
  Kokkos::fence();

  auto h = Kokkos::create_mirror_view(ptrs);
  Kokkos::deep_copy(h, ptrs);
  return {h(0), h(1), h(2), h(3)};
}

// Kernels live in free functions: nvcc rejects an extended lambda whose
// enclosing function has private access within its class, and gtest's
// TestBody() is private.
std::vector<Region> carve_and_probe(std::size_t bytes, int& contents_ok) {
  Kokkos::View<std::uintptr_t*, ES> ptrs("ptrs", kSlots);
  Kokkos::View<std::size_t*, ES>    lens("lens", kSlots);
  Kokkos::View<int, ES>             ok("ok");
  Kokkos::deep_copy(ok, 1);

  Kokkos::parallel_for(
      "carve_arena_slot_store",
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(static_cast<int>(bytes))),
      KOKKOS_LAMBDA(const team_t& team) {
        const Store s = carve_arena_slot_store<float, ES>(team, T0{}, T1{},
                                                          T2{}, T3{}, T4{});
        fill_all(team, s, std::make_index_sequence<kSlots>{});
        team.team_barrier();  // every slot written before any is read back
        Kokkos::single(Kokkos::PerTeam(team), [&] {
          inspect_all(s, ptrs, lens, ok, std::make_index_sequence<kSlots>{});
        });
      });
  Kokkos::fence();

  auto hp = Kokkos::create_mirror_view(ptrs);
  auto hl = Kokkos::create_mirror_view(lens);
  auto ho = Kokkos::create_mirror_view(ok);
  Kokkos::deep_copy(hp, ptrs);
  Kokkos::deep_copy(hl, lens);
  Kokkos::deep_copy(ho, ok);

  contents_ok = ho();
  std::vector<Region> r;
  for (std::size_t k = 0; k < kSlots; ++k) r.push_back({hp(k), hl(k)});
  return r;
}

}  // namespace

// Sizing is answerable host-side before anything is carved -- the role
// scratch_size_per_team plays for a tree -- and it is the packed total plus ONE
// allocation's alignment slack, not a sum of per-slot allocations.
//
// Spelling the expected number out by hand rather than calling the same prefix
// sum the implementation uses: this test exists to pin the arithmetic, and a
// tautology would pin nothing.
TEST(SlotStoreTest, BytesIsThePaddedSumPlusOneAlignment) {
  constexpr std::size_t kAlign = Impl::slot_arena_align<float, ES>();
  // 32 + 26 + 64 + 32 + 96 elements: only T1's 25 needs a pad, to 26.
  constexpr std::size_t kElems = 32 + 26 + 64 + 32 + 96;
  const std::size_t     total =
      arena_slot_store_bytes<float, ES>(T0{}, T1{}, T2{}, T3{}, T4{});

  EXPECT_EQ(total, kElems * sizeof(float) + kAlign);

  // And it is TIGHTER than one allocation per slot would have been, because
  // Kokkos charges its alignment slack on every allocation and the arena pays
  // it once. This is the whole reason the arena's byte figure differs.
  const std::size_t per_slot = Impl::scratch_tile_bytes<float, ES>(T0{}) +
                               Impl::scratch_tile_bytes<float, ES>(T1{}) +
                               Impl::scratch_tile_bytes<float, ES>(T2{}) +
                               Impl::scratch_tile_bytes<float, ES>(T3{}) +
                               Impl::scratch_tile_bytes<float, ES>(T4{});
  EXPECT_LT(total, per_slot);

  // Fits both backends' caps with room to spare -- the constraint that killed
  // the previous attempt at fan-out dedup was scratch, not correctness.
  EXPECT_LT(total, 32u * 1024u);
}

// A slot's type is a function of (value type, exec space, tile) ALONE. This is
// what lets every node in a graph be typed from its own tile, with no recursion
// into whatever produced it -- and hence what keeps the stage 5 driver a flat
// list rather than a left fold.
TEST(SlotStoreTest, SlotTypeDependsOnlyOnTheTile) {
  static_assert(
      std::is_same_v<SlotView<float, ES, T0>,
                     decltype(Impl::alloc_scratch_tile<float, ES>(
                         std::declval<team_t>(), std::declval<T0>()))>);

  // And it is the SAME type a producing evaluator hands over, so a slot can be
  // adopted by its producer (stage 3) and named by its consumers (stage 1)
  // without any conversion. Reached here through the allocator, i.e. by a
  // different route than the alias takes.
  using ProducerOut =
      typename ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, IntermTag,
                                float, T0>::scratch_view_t;
  static_assert(std::is_same_v<SlotView<float, ES, T0>, ProducerOut>);

  // Distinct tiles give distinct slot types; equal tiles give equal ones.
  static_assert(
      !std::is_same_v<SlotView<float, ES, T0>, SlotView<float, ES, T1>>);
  static_assert(std::is_same_v<SlotView<float, ES, T0>,
                               SlotView<float, ES, StaticTile<4, 8>>>);

  static_assert(Store::size == kSlots);
}

// The invariant, both ways.
TEST(SlotStoreTest, SlotsAreDisjoint) {
  const std::size_t bytes =
      arena_slot_store_bytes<float, ES>(T0{}, T1{}, T2{}, T3{}, T4{});
  int  contents_ok = 0;
  auto regions     = carve_and_probe(bytes, contents_ok);

  ASSERT_EQ(regions.size(), kSlots);

  // (2) CONTENTS: every slot still holds its own pattern after all five were
  // written. An overlap corrupts whichever was written first.
  EXPECT_EQ(contents_ok, 1)
      << "a slot's contents did not survive the other slots' writes";

  // (1) ADDRESSES: the used regions are pairwise disjoint. Sort by base, then
  // check each region ends at or before the next one begins.
  for (const auto& r : regions) {
    EXPECT_NE(r.begin, 0u) << "a slot was never allocated";
    EXPECT_GT(r.len, 0u);
  }
  std::sort(regions.begin(), regions.end(),
            [](const Region& a, const Region& b) { return a.begin < b.begin; });
  for (std::size_t k = 0; k + 1 < regions.size(); ++k)
    EXPECT_LE(regions[k].begin + regions[k].len, regions[k + 1].begin)
        << "slots at " << regions[k].begin << " and " << regions[k + 1].begin
        << " overlap";

  // The whole store fits inside what arena_slot_store_bytes asked for. Span
  // rather than exact sum: the per-slot round-up leaves legitimate gaps -- what
  // must not happen is running past the end. This is the host/device sizing
  // agreement, and under-requesting it faults on GPU only, because Serial's
  // 32 KB absorbs the overrun.
  const std::uintptr_t span_end = regions.back().begin + regions.back().len;
  EXPECT_LE(span_end - regions.front().begin, bytes);
}

// Every slot base sits on the scratch alignment.
//
// This used to be Kokkos's to keep: get_shmem_aligned re-aligned the cursor on
// every call, so a store of one-allocation-per-slot could not get it wrong. A
// packed arena hands out its own offsets and must therefore round each slot up
// itself. T1 is the tile that catches a missing round-up -- 25 floats is 100
// bytes, so slots 2, 3 and 4 all land 4 bytes low without it.
TEST(SlotStoreTest, EverySlotBaseIsScratchAligned) {
  constexpr std::size_t kAlign = Impl::slot_arena_align<float, ES>();
  const std::size_t     bytes =
      arena_slot_store_bytes<float, ES>(T0{}, T1{}, T2{}, T3{}, T4{});
  int        contents_ok = 0;
  const auto regions     = carve_and_probe(bytes, contents_ok);

  ASSERT_EQ(regions.size(), kSlots);
  for (std::size_t k = 0; k < regions.size(); ++k)
    EXPECT_EQ(regions[k].begin % kAlign, 0u)
        << "slot " << k << " starts at " << regions[k].begin
        << ", which is not a multiple of " << kAlign;
}

// Carving twice in one kernel must not hand back the same memory: the arena is
// still a bump allocation, so a second store has to sit after the first. A
// driver that carved slots and then let an evaluator carve its own staging
// buffers depends on exactly this -- and it is what proves the single carve
// advances the team cursor by the whole arena rather than by one slot.
TEST(SlotStoreTest, SuccessiveCarvesDoNotAlias) {
  const auto p = carve_twice();
  ASSERT_EQ(p.size(), 4u);
  for (std::size_t i = 0; i < p.size(); ++i)
    for (std::size_t j = i + 1; j < p.size(); ++j)
      EXPECT_NE(p[i], p[j]) << "slot " << i << " and " << j << " alias";
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
