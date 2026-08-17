// Tests for Impl::matrix_layout_of_t / matrix_view -- the two-mode collapse a
// contraction indexes as (row, col).
//
// THE INVARIANT, and everything here is a way of checking it: for a source
// layout of rank R split at S,
//
//     matrix(r, c)  aliases  source(i_0, ..., i_{R-1})
//
// where r enumerates axes [0, S) and c enumerates axes [S, R), each with the
// LAST axis varying fastest. That is reshape()'s collapse convention, and it is
// the one the GEMM already assumes -- so a wrong leaf order here would not fail
// to compile, it would silently transpose an operand.
//
// The interesting cases are the ones reshape() cannot express: a permuted view
// whose two axis groups are interleaved in memory.

#include <TensorOperations/TileLayout.hpp>
#include <TensorOperations/TiledLayout.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <vector>

using namespace TensorOperations;

namespace {

// A minimal backing: a flat float buffer with the interface View wants.
struct Backing {
  using value_type = float;
  float*                 p_;
  KOKKOS_FUNCTION float* data() const noexcept { return p_; }
};

template <typename Layout>
using V = View<Backing, Layout>;

// --- the invariant, checked exhaustively -----------------------------------
//
// Walk every (r, c) of the collapsed matrix, decompose them the way the
// convention says, and require the matrix's offset to equal the source's offset
// at that multi-index. Exhaustive rather than sampled: these tiles are tiny and
// an off-by-one in one leaf is exactly the bug worth catching.
template <int Split, typename Layout, int R>
void check_alias(const char* what) {
  using MatL = Impl::regroup_layout_t<Layout, Impl::split_groups_t<Split, R>>;
  static_assert(MatL::rank == 2, "the collapse must produce a rank-2 layout");

  Layout src{};
  MatL   mat{};

  const int nrow = mat.extent(0), ncol = mat.extent(1);

  // extents of the source axes, and the products the convention implies
  int e[R];
  for (int d = 0; d < R; ++d) e[d] = src.extent(d);

  int prow = 1, pcol = 1;
  for (int d = 0; d < Split; ++d) prow *= e[d];
  for (int d = Split; d < R; ++d) pcol *= e[d];
  ASSERT_EQ(nrow, prow) << what << ": row mode size";
  ASSERT_EQ(ncol, pcol) << what << ": col mode size";

  for (int r = 0; r < nrow; ++r)
    for (int c = 0; c < ncol; ++c) {
      // Decompose r over [0, Split) and c over [Split, R), LAST AXIS FASTEST.
      Impl::Index<R> coord{};
      int            rr = r;
      for (int d = Split - 1; d >= 0; --d) {
        coord[d] = rr % e[d];
        rr /= e[d];
      }
      int cc = c;
      for (int d = R - 1; d >= Split; --d) {
        coord[d] = cc % e[d];
        cc /= e[d];
      }

      const int want = src.flat_offset(coord);
      const int got  = mat.flat_offset(Impl::Index<2>{r, c});
      ASSERT_EQ(got, want) << what << ": r=" << r << " c=" << c;
    }
}

// --- the Order component, which only operator[] reads ----------------------
//
// flat_offset never touches Order, so check_alias above cannot see it at all.
// Order is the leaf MEMORY order, derived from the COALESCED strides rather
// than the source axes -- a second way to get the regroup wrong.
//
// operator[] enumerates coordinates in the layout's own memory order, so on a
// DENSE layout (offset set exactly [0, N)) the round trip is the identity:
// visiting offsets in ascending order over all of [0, N) recovers linear.
// Every source used here is a permutation of a packed tile, hence dense.
template <int Split, typename Layout, int R>
void check_roundtrip(const char* what) {
  using MatL = Impl::regroup_layout_t<Layout, Impl::split_groups_t<Split, R>>;
  MatL mat{};

  const int nrow = mat.extent(0), ncol = mat.extent(1);
  const int n = nrow * ncol;

  for (int lin = 0; lin < n; ++lin) {
    const Impl::Index<2> coord = mat[lin];

    ASSERT_GE(coord[0], 0) << what << ": lin=" << lin;
    ASSERT_LT(coord[0], nrow) << what << ": lin=" << lin;
    ASSERT_GE(coord[1], 0) << what << ": lin=" << lin;
    ASSERT_LT(coord[1], ncol) << what << ": lin=" << lin;

    ASSERT_EQ(mat.flat_offset(coord), lin)
        << what << ": lin=" << lin << " decoded to (" << coord[0] << ","
        << coord[1] << ")";
  }
}

}  // namespace

// --- contiguous sources: must agree with what reshape already produces ------

TEST(RegroupView, ContiguousRank2) {
  check_alias<1, StaticTileLayoutRight<4, 8>, 2>("right 4x8 split 1");
}

TEST(RegroupView, ContiguousRank4SplitOne) {
  check_alias<1, StaticTileLayoutRight<2, 3, 4, 5>, 4>("right rank4 split 1");
}

TEST(RegroupView, ContiguousRank4SplitTwo) {
  check_alias<2, StaticTileLayoutRight<2, 3, 4, 5>, 4>("right rank4 split 2");
}

TEST(RegroupView, ContiguousRank4SplitThree) {
  check_alias<3, StaticTileLayoutRight<2, 3, 4, 5>, 4>("right rank4 split 3");
}

TEST(RegroupView, LeftLayoutRank3) {
  check_alias<1, StaticTileLayoutLeft<3, 4, 5>, 3>("left rank3 split 1");
}

// --- the cases reshape() CANNOT express -------------------------------------
//
// A reordered tile: the axis groups are interleaved in memory, so the generic
// planner reports NotFactorable. These are the SEM gradient level's operands.

// u[e][d0][d1][d2] reordered so the contracted axis leads.
using U4 = StaticTileLayoutRight<4, 5, 5, 5>;  // (TE, N, N, N)

// contract d0 -> [d0][e, d1, d2], perm <1,0,2,3>
using Perm1 = std::integer_sequence<int, 1, 0, 2, 3>;
// contract d1 -> [d1][e, d0, d2], perm <2,0,1,3>
using Perm2 = std::integer_sequence<int, 2, 0, 1, 3>;
// contract d2 -> [d2][e, d0, d1], perm <3,0,1,2>
using Perm3 = std::integer_sequence<int, 3, 0, 1, 2>;

template <typename Perm>
using Reordered = decltype(reorder_tile(U4{}, Perm{}));

TEST(RegroupView, PermutedContractLeadingAxis) {
  check_alias<1, Reordered<Perm1>, 4>("u perm <1,0,2,3> split 1");
}

TEST(RegroupView, PermutedContractMiddleAxis) {
  // The one the reshape planner rejects: col = (e, d0, d2), whose leaves are
  // interrupted by d1 in memory.
  check_alias<1, Reordered<Perm2>, 4>("u perm <2,0,1,3> split 1");
}

TEST(RegroupView, PermutedContractTrailingAxis) {
  check_alias<1, Reordered<Perm3>, 4>("u perm <3,0,1,2> split 1");
}

// --- the point of the whole exercise ----------------------------------------
//
// Three matrices over ONE buffer, each contracting a different axis. If the
// leaf plans were wrong in a way that still compiled, two of these would alias
// the same elements.
TEST(RegroupView, ThreePermutationsOverOneBufferDiffer) {
  using M1 =
      Impl::regroup_layout_t<Reordered<Perm1>, Impl::split_groups_t<1, 4>>;
  using M2 =
      Impl::regroup_layout_t<Reordered<Perm2>, Impl::split_groups_t<1, 4>>;
  using M3 =
      Impl::regroup_layout_t<Reordered<Perm3>, Impl::split_groups_t<1, 4>>;

  M1 m1{};
  M2 m2{};
  M3 m3{};

  ASSERT_EQ(m1.extent(0), 5);
  ASSERT_EQ(m1.extent(1), 4 * 5 * 5);

  // Collect each matrix's offset map and require the three to be distinct
  // permutations of the same 500 offsets.
  std::vector<int> o1, o2, o3;
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 100; ++c) {
      o1.push_back(m1.flat_offset(Impl::Index<2>{r, c}));
      o2.push_back(m2.flat_offset(Impl::Index<2>{r, c}));
      o3.push_back(m3.flat_offset(Impl::Index<2>{r, c}));
    }

  // Each is a bijection onto [0, 500).
  for (auto* o : {&o1, &o2, &o3}) {
    std::vector<int> seen(500, 0);
    for (int v : *o) {
      ASSERT_GE(v, 0);
      ASSERT_LT(v, 500);
      ++seen[static_cast<std::size_t>(v)];
    }
    for (int s : seen) ASSERT_EQ(s, 1) << "offset map must be a bijection";
  }

  // And they are genuinely different maps.
  EXPECT_NE(o1, o2);
  EXPECT_NE(o2, o3);
  EXPECT_NE(o1, o3);
}

// --- Order round-trips, on the same sources as the alias tests --------------

TEST(RegroupOrder, ContiguousRank2RoundTrips) {
  check_roundtrip<1, StaticTileLayoutRight<4, 8>, 2>("right 4x8 split 1");
}

TEST(RegroupOrder, ContiguousRank4RoundTrips) {
  check_roundtrip<1, StaticTileLayoutRight<2, 3, 4, 5>, 4>("rank4 split 1");
  check_roundtrip<2, StaticTileLayoutRight<2, 3, 4, 5>, 4>("rank4 split 2");
  check_roundtrip<3, StaticTileLayoutRight<2, 3, 4, 5>, 4>("rank4 split 3");
}

TEST(RegroupOrder, LeftLayoutRoundTrips) {
  check_roundtrip<1, StaticTileLayoutLeft<3, 4, 5>, 3>("left rank3 split 1");
}

TEST(RegroupOrder, PermutedSourcesRoundTrip) {
  check_roundtrip<1, Reordered<Perm1>, 4>("u perm <1,0,2,3> split 1");
  check_roundtrip<1, Reordered<Perm2>, 4>("u perm <2,0,1,3> split 1");
  check_roundtrip<1, Reordered<Perm3>, 4>("u perm <3,0,1,2> split 1");
}

// The power-of-two shape reshape would have accepted while silently grouping
// the wrong axis as the row -- checklist item (3). Every gcd divides here, so
// nothing upstream fails loudly; only the offset map can tell the difference.
using U4Pow2 = StaticTileLayoutRight<4, 8, 8, 8>;

template <typename Perm>
using ReorderedPow2 = decltype(reorder_tile(U4Pow2{}, Perm{}));

TEST(RegroupView, PowerOfTwoExtentsAliasCorrectly) {
  check_alias<1, ReorderedPow2<Perm1>, 4>("pow2 perm <1,0,2,3> split 1");
  check_alias<1, ReorderedPow2<Perm2>, 4>("pow2 perm <2,0,1,3> split 1");
  check_alias<1, ReorderedPow2<Perm3>, 4>("pow2 perm <3,0,1,2> split 1");
}

TEST(RegroupOrder, PowerOfTwoExtentsRoundTrip) {
  check_roundtrip<1, ReorderedPow2<Perm1>, 4>("pow2 perm <1,0,2,3> split 1");
  check_roundtrip<1, ReorderedPow2<Perm2>, 4>("pow2 perm <2,0,1,3> split 1");
  check_roundtrip<1, ReorderedPow2<Perm3>, 4>("pow2 perm <3,0,1,2> split 1");
}

// --- zero-copy: the view keeps the same backing -----------------------------
TEST(RegroupView, ViewIsZeroCopy) {
  std::vector<float> buf(500);
  for (int i = 0; i < 500; ++i)
    buf[static_cast<std::size_t>(i)] = static_cast<float>(i);

  V<Reordered<Perm2>> v{Backing{buf.data()}, {}};
  auto                m = regroup_view(v, Split<1, 4>{});

  EXPECT_EQ(m.data(), v.data()) << "matrix_view must not copy";

  // Spot-check that reading through the matrix lands on the element the source
  // layout says it should.
  Reordered<Perm2> src{};
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 100; ++c) {
      const int a = c / 25, b = (c / 5) % 5, d = c % 5;
      const int want = src.flat_offset(Impl::Index<4>{r, a, b, d});
      EXPECT_FLOAT_EQ(m(r, c), static_cast<float>(want))
          << "r=" << r << " c=" << c;
    }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
