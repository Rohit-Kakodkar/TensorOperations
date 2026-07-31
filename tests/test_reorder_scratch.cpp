#include <Kokkos_Core.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/TiledLayout.hpp>
#include <gtest/gtest.h>

#include <vector>

using namespace TensorOperations;

// ---------------------------------------------------------------------------
// Impl::reorder_scratch_in_place — physically permute a scratch tile in place
// via a sequence of parallel axis-pair transpositions.
//
// Contract (gather convention, matching reorder_view): perm[i] is the source
// (native) axis that becomes canonical axis i, so after the reorder, reading
// the fixed physical layout at canonical coord c yields the value originally at
// native coord o with o[perm[i]] = c[i].
//
// These tests run the helper through a real TeamPolicy kernel (so the
// TeamVectorRange passes and team barriers execute) over a host buffer laid out
// under a StaticTileLayoutRight. Every case uses a cube-shaped tile so each
// transposed axis pair has equal extent (the helper's precondition).
// ---------------------------------------------------------------------------

using ExecSpace = Kokkos::DefaultHostExecutionSpace;
using Buf1D =
    Kokkos::View<float*, Kokkos::LayoutRight, ExecSpace::memory_space>;
using team_member_t = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

// Run reorder_scratch_in_place(view, Perm...) on a single team.
template <typename View, int... Perm>
void run_reorder(const View& view, std::integer_sequence<int, Perm...> perm) {
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ExecSpace>(1, Kokkos::AUTO),
      KOKKOS_LAMBDA(const team_member_t& team) {
        Impl::reorder_scratch_in_place(team, view, perm);
      });
  Kokkos::fence();
}

// ---------------------------------------------------------------------------
// transposition_plan — compile-time decomposition sanity checks.
// ---------------------------------------------------------------------------

TEST(ReorderScratch, TranspositionPlanIdentityIsEmpty) {
  constexpr auto plan =
      Impl::transposition_plan(std::integer_sequence<int, 0, 1, 2>{});
  static_assert(plan.count == 0, "identity needs no transpositions");
  SUCCEED();
}

TEST(ReorderScratch, TranspositionPlan2CycleIsOneSwap) {
  constexpr auto plan =
      Impl::transposition_plan(std::integer_sequence<int, 1, 0>{});
  static_assert(plan.count == 1);
  static_assert(plan.pairs[0][0] == 0 && plan.pairs[0][1] == 1);
  SUCCEED();
}

// ---------------------------------------------------------------------------
// Rank-2 transpose (one transposition), cube tile 4x4.
// ---------------------------------------------------------------------------

TEST(ReorderScratch, Transpose2D) {
  constexpr int N = 4;
  using Layout    = StaticTileLayoutRight<N, N>;

  Buf1D buf("buf", N * N);
  auto  h = Kokkos::create_mirror_view(buf);
  for (int k = 0; k < N * N; ++k) h(k) = static_cast<float>(k);
  Kokkos::deep_copy(buf, h);

  // Reference: out[c] = in[o], o[perm[i]] = c[i]; perm = {1,0}.
  std::vector<float> ref(N * N);
  Layout             layout{};
  for (int s = 0; s < N * N; ++s) {
    const auto     c = layout[s];  // canonical coord of flat s
    Impl::Index<2> o{c[1], c[0]};
    ref[s] = static_cast<float>(layout.flat_offset(o));
  }

  View<Buf1D, Layout> view{buf, {}};
  run_reorder(view, std::integer_sequence<int, 1, 0>{});

  Kokkos::deep_copy(h, buf);
  for (int s = 0; s < N * N; ++s)
    EXPECT_FLOAT_EQ(h(s), ref[s]) << "flat s=" << s;
}

// ---------------------------------------------------------------------------
// Rank-3 3-cycle perm {2,0,1} (decomposes into two transpositions), cube 3x3x3.
// ---------------------------------------------------------------------------

TEST(ReorderScratch, Permute3D_3Cycle) {
  constexpr int N     = 3;
  using Layout        = StaticTileLayoutRight<N, N, N>;
  constexpr int Total = N * N * N;

  Buf1D buf("buf", Total);
  auto  h = Kokkos::create_mirror_view(buf);
  for (int k = 0; k < Total; ++k) h(k) = static_cast<float>(k);
  Kokkos::deep_copy(buf, h);

  // perm = {2,0,1}: canonical c reads native o with o[perm[i]] = c[i].
  constexpr int      perm[] = {2, 0, 1};
  std::vector<float> ref(Total);
  Layout             layout{};
  for (int s = 0; s < Total; ++s) {
    const auto     c = layout[s];
    Impl::Index<3> o{0, 0, 0};
    for (int i = 0; i < 3; ++i) o[perm[i]] = c[i];
    ref[s] = static_cast<float>(layout.flat_offset(o));
  }

  View<Buf1D, Layout> view{buf, {}};
  run_reorder(view, std::integer_sequence<int, 2, 0, 1>{});

  Kokkos::deep_copy(h, buf);
  for (int s = 0; s < Total; ++s)
    EXPECT_FLOAT_EQ(h(s), ref[s]) << "flat s=" << s;
}

// ---------------------------------------------------------------------------
// Rank-3 single-pair swap {0,2,1} with a fixed point (axis 0 untouched).
// ---------------------------------------------------------------------------

TEST(ReorderScratch, Permute3D_SwapWithFixedPoint) {
  constexpr int N     = 3;
  using Layout        = StaticTileLayoutRight<N, N, N>;
  constexpr int Total = N * N * N;

  Buf1D buf("buf", Total);
  auto  h = Kokkos::create_mirror_view(buf);
  for (int k = 0; k < Total; ++k) h(k) = static_cast<float>(k);
  Kokkos::deep_copy(buf, h);

  constexpr int      perm[] = {0, 2, 1};
  std::vector<float> ref(Total);
  Layout             layout{};
  for (int s = 0; s < Total; ++s) {
    const auto     c = layout[s];
    Impl::Index<3> o{0, 0, 0};
    for (int i = 0; i < 3; ++i) o[perm[i]] = c[i];
    ref[s] = static_cast<float>(layout.flat_offset(o));
  }

  View<Buf1D, Layout> view{buf, {}};
  run_reorder(view, std::integer_sequence<int, 0, 2, 1>{});

  Kokkos::deep_copy(h, buf);
  for (int s = 0; s < Total; ++s)
    EXPECT_FLOAT_EQ(h(s), ref[s]) << "flat s=" << s;
}

// ---------------------------------------------------------------------------
// Evaluator<TeamPolicyTag, IntermTag(scratch view), plain perm_seq> — the
// scratch-view relabel specialization (mirrors Specialization 7's global-view
// relabel, but for ScratchView via reorder_tile). Unlike
// Impl::reorder_scratch_in_place above, this is a true zero-copy retype: the
// raw backing bytes are untouched, only the layout type changes.
// ---------------------------------------------------------------------------

// The kernel launches live in free functions, not directly in the TEST bodies:
// nvcc rejects an extended (KOKKOS_LAMBDA) lambda whose enclosing function has
// private access within its class, and gtest's TestBody() is private.
void run_relabel_transpose(Buf1D src_readback, Buf1D dst_readback) {
  using Tile              = StaticTile<4, 8>;
  using team_policy       = Kokkos::TeamPolicy<ExecSpace>;
  const std::size_t bytes = Impl::scratch_tile_bytes<float, ExecSpace>(Tile{});

  Kokkos::parallel_for(
      team_policy(1, Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const team_member_t& team) {
        auto scratch = Impl::alloc_scratch_tile<float, ExecSpace>(team, Tile{});
        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 8; ++j)
              scratch(i, j) = static_cast<float>(i * 8 + j);
        });
        team.team_barrier();

        auto src    = make_interm_node(scratch);  // NoHook
        using NodeT = decltype(src);
        using EvalT = Evaluator<TeamPolicyTag<ExecSpace>, NodeT,
                                std::integer_sequence<int, 1, 0>>;
        EvalT ev(std::integer_sequence<int, 1, 0>{}, team);
        auto  result = (ev(team, Kokkos::Array<int, 2>{0, 0}) = src);
        team.team_barrier();

        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 4; ++j)
              dst_readback(i * 4 + j) = result.storage_(i, j);
          for (int s = 0; s < 32; ++s) src_readback(s) = scratch.data()[s];
        });
      });
  Kokkos::fence();
}

TEST(RelabelScratchView, Transpose2DZeroCopy) {
  Buf1D src_readback("src_readback", 32);  // raw backing after relabel
  Buf1D dst_readback("dst_readback", 32);  // dst(i,j), dst is 8x4

  run_relabel_transpose(src_readback, dst_readback);

  auto h_dst = Kokkos::create_mirror_view(dst_readback);
  auto h_src = Kokkos::create_mirror_view(src_readback);
  Kokkos::deep_copy(h_dst, dst_readback);
  Kokkos::deep_copy(h_src, src_readback);

  // Zero-copy: raw backing is untouched row-major i*8+j.
  for (int s = 0; s < 32; ++s)
    EXPECT_FLOAT_EQ(h_src(s), static_cast<float>(s)) << "s=" << s;

  // dst(i,j) == original src(j,i) == j*8+i  (transpose semantics, gather
  // convention: perm={1,0} means new dim 0 reads old dim 1 and vice versa).
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 4; ++j)
      EXPECT_FLOAT_EQ(h_dst(i * 4 + j), static_cast<float>(j * 8 + i))
          << "i=" << i << " j=" << j;
}

// A named functor rather than a KOKKOS_LAMBDA: the hook is built inside the
// kernel lambda, and nvcc forbids defining an extended lambda inside another.
struct AddIndexHook {
  KOKKOS_FUNCTION void operator()(int i, int j, float& v) const {
    v += 1000.f * static_cast<float>(i + 1) + static_cast<float>(j + 1);
  }
};

void run_relabel_hook(Buf1D dst_readback) {
  using Tile              = StaticTile<2, 3>;
  using team_policy       = Kokkos::TeamPolicy<ExecSpace>;
  const std::size_t bytes = Impl::scratch_tile_bytes<float, ExecSpace>(Tile{});

  Kokkos::parallel_for(
      team_policy(1, Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const team_member_t& team) {
        auto scratch = Impl::alloc_scratch_tile<float, ExecSpace>(team, Tile{});
        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j)
              scratch(i, j) = static_cast<float>(i * 3 + j);
        });
        team.team_barrier();

        auto src    = make_interm_node(scratch, AddIndexHook{});
        using NodeT = decltype(src);
        using EvalT = Evaluator<TeamPolicyTag<ExecSpace>, NodeT,
                                std::integer_sequence<int, 1, 0>>;
        EvalT ev(std::integer_sequence<int, 1, 0>{}, team);
        auto  result = (ev(team, Kokkos::Array<int, 2>{0, 0}) = src);
        team.team_barrier();

        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 2; ++j)
              dst_readback(i * 2 + j) = result.storage_(i, j);
        });
      });
  Kokkos::fence();
}

TEST(RelabelScratchView, AppliesSourceHook) {
  Buf1D dst_readback("dst_readback", 6);  // dst is 3x2 after perm{1,0}

  run_relabel_hook(dst_readback);

  auto h = Kokkos::create_mirror_view(dst_readback);
  Kokkos::deep_copy(h, dst_readback);

  // dst(i,j) reads native src(j,i) = j*3+i, then the hook adds
  // 1000*(i+1)+(j+1) using dst's own local (i,j) as the global index
  // (tile_idx = {0,0}).
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 2; ++j) {
      const float base = static_cast<float>(j * 3 + i);
      const float expected =
          base + 1000.f * static_cast<float>(i + 1) + static_cast<float>(j + 1);
      EXPECT_FLOAT_EQ(h(i * 2 + j), expected) << "i=" << i << " j=" << j;
    }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
