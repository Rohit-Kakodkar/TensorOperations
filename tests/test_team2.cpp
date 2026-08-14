#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/TensorHandle.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <type_traits>
#include <vector>

using namespace TensorOperations;
using ES     = Kokkos::DefaultExecutionSpace;
using team_t = typename Kokkos::TeamPolicy<ES>::member_type;
using Buf1D  = Kokkos::View<float*, ES>;

struct T2 {
  static constexpr int rank = 2;
  using value_type          = float;
  using array_layout        = Kokkos::LayoutRight;
  float buf_[32]{};

  T2() {
    for (int i = 0; i < 32; ++i) buf_[i] = static_cast<float>(i);
  }
  KOKKOS_FUNCTION int extent(int i) const { return i == 0 ? 4 : 8; }
  KOKKOS_FUNCTION std::ptrdiff_t stride(int k) const { return k == 0 ? 8 : 1; }
  KOKKOS_FUNCTION float*         data() { return buf_; }
  KOKKOS_FUNCTION float* data() const { return const_cast<float*>(buf_); }
  KOKKOS_FUNCTION float  operator()(int r, int c) const {
    return buf_[r * 8 + c];
  }
};

struct T3 {
  static constexpr int rank = 3;
  using value_type          = float;
  using array_layout        = Kokkos::LayoutRight;
  float buf_[128]{};

  T3() {
    for (int i = 0; i < 128; ++i) buf_[i] = static_cast<float>(i);
  }
  KOKKOS_FUNCTION int extent(int i) const { return i == 2 ? 8 : 4; }
  KOKKOS_FUNCTION std::ptrdiff_t stride(int k) const {
    return k == 0 ? 32 : (k == 1 ? 8 : 1);
  }
  KOKKOS_FUNCTION float* data() { return buf_; }
  KOKKOS_FUNCTION float* data() const { return const_cast<float*>(buf_); }
  KOKKOS_FUNCTION float  operator()(int a, int b, int c) const {
    return buf_[a * 32 + b * 8 + c];
  }
};

struct ScaleHook {
  KOKKOS_FUNCTION void operator()(int, int, float& v) const { v *= 2.0f; }
};

struct AddIndexHook {
  KOKKOS_FUNCTION void operator()(int i, int j, float& v) const {
    v += 1000.f * static_cast<float>(i + 1) + static_cast<float>(j + 1);
  }
};

using TileT  = StaticTile<2, 4>;
using TileT3 = StaticTile<2, 2, 4>;
using NodeNH = decltype(make_input_node(make_handle<'i', 'j'>(T2{})));
using NodeH =
    decltype(make_input_node(make_handle<'i', 'j'>(T2{}), ScaleHook{}));
using Node3 = decltype(make_input_node(make_handle<'i', 'j', 'k'>(T3{})));

template <typename Node>
using Eval1 = Evaluator<TeamPolicyTag<ES>, Node, TileT>;
template <typename Node>
using Eval2 = Evaluator<TeamPolicyTag2<ES>, Node, TileT>;

// Tag2 hands back a value evaluator, not a node; the node it wraps is what
// corresponds to Tag1's result_type.
template <typename Node>
using tag2_result_t = decltype(std::declval<const Eval2<Node>&>()(
    std::declval<Kokkos::Array<int, 2>>()));
template <typename Node>
using tag2_result_node_t = typename tag2_result_t<Node>::node_type;

static_assert(std::is_same_v<tag2_result_node_t<NodeNH>,
                             typename Eval1<NodeNH>::result_type>,
              "Tag2 InputTag operator() must wrap Tag1's result_type");

static_assert(std::is_same_v<tag2_result_node_t<NodeH>,
                             typename Eval1<NodeH>::result_type>,
              "Tag2 InputTag must forward its HookOp into the interm node");

// The wrapper really is an Evaluator, keyed on the node with a void tiling.
static_assert(std::is_same_v<tag2_result_t<NodeNH>,
                             Evaluator<TeamPolicyTag2<ES>,
                                       tag2_result_node_t<NodeNH>, void>>,
              "Tag2 InputTag operator() must return a value evaluator");
static_assert(std::is_void_v<typename tag2_result_t<NodeNH>::tiling_type>);

static_assert(std::is_same_v<typename Eval2<NodeNH>::node_type, NodeNH>);

static_assert(!std::is_same_v<TeamPolicyTag<ES>, TeamPolicyTag2<ES>>);
static_assert(!std::is_same_v<Eval1<NodeNH>, Eval2<NodeNH>>);

static_assert(std::is_same_v<decltype(make_evaluator<TeamPolicyTag2<ES>>(
                                 std::declval<NodeNH>(), std::declval<TileT>(),
                                 std::declval<const team_t&>())),
                             Eval2<NodeNH>>);

TEST(Team2InputTag, ReturnTypeMatchesTeamPolicyTag) { SUCCEED(); }

int run_subview_kernel(int ti, int tj) {
  auto                  node = make_input_node(make_handle<'i', 'j'>(T2{}));
  Kokkos::View<int, ES> ok("ok");
  Kokkos::deep_copy(ok, 0);
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO),
      KOKKOS_LAMBDA(const team_t& team) {
        Eval1<NodeNH>               e1(node, TileT{}, team);
        Eval2<NodeNH>               e2(node, TileT{}, team);
        const Kokkos::Array<int, 2> idx{ti, tj};
        auto                        r1 = e1(team, idx);
        auto                        r2 = e2(idx);
        Kokkos::single(Kokkos::PerTeam(team), [&] {
          int good = 1;
          for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 4; ++b) {
              const float v2 = r2.node().storage_(a, b);
              const float expect =
                  static_cast<float>((ti * 2 + a) * 8 + (tj * 4 + b));
              if (v2 != r1.storage_(a, b) || v2 != expect) good = 0;
            }
          ok() = good;
        });
      });
  Kokkos::fence();
  int ok_h = 0;
  Kokkos::deep_copy(ok_h, ok);
  return ok_h;
}

TEST(Team2InputTag, SubviewTileAliasesExpectedElements) {
  EXPECT_EQ(run_subview_kernel(0, 0), 1);
  EXPECT_EQ(run_subview_kernel(1, 1), 1);
  EXPECT_EQ(run_subview_kernel(0, 1), 1);
  EXPECT_EQ(run_subview_kernel(1, 0), 1);
}

// ---------------------------------------------------------------------------
// Tag2 IntermTag + perm seq — relabel, both storage families
// ---------------------------------------------------------------------------

using Swap   = std::integer_sequence<int, 1, 0>;
using Ident2 = std::integer_sequence<int, 0, 1>;

// Tag1 keys the relabel on the NODE; Tag2 keys it on the value evaluator that
// wraps it, so each family gets its source spelled in its own terms.
using SrcNode  = tag2_result_node_t<NodeNH>;
using SrcNodeH = tag2_result_node_t<NodeH>;
using SrcVal   = tag2_result_t<NodeNH>;
using SrcValH  = tag2_result_t<NodeH>;

using ScratchTile  = StaticTile<4, 8>;
using ScratchViewT = decltype(Impl::alloc_scratch_tile<float, ES>(
    std::declval<const team_t&>(), std::declval<const ScratchTile&>()));
using ScratchNode  = decltype(make_interm_node(std::declval<ScratchViewT>()));
using ScratchNodeH =
    decltype(make_interm_node(std::declval<ScratchViewT>(), AddIndexHook{}));
using ScratchVal  = decltype(make_value_evaluator(
    std::declval<ScratchNode>(), std::declval<const team_t&>()));
using ScratchValH = decltype(make_value_evaluator(
    std::declval<ScratchNodeH>(), std::declval<const team_t&>()));

template <typename Node, typename Perm>
using Relabel1 = Evaluator<TeamPolicyTag<ES>, Node, Perm>;
template <typename Src, typename Perm>
using Relabel2 = Evaluator<TeamPolicyTag2<ES>, Src, Perm>;

// The relabel also yields a value evaluator, so a relabel composes onto the
// next step like any other Tag2 result.
template <typename Src, typename Perm>
using relabel2_result_t = decltype(std::declval<const Relabel2<Src, Perm>&>() =
                                       std::declval<const Src&>());
template <typename Src, typename Perm>
using relabel2_result_node_t = typename relabel2_result_t<Src, Perm>::node_type;

static_assert(
    std::is_same_v<relabel2_result_t<SrcVal, Swap>,
                   Evaluator<TeamPolicyTag2<ES>,
                             relabel2_result_node_t<SrcVal, Swap>, void>>,
    "Tag2 relabel must return a value evaluator");

static_assert(
    std::is_same_v<typename relabel2_result_node_t<SrcVal, Swap>::storage_type,
                   typename Relabel1<SrcNode, Swap>::dest_view_t>);
static_assert(std::is_same_v<
              typename relabel2_result_node_t<ScratchVal, Swap>::storage_type,
              typename Relabel1<ScratchNode, Swap>::dest_view_t>);

static_assert(
    std::is_same_v<
        decltype(std::declval<relabel2_result_node_t<SrcValH, Swap>>().hook_op),
        ScaleHook>);
static_assert(
    std::is_same_v<
        decltype(std::declval<typename Relabel1<SrcNodeH, Swap>::interm_type>()
                     .hook_op),
        NoHook>);
static_assert(std::is_same_v<
              decltype(std::declval<relabel2_result_node_t<ScratchValH, Swap>>()
                           .hook_op),
              AddIndexHook>);

// An identity relabel round-trips to the very same value evaluator type.
static_assert(std::is_same_v<relabel2_result_t<SrcVal, Ident2>, SrcVal>);

static_assert(std::is_same_v<decltype(make_evaluator<TeamPolicyTag2<ES>>(
                                 std::declval<SrcVal>(), std::declval<Swap>(),
                                 std::declval<const team_t&>())),
                             Relabel2<SrcVal, Swap>>);

TEST(Team2Relabel, TypesMatchTeamPolicyTagSpecialization7) { SUCCEED(); }

void run_relabel_global(int ti, int tj, Buf1D dst_readback,
                        Kokkos::View<int, ES> same_data) {
  auto node = make_input_node(make_handle<'i', 'j'>(T2{}));
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO),
      KOKKOS_LAMBDA(const team_t& team) {
        Eval2<NodeNH> e2(node, TileT{}, team);
        auto          src = e2(Kokkos::Array<int, 2>{ti, tj});
        auto rel = make_evaluator<TeamPolicyTag2<ES>>(src, Swap{}, team);
        auto dst = (rel = src);
        Kokkos::single(Kokkos::PerTeam(team), [&] {
          const auto dv = dst.node().storage_;
          const auto sv = src.node().storage_;
          for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 2; ++b) dst_readback(a * 2 + b) = dv(a, b);
          const auto d_off = dv.data() - dv.backing_.data();
          const auto s_off = sv.data() - sv.backing_.data();
          same_data()      = (d_off == s_off) ? 1 : 0;
        });
      });
  Kokkos::fence();
}

TEST(Team2Relabel, GlobalTransposeIsZeroCopy) {
  const int             ti = 1, tj = 1;
  Buf1D                 dst("dst", 8);
  Kokkos::View<int, ES> same("same");
  run_relabel_global(ti, tj, dst, same);

  auto h_dst = Kokkos::create_mirror_view(dst);
  Kokkos::deep_copy(h_dst, dst);
  int h_same = 0;
  Kokkos::deep_copy(h_same, same);

  EXPECT_EQ(h_same, 1);
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 2; ++b)
      EXPECT_FLOAT_EQ(h_dst(a * 2 + b),
                      static_cast<float>((ti * 2 + b) * 8 + (tj * 4 + a)))
          << "a=" << a << " b=" << b;
}

void run_relabel_global3(Buf1D dst_readback, Buf1D src_readback) {
  auto node = make_input_node(make_handle<'i', 'j', 'k'>(T3{}));
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO),
      KOKKOS_LAMBDA(const team_t& team) {
        Evaluator<TeamPolicyTag2<ES>, Node3, TileT3> e2(node, TileT3{}, team);
        auto src = e2(Kokkos::Array<int, 3>{1, 1, 1});
        auto rel = make_evaluator<TeamPolicyTag2<ES>>(
            src, std::integer_sequence<int, 2, 0, 1>{}, team);
        auto dst = (rel = src);
        Kokkos::single(Kokkos::PerTeam(team), [&] {
          const auto dv = dst.node().storage_;
          const auto sv = src.node().storage_;
          for (int x = 0; x < 4; ++x)
            for (int y = 0; y < 2; ++y)
              for (int z = 0; z < 2; ++z)
                dst_readback(x * 4 + y * 2 + z) = dv(x, y, z);
          for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b)
              for (int c = 0; c < 4; ++c)
                src_readback(a * 8 + b * 4 + c) = sv(a, b, c);
        });
      });
  Kokkos::fence();
}

TEST(Team2Relabel, GlobalRank3CyclePermutation) {
  Buf1D dst("dst", 16), src("src", 16);
  run_relabel_global3(dst, src);

  auto h_dst = Kokkos::create_mirror_view(dst);
  auto h_src = Kokkos::create_mirror_view(src);
  Kokkos::deep_copy(h_dst, dst);
  Kokkos::deep_copy(h_src, src);

  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 2; ++b)
      for (int c = 0; c < 4; ++c)
        EXPECT_FLOAT_EQ(
            h_src(a * 8 + b * 4 + c),
            static_cast<float>((2 + a) * 32 + (2 + b) * 8 + (4 + c)))
            << "a=" << a << " b=" << b << " c=" << c;

  for (int x = 0; x < 4; ++x)
    for (int y = 0; y < 2; ++y)
      for (int z = 0; z < 2; ++z)
        EXPECT_FLOAT_EQ(h_dst(x * 4 + y * 2 + z), h_src(y * 8 + z * 4 + x))
            << "x=" << x << " y=" << y << " z=" << z;
}

template <typename Hook>
void run_relabel_scratch(Buf1D dst_readback, Buf1D src_readback, Hook hook) {
  const std::size_t bytes = Impl::scratch_tile_bytes<float, ES>(ScratchTile{});
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const team_t& team) {
        auto scratch = Impl::alloc_scratch_tile<float, ES>(team, ScratchTile{});
        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 8; ++j)
              scratch(i, j) = static_cast<float>(i * 8 + j);
        });
        team.team_barrier();

        auto src = make_value_evaluator(make_interm_node(scratch, hook), team);
        auto rel = make_evaluator<TeamPolicyTag2<ES>>(src, Swap{}, team);
        auto dst = (rel = src);
        team.team_barrier();

        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          const auto dv = dst.node().storage_;
          for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 4; ++j) dst_readback(i * 4 + j) = dv(i, j);
          for (int s = 0; s < 32; ++s) src_readback(s) = scratch.data()[s];
        });
      });
  Kokkos::fence();
}

TEST(Team2Relabel, ScratchTransposeIsZeroCopyRetype) {
  Buf1D dst("dst", 32), src("src", 32);
  run_relabel_scratch(dst, src, NoHook{});

  auto h_dst = Kokkos::create_mirror_view(dst);
  auto h_src = Kokkos::create_mirror_view(src);
  Kokkos::deep_copy(h_dst, dst);
  Kokkos::deep_copy(h_src, src);

  for (int s = 0; s < 32; ++s)
    EXPECT_FLOAT_EQ(h_src(s), static_cast<float>(s)) << "s=" << s;
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 4; ++j)
      EXPECT_FLOAT_EQ(h_dst(i * 4 + j), static_cast<float>(j * 8 + i))
          << "i=" << i << " j=" << j;
}

TEST(Team2Relabel, ScratchSourceHookIsDeferred) {
  Buf1D dst("dst", 32), src("src", 32);
  run_relabel_scratch(dst, src, AddIndexHook{});

  auto h_dst = Kokkos::create_mirror_view(dst);
  auto h_src = Kokkos::create_mirror_view(src);
  Kokkos::deep_copy(h_dst, dst);
  Kokkos::deep_copy(h_src, src);

  for (int s = 0; s < 32; ++s)
    EXPECT_FLOAT_EQ(h_src(s), static_cast<float>(s)) << "s=" << s;
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 4; ++j)
      EXPECT_FLOAT_EQ(h_dst(i * 4 + j), static_cast<float>(j * 8 + i))
          << "i=" << i << " j=" << j;
}

using FullTile  = StaticTile<4, 8>;
using PartTile  = StaticTile<2, 4>;
using FullTile3 = StaticTile<2, 4, 8>;
using PartTile3 = StaticTile<2, 2, 4>;
using TransTile = StaticTile<8, 4>;

template <typename Node, typename Tile>
using stage_src_t =
    decltype(std::declval<const Evaluator<TeamPolicyTag2<ES>, Node, Tile>&>()(
        std::declval<Kokkos::Array<int, Tile::rank>>()));

template <typename Tile>
using scratch_view_for_t = decltype(Impl::alloc_scratch_tile<float, ES>(
    std::declval<const team_t&>(), std::declval<const Tile&>()));
template <typename Tile>
using scratch_layout_for_t = typename scratch_view_for_t<Tile>::layout_t;
template <typename Tile>
using Stager = Evaluator<TeamPolicyTag2<ES>,
                         decltype(make_interm_node(
                             std::declval<scratch_view_for_t<Tile>>())),
                         StageTag>;

template <typename Node, typename Tile>
using stage_result_t =
    decltype(std::declval<const Stager<Tile>&>() =
                 std::declval<const stage_src_t<Node, Tile>&>());
static_assert(
    std::is_same_v<
        stage_result_t<NodeNH, FullTile>,
        Evaluator<TeamPolicyTag2<ES>,
                  typename stage_result_t<NodeNH, FullTile>::node_type, void>>,
    "Tag2 staging must return a value evaluator");
static_assert(std::is_same_v<decltype(std::declval<typename stage_result_t<
                                          NodeH, FullTile>::node_type>()
                                          .hook_op),
                             ScaleHook>,
              "Tag2 staging must carry the source hook forward unapplied");
static_assert(
    std::is_same_v<typename stage_result_t<NodeNH, FullTile>::storage_type,
                   scratch_view_for_t<FullTile>>);

TEST(Team2Stage, ResultTypesAreAsExpected) { SUCCEED(); }

template <typename NodeT, typename Tile, std::size_t R>
void run_stage(NodeT node, Kokkos::Array<int, R> tidx, Buf1D out) {
  using Eval              = Evaluator<TeamPolicyTag2<ES>, NodeT, Tile>;
  const std::size_t bytes = Impl::scratch_tile_bytes<float, ES>(Tile{});
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const team_t& team) {
        Eval e(node, Tile{}, team);
        auto src    = e(tidx);
        auto stager = make_evaluator<TeamPolicyTag2<ES>>(
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, Tile{})),
            StageTag{}, team);
        auto res = (stager = src);
        team.team_barrier();

        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          const auto v = res.node().storage_;
          for (int s = 0; s < scratch_layout_for_t<Tile>::size(); ++s)
            out(s) = v.data()[s];
        });
      });
  Kokkos::fence();
}

template <typename NodeT, typename Tile, std::size_t R>
std::vector<float> stage(NodeT node, Kokkos::Array<int, R> tidx) {
  constexpr int n = scratch_layout_for_t<Tile>::size();
  Buf1D         out("out", n);
  run_stage<NodeT, Tile>(node, tidx, out);

  auto h_out = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(h_out, out);
  std::vector<float> vals(n);
  for (int i = 0; i < n; ++i) vals[i] = h_out(i);
  return vals;
}

TEST(Team2Stage, FullTileCopiesEveryElement) {
  const auto vals =
      stage<NodeNH, FullTile>(make_input_node(make_handle<'i', 'j'>(T2{})),
                              Kokkos::Array<int, 2>{0, 0});
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j)
      EXPECT_FLOAT_EQ(vals[i * 8 + j], static_cast<float>(i * 8 + j))
          << "i=" << i << " j=" << j;
}

TEST(Team2Stage, PartialTileCopiesItsWindow) {
  const auto vals =
      stage<NodeNH, PartTile>(make_input_node(make_handle<'i', 'j'>(T2{})),
                              Kokkos::Array<int, 2>{1, 1});
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 4; ++b)
      EXPECT_FLOAT_EQ(vals[a * 4 + b],
                      static_cast<float>((2 + a) * 8 + (4 + b)))
          << "a=" << a << " b=" << b;
}

TEST(Team2Stage, Rank3FullTileCopiesEveryElement) {
  const auto vals =
      stage<Node3, FullTile3>(make_input_node(make_handle<'i', 'j', 'k'>(T3{})),
                              Kokkos::Array<int, 3>{1, 0, 0});
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 4; ++b)
      for (int c = 0; c < 8; ++c)
        EXPECT_FLOAT_EQ(vals[a * 32 + b * 8 + c],
                        static_cast<float>((2 + a) * 32 + b * 8 + c))
            << "a=" << a << " b=" << b << " c=" << c;
}

TEST(Team2Stage, Rank3PartialTileCopiesItsWindow) {
  const auto vals =
      stage<Node3, PartTile3>(make_input_node(make_handle<'i', 'j', 'k'>(T3{})),
                              Kokkos::Array<int, 3>{1, 1, 1});
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 2; ++b)
      for (int c = 0; c < 4; ++c)
        EXPECT_FLOAT_EQ(
            vals[a * 8 + b * 4 + c],
            static_cast<float>((2 + a) * 32 + (2 + b) * 8 + (4 + c)))
            << "a=" << a << " b=" << b << " c=" << c;
}

TEST(Team2Stage, HookedSourceStagesRawValues) {
  const auto vals = stage<NodeH, FullTile>(
      make_input_node(make_handle<'i', 'j'>(T2{}), ScaleHook{}),
      Kokkos::Array<int, 2>{0, 0});
  for (int s = 0; s < 32; ++s)
    EXPECT_FLOAT_EQ(vals[s], static_cast<float>(s)) << "s=" << s;
}

void run_stage_relabeled(Buf1D out) {
  auto              node  = make_input_node(make_handle<'i', 'j'>(T2{}));
  const std::size_t bytes = Impl::scratch_tile_bytes<float, ES>(TransTile{});
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const team_t& team) {
        Evaluator<TeamPolicyTag2<ES>, NodeNH, FullTile> e(node, FullTile{},
                                                          team);
        auto src = e(Kokkos::Array<int, 2>{0, 0});
        auto rel = make_evaluator<TeamPolicyTag2<ES>>(src, Swap{}, team);
        auto tr  = (rel = src);

        auto stager = make_evaluator<TeamPolicyTag2<ES>>(
            make_interm_node(
                Impl::alloc_scratch_tile<float, ES>(team, TransTile{})),
            StageTag{}, team);
        auto res = (stager = tr);
        team.team_barrier();

        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          const auto v = res.node().storage_;
          for (int s = 0; s < 32; ++s) out(s) = v.data()[s];
        });
      });
  Kokkos::fence();
}

TEST(Team2Stage, RelabeledSourceTransposes) {
  Buf1D out("out", 32);
  run_stage_relabeled(out);
  auto h_out = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(h_out, out);

  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 4; ++j)
      EXPECT_FLOAT_EQ(h_out(i * 4 + j), static_cast<float>(j * 8 + i))
          << "i=" << i << " j=" << j;
}

namespace contract {

using V2 = Kokkos::View<float**, Kokkos::LayoutRight, ES>;
using V3 = Kokkos::View<float***, Kokkos::LayoutRight, ES>;

KOKKOS_INLINE_FUNCTION float a_val(int i, int k) {
  return 0.75f + 0.25f * i - 0.5f * k + 0.125f * i * k;
}
KOKKOS_INLINE_FUNCTION float b_val(int k, int l) {
  return -1.5f + 0.5f * k + 0.75f * l - 0.0625f * k * l;
}

template <typename Tile, typename Node, typename Team>
KOKKOS_FUNCTION auto stage_full(Node node, Tile tile, const Team& team) {
  auto src = make_evaluator<TeamPolicyTag2<ES>>(
      node, tile, team)(Kokkos::Array<int, Tile::rank>{});
  auto dst = make_evaluator<TeamPolicyTag2<ES>>(
      make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, tile)),
      StageTag{}, team);
  return (dst = src);
}

template <typename... Tiles>
std::size_t scratch_for() {
  return (Impl::scratch_tile_bytes<float, ES>(Tiles{}) + ... + std::size_t{0});
}

struct ScaleC {
  KOKKOS_FUNCTION void operator()(int, int, float& v) const { v *= 3.0f; }
};

constexpr int kI = 4, kK = 5, kL = 3;
using TA = StaticTile<kI, kK>;
using TB = StaticTile<kK, kL>;
using TC = StaticTile<kI, kL>;

template <typename Hook>
void run_matmul(V2 a, V2 b, Buf1D team_out, Hook hook) {
  auto node = make_contraction_node<'i', 'l'>(
      make_input_node(make_handle<'i', 'k'>(a)),
      make_input_node(make_handle<'k', 'l'>(b)), hook);
  const std::size_t bytes = scratch_for<TA, TB, TC>();
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const team_t& team) {
        auto A = stage_full(node.node_a, TA{}, team);
        auto B = stage_full(node.node_b, TB{}, team);
        auto C =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, TC{}));
        auto gemm = make_evaluator<TeamPolicyTag2<ES>>(
            node, ContractOperands{A, B, C}, team);
        team.team_barrier();
        auto res = gemm();
        team.team_barrier();

        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          const auto cv = res.node().storage_;
          for (int i = 0; i < kI; ++i)
            for (int l = 0; l < kL; ++l) team_out(i * kL + l) = cv(i, l);
        });
      });
  Kokkos::fence();
}

constexpr int mI = 3, mJ = 2, mK = 5, mL = 4;
using MTA = StaticTile<mI, mJ, mK>;
using MTB = StaticTile<mJ, mK, mL>;
using MTC = StaticTile<mI, mL>;

void run_multi_k(V3 a, V3 b, Buf1D out) {
  auto node = make_contraction_node<'i', 'l'>(
      make_input_node(make_handle<'i', 'j', 'k'>(a)),
      make_input_node(make_handle<'j', 'k', 'l'>(b)));
  const std::size_t bytes = scratch_for<MTA, MTB, MTC>();
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const team_t& team) {
        auto A = stage_full(node.node_a, MTA{}, team);
        auto B = stage_full(node.node_b, MTB{}, team);
        auto C =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, MTC{}));
        auto res = make_evaluator<TeamPolicyTag2<ES>>(
            node, ContractOperands{A, B, C}, team)();
        team.team_barrier();
        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          const auto cv = res.node().storage_;
          for (int i = 0; i < mI; ++i)
            for (int l = 0; l < mL; ++l) out(i * mL + l) = cv(i, l);
        });
      });
  Kokkos::fence();
}

constexpr int rI = 2, rJ = 3, rK = 5, rL = 4;
using RTA = StaticTile<rI, rJ, rK>;
using RTB = StaticTile<rK, rL>;
using RTC = StaticTile<rI, rJ, rL>;

void run_rank3_out(V3 a, V2 b, Buf1D out) {
  auto node = make_contraction_node<'i', 'j', 'l'>(
      make_input_node(make_handle<'i', 'j', 'k'>(a)),
      make_input_node(make_handle<'k', 'l'>(b)));
  const std::size_t bytes = scratch_for<RTA, RTB, RTC>();
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const team_t& team) {
        auto A = stage_full(node.node_a, RTA{}, team);
        auto B = stage_full(node.node_b, RTB{}, team);
        auto C =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, RTC{}));
        auto res = make_evaluator<TeamPolicyTag2<ES>>(
            node, ContractOperands{A, B, C}, team)();
        team.team_barrier();
        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          const auto cv = res.node().storage_;
          for (int i = 0; i < rI; ++i)
            for (int j = 0; j < rJ; ++j)
              for (int l = 0; l < rL; ++l)
                out((i * rJ + j) * rL + l) = cv(i, j, l);
        });
      });
  Kokkos::fence();
}

std::vector<float> to_host(Buf1D v) {
  auto h = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(h, v);
  std::vector<float> out(v.extent(0));
  for (std::size_t i = 0; i < v.extent(0); ++i) out[i] = h(i);
  return out;
}

// P1: drive the SAME contraction two ways -- once through operator()(), once
// through a caller-written range calling the element operator()(i, j). The
// second shape is what the fused level driver will emit, so they must agree.
void run_element_vs_driver_matmul(V2 a, V2 b, Buf1D driver_out,
                                  Buf1D manual_out) {
  auto node = make_contraction_node<'i', 'l'>(
      make_input_node(make_handle<'i', 'k'>(a)),
      make_input_node(make_handle<'k', 'l'>(b)));
  const std::size_t bytes = scratch_for<TA, TB, TC, TC>();
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const team_t& team) {
        auto A = stage_full(node.node_a, TA{}, team);
        auto B = stage_full(node.node_b, TB{}, team);
        auto C0 =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, TC{}));
        auto C1 =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, TC{}));
        auto driver = make_evaluator<TeamPolicyTag2<ES>>(
            node, ContractOperands{A, B, C0}, team);
        auto manual = make_evaluator<TeamPolicyTag2<ES>>(
            node, ContractOperands{A, B, C1}, team);
        team.team_barrier();

        auto res = driver();

        constexpr int SA   = decltype(manual)::SA;
        constexpr int SB   = decltype(manual)::SB;
        const auto    self = manual;
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team, SA * SB),
                             [=](int t) { self(t / SB, t % SB); });
        team.team_barrier();

        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          const auto dv = res.node().storage_;
          const auto mv = C1.storage_;
          for (int i = 0; i < kI; ++i)
            for (int l = 0; l < kL; ++l) {
              driver_out(i * kL + l) = dv(i, l);
              manual_out(i * kL + l) = mv(i, l);
            }
        });
      });
  Kokkos::fence();
}

}  // namespace contract

using contract::a_val;
using contract::b_val;

TEST(Team2Contract, MatmulMatchesHostReference) {
  using namespace contract;
  V2   a("a", kI, kK), b("b", kK, kL);
  auto ah = Kokkos::create_mirror_view(a);
  auto bh = Kokkos::create_mirror_view(b);
  for (int i = 0; i < kI; ++i)
    for (int k = 0; k < kK; ++k) ah(i, k) = a_val(i, k);
  for (int k = 0; k < kK; ++k)
    for (int l = 0; l < kL; ++l) bh(k, l) = b_val(k, l);
  Kokkos::deep_copy(a, ah);
  Kokkos::deep_copy(b, bh);

  Buf1D team_out("team", kI * kL);
  run_matmul(a, b, team_out, NoHook{});
  const auto got = to_host(team_out);

  for (int i = 0; i < kI; ++i)
    for (int l = 0; l < kL; ++l) {
      double acc = 0.0;
      for (int k = 0; k < kK; ++k)
        acc += static_cast<double>(ah(i, k)) * bh(k, l);
      EXPECT_NEAR(got[i * kL + l], static_cast<float>(acc), 1e-4f)
          << "i=" << i << " l=" << l;
    }
}

TEST(Team2Contract, HookRidesForwardUnapplied) {
  using namespace contract;
  V2   a("a", kI, kK), b("b", kK, kL);
  auto ah = Kokkos::create_mirror_view(a);
  auto bh = Kokkos::create_mirror_view(b);
  for (int i = 0; i < kI; ++i)
    for (int k = 0; k < kK; ++k) ah(i, k) = a_val(i, k);
  for (int k = 0; k < kK; ++k)
    for (int l = 0; l < kL; ++l) bh(k, l) = b_val(k, l);
  Kokkos::deep_copy(a, ah);
  Kokkos::deep_copy(b, bh);

  Buf1D team_out("team", kI * kL);
  run_matmul(a, b, team_out, ScaleC{});
  const auto got = to_host(team_out);

  // ScaleC triples. The scratch must hold the RAW contraction; the hook rides
  // forward on the returned node for a later step to apply.
  for (int i = 0; i < kI; ++i)
    for (int l = 0; l < kL; ++l) {
      double acc = 0.0;
      for (int k = 0; k < kK; ++k)
        acc += static_cast<double>(ah(i, k)) * bh(k, l);
      EXPECT_NEAR(got[i * kL + l], static_cast<float>(acc), 1e-4f)
          << "i=" << i << " l=" << l;
    }
}

TEST(Team2Contract, MultipleContractedModesCollapseIntoSK) {
  using namespace contract;
  V3   a("a", mI, mJ, mK), b("b", mJ, mK, mL);
  auto ah = Kokkos::create_mirror_view(a);
  auto bh = Kokkos::create_mirror_view(b);
  for (int i = 0; i < mI; ++i)
    for (int j = 0; j < mJ; ++j)
      for (int k = 0; k < mK; ++k)
        ah(i, j, k) = static_cast<float>((i + 2 * j + 3 * k) % 5 + 1) * 0.5f;
  for (int j = 0; j < mJ; ++j)
    for (int k = 0; k < mK; ++k)
      for (int l = 0; l < mL; ++l)
        bh(j, k, l) = static_cast<float>((3 * j + k + 2 * l) % 4 + 1) * 0.25f;
  Kokkos::deep_copy(a, ah);
  Kokkos::deep_copy(b, bh);

  Buf1D out("out", mI * mL);
  run_multi_k(a, b, out);
  const auto got = to_host(out);

  for (int i = 0; i < mI; ++i)
    for (int l = 0; l < mL; ++l) {
      double acc = 0.0;
      for (int j = 0; j < mJ; ++j)
        for (int k = 0; k < mK; ++k)
          acc += static_cast<double>(ah(i, j, k)) * bh(j, k, l);
      EXPECT_NEAR(got[i * mL + l], static_cast<float>(acc), 1e-4f)
          << "i=" << i << " l=" << l;
    }
}

TEST(Team2Contract, MultipleFreeAModesCollapseIntoSA) {
  using namespace contract;
  V3   a("a", rI, rJ, rK);
  V2   b("b", rK, rL);
  auto ah = Kokkos::create_mirror_view(a);
  auto bh = Kokkos::create_mirror_view(b);
  for (int i = 0; i < rI; ++i)
    for (int j = 0; j < rJ; ++j)
      for (int k = 0; k < rK; ++k)
        ah(i, j, k) = static_cast<float>((2 * i + j + 3 * k) % 7 + 1) * 0.25f;
  for (int k = 0; k < rK; ++k)
    for (int l = 0; l < rL; ++l) bh(k, l) = b_val(k, l);
  Kokkos::deep_copy(a, ah);
  Kokkos::deep_copy(b, bh);

  Buf1D out("out", rI * rJ * rL);
  run_rank3_out(a, b, out);
  const auto got = to_host(out);

  for (int i = 0; i < rI; ++i)
    for (int j = 0; j < rJ; ++j)
      for (int l = 0; l < rL; ++l) {
        double acc = 0.0;
        for (int k = 0; k < rK; ++k)
          acc += static_cast<double>(ah(i, j, k)) * bh(k, l);
        EXPECT_NEAR(got[(i * rJ + j) * rL + l], static_cast<float>(acc), 1e-4f)
            << "i=" << i << " j=" << j << " l=" << l;
      }
}

// ---------------------------------------------------------------------------
// Tag2 ContractionTag over PERMUTED operands — the capability the regroup
// exists for. A relabeled operand is a zero-copy strided view, so the row and
// column groups the contraction collapses are no longer contiguous runs of the
// operand's memory-order stream; reshape could not express these at all.
//
// reorder_view's permutation reads perm[d] = the SOURCE axis that becomes
// destination axis d, matching Team2Relabel.GlobalRank3CyclePermutation.
// ---------------------------------------------------------------------------
TEST(Team2Contract, ElementOperatorMatchesDriver) {
  using namespace contract;
  V2   a("a", kI, kK), b("b", kK, kL);
  auto ah = Kokkos::create_mirror_view(a);
  auto bh = Kokkos::create_mirror_view(b);
  for (int i = 0; i < kI; ++i)
    for (int k = 0; k < kK; ++k) ah(i, k) = a_val(i, k);
  for (int k = 0; k < kK; ++k)
    for (int l = 0; l < kL; ++l) bh(k, l) = b_val(k, l);
  Kokkos::deep_copy(a, ah);
  Kokkos::deep_copy(b, bh);

  Buf1D driver_out("driver", kI * kL), manual_out("manual", kI * kL);
  run_element_vs_driver_matmul(a, b, driver_out, manual_out);
  const auto d = to_host(driver_out);
  const auto m = to_host(manual_out);

  for (int i = 0; i < kI; ++i)
    for (int l = 0; l < kL; ++l) {
      double acc = 0.0;
      for (int k = 0; k < kK; ++k)
        acc += static_cast<double>(ah(i, k)) * bh(k, l);
      EXPECT_NEAR(d[i * kL + l], static_cast<float>(acc), 1e-4f)
          << "i=" << i << " l=" << l;
      EXPECT_FLOAT_EQ(m[i * kL + l], d[i * kL + l]) << "i=" << i << " l=" << l;
    }
}

namespace permuted {

using contract::scratch_for;
using contract::stage_full;
using contract::to_host;

using PV2 = Kokkos::View<float**, Kokkos::LayoutRight, ES>;
using PV3 = Kokkos::View<float***, Kokkos::LayoutRight, ES>;

KOKKOS_INLINE_FUNCTION float u_val(int p, int q, int r) {
  return 0.5f + 0.25f * p - 0.125f * q + 0.375f * r + 0.0625f * p * r -
         0.03125f * q * r;
}
KOKKOS_INLINE_FUNCTION float h_val(int d, int m, int k) {
  return 1.25f - 0.5f * d + 0.75f * m - 0.25f * k + 0.125f * m * k * (d + 1);
}

// The SEM gradient level: one tensor staged once, three contractions each
// taking a different axis as k. Directions 1 and 2 reach it through a permuted
// view of the SAME buffer instead of a transposed copy.
constexpr int uP = 3, uQ = 4, uR = 5, hM = 2;

using TU  = StaticTile<uP, uQ, uR>;
using TH0 = StaticTile<hM, uP>;
using TH1 = StaticTile<hM, uQ>;
using TH2 = StaticTile<hM, uR>;
using TC0 = StaticTile<hM, uQ, uR>;
using TC1 = StaticTile<hM, uP, uR>;
using TC2 = StaticTile<hM, uP, uQ>;

void run_three_directions(PV3 u, PV2 h0, PV2 h1, PV2 h2, Buf1D o0, Buf1D o1,
                          Buf1D o2) {
  auto un  = make_input_node(make_handle<'p', 'q', 'r'>(u));
  auto h0n = make_input_node(make_handle<'m', 'p'>(h0));
  auto h1n = make_input_node(make_handle<'m', 'q'>(h1));
  auto h2n = make_input_node(make_handle<'m', 'r'>(h2));

  auto n0 = make_contraction_node<'m', 'q', 'r'>(h0n, un);
  auto n1 = make_contraction_node<'m', 'p', 'r'>(h1n, un);
  auto n2 = make_contraction_node<'m', 'p', 'q'>(h2n, un);

  const std::size_t bytes = scratch_for<TU, TH0, TH1, TH2, TC0, TC1, TC2>();
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const team_t& team) {
        auto U  = stage_full(un, TU{}, team);
        auto A0 = stage_full(h0n, TH0{}, team);
        auto A1 = stage_full(h1n, TH1{}, team);
        auto A2 = stage_full(h2n, TH2{}, team);

        // (p,q,r) -> (q,p,r): k = q, free (p,r), both groups gapped.
        auto B1 = (make_evaluator<TeamPolicyTag2<ES>>(
                       U, std::integer_sequence<int, 1, 0, 2>{}, team) = U);
        // (p,q,r) -> (r,p,q): k = r is the unit-stride axis, free (p,q).
        auto B2 = (make_evaluator<TeamPolicyTag2<ES>>(
                       U, std::integer_sequence<int, 2, 0, 1>{}, team) = U);

        auto C0 =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, TC0{}));
        auto C1 =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, TC1{}));
        auto C2 =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, TC2{}));
        team.team_barrier();

        make_evaluator<TeamPolicyTag2<ES>>(n0, ContractOperands{A0, U, C0},
                                           team)();
        make_evaluator<TeamPolicyTag2<ES>>(n1, ContractOperands{A1, B1, C1},
                                           team)();
        make_evaluator<TeamPolicyTag2<ES>>(n2, ContractOperands{A2, B2, C2},
                                           team)();
        team.team_barrier();

        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          const auto v0 = C0.storage_;
          const auto v1 = C1.storage_;
          const auto v2 = C2.storage_;
          for (int m = 0; m < hM; ++m) {
            for (int q = 0; q < uQ; ++q)
              for (int r = 0; r < uR; ++r)
                o0((m * uQ + q) * uR + r) = v0(m, q, r);
            for (int p = 0; p < uP; ++p)
              for (int r = 0; r < uR; ++r)
                o1((m * uP + p) * uR + r) = v1(m, p, r);
            for (int p = 0; p < uP; ++p)
              for (int q = 0; q < uQ; ++q)
                o2((m * uP + p) * uQ + q) = v2(m, p, q);
          }
        });
      });
  Kokkos::fence();
}

// A permuted A operand, where it is the ROW group that is gapped: W(x,y,z)
// presented as (x,z,y), so the free modes straddle the contracted one.
constexpr int wX = 3, wY = 4, wZ = 2, wL = 5;

using TW  = StaticTile<wX, wY, wZ>;
using TWB = StaticTile<wY, wL>;
using TWC = StaticTile<wX, wZ, wL>;

void run_permuted_a(PV3 w, PV2 b, Buf1D out) {
  auto wn = make_input_node(make_handle<'x', 'y', 'z'>(w));
  auto bn = make_input_node(make_handle<'y', 'l'>(b));
  auto n  = make_contraction_node<'x', 'z', 'l'>(wn, bn);

  const std::size_t bytes = scratch_for<TW, TWB, TWC>();
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const team_t& team) {
        auto W = stage_full(wn, TW{}, team);
        auto B = stage_full(bn, TWB{}, team);
        auto A = (make_evaluator<TeamPolicyTag2<ES>>(
                      W, std::integer_sequence<int, 0, 2, 1>{}, team) = W);
        auto C =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, TWC{}));
        team.team_barrier();

        make_evaluator<TeamPolicyTag2<ES>>(n, ContractOperands{A, B, C},
                                           team)();
        team.team_barrier();

        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          const auto cv = C.storage_;
          for (int x = 0; x < wX; ++x)
            for (int z = 0; z < wZ; ++z)
              for (int l = 0; l < wL; ++l)
                out((x * wZ + z) * wL + l) = cv(x, z, l);
        });
      });
  Kokkos::fence();
}

// The extent static_asserts reject most mis-groupings at compile time, but a
// group whose axes have EQUAL extents can be collapsed in either order and
// still typecheck. That is where a silent transpose lives, so the free group
// here is square: S(a,b,c) with extent(a) == extent(c), contracted over b.
constexpr int sA = 3, sB = 2, sC = 3, sM = 2;

using TS  = StaticTile<sA, sB, sC>;
using TSH = StaticTile<sM, sB>;
using TSC = StaticTile<sM, sA, sC>;

void run_square_free_group(PV3 s, PV2 h, Buf1D out) {
  auto sn = make_input_node(make_handle<'a', 'b', 'c'>(s));
  auto hn = make_input_node(make_handle<'m', 'b'>(h));
  auto n  = make_contraction_node<'m', 'a', 'c'>(hn, sn);

  const std::size_t bytes = scratch_for<TS, TSH, TSC>();
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const team_t& team) {
        auto S = stage_full(sn, TS{}, team);
        auto H = stage_full(hn, TSH{}, team);
        auto B = (make_evaluator<TeamPolicyTag2<ES>>(
                      S, std::integer_sequence<int, 1, 0, 2>{}, team) = S);
        auto C =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, TSC{}));
        team.team_barrier();

        make_evaluator<TeamPolicyTag2<ES>>(n, ContractOperands{H, B, C},
                                           team)();
        team.team_barrier();

        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          const auto cv = C.storage_;
          for (int m = 0; m < sM; ++m)
            for (int a = 0; a < sA; ++a)
              for (int c = 0; c < sC; ++c)
                out((m * sA + a) * sC + c) = cv(m, a, c);
        });
      });
  Kokkos::fence();
}

}  // namespace permuted

TEST(Team2ContractPermuted, SquareFreeGroupKeepsItsAxisOrder) {
  using namespace permuted;
  PV3  s("s", sA, sB, sC);
  PV2  h("h", sM, sB);
  auto sh = Kokkos::create_mirror_view(s);
  auto hh = Kokkos::create_mirror_view(h);
  for (int a = 0; a < sA; ++a)
    for (int b = 0; b < sB; ++b)
      for (int c = 0; c < sC; ++c) sh(a, b, c) = u_val(a, b, c);
  for (int m = 0; m < sM; ++m)
    for (int b = 0; b < sB; ++b) hh(m, b) = h_val(2, m, b);
  Kokkos::deep_copy(s, sh);
  Kokkos::deep_copy(h, hh);

  Buf1D out("out", sM * sA * sC);
  run_square_free_group(s, h, out);
  const auto got = to_host(out);

  // u_val is not symmetric in its first and third arguments, so a group
  // collapsed as (c,a) rather than (a,c) shows up as a transposed C.
  for (int m = 0; m < sM; ++m)
    for (int a = 0; a < sA; ++a)
      for (int c = 0; c < sC; ++c) {
        double acc = 0.0;
        for (int b = 0; b < sB; ++b)
          acc += static_cast<double>(hh(m, b)) * sh(a, b, c);
        EXPECT_NEAR(got[(m * sA + a) * sC + c], static_cast<float>(acc), 1e-4f)
            << "m=" << m << " a=" << a << " c=" << c;
      }
}

TEST(Team2ContractPermuted, ThreeDirectionsShareOneStagedTensor) {
  using namespace permuted;
  PV3  u("u", uP, uQ, uR);
  PV2  h0("h0", hM, uP), h1("h1", hM, uQ), h2("h2", hM, uR);
  auto uh  = Kokkos::create_mirror_view(u);
  auto h0h = Kokkos::create_mirror_view(h0);
  auto h1h = Kokkos::create_mirror_view(h1);
  auto h2h = Kokkos::create_mirror_view(h2);
  for (int p = 0; p < uP; ++p)
    for (int q = 0; q < uQ; ++q)
      for (int r = 0; r < uR; ++r) uh(p, q, r) = u_val(p, q, r);
  for (int m = 0; m < hM; ++m) {
    for (int p = 0; p < uP; ++p) h0h(m, p) = h_val(0, m, p);
    for (int q = 0; q < uQ; ++q) h1h(m, q) = h_val(1, m, q);
    for (int r = 0; r < uR; ++r) h2h(m, r) = h_val(2, m, r);
  }
  Kokkos::deep_copy(u, uh);
  Kokkos::deep_copy(h0, h0h);
  Kokkos::deep_copy(h1, h1h);
  Kokkos::deep_copy(h2, h2h);

  Buf1D o0("o0", hM * uQ * uR), o1("o1", hM * uP * uR), o2("o2", hM * uP * uQ);
  run_three_directions(u, h0, h1, h2, o0, o1, o2);
  const auto g0 = to_host(o0);
  const auto g1 = to_host(o1);
  const auto g2 = to_host(o2);

  for (int m = 0; m < hM; ++m) {
    for (int q = 0; q < uQ; ++q)
      for (int r = 0; r < uR; ++r) {
        double acc = 0.0;
        for (int p = 0; p < uP; ++p)
          acc += static_cast<double>(h0h(m, p)) * uh(p, q, r);
        EXPECT_NEAR(g0[(m * uQ + q) * uR + r], static_cast<float>(acc), 1e-4f)
            << "d=0 m=" << m << " q=" << q << " r=" << r;
      }
    for (int p = 0; p < uP; ++p)
      for (int r = 0; r < uR; ++r) {
        double acc = 0.0;
        for (int q = 0; q < uQ; ++q)
          acc += static_cast<double>(h1h(m, q)) * uh(p, q, r);
        EXPECT_NEAR(g1[(m * uP + p) * uR + r], static_cast<float>(acc), 1e-4f)
            << "d=1 m=" << m << " p=" << p << " r=" << r;
      }
    for (int p = 0; p < uP; ++p)
      for (int q = 0; q < uQ; ++q) {
        double acc = 0.0;
        for (int r = 0; r < uR; ++r)
          acc += static_cast<double>(h2h(m, r)) * uh(p, q, r);
        EXPECT_NEAR(g2[(m * uP + p) * uQ + q], static_cast<float>(acc), 1e-4f)
            << "d=2 m=" << m << " p=" << p << " q=" << q;
      }
  }
}

TEST(Team2ContractPermuted, PermutedAOperandGapsTheRowGroup) {
  using namespace permuted;
  PV3  w("w", wX, wY, wZ);
  PV2  b("b", wY, wL);
  auto wh = Kokkos::create_mirror_view(w);
  auto bh = Kokkos::create_mirror_view(b);
  for (int x = 0; x < wX; ++x)
    for (int y = 0; y < wY; ++y)
      for (int z = 0; z < wZ; ++z) wh(x, y, z) = u_val(x, y, z);
  for (int y = 0; y < wY; ++y)
    for (int l = 0; l < wL; ++l) bh(y, l) = h_val(1, y, l);
  Kokkos::deep_copy(w, wh);
  Kokkos::deep_copy(b, bh);

  Buf1D out("out", wX * wZ * wL);
  run_permuted_a(w, b, out);
  const auto got = to_host(out);

  for (int x = 0; x < wX; ++x)
    for (int z = 0; z < wZ; ++z)
      for (int l = 0; l < wL; ++l) {
        double acc = 0.0;
        for (int y = 0; y < wY; ++y)
          acc += static_cast<double>(wh(x, y, z)) * bh(y, l);
        EXPECT_NEAR(got[(x * wZ + z) * wL + l], static_cast<float>(acc), 1e-4f)
            << "x=" << x << " z=" << z << " l=" << l;
      }
}

// ---------------------------------------------------------------------------
// Tag2 CombineTag — pointwise, N-ary, multi-output
// ---------------------------------------------------------------------------
namespace combine {

using contract::scratch_for;
using contract::stage_full;
using contract::to_host;

using V2 = Kokkos::View<float**, Kokkos::LayoutRight, ES>;
using V3 = Kokkos::View<float***, Kokkos::LayoutRight, ES>;

// Deliberately non-symmetric, non-square, and distinct between operands: an
// all-ones fixture cannot catch a transposed or swapped index.
KOKKOS_INLINE_FUNCTION float p_val(int i, int j) {
  return 0.5f + 0.25f * i - 0.75f * j + 0.125f * i * j;
}
KOKKOS_INLINE_FUNCTION float q_val(int i, int j) {
  return -1.25f + 0.5f * j - 0.375f * i + 0.0625f * i * j;
}
KOKKOS_INLINE_FUNCTION float r_val(int i, int j) {
  return 2.0f - 0.125f * i + 0.25f * j;
}

// Every fn below folds the COORDINATE in, so a wrong coordinate (tile-local
// where global was meant, or i/j swapped) changes the answer.
struct Mix {
  KOKKOS_FUNCTION float operator()(int i, int j, float a, float b) const {
    return a * b + 0.5f * static_cast<float>(i) - static_cast<float>(j);
  }
};
struct SumDiff {
  KOKKOS_FUNCTION Kokkos::Array<float, 2> operator()(int i, int j, float a,
                                                     float b) const {
    return {a + b + static_cast<float>(i), a - b - static_cast<float>(j)};
  }
};
struct Tri {
  KOKKOS_FUNCTION float operator()(int i, int j, float a, float b,
                                   float c) const {
    return a + 2.0f * b - 3.0f * c + static_cast<float>(10 * i + j);
  }
};
struct Mix3 {
  KOKKOS_FUNCTION float operator()(int i, int j, int k, float a,
                                   float b) const {
    return a * b + static_cast<float>(100 * i + 10 * j + k);
  }
};

constexpr int cI = 3, cJ = 5;
using CT  = StaticTile<cI, cJ>;
using CTt = StaticTile<cJ, cI>;

// stage_full's sibling for a tile that is NOT at the origin: the combine's
// global-coordinate contract is only observable off tile {0, ...}.
template <typename Tile, typename Node, typename Team, std::size_t R>
KOKKOS_FUNCTION auto stage_at(Node node, Tile tile, Kokkos::Array<int, R> tidx,
                              const Team& team) {
  auto src = make_evaluator<TeamPolicyTag2<ES>>(node, tile, team)(tidx);
  auto dst = make_evaluator<TeamPolicyTag2<ES>>(
      make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, tile)),
      StageTag{}, team);
  return (dst = src);
}

// --- compile-time surface ---------------------------------------------------

using CScratch = scratch_view_for_t<CT>;
using CNode    = decltype(make_interm_node(std::declval<CScratch>()));
using CNodeH =
    decltype(make_interm_node(std::declval<CScratch>(), AddIndexHook{}));
using CVal  = Impl::value_evaluator_t<CNode>;
using CValH = Impl::value_evaluator_t<CNodeH>;

using CombNode1 = decltype(make_combine_node<'i', 'j'>(
    std::declval<NodeNH>(), std::declval<NodeNH>(), Mix{}));
using CombNode2 = decltype(make_combine_node<'i', 'j'>(
    std::declval<NodeNH>(), std::declval<NodeNH>(), SumDiff{}));

using Ops1 = decltype(make_combine_operands(
    std::declval<CVal>(), std::declval<CVal>(), std::declval<CNode>()));
using Ops2 =
    decltype(make_combine_operands(std::declval<CVal>(), std::declval<CVal>(),
                                   std::declval<Kokkos::Array<CNode, 2>>()));

using Comb1 = Evaluator<TeamPolicyTag2<ES>, CombNode1, Ops1>;
using Comb2 = Evaluator<TeamPolicyTag2<ES>, CombNode2, Ops2>;

// A bare destination node and a one-element array of it are the same request.
static_assert(
    std::is_same_v<Ops1, decltype(make_combine_operands(
                             std::declval<CVal>(), std::declval<CVal>(),
                             std::declval<Kokkos::Array<CNode, 1>>()))>,
    "a bare destination node must normalize to a 1-output pack");

// Both operators return one component per output, exactly like Tag1's combine.
static_assert(
    std::is_same_v<typename Comb1::result_type, Kokkos::Array<CVal, 1>>,
    "a scalar-returning fn is the NumOut == 1 case");
static_assert(
    std::is_same_v<typename Comb2::result_type, Kokkos::Array<CVal, 2>>,
    "a Kokkos::Array-returning fn emits one value evaluator per "
    "component");
static_assert(std::is_same_v<decltype(std::declval<const Comb1&>()(0, 0)),
                             Kokkos::Array<float, 1>>);
static_assert(std::is_same_v<decltype(std::declval<const Comb2&>()(0, 0)),
                             Kokkos::Array<float, 2>>);
static_assert(Comb1::Rank == 2 && Comb1::NumOps == 2 && Comb1::NumOut == 1);
static_assert(Comb2::NumOut == 2);

// --- P2: PermutedAt's compile-time surface ---------------------------------

using CScratchT = scratch_view_for_t<CTt>;
using CNodeT    = decltype(make_interm_node(std::declval<CScratchT>()));
using CValT     = Impl::value_evaluator_t<CNodeT>;
using PermT     = Impl::PermutedAt<Swap, CValT>;

// A plain evaluator reports the identity read permutation, so every existing
// operand keeps today's meaning.
static_assert(!Impl::is_permuted_at_v<CVal>);
static_assert(
    std::is_same_v<Impl::read_perm_t<CVal>, std::integer_sequence<int, 0, 1>>,
    "a plain evaluator's read permutation must be the identity");
static_assert(Impl::is_permuted_at_v<PermT>);
static_assert(std::is_same_v<Impl::read_perm_t<PermT>, Swap>);
static_assert(std::is_same_v<typename PermT::storage_type, CScratchT>,
              "PermutedAt must present the operand's NATIVE storage");

// An identity permutation is not a wrapper: make_permuted_at hands the plain
// evaluator straight back, so the aligned and permuted policies produce the
// SAME type wherever the operand is already in output order.
static_assert(std::is_same_v<
              decltype(Impl::make_permuted_at<std::integer_sequence<int, 0, 1>>(
                  std::declval<CVal>())),
              CVal>);

using CTLayout  = scratch_layout_for_t<CT>;
using CTtLayout = scratch_layout_for_t<CTt>;

// A cI x cJ operand read at Swap does NOT present a cI x cJ output's extents
// (cI != cJ), while the cJ x cI one does. This is the check that a transposed
// operand slipped in unpermuted must fail.
static_assert(Impl::combine_op_aligned_v<2, CTLayout, CVal>);
static_assert(Impl::combine_op_aligned_v<2, CTLayout, PermT>);
static_assert(!Impl::combine_op_aligned_v<2, CTLayout, CValT>,
              "a {j,i}-shaped operand must not pass as aligned to an {i,j} "
              "output");
static_assert(!Impl::combine_op_aligned_v<2, CTtLayout, PermT>,
              "PermutedAt<Swap> over a cJ x cI tile presents cI x cJ, not "
              "cJ x cI");

// The ill-formed-not-false case the if constexpr chain protects: a rank-1
// operand layout against a rank-2 output must answer false. Reached with `&&`
// it would not be false, it would fail to compile -- the operand's read
// permutation is one element long and the output index pack is two, and a fold
// over packs of different lengths is an error, not a value.
using C1     = scratch_view_for_t<StaticTile<cI>>;
using CNode1 = decltype(make_interm_node(std::declval<C1>()));
static_assert(
    !Impl::combine_op_aligned_v<2, CTLayout, Impl::value_evaluator_t<CNode1>>,
    "a short-rank check must short-circuit before any extent is read");

// A hooked destination keeps its hook on the node handed back.
using Ops1H = decltype(make_combine_operands(
    std::declval<CVal>(), std::declval<CVal>(), std::declval<CNodeH>()));
static_assert(std::is_same_v<typename Evaluator<TeamPolicyTag2<ES>, CombNode1,
                                                Ops1H>::result_type,
                             Kokkos::Array<CValH, 1>>,
              "a destination hook must ride forward on the returned node");

// --- runners ----------------------------------------------------------------

void run_binary(V2 a, V2 b, Buf1D team_out, Buf1D scalar_out) {
  auto node = make_combine_node<'i', 'j'>(
      make_input_node(make_handle<'i', 'j'>(a)),
      make_input_node(make_handle<'i', 'j'>(b)), Mix{});
  const std::size_t bytes = scratch_for<CT, CT, CT>();
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const team_t& team) {
        auto A = stage_full(node.operands.get<0>(), CT{}, team);
        auto B = stage_full(node.operands.get<1>(), CT{}, team);
        auto P =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, CT{}));
        auto comb = make_evaluator<TeamPolicyTag2<ES>>(
            node, make_combine_operands(A, B, P), team);
        team.team_barrier();
        auto res = comb();
        team.team_barrier();

        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          const auto pv = res[0].node().storage_;
          for (int i = 0; i < cI; ++i)
            for (int j = 0; j < cJ; ++j) {
              team_out(i * cJ + j)   = pv(i, j);
              scalar_out(i * cJ + j) = comb(i, j)[0];
            }
        });
      });
  Kokkos::fence();
}

// Same combine, but on tile {1, 0} of a 2*cI x cJ input, with the tile's global
// offset handed in. fn must see the GLOBAL row index.
void run_origin(V2 a, V2 b, Buf1D with_origin, Buf1D without_origin) {
  auto node = make_combine_node<'i', 'j'>(
      make_input_node(make_handle<'i', 'j'>(a)),
      make_input_node(make_handle<'i', 'j'>(b)), Mix{});
  const std::size_t bytes = scratch_for<CT, CT, CT, CT>();
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const team_t& team) {
        const Kokkos::Array<int, 2> tidx{1, 0};
        auto A = stage_at(node.operands.get<0>(), CT{}, tidx, team);
        auto B = stage_at(node.operands.get<1>(), CT{}, tidx, team);
        auto P0 =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, CT{}));
        auto P1 =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, CT{}));
        const Kokkos::Array<int, 2> origin{tidx[0] * cI, tidx[1] * cJ};

        auto shifted = make_evaluator<TeamPolicyTag2<ES>>(
            node, make_combine_operands(A, B, P0).at(origin), team);
        auto local = make_evaluator<TeamPolicyTag2<ES>>(
            node, make_combine_operands(A, B, P1), team);
        team.team_barrier();
        auto rs = shifted();
        auto rl = local();
        team.team_barrier();

        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          const auto sv = rs[0].node().storage_;
          const auto lv = rl[0].node().storage_;
          for (int i = 0; i < cI; ++i)
            for (int j = 0; j < cJ; ++j) {
              with_origin(i * cJ + j)    = sv(i, j);
              without_origin(i * cJ + j) = lv(i, j);
            }
        });
      });
  Kokkos::fence();
}

void run_multi_out(V2 a, V2 b, Buf1D out0, Buf1D out1) {
  auto node = make_combine_node<'i', 'j'>(
      make_input_node(make_handle<'i', 'j'>(a)),
      make_input_node(make_handle<'i', 'j'>(b)), SumDiff{});
  const std::size_t bytes = scratch_for<CT, CT, CT, CT>();
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const team_t& team) {
        auto A = stage_full(node.operands.get<0>(), CT{}, team);
        auto B = stage_full(node.operands.get<1>(), CT{}, team);
        auto P0 =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, CT{}));
        auto P1 =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, CT{}));
        const Kokkos::Array<decltype(P0), 2> outs{P0, P1};
        auto res = make_evaluator<TeamPolicyTag2<ES>>(
            node, make_combine_operands(A, B, outs), team)();
        team.team_barrier();

        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          const auto v0 = res[0].node().storage_;
          const auto v1 = res[1].node().storage_;
          for (int i = 0; i < cI; ++i)
            for (int j = 0; j < cJ; ++j) {
              out0(i * cJ + j) = v0(i, j);
              out1(i * cJ + j) = v1(i, j);
            }
        });
      });
  Kokkos::fence();
}

void run_ternary(V2 a, V2 b, V2 c, Buf1D out) {
  auto node = make_combine_node<'i', 'j'>(
      make_input_node(make_handle<'i', 'j'>(a)),
      make_input_node(make_handle<'i', 'j'>(b)),
      make_input_node(make_handle<'i', 'j'>(c)), Tri{});
  const std::size_t bytes = scratch_for<CT, CT, CT, CT>();
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const team_t& team) {
        auto A = stage_full(node.operands.get<0>(), CT{}, team);
        auto B = stage_full(node.operands.get<1>(), CT{}, team);
        auto C = stage_full(node.operands.get<2>(), CT{}, team);
        auto P =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, CT{}));
        auto res = make_evaluator<TeamPolicyTag2<ES>>(
            node, make_combine_operands(A, B, C, P), team)();
        team.team_barrier();

        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          const auto pv = res[0].node().storage_;
          for (int i = 0; i < cI; ++i)
            for (int j = 0; j < cJ; ++j) out(i * cJ + j) = pv(i, j);
        });
      });
  Kokkos::fence();
}

constexpr int gI = 2, gJ = 3, gK = 4;
using GT = StaticTile<gI, gJ, gK>;

void run_rank3(V3 a, V3 b, Buf1D out) {
  auto node = make_combine_node<'i', 'j', 'k'>(
      make_input_node(make_handle<'i', 'j', 'k'>(a)),
      make_input_node(make_handle<'i', 'j', 'k'>(b)), Mix3{});
  const std::size_t bytes = scratch_for<GT, GT, GT>();
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const team_t& team) {
        auto A = stage_full(node.operands.get<0>(), GT{}, team);
        auto B = stage_full(node.operands.get<1>(), GT{}, team);
        auto P =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, GT{}));
        auto res = make_evaluator<TeamPolicyTag2<ES>>(
            node, make_combine_operands(A, B, P), team)();
        team.team_barrier();

        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          const auto pv = res[0].node().storage_;
          for (int i = 0; i < gI; ++i)
            for (int j = 0; j < gJ; ++j)
              for (int k = 0; k < gK; ++k)
                out((i * gJ + j) * gK + k) = pv(i, j, k);
        });
      });
  Kokkos::fence();
}

// Operand B is declared {'j','i'} over a cJ x cI view. Tag2 gathers no axes, so
// the caller relabels and stages it into the output order first.
void run_relabeled(V2 a, V2 bt, Buf1D out) {
  auto node = make_combine_node<'i', 'j'>(
      make_input_node(make_handle<'i', 'j'>(a)),
      make_input_node(make_handle<'j', 'i'>(bt)), Mix{});
  const std::size_t bytes = scratch_for<CT, CT, CT>();
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const team_t& team) {
        auto A = stage_full(node.operands.get<0>(), CT{}, team);

        // subview {j,i} -> relabel to {i,j} -> stage into the output tile.
        auto bsrc = make_evaluator<TeamPolicyTag2<ES>>(
            node.operands.get<1>(), CTt{}, team)(Kokkos::Array<int, 2>{});
        auto brel = make_evaluator<TeamPolicyTag2<ES>>(bsrc, Swap{}, team);
        auto bdst = make_evaluator<TeamPolicyTag2<ES>>(
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, CT{})),
            StageTag{}, team);
        auto B = (bdst = (brel = bsrc));

        auto P =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, CT{}));
        auto res = make_evaluator<TeamPolicyTag2<ES>>(
            node, make_combine_operands(A, B, P), team)();
        team.team_barrier();

        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          const auto pv = res[0].node().storage_;
          for (int i = 0; i < cI; ++i)
            for (int j = 0; j < cJ; ++j) out(i * cJ + j) = pv(i, j);
        });
      });
  Kokkos::fence();
}

// P2: the same {'j','i'} operand, but held in its NATIVE order and read at a
// permuted coordinate instead of relabeled into a strided view. Both arms run
// in one kernel off one staged native tile, so a divergence is the read form
// and nothing else.
void run_permuted_vs_aligned(V2 a, V2 bt, Buf1D aligned, Buf1D permuted) {
  auto node = make_combine_node<'i', 'j'>(
      make_input_node(make_handle<'i', 'j'>(a)),
      make_input_node(make_handle<'j', 'i'>(bt)), Mix{});
  const std::size_t bytes = scratch_for<CT, CTt, CT, CT, CT>();
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const team_t& team) {
        auto A    = stage_full(node.operands.get<0>(), CT{}, team);
        auto Bnat = stage_full(node.operands.get<1>(), CTt{}, team);

        auto Bal = Impl::combine_operand<AlignedOperands, Swap>(Bnat, team);
        auto Bpm = Impl::combine_operand<PermutedOperands, Swap>(Bnat, team);

        static_assert(!Impl::is_permuted_at_v<decltype(Bal)>,
                      "AlignedOperands must keep today's reorder_view operand");
        static_assert(Impl::is_permuted_at_v<decltype(Bpm)>,
                      "PermutedOperands must wrap the native storage");
        static_assert(std::is_same_v<typename decltype(Bpm)::storage_type,
                                     typename decltype(Bnat)::storage_type>,
                      "PermutedAt must hold the operand's NATIVE storage type");

        auto P0 =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, CT{}));
        auto P1 =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, CT{}));
        auto ra = make_evaluator<TeamPolicyTag2<ES>>(
            node, make_combine_operands(A, Bal, P0), team)();
        auto rp = make_evaluator<TeamPolicyTag2<ES>>(
            node, make_combine_operands(A, Bpm, P1), team)();
        team.team_barrier();

        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          const auto av = ra[0].node().storage_;
          const auto pv = rp[0].node().storage_;
          for (int i = 0; i < cI; ++i)
            for (int j = 0; j < cJ; ++j) {
              aligned(i * cJ + j)  = av(i, j);
              permuted(i * cJ + j) = pv(i, j);
            }
        });
      });
  Kokkos::fence();
}

// Rank 2 cannot pin the SCATTER DIRECTION: <1,0> is its own inverse, so
// permuting where the inverse belongs reads the same element. This is the same
// A/B at rank 3 with <1,2,0>, whose inverse <2,0,1> differs, over three
// distinct extents.
void run_permuted3_vs_aligned(V3 a, V3 bp, Buf1D aligned, Buf1D permuted) {
  auto node = make_combine_node<'i', 'j', 'k'>(
      make_input_node(make_handle<'i', 'j', 'k'>(a)),
      make_input_node(make_handle<'k', 'i', 'j'>(bp)), Mix3{});
  using GTn = StaticTile<gK, gI, gJ>;
  using Rot =
      Impl::label_perm_seq_t<std::integer_sequence<int32_t, 'i', 'j', 'k'>,
                             std::integer_sequence<int32_t, 'k', 'i', 'j'>>;
  static_assert(std::is_same_v<Rot, std::integer_sequence<int, 1, 2, 0>>);

  const std::size_t bytes = scratch_for<GT, GTn, GT, GT>();
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const team_t& team) {
        auto A    = stage_full(node.operands.get<0>(), GT{}, team);
        auto Bnat = stage_full(node.operands.get<1>(), GTn{}, team);

        auto Bal = Impl::combine_operand<AlignedOperands, Rot>(Bnat, team);
        auto Bpm = Impl::combine_operand<PermutedOperands, Rot>(Bnat, team);

        auto P0 =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, GT{}));
        auto P1 =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, GT{}));
        auto ra = make_evaluator<TeamPolicyTag2<ES>>(
            node, make_combine_operands(A, Bal, P0), team)();
        auto rp = make_evaluator<TeamPolicyTag2<ES>>(
            node, make_combine_operands(A, Bpm, P1), team)();
        team.team_barrier();

        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          const auto av = ra[0].node().storage_;
          const auto pv = rp[0].node().storage_;
          for (int i = 0; i < gI; ++i)
            for (int j = 0; j < gJ; ++j)
              for (int k = 0; k < gK; ++k) {
                aligned((i * gJ + j) * gK + k)  = av(i, j, k);
                permuted((i * gJ + j) * gK + k) = pv(i, j, k);
              }
        });
      });
  Kokkos::fence();
}

void run_hooked_dest(V2 a, V2 b, Buf1D out) {
  auto node = make_combine_node<'i', 'j'>(
      make_input_node(make_handle<'i', 'j'>(a)),
      make_input_node(make_handle<'i', 'j'>(b)), Mix{});
  const std::size_t bytes = scratch_for<CT, CT, CT>();
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const team_t& team) {
        auto A = stage_full(node.operands.get<0>(), CT{}, team);
        auto B = stage_full(node.operands.get<1>(), CT{}, team);
        auto P = make_interm_node(
            Impl::alloc_scratch_tile<float, ES>(team, CT{}), AddIndexHook{});
        auto res = make_evaluator<TeamPolicyTag2<ES>>(
            node, make_combine_operands(A, B, P), team)();
        team.team_barrier();
        static_assert(
            std::is_same_v<decltype(res[0].node().hook_op), AddIndexHook>,
            "the destination hook must survive on the returned node");

        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          const auto pv = res[0].node().storage_;
          for (int i = 0; i < cI; ++i)
            for (int j = 0; j < cJ; ++j) out(i * cJ + j) = pv(i, j);
        });
      });
  Kokkos::fence();
}

// Fill a rank-2 device view from a host callable, returning the host mirror.
template <typename V, typename F>
auto fill2(V v, F f) {
  auto h = Kokkos::create_mirror_view(v);
  for (std::size_t i = 0; i < v.extent(0); ++i)
    for (std::size_t j = 0; j < v.extent(1); ++j)
      h(i, j) = f(static_cast<int>(i), static_cast<int>(j));
  Kokkos::deep_copy(v, h);
  return h;
}

// P1: the multi-output combine driven two ways -- operator()() versus a flat
// caller-written range that builds its own coordinate and calls the element
// operator()(coord). The second is the fused level driver's shape.
void run_element_vs_driver_multi_out(V2 a, V2 b, Buf1D d0, Buf1D d1, Buf1D m0,
                                     Buf1D m1) {
  auto node = make_combine_node<'i', 'j'>(
      make_input_node(make_handle<'i', 'j'>(a)),
      make_input_node(make_handle<'i', 'j'>(b)), SumDiff{});
  const std::size_t bytes = scratch_for<CT, CT, CT, CT, CT, CT>();
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const team_t& team) {
        auto A = stage_full(node.operands.get<0>(), CT{}, team);
        auto B = stage_full(node.operands.get<1>(), CT{}, team);
        auto D0 =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, CT{}));
        auto D1 =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, CT{}));
        auto M0 =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, CT{}));
        auto M1 =
            make_interm_node(Impl::alloc_scratch_tile<float, ES>(team, CT{}));
        const Kokkos::Array<decltype(D0), 2> douts{D0, D1};
        const Kokkos::Array<decltype(M0), 2> mouts{M0, M1};

        auto driver = make_evaluator<TeamPolicyTag2<ES>>(
            node, make_combine_operands(A, B, douts), team);
        auto manual = make_evaluator<TeamPolicyTag2<ES>>(
            node, make_combine_operands(A, B, mouts), team);
        team.team_barrier();

        auto res = driver();

        const auto self = manual;
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team, cI * cJ),
            [=](int t) { self(Impl::Index<2>{t / cJ, t % cJ}); });
        team.team_barrier();

        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          const auto dv0 = res[0].node().storage_;
          const auto dv1 = res[1].node().storage_;
          const auto mv0 = M0.storage_;
          const auto mv1 = M1.storage_;
          for (int i = 0; i < cI; ++i)
            for (int j = 0; j < cJ; ++j) {
              d0(i * cJ + j) = dv0(i, j);
              d1(i * cJ + j) = dv1(i, j);
              m0(i * cJ + j) = mv0(i, j);
              m1(i * cJ + j) = mv1(i, j);
            }
        });
      });
  Kokkos::fence();
}

}  // namespace combine

TEST(Team2Combine, ResultTypesAreAsExpected) { SUCCEED(); }

TEST(Team2Combine, PointwiseMatchesHostReference) {
  using namespace combine;
  V2   a("a", cI, cJ), b("b", cI, cJ);
  auto ah = fill2(a, p_val);
  auto bh = fill2(b, q_val);

  Buf1D team_out("team", cI * cJ), scalar_out("scalar", cI * cJ);
  run_binary(a, b, team_out, scalar_out);
  const auto got    = to_host(team_out);
  const auto scalar = to_host(scalar_out);

  for (int i = 0; i < cI; ++i)
    for (int j = 0; j < cJ; ++j) {
      const float want = ah(i, j) * bh(i, j) + 0.5f * i - static_cast<float>(j);
      EXPECT_NEAR(got[i * cJ + j], want, 1e-5f) << "i=" << i << " j=" << j;
      // The two operators must agree: operator()() is only a parallel wrapper
      // around the per-element operator.
      EXPECT_FLOAT_EQ(scalar[i * cJ + j], got[i * cJ + j])
          << "i=" << i << " j=" << j;
    }
}

TEST(Team2Combine, GlobalOriginIsVisibleToFn) {
  using namespace combine;
  V2   a("a", 2 * cI, cJ), b("b", 2 * cI, cJ);
  auto ah = fill2(a, p_val);
  auto bh = fill2(b, q_val);

  Buf1D shifted("shifted", cI * cJ), local("local", cI * cJ);
  run_origin(a, b, shifted, local);
  const auto got_g = to_host(shifted);
  const auto got_l = to_host(local);

  for (int i = 0; i < cI; ++i)
    for (int j = 0; j < cJ; ++j) {
      const int   gi   = cI + i;  // tile {1, 0}
      const float prod = ah(gi, j) * bh(gi, j);
      EXPECT_NEAR(got_g[i * cJ + j], prod + 0.5f * gi - static_cast<float>(j),
                  1e-5f)
          << "i=" << i << " j=" << j;
      // Without the origin, fn sees the tile-local row instead.
      EXPECT_NEAR(got_l[i * cJ + j], prod + 0.5f * i - static_cast<float>(j),
                  1e-5f)
          << "i=" << i << " j=" << j;
    }
  // The two must actually differ, or the test proves nothing.
  EXPECT_GT(std::abs(got_g[0] - got_l[0]), 0.0f);
}

TEST(Team2Combine, MultiOutputWritesEveryTile) {
  using namespace combine;
  V2   a("a", cI, cJ), b("b", cI, cJ);
  auto ah = fill2(a, p_val);
  auto bh = fill2(b, q_val);

  Buf1D o0("o0", cI * cJ), o1("o1", cI * cJ);
  run_multi_out(a, b, o0, o1);
  const auto g0 = to_host(o0);
  const auto g1 = to_host(o1);

  for (int i = 0; i < cI; ++i)
    for (int j = 0; j < cJ; ++j) {
      EXPECT_NEAR(g0[i * cJ + j], ah(i, j) + bh(i, j) + static_cast<float>(i),
                  1e-5f)
          << "i=" << i << " j=" << j;
      EXPECT_NEAR(g1[i * cJ + j], ah(i, j) - bh(i, j) - static_cast<float>(j),
                  1e-5f)
          << "i=" << i << " j=" << j;
    }
}

TEST(Team2Combine, ThreeOperandCombine) {
  using namespace combine;
  V2   a("a", cI, cJ), b("b", cI, cJ), c("c", cI, cJ);
  auto ah = fill2(a, p_val);
  auto bh = fill2(b, q_val);
  auto ch = fill2(c, r_val);

  Buf1D out("out", cI * cJ);
  run_ternary(a, b, c, out);
  const auto got = to_host(out);

  for (int i = 0; i < cI; ++i)
    for (int j = 0; j < cJ; ++j)
      EXPECT_NEAR(got[i * cJ + j],
                  ah(i, j) + 2.0f * bh(i, j) - 3.0f * ch(i, j) +
                      static_cast<float>(10 * i + j),
                  1e-5f)
          << "i=" << i << " j=" << j;
}

TEST(Team2Combine, Rank3Combine) {
  using namespace combine;
  V3   a("a", gI, gJ, gK), b("b", gI, gJ, gK);
  auto ah = Kokkos::create_mirror_view(a);
  auto bh = Kokkos::create_mirror_view(b);
  for (int i = 0; i < gI; ++i)
    for (int j = 0; j < gJ; ++j)
      for (int k = 0; k < gK; ++k) {
        ah(i, j, k) = 0.5f + 0.25f * i - 0.125f * j + 0.75f * k;
        bh(i, j, k) = -1.0f + 0.375f * i + 0.5f * j - 0.25f * k;
      }
  Kokkos::deep_copy(a, ah);
  Kokkos::deep_copy(b, bh);

  Buf1D out("out", gI * gJ * gK);
  run_rank3(a, b, out);
  const auto got = to_host(out);

  for (int i = 0; i < gI; ++i)
    for (int j = 0; j < gJ; ++j)
      for (int k = 0; k < gK; ++k)
        EXPECT_NEAR(got[(i * gJ + j) * gK + k],
                    ah(i, j, k) * bh(i, j, k) +
                        static_cast<float>(100 * i + 10 * j + k),
                    1e-4f)
            << "i=" << i << " j=" << j << " k=" << k;
}

TEST(Team2Combine, RelabeledOperandAligns) {
  using namespace combine;
  V2   a("a", cI, cJ), bt("bt", cJ, cI);
  auto ah = fill2(a, p_val);
  // bt is stored {j, i}: bt(j, i) is the operand's value at output (i, j).
  auto bth = fill2(bt, [](int j, int i) { return q_val(i, j); });

  Buf1D out("out", cI * cJ);
  run_relabeled(a, bt, out);
  const auto got = to_host(out);

  for (int i = 0; i < cI; ++i)
    for (int j = 0; j < cJ; ++j)
      EXPECT_NEAR(got[i * cJ + j],
                  ah(i, j) * bth(j, i) + 0.5f * i - static_cast<float>(j),
                  1e-5f)
          << "i=" << i << " j=" << j;
}

TEST(Team2Combine, PermutedOperandMatchesAligned) {
  using namespace combine;
  V2   a("a", cI, cJ), bt("bt", cJ, cI);
  auto ah = fill2(a, p_val);
  // bt is stored {j, i}: bt(j, i) is the operand's value at output (i, j).
  auto bth = fill2(bt, [](int j, int i) { return q_val(i, j); });

  Buf1D aligned("aligned", cI * cJ), permuted("permuted", cI * cJ);
  run_permuted_vs_aligned(a, bt, aligned, permuted);
  const auto ga = to_host(aligned);
  const auto gp = to_host(permuted);

  for (int i = 0; i < cI; ++i)
    for (int j = 0; j < cJ; ++j) {
      const float want =
          ah(i, j) * bth(j, i) + 0.5f * i - static_cast<float>(j);
      EXPECT_NEAR(ga[i * cJ + j], want, 1e-5f) << "i=" << i << " j=" << j;
      EXPECT_EQ(gp[i * cJ + j], ga[i * cJ + j]) << "i=" << i << " j=" << j;
    }
}

TEST(Team2Combine, PermutedOperandMatchesAlignedRank3) {
  using namespace combine;
  V3   a("a", gI, gJ, gK), bp("bp", gK, gI, gJ);
  auto ah = Kokkos::create_mirror_view(a);
  auto bh = Kokkos::create_mirror_view(bp);
  for (int i = 0; i < gI; ++i)
    for (int j = 0; j < gJ; ++j)
      for (int k = 0; k < gK; ++k) {
        ah(i, j, k) = 0.5f + 0.25f * i - 0.125f * j + 0.75f * k;
        // bp is stored {k, i, j}: bp(k, i, j) is the value at output (i, j, k).
        bh(k, i, j) =
            -1.0f + 0.375f * i + 0.5f * j - 0.25f * k + 0.0625f * i * k;
      }
  Kokkos::deep_copy(a, ah);
  Kokkos::deep_copy(bp, bh);

  Buf1D aligned("aligned", gI * gJ * gK), permuted("permuted", gI * gJ * gK);
  run_permuted3_vs_aligned(a, bp, aligned, permuted);
  const auto ga = to_host(aligned);
  const auto gp = to_host(permuted);

  for (int i = 0; i < gI; ++i)
    for (int j = 0; j < gJ; ++j)
      for (int k = 0; k < gK; ++k) {
        const int   f    = (i * gJ + j) * gK + k;
        const float want = ah(i, j, k) * bh(k, i, j) +
                           static_cast<float>(100 * i + 10 * j + k);
        EXPECT_NEAR(ga[f], want, 1e-4f)
            << "i=" << i << " j=" << j << " k=" << k;
        EXPECT_EQ(gp[f], ga[f]) << "i=" << i << " j=" << j << " k=" << k;
      }
}

TEST(Team2Combine, DestHookRidesForwardUnapplied) {
  using namespace combine;
  V2   a("a", cI, cJ), b("b", cI, cJ);
  auto ah = fill2(a, p_val);
  auto bh = fill2(b, q_val);

  Buf1D out("out", cI * cJ);
  run_hooked_dest(a, b, out);
  const auto got = to_host(out);

  // AddIndexHook would add 1000*(i+1) + (j+1). The scratch must hold the RAW
  // combine; the hook rides forward on the returned node for a later step.
  for (int i = 0; i < cI; ++i)
    for (int j = 0; j < cJ; ++j)
      EXPECT_NEAR(got[i * cJ + j],
                  ah(i, j) * bh(i, j) + 0.5f * i - static_cast<float>(j), 1e-5f)
          << "i=" << i << " j=" << j;
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}

TEST(Team2Combine, ElementOperatorMatchesDriver) {
  using namespace combine;
  V2   a("a", cI, cJ), b("b", cI, cJ);
  auto ah = Kokkos::create_mirror_view(a);
  auto bh = Kokkos::create_mirror_view(b);
  for (int i = 0; i < cI; ++i)
    for (int j = 0; j < cJ; ++j) {
      ah(i, j) = p_val(i, j);
      bh(i, j) = q_val(i, j);
    }
  Kokkos::deep_copy(a, ah);
  Kokkos::deep_copy(b, bh);

  Buf1D d0("d0", cI * cJ), d1("d1", cI * cJ);
  Buf1D m0("m0", cI * cJ), m1("m1", cI * cJ);
  run_element_vs_driver_multi_out(a, b, d0, d1, m0, m1);
  const auto hd0 = to_host(d0), hd1 = to_host(d1);
  const auto hm0 = to_host(m0), hm1 = to_host(m1);

  for (int i = 0; i < cI; ++i)
    for (int j = 0; j < cJ; ++j) {
      const int   n  = i * cJ + j;
      const float e0 = ah(i, j) + bh(i, j) + static_cast<float>(i);
      const float e1 = ah(i, j) - bh(i, j) - static_cast<float>(j);
      EXPECT_FLOAT_EQ(hd0[n], e0) << "i=" << i << " j=" << j;
      EXPECT_FLOAT_EQ(hd1[n], e1) << "i=" << i << " j=" << j;
      EXPECT_FLOAT_EQ(hm0[n], hd0[n]) << "i=" << i << " j=" << j;
      EXPECT_FLOAT_EQ(hm1[n], hd1[n]) << "i=" << i << " j=" << j;
    }
}
