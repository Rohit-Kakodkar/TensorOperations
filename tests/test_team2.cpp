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
void run_matmul(V2 a, V2 b, Buf1D team_out, Buf1D scalar_out, Hook hook) {
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
            for (int l = 0; l < kL; ++l) {
              team_out(i * kL + l)   = cv(i, l);
              scalar_out(i * kL + l) = gemm(i, l);
            }
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

  Buf1D team_out("team", kI * kL), scalar_out("scalar", kI * kL);
  run_matmul(a, b, team_out, scalar_out, NoHook{});
  const auto got    = to_host(team_out);
  const auto scalar = to_host(scalar_out);

  for (int i = 0; i < kI; ++i)
    for (int l = 0; l < kL; ++l) {
      double acc = 0.0;
      for (int k = 0; k < kK; ++k)
        acc += static_cast<double>(ah(i, k)) * bh(k, l);
      EXPECT_NEAR(got[i * kL + l], static_cast<float>(acc), 1e-4f)
          << "i=" << i << " l=" << l;
      // The two operators must agree: operator()() is only a parallel wrapper
      // around operator()(i, j).
      EXPECT_FLOAT_EQ(scalar[i * kL + l], got[i * kL + l])
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

  Buf1D team_out("team", kI * kL), scalar_out("scalar", kI * kL);
  run_matmul(a, b, team_out, scalar_out, ScaleC{});
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
