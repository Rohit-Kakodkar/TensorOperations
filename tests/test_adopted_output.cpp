// ===========================================================================
// test_adopted_output.cpp — writing into a buffer the caller carved.
//
// Stage 3 of the DAG evaluator. A producer evaluator normally carves its output
// tile from the team cursor in its own constructor, which makes the buffer's
// identity a function of CONSTRUCTION ORDER. A shared graph cannot work that
// way: a node's result has to outlive its evaluator so other nodes can name it,
// which means the DRIVER must own the buffers and hand each evaluator its own.
//
// The bar is bit-identity. Adoption changes who allocated the memory and
// nothing else, so an adopting evaluator must produce exactly the bits a
// carving one does -- not merely agree within tolerance, which would leave room
// for it to be reading or writing something subtly different.
//
// Data is non-symmetric with mutually distinct fills, for the reason spelled
// out in test_slot_node.cpp: an all-ones fill cannot tell a correct contraction
// from one that transposed an index.
// ===========================================================================
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/Graph.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

using namespace TensorOperations;
using ES     = Kokkos::DefaultExecutionSpace;
using team_t = typename Kokkos::TeamPolicy<ES>::member_type;

namespace {

// SA = I = 16, SK = K = 8, SB = L = 32: divides the register GEMM's block
// factors on both backends (GPU MT=4/NT=2/NR=2, CPU MT=NT=8/NR=2*simd_width).
constexpr int kI = 16, kK = 8, kL = 32;

using ViewH = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace>;
using View2 = Kokkos::View<float**, Kokkos::LayoutRight, ES>;

using TileIK = StaticTile<kI, kK>;
using TileKL = StaticTile<kK, kL>;
using TileIL = StaticTile<kI, kL>;
using TileC  = Tile<TileIK, TileKL, TileIL>;

using NodeA =
    decltype(make_input_node(make_handle<'i', 'k'>(std::declval<View2>())));
using NodeB =
    decltype(make_input_node(make_handle<'k', 'l'>(std::declval<View2>())));
using NodeC = decltype(make_contraction_node<'i', 'l'>(std::declval<NodeA>(),
                                                       std::declval<NodeB>()));
using EvalC = Evaluator<TeamPolicyTag<ES>, NodeC, TileC>;

float a_val(int i, int k) {
  return 0.75f + 0.25f * i - 0.5f * k + 0.125f * i * k;
}
float b_val(int k, int l) {
  return -1.5f + 0.5f * k + 0.75f * l - 0.0625f * k * l;
}

View2 filled_a() {
  ViewH h("a", kI, kK);
  for (int i = 0; i < kI; ++i)
    for (int k = 0; k < kK; ++k) h(i, k) = a_val(i, k);
  View2 d("ad", kI, kK);
  Kokkos::deep_copy(d, h);
  return d;
}
View2 filled_b() {
  ViewH h("b", kK, kL);
  for (int k = 0; k < kK; ++k)
    for (int l = 0; l < kL; ++l) h(k, l) = b_val(k, l);
  View2 d("bd", kK, kL);
  Kokkos::deep_copy(d, h);
  return d;
}

KOKKOS_INLINE_FUNCTION Kokkos::Array<int, 2> zero_idx() { return {0, 0}; }

// CARVING: the evaluator takes its output from the team cursor, as it always
// has. Charges scratch_size_per_team().
void run_carving(View2 a, View2 b, View2 out) {
  const auto node = make_contraction_node<'i', 'l'>(
      make_input_node(make_handle<'i', 'k'>(a)),
      make_input_node(make_handle<'k', 'l'>(b)));
  const std::size_t bytes = EvalC::scratch_size_per_team(TileC{});

  Kokkos::parallel_for(
      "carving",
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(static_cast<int>(bytes))),
      KOKKOS_LAMBDA(const team_t& team) {
        auto eval = make_evaluator<TeamPolicyTag<ES>>(node, TileC{}, team);
        const auto interm = eval(team, zero_idx());
        auto       store  = make_evaluator<TeamPolicyTag<ES>>(interm, TileIL{});
        store(team, zero_idx(), out, Impl::output_perm_seq<NodeC>());
      });
  Kokkos::fence();
}

// ADOPTING: the caller carves the output first and hands it in. Charges
// operand_scratch_size_per_team() ON TOP of that buffer -- asking for
// scratch_size_per_team() here would reserve the output twice.
void run_adopting(View2 a, View2 b, View2 out) {
  const auto node = make_contraction_node<'i', 'l'>(
      make_input_node(make_handle<'i', 'k'>(a)),
      make_input_node(make_handle<'k', 'l'>(b)));
  const std::size_t bytes = Impl::scratch_tile_bytes<float, ES>(TileIL{}) +
                            EvalC::operand_scratch_size_per_team(TileC{});

  Kokkos::parallel_for(
      "adopting",
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(static_cast<int>(bytes))),
      KOKKOS_LAMBDA(const team_t& team) {
        // The driver's job, done by hand here: carve the output up front, so it
        // belongs to the caller rather than to the evaluator.
        auto out_buf = Impl::alloc_scratch_tile<float, ES>(team, TileIL{});
        auto eval =
            make_evaluator<TeamPolicyTag<ES>>(node, TileC{}, team, out_buf);
        const auto interm = eval(team, zero_idx());
        auto       store  = make_evaluator<TeamPolicyTag<ES>>(interm, TileIL{});
        store(team, zero_idx(), out, Impl::output_perm_seq<NodeC>());
      });
  Kokkos::fence();
}

// A 2-output combine, P{i,k} = [s + x, s - x], over two input operands.
struct TwoOut {
  KOKKOS_FUNCTION Kokkos::Array<float, 2> operator()(int, int, float s,
                                                     float x) const {
    return {s + x, s - x};
  }
};

using NodeIK =
    decltype(make_input_node(make_handle<'i', 'k'>(std::declval<View2>())));
using NodeP = decltype(make_combine_node<'i', 'k'>(
    std::declval<NodeIK>(), std::declval<NodeIK>(), TwoOut{}));
using EvalP = Evaluator<TeamPolicyTag<ES>, NodeP, TileIK>;

// The multi-output case: a combine owns NumOut distinct tiles, so adoption
// hands it an ARRAY of buffers. Writes both outputs so neither slot can be
// silently aliased to the other.
void run_combine_adopting(View2 s, View2 x, View2 out0, View2 out1) {
  const auto node = make_combine_node<'i', 'k'>(
      make_input_node(make_handle<'i', 'k'>(s)),
      make_input_node(make_handle<'i', 'k'>(x)), TwoOut{});
  static_assert(EvalP::NumOut == 2);

  const std::size_t bytes = 2 * Impl::scratch_tile_bytes<float, ES>(TileIK{}) +
                            EvalP::operand_scratch_size_per_team(TileIK{});

  Kokkos::parallel_for(
      "combine_adopting",
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(static_cast<int>(bytes))),
      KOKKOS_LAMBDA(const team_t& team) {
        Kokkos::Array<typename EvalP::scratch_view_t, 2> bufs{
            Impl::alloc_scratch_tile<float, ES>(team, TileIK{}),
            Impl::alloc_scratch_tile<float, ES>(team, TileIK{})};
        auto eval =
            make_evaluator<TeamPolicyTag<ES>>(node, TileIK{}, team, bufs);
        const auto interms = eval(team, zero_idx());

        auto st0 = make_evaluator<TeamPolicyTag<ES>>(interms[0], TileIK{});
        st0(team, zero_idx(), out0, Impl::output_perm_seq<NodeP>());
        auto st1 = make_evaluator<TeamPolicyTag<ES>>(interms[1], TileIK{});
        st1(team, zero_idx(), out1, Impl::output_perm_seq<NodeP>());
      });
  Kokkos::fence();
}

ViewH to_host(const View2& d) {
  ViewH h("h", d.extent(0), d.extent(1));
  Kokkos::deep_copy(h, d);
  return h;
}

}  // namespace

// The sizing invariant, host-side and free. Stated as an equality rather than
// as two independent numbers because the evaluators define
// scratch_size_per_team AS output + operands -- this asserts that the
// definition has not been un-factored into a third parallel sum.
TEST(AdoptedOutputTest, OperandSizePlusOutputEqualsTotal) {
  // Locals rather than inline calls: `scratch_tile_bytes<float, ES>` carries a
  // comma inside its template brackets, which the preprocessor would read as a
  // macro argument separator.
  const std::size_t c_operands = EvalC::operand_scratch_size_per_team(TileC{});
  const std::size_t c_total    = EvalC::scratch_size_per_team(TileC{});
  const std::size_t c_out      = Impl::scratch_tile_bytes<float, ES>(TileIL{});
  EXPECT_EQ(c_operands + c_out, c_total);

  // And the adopting form really is cheaper from the team's cursor: the whole
  // point is that the driver, not the evaluator, pays for the output.
  EXPECT_LT(c_operands, c_total);

  // Multi-output: the combine charges NumOut tiles, so the gap is 2 tiles.
  const std::size_t p_operands = EvalP::operand_scratch_size_per_team(TileIK{});
  const std::size_t p_total    = EvalP::scratch_size_per_team(TileIK{});
  const std::size_t p_out      = Impl::scratch_tile_bytes<float, ES>(TileIK{});
  EXPECT_EQ(p_operands + 2 * p_out, p_total);
}

// Adoption is a constructor choice, not a type-level one: both forms must yield
// the SAME evaluator type, or every downstream consumer would have to know
// which constructor ran.
TEST(AdoptedOutputTest, AdoptionIsNotVisibleInTheType) {
  static_assert(std::is_constructible_v<EvalC, NodeC, TileC, team_t>);
  static_assert(std::is_constructible_v<EvalC, NodeC, TileC, team_t,
                                        typename EvalC::scratch_view_t>);
  // The buffer a caller must carve to adopt is exactly the one the evaluator
  // would have carved for itself.
  static_assert(
      std::is_same_v<typename EvalC::scratch_view_t,
                     decltype(Impl::alloc_scratch_tile<float, ES>(
                         std::declval<team_t>(), std::declval<TileIL>()))>);
}

// The decisive test.
TEST(AdoptedOutputTest, AdoptedMatchesCarvedBitForBit) {
  const auto a = filled_a();
  const auto b = filled_b();

  View2 out_carved("out_carved", kI, kL);
  View2 out_adopted("out_adopted", kI, kL);

  run_carving(a, b, out_carved);
  run_adopting(a, b, out_adopted);
  Kokkos::fence();

  const auto hc = to_host(out_carved);
  const auto ha = to_host(out_adopted);

  // Non-triviality: a kernel that silently wrote nothing would otherwise pass
  // this test twice over.
  double mag = 0.0;
  for (int i = 0; i < kI; ++i)
    for (int l = 0; l < kL; ++l) mag += std::abs(static_cast<double>(hc(i, l)));
  EXPECT_GT(mag, 1.0);

  for (int i = 0; i < kI; ++i)
    for (int l = 0; l < kL; ++l)
      EXPECT_EQ(hc(i, l), ha(i, l)) << "at (" << i << "," << l << ")";
}

// Multi-output adoption, against a closed-form reference: the two outputs must
// be distinct buffers, which s+x and s-x would expose if they were aliased.
TEST(AdoptedOutputTest, MultiOutputCombineAdoptsOneBufferPerOutput) {
  ViewH sh("s", kI, kK), xh("x", kI, kK);
  for (int i = 0; i < kI; ++i)
    for (int k = 0; k < kK; ++k) {
      sh(i, k) = a_val(i, k);
      xh(i, k) = b_val(k, i % kK) + 0.5f * i;
    }
  View2 s("sd", kI, kK), x("xd", kI, kK);
  Kokkos::deep_copy(s, sh);
  Kokkos::deep_copy(x, xh);

  View2 out0("out0", kI, kK), out1("out1", kI, kK);
  run_combine_adopting(s, x, out0, out1);
  Kokkos::fence();

  const auto h0 = to_host(out0);
  const auto h1 = to_host(out1);
  for (int i = 0; i < kI; ++i)
    for (int k = 0; k < kK; ++k) {
      EXPECT_FLOAT_EQ(h0(i, k), sh(i, k) + xh(i, k)) << i << "," << k;
      EXPECT_FLOAT_EQ(h1(i, k), sh(i, k) - xh(i, k)) << i << "," << k;
    }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
