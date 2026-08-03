// ===========================================================================
// test_slot_node.cpp — SlotTag: naming a buffer another node owns.
//
// Stage 1 of the DAG evaluator. A slot node lets a consumer NAME its operand
// instead of nesting that operand's whole subtree by value, which is what makes
// a shared subtree evaluate once rather than once per consumer.
//
// There is no DAG driver yet (that is Stage 5), so these tests hand-roll the
// kernel: carve a scratch tile, fill it, wrap it in a slot node, build the
// consumer's evaluator around that slot, run it, and compare against the SAME
// contraction spelled with the operand as an ordinary input node. Bit-identical
// is the bar -- a slot changes who owns the buffer, and nothing else.
//
// DATA. Every tensor here is non-symmetric and non-square-in-content, and no
// two of them share a fill. The team contraction tests grew up using A=B=1.0,
// which cannot distinguish a correct contraction from one that transposed an
// index or summed the wrong axis (see the all-ones blindspot); this pipeline is
// nothing but transposed indices, so an all-ones fill here would assert almost
// nothing.
// ===========================================================================
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/Graph.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <vector>

using namespace TensorOperations;
using ES     = Kokkos::DefaultExecutionSpace;
using team_t = typename Kokkos::TeamPolicy<ES>::member_type;

namespace {

// Shapes chosen for the register GEMM's block factors on BOTH backends:
// SA = I = 16, SK = K = 8, SB = L = 32. GPU wants MT=4/NT=2/NR=2, CPU wants
// MT=NT=8 and NR=2*simd_width (32 on AVX-512); all three dims divide on both.
constexpr int kI = 16, kK = 8, kL = 32;

using ViewH = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace>;
using View2 = Kokkos::View<float**, Kokkos::LayoutRight, ES>;

using TileIK = StaticTile<kI, kK>;
using TileKL = StaticTile<kK, kL>;
using TileIL = StaticTile<kI, kL>;

// The scratch view type a slot over an [I,K] tile carries. This is the claim
// the whole design rests on: a slot's storage type is a function of
// (value_type, exec space, tile) ALONE -- the node that produced it does not
// enter. If this ever stops holding, a slot can no longer be typed without
// recursing into its producer, and the flat node list collapses back into a
// left fold.
using SlotViewIK = decltype(Impl::alloc_scratch_tile<float, ES>(
    std::declval<team_t>(), std::declval<TileIK>()));

// The same view type, reached instead through a producing evaluator's own
// output allocator. Equality of the two is the load-bearing static_assert.
using ProducerCOut =
    typename ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, IntermTag,
                              float, TileIK>::scratch_view_t;
static_assert(std::is_same_v<SlotViewIK, ProducerCOut>,
              "a pre-carved slot buffer must be type-identical to the buffer a "
              "producer hands over, or Specialization 8's zero-copy "
              "passthrough will not fire and the DAG design does not hold");

using SlotIK = decltype(make_slot_node<0, 'i', 'k'>(
    std::declval<SlotViewIK>(), std::declval<Kokkos::Array<int, 2>>()));

// Does this node carry a hook? Spelled as a concept rather than a bare
// requires-expression at the use site so the check sits in a dependent -- hence
// SFINAE-friendly -- context: probing for a member that genuinely does not
// exist is a HARD error on a concrete type, not a false. Same reason
// test_scratch_allocator.cpp spells its HasStage probe this way.
template <typename N>
concept HasHook = requires(const N& n) { n.hook_op; };

// Deterministic, index-dependent, mutually distinct fills.
float a_val(int i, int k) {
  return 1.f + 0.25f * i - 0.5f * k + 0.125f * i * k;
}
float b_val(int k, int l) {
  return -2.f + 0.5f * k + 0.75f * l - 0.0625f * k * l;
}

ViewH host_a() {
  ViewH a("a", kI, kK);
  for (int i = 0; i < kI; ++i)
    for (int k = 0; k < kK; ++k) a(i, k) = a_val(i, k);
  return a;
}
ViewH host_b() {
  ViewH b("b", kK, kL);
  for (int k = 0; k < kK; ++k)
    for (int l = 0; l < kL; ++l) b(k, l) = b_val(k, l);
  return b;
}

View2 device_copy(const ViewH& h) {
  View2 d("d", h.extent(0), h.extent(1));
  Kokkos::deep_copy(d, h);
  return d;
}

// C[i,l] = sum_k A[i,k] B[k,l], on the host, in double. The independent
// oracle: neither implementation under test can drift from it silently.
std::vector<double> reference_c(const ViewH& a, const ViewH& b) {
  std::vector<double> c(static_cast<std::size_t>(kI) * kL, 0.0);
  for (int i = 0; i < kI; ++i)
    for (int l = 0; l < kL; ++l) {
      double s = 0.0;
      for (int k = 0; k < kK; ++k)
        s += static_cast<double>(a(i, k)) * static_cast<double>(b(k, l));
      c[static_cast<std::size_t>(i) * kL + l] = s;
    }
  return c;
}

// ---------------------------------------------------------------------------
// The two spellings of one contraction.
//
// NESTED: C = A{i,k} x B{k,l}, A an ordinary input node read from global.
// SLOT:   the same, except A is a team-scratch buffer filled in-kernel and then
//         named by a slot node. Same GEMM, same tile, different operand owner.
// ---------------------------------------------------------------------------

// Kernels live in free functions: nvcc rejects an extended (KOKKOS_LAMBDA)
// lambda whose enclosing function has private access within its class, and
// gtest's TestBody() is private.
void run_nested(View2 a, View2 b, View2 out) {
  auto na      = make_input_node(make_handle<'i', 'k'>(a));
  auto nb      = make_input_node(make_handle<'k', 'l'>(b));
  auto g       = make_graph();
  auto [g1, t] = g.ops(make_contraction_node<'i', 'l'>(na, nb));
  g1.execute(TeamPolicyTag<ES>{}, Tile<TileIK, TileKL, TileIL>{}, out);
}

// BUILD ON THE HOST, BIND ON THE DEVICE.
//
// The node factories (make_contraction_node, make_combine_node) are host-only:
// they run extent assertions and touch std::array. A slot's storage, though,
// only exists INSIDE the kernel -- it is team scratch. So the graph cannot be
// assembled where the buffer lives, and the buffer cannot be obtained where the
// graph is assembled.
//
// The resolution, and the shape the eventual DAG driver has to use: build the
// whole node on the HOST with a value-initialized placeholder view, pass it in
// by value, and assign the real buffer into `storage_` on the DEVICE before
// constructing the evaluator. Only the pointer is deferred; every label, shape
// and tile is fixed at build time, which is where the compile-time machinery
// wants them.
//
// This is not a detail of the test. Calling a host-only factory from a kernel
// is diagnosed ONLY as nvcc warning #20011 -- it compiles, the Serial build
// stays green, and the CUDA build traps at runtime as "unspecified launch
// failure". Grep build logs for 20011.
KOKKOS_INLINE_FUNCTION Kokkos::Array<int, 2> zero_idx() { return {0, 0}; }

void run_slot(View2 a, View2 b, View2 out) {
  using TileC = Tile<TileIK, TileKL, TileIL>;

  // Host-side: a slot node over a PLACEHOLDER view, carrying the producer's
  // real shape and labels. Everything the type system needs is here.
  const auto slot0 =
      make_slot_node<0, 'i', 'k'>(SlotViewIK{}, Kokkos::Array<int, 2>{kI, kK});
  const auto node = make_contraction_node<'i', 'l'>(
      slot0, make_input_node(make_handle<'k', 'l'>(b)));

  using NodeC = std::decay_t<decltype(node)>;
  using EvalC = Evaluator<TeamPolicyTag<ES>, NodeC, TileC>;

  // Scratch = the slot buffer + everything the consumer carves for itself. The
  // slot contributes its tile ONCE, here, rather than once per consumer -- its
  // ScratchAllocator::bytes() is 0.
  const std::size_t bytes = Impl::scratch_tile_bytes<float, ES>(TileIK{}) +
                            EvalC::scratch_size_per_team(TileC{});

  Kokkos::parallel_for(
      "slot_contraction",
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(static_cast<int>(bytes))),
      KOKKOS_LAMBDA(const team_t& team) {
        // The "producer": carve the slot buffer and fill it. In a real DAG this
        // is whatever node the slot names; here it stands in for one.
        auto slot_buf = Impl::alloc_scratch_tile<float, ES>(team, TileIK{});
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team, kI * kK),
            [&](int t) { slot_buf(t / kK, t % kK) = a(t / kK, t % kK); });
        team.team_barrier();

        // Bind: the only thing deferred to device is the pointer.
        NodeC bound           = node;
        bound.node_a.storage_ = slot_buf;

        auto eval = make_evaluator<TeamPolicyTag<ES>>(bound, TileC{}, team);
        const auto interm = eval(team, zero_idx());

        auto store = make_evaluator<TeamPolicyTag<ES>>(interm, TileIL{});
        store(team, zero_idx(), out, Impl::output_perm_seq<NodeC>());
      });
  Kokkos::fence();
}

// ---------------------------------------------------------------------------
// A combine reading a slot whose labels do NOT match its output order, which
// must resolve through the zero-copy relabel rather than by permuting the
// (possibly shared) buffer in place.
//
// P{i,k} = 10*S{k,i} + X{i,k}. The slot's storage is [K,I]; the combine's
// output is [I,K].
// ---------------------------------------------------------------------------
struct SlotCombine {
  KOKKOS_FUNCTION float operator()(int, int, float s, float x) const {
    return 10.f * s + x;
  }
};

using TileKI     = StaticTile<kK, kI>;
using SlotViewKI = decltype(Impl::alloc_scratch_tile<float, ES>(
    std::declval<team_t>(), std::declval<TileKI>()));

void run_combine_relabel(View2 src_ki, View2 x_ik, View2 out) {
  using CTile = CombineTile<TileIK, TileIK, TileIK>;

  const auto slot0 =
      make_slot_node<0, 'k', 'i'>(SlotViewKI{}, Kokkos::Array<int, 2>{kK, kI});
  const auto node = make_combine_node<'i', 'k'>(
      slot0, make_input_node(make_handle<'i', 'k'>(x_ik)), SlotCombine{});

  using NodeP = std::decay_t<decltype(node)>;
  using EvalP = Evaluator<TeamPolicyTag<ES>, NodeP, CTile>;

  // The slot must take the relabel path, not the in-place reorder: [8,16]
  // cannot be transposed inside its own storage anyway, but the reason it is
  // forbidden here is aliasing, not extents.
  //
  // Spelled on the CV-QUALIFIED `decltype(slot0)` on purpose (slot0 is const),
  // because that is how a driver holding nodes by const reference will ask.
  // Before has_node_tag_v decayed its argument this answered "stageable,
  // unconstrained" -- the permissive fallthrough -- for a node that must not be
  // reordered at all.
  using Perm = Impl::label_perm_seq_t<std::integer_sequence<int32_t, 'i', 'k'>,
                                      std::integer_sequence<int32_t, 'k', 'i'>>;
  static_assert(std::is_const_v<std::remove_reference_t<decltype(slot0)>>);
  static_assert(Impl::operand_relabelable_v<decltype(slot0), Perm>);
  static_assert(!Impl::operand_stageable_v<decltype(slot0), TileKI, Perm>);

  const std::size_t bytes = Impl::scratch_tile_bytes<float, ES>(TileKI{}) +
                            EvalP::scratch_size_per_team(CTile{});

  Kokkos::parallel_for(
      "slot_combine_relabel",
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(static_cast<int>(bytes))),
      KOKKOS_LAMBDA(const team_t& team) {
        auto slot_buf = Impl::alloc_scratch_tile<float, ES>(team, TileKI{});
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team, kK * kI),
            [&](int t) { slot_buf(t / kI, t % kI) = src_ki(t / kI, t % kI); });
        team.team_barrier();

        NodeP bound                               = node;
        bound.operands.template get<0>().storage_ = slot_buf;

        auto eval = make_evaluator<TeamPolicyTag<ES>>(bound, CTile{}, team);
        const auto interms = eval(team, zero_idx());

        auto store = make_evaluator<TeamPolicyTag<ES>>(interms[0], TileIK{});
        store(team, zero_idx(), out, Impl::output_perm_seq<NodeP>());
      });
  Kokkos::fence();
}

double max_abs_diff(const View2& d, const std::vector<double>& ref) {
  auto h = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(h, d);
  double m = 0.0;
  for (int i = 0; i < kI; ++i)
    for (int l = 0; l < kL; ++l)
      m = std::max(m, std::abs(static_cast<double>(h(i, l)) -
                               ref[static_cast<std::size_t>(i) * kL + l]));
  return m;
}

}  // namespace

// ---------------------------------------------------------------------------
// Compile-time: the predicates that admit and constrain a slot.
// ---------------------------------------------------------------------------
TEST(SlotNodeTest, Predicates) {
  using Id   = std::integer_sequence<int, 0, 1>;
  using Swap = std::integer_sequence<int, 1, 0>;

  // A slot is a legal operand...
  static_assert(Impl::fusable_operand_v<SlotIK>);
  // ...and is NOT "produces its own scratch": it evaluates nothing. Getting
  // this wrong would charge a subtree's recursive bytes and try to run it.
  static_assert(!Impl::produces_own_scratch_v<SlotIK>);
  static_assert(Impl::reads_foreign_scratch_v<SlotIK>);
  static_assert(Impl::is_node_handle_v<SlotIK>);
  static_assert(Impl::has_node_tag_v<SlotTag, SlotIK>);

  // GUARD (Stage 2's rule, landed early because widening fusable_operand_v is
  // what makes the unsound path reachable): a slot is stageable ONLY under the
  // identity. Any other permutation would reorder a possibly-shared buffer in
  // place, permuting it under its other consumers.
  static_assert(Impl::operand_stageable_v<SlotIK, TileIK, Id>);
  static_assert(!Impl::operand_stageable_v<SlotIK, TileIK, Swap>);

  // The escape hatch for a differently-ordered consumer: relabel, zero-copy.
  static_assert(!Impl::operand_relabelable_v<SlotIK, Id>);
  static_assert(Impl::operand_relabelable_v<SlotIK, Swap>);

  // A slot node carries no hook member at all -- the two hook paths that write
  // back into a source's own buffer are structurally unreachable from it. An
  // input node, by contrast, does carry one, so this is a real distinction and
  // not a probe that happens to fail for everything.
  static_assert(!HasHook<SlotIK>);
  static_assert(HasHook<decltype(make_input_node(
                    make_handle<'i', 'k'>(std::declval<View2>())))>);
}

// A slot costs its consumer nothing. This is the property the whole DAG rests
// on: today a fused operand charges its full recursive scratch PER CONSUMER,
// which is why N consumers cost N copies.
TEST(SlotNodeTest, CostsTheConsumerNothing) {
  using Alloc = ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, SlotIK,
                                 float, TileIK>;
  EXPECT_EQ(Alloc::bytes(TileIK{}), 0u);

  using CombAlloc =
      ScratchAllocator<TeamPolicyTag<ES>, CombineTag, SlotIK, float, TileIK>;
  EXPECT_EQ(CombAlloc::bytes(TileIK{}), 0u);

  // get() hands back the slot's own view type, unchanged -- no staging buffer
  // is carved and no copy is implied.
  static_assert(
      std::is_same_v<decltype(std::declval<const Alloc&>().get()), SlotViewIK>);

  // stage() must return a node whose storage_type EQUALS the destination
  // scratch type, because that equality is what selects Specialization 8's
  // zero-copy passthrough over its copy branch.
  using Staged = decltype(std::declval<const Alloc&>().stage(
      std::declval<team_t>(), std::declval<Kokkos::Array<int, 2>>()));
  static_assert(std::is_same_v<typename Staged::storage_type, SlotViewIK>);
}

// ---------------------------------------------------------------------------
// Numerical: a slot operand computes the same contraction as a nested one.
// ---------------------------------------------------------------------------
TEST(SlotNodeTest, SlotOperandMatchesNestedSpelling) {
  const auto ah  = host_a();
  const auto bh  = host_b();
  const auto a   = device_copy(ah);
  const auto b   = device_copy(bh);
  const auto ref = reference_c(ah, bh);

  View2 out_nested("out_nested", kI, kL);
  View2 out_slot("out_slot", kI, kL);

  run_nested(a, b, out_nested);
  run_slot(a, b, out_slot);
  Kokkos::fence();

  // Both must match the independent host oracle...
  EXPECT_LT(max_abs_diff(out_nested, ref), 1e-3);
  EXPECT_LT(max_abs_diff(out_slot, ref), 1e-3);

  // ...and each other BIT for bit. The slot changes who owns the operand
  // buffer and nothing else, so the GEMM sees identical bytes in identical
  // order; anything less than exact equality means it did not.
  auto hn = Kokkos::create_mirror_view(out_nested);
  auto hs = Kokkos::create_mirror_view(out_slot);
  Kokkos::deep_copy(hn, out_nested);
  Kokkos::deep_copy(hs, out_slot);
  for (int i = 0; i < kI; ++i)
    for (int l = 0; l < kL; ++l)
      EXPECT_EQ(hn(i, l), hs(i, l)) << "at (" << i << "," << l << ")";
}

// ---------------------------------------------------------------------------
// The property the SEM graph actually needs: ONE buffer, TWO slot nodes with
// different labels, read by two different consumers.
//
// This is the relabel mechanism. SEM's gradient is spelled {q,e,j} by one
// consumer and {i,e,q} by another -- same tensor, same physical order, two
// label sets -- and rather than moving data, it is named twice. Here the same
// [I,K] buffer is read once as {i,k} (identity, passthrough) and once as {k,i}
// by a consumer wanting the transpose (non-identity, relabel path).
// ---------------------------------------------------------------------------
TEST(SlotNodeTest, TwoSlotNodesMayAliasOneBuffer) {
  using SlotKI = decltype(make_slot_node<0, 'k', 'i'>(
      std::declval<SlotViewIK>(), std::declval<Kokkos::Array<int, 2>>()));

  // Same storage type, different labels -- so they can name one buffer.
  static_assert(std::is_same_v<typename SlotIK::storage_type,
                               typename SlotKI::storage_type>);
  static_assert(
      !std::is_same_v<typename SlotIK::modes_seq, typename SlotKI::modes_seq>);

  // A consumer whose canonical order is {i,k} reads SlotIK under the identity
  // (passthrough) and SlotKI under a swap. The swap must take the RELABEL path,
  // never the in-place reorder -- that is what makes aliasing safe.
  using Target     = std::integer_sequence<int32_t, 'i', 'k'>;
  using PermFromIK = Impl::label_perm_seq_t<Target, typename SlotIK::modes_seq>;
  using PermFromKI = Impl::label_perm_seq_t<Target, typename SlotKI::modes_seq>;
  static_assert(Impl::is_identity_seq(PermFromIK{}));
  static_assert(!Impl::is_identity_seq(PermFromKI{}));

  static_assert(Impl::operand_stageable_v<SlotIK, TileIK, PermFromIK>);
  static_assert(!Impl::operand_relabelable_v<SlotIK, PermFromIK>);

  static_assert(!Impl::operand_stageable_v<SlotKI, TileIK, PermFromKI>);
  static_assert(Impl::operand_relabelable_v<SlotKI, PermFromKI>);
}

// The relabel path, numerically. A combine whose output is {i,k} reads a slot
// labelled {k,i} over [K,I] storage: the axis mismatch must be resolved by
// retyping the layout, not by permuting the buffer.
TEST(SlotNodeTest, CombineReadsRelabeledSlot) {
  ViewH s("s", kK, kI), x("x", kI, kK);
  for (int k = 0; k < kK; ++k)
    for (int i = 0; i < kI; ++i) s(k, i) = b_val(k, i);
  for (int i = 0; i < kI; ++i)
    for (int k = 0; k < kK; ++k) x(i, k) = a_val(i, k);

  const auto sd = device_copy(s);
  const auto xd = device_copy(x);

  View2 out("out", kI, kK);
  run_combine_relabel(sd, xd, out);
  Kokkos::fence();

  auto h = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(h, out);
  for (int i = 0; i < kI; ++i)
    for (int k = 0; k < kK; ++k)
      EXPECT_FLOAT_EQ(h(i, k), 10.f * s(k, i) + x(i, k))
          << "at (" << i << "," << k << ")";
}

// ---------------------------------------------------------------------------
// GUARD B — a slot operand must be demanded at ONE tile index.
//
// A slot's stage() ignores the index it is handed and returns the producer's
// buffer, which holds one tile. The contraction's k-loop re-invokes each
// operand once per contracted tile at a DIFFERENT index. Compatible only when
// there is a single contracted tile; otherwise every iteration re-reads the
// same data and the sum is silently wrong.
//
// This one cannot be a static_assert -- the tile extent is compile-time but the
// operand's SHAPE is a runtime value, so the tile count is not formable at
// compile time. It is therefore a runtime predicate that a driver is expected
// to ask host-side, before launching. These tests are the reason it is public.
// ---------------------------------------------------------------------------
TEST(SlotNodeTest, GuardBAcceptsSingleContractedTile) {
  // Slot {i,k} with shape [16,8] and a [16,8] tile: one contracted tile.
  const auto slot =
      make_slot_node<0, 'i', 'k'>(SlotViewIK{}, Kokkos::Array<int, 2>{kI, kK});
  const auto node = make_contraction_node<'i', 'l'>(
      slot, make_input_node(make_handle<'k', 'l'>(View2{})));
  using Eval = Evaluator<TeamPolicyTag<ES>, std::decay_t<decltype(node)>,
                         Tile<TileIK, TileKL, TileIL>>;

  static_assert(Eval::kHasSlotOperand);
  EXPECT_TRUE(
      Eval::slot_operands_single_k_tile(node, Tile<TileIK, TileKL, TileIL>{}));
}

TEST(SlotNodeTest, GuardBRejectsMultipleContractedTiles) {
  // Same tile, but the slot claims a contracted extent of 2*kK -- so the k-loop
  // would run twice and demand two different tiles from a buffer holding one.
  //
  // The shape is what varies, not the tile, which is exactly why this is a
  // runtime check: both configurations have identical types.
  const auto bad = make_slot_node<0, 'i', 'k'>(
      SlotViewIK{}, Kokkos::Array<int, 2>{kI, 2 * kK});
  const auto node = make_contraction_node<'i', 'l'>(
      bad, make_input_node(make_handle<'k', 'l'>(View2{})));
  using Eval = Evaluator<TeamPolicyTag<ES>, std::decay_t<decltype(node)>,
                         Tile<TileIK, TileKL, TileIL>>;

  EXPECT_FALSE(
      Eval::slot_operands_single_k_tile(node, Tile<TileIK, TileKL, TileIL>{}));
}

// Guard B applies only where there is a k-loop to disagree with. A contraction
// with no slot operand is unconstrained -- the fused-operand recompute the
// k-loop performs is exactly what it is for.
TEST(SlotNodeTest, GuardBIgnoresNonSlotOperands) {
  const auto node = make_contraction_node<'i', 'l'>(
      make_input_node(make_handle<'i', 'k'>(View2{})),
      make_input_node(make_handle<'k', 'l'>(View2{})));
  using Eval = Evaluator<TeamPolicyTag<ES>, std::decay_t<decltype(node)>,
                         Tile<TileIK, TileKL, TileIL>>;

  static_assert(!Eval::kHasSlotOperand);
  // Vacuously true even for a shape that WOULD span several contracted tiles.
  EXPECT_TRUE(
      Eval::slot_operands_single_k_tile(node, Tile<TileIK, TileKL, TileIL>{}));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
