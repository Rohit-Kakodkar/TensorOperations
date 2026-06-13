#include <Kokkos_Core.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/TensorHandle.hpp>
#include <TensorOperations/Tiling.hpp>
#include <array>
#include <gtest/gtest.h>

using namespace TensorOperations;

// ---------------------------------------------------------------------------
// GEMM-shaped contraction  C_{ik} = sum_j A_{ij} B_{jk},  all 4x4.
// A is stored [i,j] (= [freeA, contracted]), B is stored [j,k] (= [contracted,
// freeB]), out_modes [i,k] (= [freeA, freeB]) — the Option A / GEMM convention.
// ---------------------------------------------------------------------------
struct AGrid {  // A(i,j)
  static constexpr int rank = 2;
  using value_type          = float;
  int   extent(int) const { return 4; }
  float operator()(int i, int j) const { return static_cast<float>(i * 4 + j); }
};
struct BGrid {  // B(j,k)
  static constexpr int rank = 2;
  using value_type          = float;
  int   extent(int) const { return 4; }
  float operator()(int j, int k) const { return static_cast<float>(j * 7 + k); }
};
struct Doubler {
  float operator()(float v) const { return 2.0f * v; }
};

static float reference(int i, int k) {
  float s = 0.f;
  for (int j = 0; j < 4; ++j) s += AGrid{}(i, j) * BGrid{}(j, k);
  return s;
}

// Build a contraction node from A, B grids with the given output hook.
template <typename HookOp = NoHook>
static auto make_nc(HookOp hook = {}) {
  auto na =
      make_input_node(make_handle(AGrid{}, std::array<int32_t, 2>{'i', 'j'}));
  auto nb =
      make_input_node(make_handle(BGrid{}, std::array<int32_t, 2>{'j', 'k'}));
  return make_contraction_node<float>(na, nb, std::array<int32_t, 2>{'i', 'k'},
                                      hook);
}

// Drive the register-tier contraction Evaluator over the whole 4x4 output with
// the participating tile TileT = StaticTile<Ti, Tk, Tj>. For each output tile:
// zero the accumulator, loop the contracted blocks accumulating, then store the
// hook-applied result — exactly the caller responsibilities the evaluator
// omits.
template <typename TileT, typename NC>
static std::array<std::array<float, 4>, 4> compute(const NC& nc) {
  using Eval = Evaluator<RangePolicyTag, NC, TileT>;
  Eval          ev{nc};
  constexpr int N  = 4;
  constexpr int Ti = TileT::extent(0), Tk = TileT::extent(1),
                Tj = TileT::extent(2);
  std::array<std::array<float, 4>, 4> out{};

  for (int off_i = 0; off_i < N; off_i += Ti)
    for (int off_k = 0; off_k < N; off_k += Tk) {
      typename Eval::accumulator_t acc{};
      acc.fill(0.f);
      for (int off_j = 0; off_j < N; off_j += Tj)
        ev(std::array<int, 2>{off_i, off_j}, std::array<int, 2>{off_j, off_k},
           acc);
      for (int m = 0; m < Ti; ++m)
        for (int n = 0; n < Tk; ++n)
          out[off_i + m][off_k + n] = Impl::apply_hook(nc.hook_op, acc(m, n));
    }
  return out;
}

// Tile contracted extent 2 < K=4 -> two contracted blocks (accumulation across
// operator() calls is exercised).
TEST(ContractionEvalTest, GemmTwoContractedBlocks) {
  auto nc  = make_nc();
  auto out = compute<StaticTile<2, 2, 2>>(nc);
  for (int i = 0; i < 4; ++i)
    for (int k = 0; k < 4; ++k)
      EXPECT_FLOAT_EQ(out[i][k], reference(i, k)) << "i=" << i << " k=" << k;
}

// Tile contracted extent 4 == K -> a single contracted block per output tile.
TEST(ContractionEvalTest, GemmSingleContractedBlock) {
  auto nc  = make_nc();
  auto out = compute<StaticTile<2, 2, 4>>(nc);
  for (int i = 0; i < 4; ++i)
    for (int k = 0; k < 4; ++k)
      EXPECT_FLOAT_EQ(out[i][k], reference(i, k)) << "i=" << i << " k=" << k;
}

// The contraction node's output hook is applied by the caller at store time.
TEST(ContractionEvalTest, AppliesHookAtStore) {
  auto nc  = make_nc(Doubler{});
  auto out = compute<StaticTile<2, 2, 2>>(nc);
  for (int i = 0; i < 4; ++i)
    for (int k = 0; k < 4; ++k)
      EXPECT_FLOAT_EQ(out[i][k], 2.0f * reference(i, k));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
