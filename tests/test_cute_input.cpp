#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/TensorHandle.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

using namespace TensorOperations;

template <typename Layout>
int count_tile_mismatches() {
  using View      = Kokkos::View<float**, Layout, Kokkos::Cuda>;
  using Tiler     = cute::Shape<cute::_4, cute::_4>;
  constexpr int I = 8, J = 12, TI = 4, TJ = 4;

  View v("v", I, J);
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::Rank<2>>({0, 0}, {I, J}),
      KOKKOS_LAMBDA(int i, int j) { v(i, j) = 100.0f * i + j; });

  auto node = make_input_node(make_handle<'i', 'j'>(v));
  int  bad  = 0;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<Kokkos::Cuda>(0, (I / TI) * (J / TJ)),
      KOKKOS_LAMBDA(int t, int& acc) {
        const int ti = t / (J / TJ), tj = t % (J / TJ);
        auto      ev   = make_evaluator<CutePolicyTag<>>(node, Tiler{});
        auto      tile = ev(cute::make_coord(ti, tj)).node().storage_;
        for (int a = 0; a < TI; ++a)
          for (int b = 0; b < TJ; ++b)
            if (tile(a, b) != v(TI * ti + a, TJ * tj + b)) ++acc;
      },
      bad);
  return bad;
}

TEST(CuteInput, TilesMatchView) {
  EXPECT_EQ(count_tile_mismatches<Kokkos::LayoutRight>(), 0);
  EXPECT_EQ(count_tile_mismatches<Kokkos::LayoutLeft>(), 0);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
