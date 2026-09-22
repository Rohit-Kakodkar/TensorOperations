#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/TensorHandle.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

using namespace TensorOperations;

int count_staged_mismatches() {
  using View      = Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::Cuda>;
  using Tiler     = cute::Shape<cute::_4, cute::_8>;
  using ThrLayout = cute::Layout<cute::Shape<cute::_2, cute::_8>>;
  using Policy    = Kokkos::TeamPolicy<Kokkos::Cuda>;
  constexpr int I = 8, J = 16, TI = 4, TJ = 8, NT = 16;

  View v("v", I, J);
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::Rank<2>>({0, 0}, {I, J}),
      KOKKOS_LAMBDA(int i, int j) { v(i, j) = 100.0f * i + j; });

  auto node = make_input_node(make_handle<'i', 'j'>(v));
  int  bad  = 0;
  Kokkos::parallel_reduce(
      Policy((I / TI) * (J / TJ), NT)
          .set_scratch_size(0, Kokkos::PerTeam(TI * TJ * sizeof(float))),
      KOKKOS_LAMBDA(const Policy::member_type& team, int& acc) {
        const int t  = team.league_rank();
        const int ti = t / (J / TJ), tj = t % (J / TJ);

        auto* ptr = static_cast<float*>(
            team.team_scratch(0).get_shmem(TI * TJ * sizeof(float)));
        auto stile =
            cute::make_tensor(cute::make_smem_ptr(ptr),
                              cute::make_layout(Tiler{}, cute::LayoutRight{}));

        auto src = make_evaluator<CutePolicyTag<>>(
            node, Tiler{})(cute::make_coord(ti, tj));
        auto stager = make_evaluator<CutePolicyTag<>>(
            make_cute_interm_node<Kokkos::Cuda>(stile),
            CuteStageTag<ThrLayout>{ThrLayout{}, team.team_rank()});
        auto staged = (stager = src);
        team.team_barrier();

        if (team.team_rank() == 0) {
          const auto s = staged.node().storage_;
          for (int a = 0; a < TI; ++a)
            for (int b = 0; b < TJ; ++b)
              if (s(a, b) != v(TI * ti + a, TJ * tj + b)) ++acc;
        }
      },
      bad);
  return bad;
}

TEST(CuteStage, StagedTileMatchesView) {
  EXPECT_EQ(count_staged_mismatches(), 0);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
