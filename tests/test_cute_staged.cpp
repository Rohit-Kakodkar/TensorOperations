#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/TensorHandle.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

using namespace TensorOperations;

int count_staged_node_mismatches() {
  using View      = Kokkos::View<float***, Kokkos::LayoutLeft, Kokkos::Cuda>;
  using Tiler     = cute::Shape<cute::_3, cute::_4, cute::_4>;
  using ThrLayout = cute::Layout<cute::Shape<cute::_3, cute::_2, cute::_4>>;
  using Policy    = Kokkos::TeamPolicy<Kokkos::Cuda>;
  constexpr int P = 6, Q = 8, R = 12, TP = 3, TQ = 4, TR = 4, NT = 24;
  constexpr int NQ = Q / TQ, NR = R / TR, TS = TP * TQ * TR;

  View v("v", P, Q, R);
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::Rank<3>>({0, 0, 0},
                                                           {P, Q, R}),
      KOKKOS_LAMBDA(int p, int q, int r) {
        v(p, q, r) = 100.0f * p + 10.0f * q + r + 0.5f * p * r;
      });

  auto sn  = make_stage_node(make_input_node(make_handle<'p', 'q', 'r'>(v)));
  int  bad = 0;
  Kokkos::parallel_reduce(
      Policy((P / TP) * NQ * NR, NT)
          .set_scratch_size(0, Kokkos::PerTeam(TS * sizeof(float))),
      KOKKOS_LAMBDA(const Policy::member_type& team, int& acc) {
        const int t  = team.league_rank();
        const int tp = t / (NQ * NR), tq = (t / NR) % NQ, tr = t % NR;

        auto* ptr = static_cast<float*>(
            team.team_scratch(0).get_shmem(TS * sizeof(float)));
        auto stile =
            cute::make_tensor(cute::make_smem_ptr(ptr),
                              cute::make_layout(Tiler{}, cute::LayoutRight{}));

        auto ev = make_evaluator<CutePolicyTag<>>(
            sn, CuteStagedTag<decltype(stile), ThrLayout>{
                    {ThrLayout{}, team.team_rank()}, stile});
        auto staged = ev(cute::make_coord(tp, tq, tr));
        team.team_barrier();

        if (team.team_rank() == 0) {
          const auto s = staged.node().storage_;
          const auto e = ev.storage();
          for (int a = 0; a < TP; ++a)
            for (int b = 0; b < TQ; ++b)
              for (int c = 0; c < TR; ++c) {
                const float want = v(TP * tp + a, TQ * tq + b, TR * tr + c);
                if (s(a, b, c) != want || e(a, b, c) != want) ++acc;
              }
        }
      },
      bad);
  return bad;
}

TEST(CuteStaged, StagedNodeTileMatchesView) {
  EXPECT_EQ(count_staged_node_mismatches(), 0);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
