#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/TensorHandle.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <utility>

using namespace TensorOperations;

namespace {

using InView    = Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::Cuda>;
using OutView   = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::Cuda>;
using Tiler     = cute::Shape<cute::_4, cute::_8>;
using ThrLayout = cute::Layout<cute::Shape<cute::_2, cute::_8>>;
using Policy    = Kokkos::TeamPolicy<Kokkos::Cuda>;
constexpr int I = 8, J = 16, TI = 4, TJ = 8, NT = 16;

struct ShiftHook {
  KOKKOS_FUNCTION void operator()(int i, int j, float& v) const {
    v += 0.5f * i - j;
  }
};

InView make_input() {
  InView v("in", I, J);
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::Rank<2>>({0, 0}, {I, J}),
      KOKKOS_LAMBDA(int i, int j) { v(i, j) = 100.0f * i + j; });
  return v;
}

template <typename Hook, typename OutHandle, int... Perm>
void round_trip(const InView& in, const OutHandle& out, Hook hook,
                std::integer_sequence<int, Perm...> perm) {
  auto node = make_input_node(make_handle<'i', 'j'>(in));
  Kokkos::parallel_for(
      Policy((I / TI) * (J / TJ), NT)
          .set_scratch_size(0, Kokkos::PerTeam(TI * TJ * sizeof(float))),
      KOKKOS_LAMBDA(const Policy::member_type& team) {
        const int t  = team.league_rank();
        const int ti = t / (J / TJ), tj = t % (J / TJ);

        auto* ptr = static_cast<float*>(
            team.team_scratch(0).get_shmem(TI * TJ * sizeof(float)));
        auto stile =
            cute::make_tensor(cute::make_smem_ptr(ptr),
                              cute::make_layout(Tiler{}, cute::LayoutRight{}));

        const auto coord = cute::make_coord(ti, tj);
        auto       src = make_evaluator<CutePolicyTag<>>(node, Tiler{})(coord);
        auto       stager = make_evaluator<CutePolicyTag<>>(
            make_cute_interm_node<Kokkos::Cuda>(stile),
            CuteStageTag<ThrLayout>{ThrLayout{}, team.team_rank()});
        stager = src;
        team.team_barrier();

        auto store = make_evaluator<CutePolicyTag<>>(
            make_cute_interm_node<Kokkos::Cuda>(stile, hook),
            CuteStoreTag<ThrLayout>{ThrLayout{}, team.team_rank()});
        store(coord, out, perm);
      });
  Kokkos::fence();
}

int count_hooked_mismatches() {
  const auto in = make_input();
  OutView    out("out", I, J);
  round_trip(in, make_handle<'i', 'j'>(out), ShiftHook{},
             std::integer_sequence<int, 0, 1>{});

  auto h_in  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, in);
  auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  int  bad   = 0;
  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j)
      if (h_out(i, j) != h_in(i, j) + 0.5f * i - j) ++bad;
  return bad;
}

int count_permuted_mismatches() {
  const auto in = make_input();
  OutView    out("out", J, I);
  round_trip(in, make_handle<'j', 'i'>(out), NoHook{},
             std::integer_sequence<int, 1, 0>{});

  auto h_in  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, in);
  auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  int  bad   = 0;
  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j)
      if (h_out(j, i) != h_in(i, j)) ++bad;
  return bad;
}

}  // namespace

TEST(CuteStore, HookedRoundTripMatchesInput) {
  EXPECT_EQ(count_hooked_mismatches(), 0);
}

TEST(CuteStore, PermutedOutputIsTransposed) {
  EXPECT_EQ(count_permuted_mismatches(), 0);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
