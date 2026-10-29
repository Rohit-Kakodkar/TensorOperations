#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/NodeHandle.hpp>

#include <Kokkos_Core.hpp>

#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include <sys/wait.h>
#include <unistd.h>

using namespace TensorOperations;

namespace {

using ES   = Kokkos::Cuda;
using V4   = Kokkos::View<float****, Kokkos::LayoutRight, ES>;
using Map4 = LabelTiles<LabelTile<'e', 2>, LabelWhole<'a', 5>,
                        LabelWhole<'b', 5>, LabelWhole<'c', 5>>;

void run() {
  V4         big("big", 8, 5, 5, 8);
  V4         out("out", 8, 5, 5, 5);
  const auto u = Kokkos::subview(big, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                 Kokkos::pair<int, int>(0, 5));
  static_assert(std::is_same_v<decltype(u)::array_layout, Kokkos::LayoutRight>);
  std::printf("contiguous %d\n", static_cast<int>(u.span_is_contiguous()));

  auto g0      = make_level_graph<float, ES>(Map4{});
  auto [g1, s] = g0.add(
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(u))));
  g1.outputs(s).execute(CutePolicyTag<>{}, out);
  Kokkos::fence();
  std::printf("executed without aborting\n");
}

}  // namespace

int main(int argc, char** argv) {
  const pid_t pid = fork();
  if (pid == 0) {
    Kokkos::initialize(argc, argv);
    run();
    Kokkos::finalize();
    std::fflush(stdout);
    std::_Exit(0);
  }
  int status = 0;
  waitpid(pid, &status, 0);
  std::printf(WIFSIGNALED(status) ? "child aborted\n" : "child exited\n");
  return 0;
}
