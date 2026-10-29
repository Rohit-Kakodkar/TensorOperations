#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/TensorHandle.hpp>

#include <Kokkos_Core.hpp>

using namespace TensorOperations;
using ES = Kokkos::DefaultExecutionSpace;

#ifndef REGISTER_SRC_NEG_CASE
#error "define REGISTER_SRC_NEG_CASE"
#endif

// make_register_node<Modes...>(src, tv) with a view or handle source. Each case
// must fail to compile with the guard's own message.
//   0: a handle whose labels differ from the labels given
//   1: a view with no labels at all
//   2: fewer labels than the view has axes
namespace {

using ValV = Kokkos::View<float***, Kokkos::LayoutRight, ES>;

struct SomeTV {};

[[maybe_unused]] void build(const ValV& u) {
#if REGISTER_SRC_NEG_CASE == 0
  (void)make_register_node<'e', 'a', 'b'>(make_handle<'e', 'b', 'a'>(u),
                                          SomeTV{});
#elif REGISTER_SRC_NEG_CASE == 1
  (void)make_register_node(u, SomeTV{});
#elif REGISTER_SRC_NEG_CASE == 2
  (void)make_register_node<'e', 'a'>(u, SomeTV{});
#else
#error "REGISTER_SRC_NEG_CASE must be 0..2"
#endif
}

}  // namespace

int main() { return 0; }
