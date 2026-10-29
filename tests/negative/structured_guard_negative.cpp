// ===========================================================================
// structured_guard_negative.cpp — structured operands that MUST NOT COMPILE.
//
// make_stack_node<r, l...> and make_outer_product_node<l...> take every label,
// positionally over the natural order (for a stack: r, then branch 0's labels;
// for an outer product: the operands' labels, concatenated). A new name
// relabels that axis; a REPEATED name must keep its position, so a swapped
// order is rejected rather than read as a silent transpose. At equal extents
// nothing downstream could catch that.
//
// CMake builds each case as its own target, and ctest asserts the build fails
// WITH THAT GUARD'S OWN DIAGNOSTIC.
//
// Selected by -DSTRUCTURED_NEG_CASE=<n>; exactly one case per target.
//   1  a stack whose branch labels are declared swapped: <'r','b','a'>
//   2  an outer product declaring too few labels
//   3  an outer product whose operand labels are declared swapped
//   4  a stack declaring only its stacking label, the removed form
//   0  the control: the same operands correctly declared (one of them
//      relabelled at construction), which must COMPILE.
// ===========================================================================
#include <TensorOperations/Structured.hpp>

using namespace TensorOperations;

#ifndef STRUCTURED_NEG_CASE
#error "define STRUCTURED_NEG_CASE"
#endif

namespace {

constexpr auto ab = make_delta_node<'a', 'b'>();
constexpr auto cd = make_delta_node<'c', 'd'>();

#if STRUCTURED_NEG_CASE == 0
[[maybe_unused]] const auto s = make_stack_node<'r', 'a', 'b'>(ab, ab);
[[maybe_unused]] const auto o = make_outer_product_node<'a', 'b', 'c', 'e'>(ab,
                                                                           cd);
#elif STRUCTURED_NEG_CASE == 1
[[maybe_unused]] const auto s = make_stack_node<'r', 'b', 'a'>(ab, ab);
#elif STRUCTURED_NEG_CASE == 2
[[maybe_unused]] const auto o = make_outer_product_node<'a', 'b'>(ab, cd);
#elif STRUCTURED_NEG_CASE == 3
[[maybe_unused]] const auto o = make_outer_product_node<'a', 'b', 'd', 'c'>(ab,
                                                                           cd);
#elif STRUCTURED_NEG_CASE == 4
[[maybe_unused]] const auto s = make_stack_node<'r'>(ab, ab);
#else
#error "STRUCTURED_NEG_CASE must be 0..4"
#endif

}  // namespace

int main() { return 0; }
