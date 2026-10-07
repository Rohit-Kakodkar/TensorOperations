#include <TensorOperations/LabelTiles.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>

using namespace TensorOperations;
using ES = Kokkos::DefaultExecutionSpace;

#ifndef STAGE_TV_NEG_CASE
#error "define STAGE_TV_NEG_CASE"
#endif

namespace {

using Map  = LabelTiles<LabelTile<'e', 2>, LabelWhole<'a', 5>>;
using ValV = Kokkos::View<float**, Kokkos::LayoutRight, ES>;

#if STAGE_TV_NEG_CASE == 0
using TV = cute::Layout<cute::Shape<cute::_5, cute::_2>,
                        cute::Stride<cute::_2, cute::_1>>;
#elif STAGE_TV_NEG_CASE == 1
using TV = cute::Layout<cute::Shape<cute::_5, cute::_4>,
                        cute::Stride<cute::_4, cute::_1>>;
#elif STAGE_TV_NEG_CASE == 2
using TV = cute::Layout<cute::Shape<cute::_5, cute::_2>,
                        cute::Stride<cute::_1, cute::_1>>;
#elif STAGE_TV_NEG_CASE == 3
using TV = cute::Layout<cute::Shape<cute::_5, cute::_2, cute::_1>,
                        cute::Stride<cute::_2, cute::_1, cute::_0>>;
#elif STAGE_TV_NEG_CASE == 4
using TV = cute::Layout<cute::Shape<int, int>, cute::Stride<int, cute::_1>>;
#else
#error "STAGE_TV_NEG_CASE must be 0..4"
#endif

using Raw  = decltype(make_stage_node(
    make_input_node(make_handle<'e', 'a'>(std::declval<ValV>())),
    std::declval<TV>()));
using Node = typename Impl::lg_resolve_member<Map, Raw>::type;
using Part = typename Impl::lg_cute_stage<Node, 128>::part;
static_assert(sizeof(Part) > 0, "forces the stage's guards to instantiate");

}  // namespace

int main() { return 0; }
