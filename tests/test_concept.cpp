#include <TensorOperations/Concept.hpp>
#include <TensorOperations/TensorHandle.hpp>
#include <cassert>

using namespace TensorOperations;

struct Good2D {
    static constexpr int rank() { return 2; }
    int extent(int) const { return 4; }
    float operator()(int, int) const { return 1.f; }
};

struct BadNoRank {
    int extent(int) const { return 4; }
    float operator()(int, int) const { return 1.f; }
};

struct BadWrongArity {
    static constexpr int rank() { return 2; }
    int extent(int) const { return 4; }
    float operator()(int, int, int) const { return 1.f; }
};

static_assert( TensorLike<Good2D>);
static_assert(!TensorLike<BadNoRank>);
static_assert(!TensorLike<BadWrongArity>);

int main() {
    auto h = make_handle(Good2D{}, std::array<int32_t, 2>{'i', 'j'});
    assert(h.extent(0) == 4);
    assert(h(0, 0) == 1.f);
    assert(h.modes[0] == 'i');
    assert(h.modes[1] == 'j');
    return 0;
}
