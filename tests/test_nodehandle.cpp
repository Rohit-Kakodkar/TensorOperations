#include <gtest/gtest.h>
#include <TensorOperations/Concept.hpp>
#include <TensorOperations/TensorHandle.hpp>
#include <TensorOperations/NodeHandle.hpp>

#include <Kokkos_Core.hpp>

using namespace TensorOperations;

struct MyTensor {
    static constexpr int rank = 2;
    int extent(int i) const { return (i == 0) ? 4 : 3; }
    float operator()(int i, int j) const { return static_cast<float>(i * 3 + j); }
};

static_assert(TensorLike<Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace>>);
static_assert(WritableTensorLike<Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace>>);
static_assert(TensorLike<MyTensor>);

TEST(NodeHandleTest, InputNode) {
    auto h   = make_handle(MyTensor{}, std::array<int32_t, 2>{'i', 'j'});
    auto inp = make_input_node(h);

    auto s = inp.shape();
    EXPECT_EQ(s[0], 4);
    EXPECT_EQ(s[1], 3);

    auto m = inp.modes();
    EXPECT_EQ(m[0], 'i');
    EXPECT_EQ(m[1], 'j');

    const auto& ch = inp.concretize();
    EXPECT_FLOAT_EQ(ch(1, 2), 5.f);  // MyTensor: 1*3+2 = 5
}

TEST(NodeHandleTest, IntermediateNodeGlobalAlloc) {
    auto interm = make_interm_node<float, 2, Kokkos::Serial>(
        0,
        std::array<int, 2>{4, 3},
        std::array<int32_t, 2>{'a', 'b'});

    auto s = interm.shape();
    EXPECT_EQ(s[0], 4);
    EXPECT_EQ(s[1], 3);

    auto m = interm.modes();
    EXPECT_EQ(m[0], 'a');
    EXPECT_EQ(m[1], 'b');

    auto cv = interm.concretize();
    EXPECT_EQ(cv.extent(0), 4);
    EXPECT_EQ(cv.extent(1), 3);
    EXPECT_EQ(cv.modes[0], 'a');
    EXPECT_EQ(cv.modes[1], 'b');

    cv(1, 2) = 9.f;
    EXPECT_FLOAT_EQ(cv(1, 2), 9.f);
    EXPECT_FLOAT_EQ(cv(0, 0), 0.f);
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::initialize(argc, argv);
    int result = RUN_ALL_TESTS();
    Kokkos::finalize();
    return result;
}
