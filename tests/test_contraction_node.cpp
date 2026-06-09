#include <gtest/gtest.h>
#include <TensorOperations/NodeHandle.hpp>
#include <Kokkos_Core.hpp>

using namespace TensorOperations;

// A_{i,j,k}: shape (3, 4, 5)
struct ATensor {
    static constexpr int rank = 3;
    int extent(int i) const {
        if (i == 0) return 3;
        if (i == 1) return 4;
        return 5;
    }
    float operator()(int, int, int) const { return 0.f; }
};

// B_{j,k,l}: shape (4, 5, 6)
struct BTensor {
    static constexpr int rank = 3;
    int extent(int i) const {
        if (i == 0) return 4;
        if (i == 1) return 5;
        return 6;
    }
    float operator()(int, int, int) const { return 0.f; }
};

static_assert(TensorLike<ATensor>);
static_assert(TensorLike<BTensor>);

TEST(ContractionNodeTest, ShapeAndModes) {
    auto ha = make_handle(ATensor{}, std::array<int32_t, 3>{'i', 'j', 'k'});
    auto hb = make_handle(BTensor{}, std::array<int32_t, 3>{'j', 'k', 'l'});
    auto na = make_input_node(ha);
    auto nb = make_input_node(hb);

    // Contract over {j, k} → C_{i,l} with shape (3, 6)
    auto nc = make_contraction_node<float>(
        na, nb, std::array<int32_t, 2>{'i', 'l'});

    static_assert(decltype(nc)::Rank == 2);
    static_assert(decltype(nc)::NumContracted == 2);

    auto s = nc.shape();
    EXPECT_EQ(s[0], 3);
    EXPECT_EQ(s[1], 6);

    auto m = nc.modes();
    EXPECT_EQ(m[0], 'i');
    EXPECT_EQ(m[1], 'l');
}

TEST(ContractionNodeTest, HookIsStored) {
    auto ha = make_handle(ATensor{}, std::array<int32_t, 3>{'i', 'j', 'k'});
    auto hb = make_handle(BTensor{}, std::array<int32_t, 3>{'j', 'k', 'l'});
    auto na = make_input_node(ha);
    auto nb = make_input_node(hb);

    bool hook_constructed = false;
    auto hook = [&](int, int) { hook_constructed = true; };

    auto nc = make_contraction_node<float>(
        na, nb, std::array<int32_t, 2>{'i', 'l'}, hook);

    static_assert(!std::is_same_v<decltype(nc)::hook_type, void>);
    (void)hook_constructed;
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::initialize(argc, argv);
    int result = RUN_ALL_TESTS();
    Kokkos::finalize();
    return result;
}
