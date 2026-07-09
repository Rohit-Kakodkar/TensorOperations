#include <Kokkos_Core.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <gtest/gtest.h>

using namespace TensorOperations;

// A_{i,j,k}: shape (3, 4, 5)
struct ATensor {
  static constexpr int rank = 3;
  float                buf_[60]{};
  int                  extent(int i) const {
    if (i == 0) return 3;
    if (i == 1) return 4;
    return 5;
  }
  std::ptrdiff_t stride(int k) const {
    if (k == 0) return 20;
    if (k == 1) return 5;
    return 1;
  }
  float*       data() { return buf_; }
  const float* data() const { return buf_; }
  float        operator()(int, int, int) const { return 0.f; }
};

// B_{j,k,l}: shape (4, 5, 6)
struct BTensor {
  static constexpr int rank = 3;
  float                buf_[120]{};
  int                  extent(int i) const {
    if (i == 0) return 4;
    if (i == 1) return 5;
    return 6;
  }
  std::ptrdiff_t stride(int k) const {
    if (k == 0) return 30;
    if (k == 1) return 6;
    return 1;
  }
  float*       data() { return buf_; }
  const float* data() const { return buf_; }
  float        operator()(int, int, int) const { return 0.f; }
};

static_assert(TensorLike<ATensor>);
static_assert(TensorLike<BTensor>);

TEST(ContractionNodeTest, ShapeAndModes) {
  auto ha = make_handle<'i', 'j', 'k'>(ATensor{});
  auto hb = make_handle<'j', 'k', 'l'>(BTensor{});
  auto na = make_input_node(ha);
  auto nb = make_input_node(hb);

  // Contract over {j, k} → C_{i,l} with shape (3, 6)
  auto nc =
      make_contraction_node<float, Kokkos::DefaultExecutionSpace, 'i', 'l'>(na,
                                                                            nb);

  static_assert(decltype(nc)::Rank == 2);
  static_assert(decltype(nc)::NumContracted == 2);
  static_assert(std::is_same_v<decltype(nc)::modes_seq,
                               std::integer_sequence<int32_t, 'i', 'l'>>);

  auto s = nc.shape();
  EXPECT_EQ(s[0], 3);
  EXPECT_EQ(s[1], 6);
}

TEST(ContractionNodeTest, HookIsStored) {
  auto ha = make_handle<'i', 'j', 'k'>(ATensor{});
  auto hb = make_handle<'j', 'k', 'l'>(BTensor{});
  auto na = make_input_node(ha);
  auto nb = make_input_node(hb);

  [[maybe_unused]] bool hook_constructed = false;
  auto                  hook = [&](int, int) { hook_constructed = true; };

  auto nc =
      make_contraction_node<float, Kokkos::DefaultExecutionSpace, 'i', 'l'>(
          na, nb, hook);

  static_assert(!std::is_same_v<decltype(nc)::hook_type, void>);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
