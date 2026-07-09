#include <gtest/gtest.h>
#include <TensorOperations/Concept.hpp>
#include <TensorOperations/TensorHandle.hpp>
#include <TensorOperations/NodeHandle.hpp>

#include <Kokkos_Core.hpp>

using namespace TensorOperations;

struct MyTensor {
  static constexpr int rank = 2;
  float                buf_[12]{};
  int                  extent(int i) const { return (i == 0) ? 4 : 3; }
  std::ptrdiff_t       stride(int k) const { return k == 0 ? 3 : 1; }
  float*               data() { return buf_; }
  const float*         data() const { return buf_; }
  float operator()(int i, int j) const { return static_cast<float>(i * 3 + j); }
};

static_assert(
    TensorLike<Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace>>);
static_assert(TensorLike<MyTensor>);

TEST(NodeHandleTest, InputNode) {
  auto h   = make_handle<'i', 'j'>(MyTensor{});
  auto inp = make_input_node(h);

  static_assert(std::is_same_v<decltype(inp)::modes_seq,
                               std::integer_sequence<int32_t, 'i', 'j'>>);

  auto s = inp.shape();
  EXPECT_EQ(s[0], 4);
  EXPECT_EQ(s[1], 3);

  EXPECT_FLOAT_EQ(inp.handle(1, 2), 5.f);  // MyTensor: 1*3+2 = 5
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
