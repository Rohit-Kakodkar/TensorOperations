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

TEST(NodeHandleTest, NodeTagTypedef) {
  // InputTag
  using ha_t  = decltype(make_handle<'i', 'j'>(MyTensor{}));
  using inp_t = decltype(make_input_node(std::declval<ha_t>()));
  static_assert(std::is_same_v<inp_t::node_tag, InputTag>);

  // ContractionTag — use declval so no runtime extent assert fires
  using hb_t = decltype(make_handle<'k', 'j'>(MyTensor{}));
  using nb_t = decltype(make_input_node(std::declval<hb_t>()));
  using con_t =
      decltype(make_contraction_node<float, Kokkos::DefaultExecutionSpace, 'i',
                                     'k'>(std::declval<inp_t>(),
                                          std::declval<nb_t>()));
  static_assert(std::is_same_v<con_t::node_tag, ContractionTag>);

  // IntermTag
  using storage_t =
      Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace>;
  using interm_t = decltype(make_interm_node(std::declval<storage_t>()));
  static_assert(std::is_same_v<interm_t::node_tag, IntermTag>);

  // CombineTag — both operands share the same handle so extents match at
  // runtime
  auto fn  = [](int, int, float a, float b) { return a + b; };
  auto ha  = make_handle<'i', 'j'>(MyTensor{});
  auto inp = make_input_node(ha);
  auto com = make_combine_node<float, Kokkos::DefaultExecutionSpace, 'i', 'j'>(
      inp, make_input_node(ha), fn);
  static_assert(std::is_same_v<typename decltype(com)::node_tag, CombineTag>);
}

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
