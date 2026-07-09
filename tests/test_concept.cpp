#include <gtest/gtest.h>
#include <TensorOperations/Concept.hpp>
#include <TensorOperations/TensorHandle.hpp>

using namespace TensorOperations;

struct Good2D {
  static constexpr int rank = 2;
  float                buf_[16]{};
  int                  extent(int) const { return 4; }
  std::ptrdiff_t       stride(int k) const { return k == 0 ? 4 : 1; }
  float*               data() { return buf_; }
  const float*         data() const { return buf_; }
  float                operator()(int, int) const { return 1.f; }
};

struct BadNoRank {
  int   extent(int) const { return 4; }
  float operator()(int, int) const { return 1.f; }
};

struct BadWrongArity {
  static constexpr int rank = 2;
  int                  extent(int) const { return 4; }
  float                operator()(int, int, int) const { return 1.f; }
};

static_assert(TensorLike<Good2D>);
static_assert(!TensorLike<BadNoRank>);
static_assert(!TensorLike<BadWrongArity>);

TEST(ConceptTest, TensorHandleBasicOps) {
  auto h = make_handle<'i', 'j'>(Good2D{});
  static_assert(std::is_same_v<decltype(h)::modes_seq,
                               std::integer_sequence<int32_t, 'i', 'j'>>);
  EXPECT_EQ(h.extent(0), 4);
  EXPECT_FLOAT_EQ(h(0, 0), 1.f);
}
