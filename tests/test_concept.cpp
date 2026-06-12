#include <gtest/gtest.h>
#include <TensorOperations/Concept.hpp>
#include <TensorOperations/TensorHandle.hpp>

using namespace TensorOperations;

struct Good2D {
  static constexpr int rank = 2;
  int                  extent(int) const { return 4; }
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
  auto h = make_handle(Good2D{}, std::array<int32_t, 2>{'i', 'j'});
  EXPECT_EQ(h.extent(0), 4);
  EXPECT_FLOAT_EQ(h(0, 0), 1.f);
  EXPECT_EQ(h.modes[0], 'i');
  EXPECT_EQ(h.modes[1], 'j');
}
