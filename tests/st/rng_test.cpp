#include <gtest/gtest.h>
#include <ax/st/rng.hpp>

#include <array>
#include <cmath>
#include <cstdint>

using ax::st::rng;

TEST(rng, deterministic_same_seed) {
  rng a{42}, b{42};
  for (int i = 0; i < 100; ++i) EXPECT_EQ(a.next_u64(), b.next_u64());
}

TEST(rng, different_seeds_differ) {
  rng a{1}, b{2};
  int same = 0;
  for (int i = 0; i < 100; ++i)
    if (a.next_u64() == b.next_u64()) ++same;
  EXPECT_LT(same, 2);
}

TEST(rng, next_double_range_and_moments) {
  rng g{7};
  const int n = 100000;
  double sum = 0, sum2 = 0;
  for (int i = 0; i < n; ++i) {
    const double x = g.next_double();
    ASSERT_GE(x, 0.0);
    ASSERT_LT(x, 1.0);
    sum += x;
    sum2 += x * x;
  }
  const double mean = sum / n;
  const double var = sum2 / n - mean * mean;
  EXPECT_NEAR(mean, 0.5, 0.01);
  EXPECT_NEAR(var, 1.0 / 12.0, 0.01);
}

TEST(rng, below_unbiased_covers_faces) {
  rng g{13};
  std::array<int, 6> counts{};
  for (int i = 0; i < 10000; ++i) {
    const std::uint64_t v = g.below(6);
    ASSERT_LT(v, 6u);
    ++counts[static_cast<std::size_t>(v)];
  }
  for (const int c : counts) EXPECT_GT(c, 1400);  // fair-ish die
}

TEST(rng, uniform_bounds) {
  rng g{99};
  for (int i = 0; i < 10000; ++i) {
    const double x = g.uniform(2.0, 5.0);
    ASSERT_GE(x, 2.0);
    ASSERT_LT(x, 5.0);
  }
}
