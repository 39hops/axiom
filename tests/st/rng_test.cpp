#include <gtest/gtest.h>
#include <ax/st/rng.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

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

namespace {
struct stats {
  double mean, var, skew;
};
template <typename F>
stats sample_stats(int n, F draw) {
  double s1 = 0;
  std::vector<double> xs(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    xs[static_cast<std::size_t>(i)] = draw();
    s1 += xs[static_cast<std::size_t>(i)];
  }
  const double mean = s1 / n;
  double m2 = 0, m3 = 0;
  for (const double x : xs) {
    const double d = x - mean;
    m2 += d * d;
    m3 += d * d * d;
  }
  m2 /= n;
  m3 /= n;
  return {mean, m2, m3 / std::pow(m2, 1.5)};
}
}  // namespace

TEST(rng_normal, moments_and_tails) {
  rng g{101};
  const int n = 200000;
  const stats s = sample_stats(n, [&] {
    const double x = g.normal();
    EXPECT_LT(std::abs(x), 6.0);  // |x|>6 prob ~ 2e-9, never in 200k
    return x;
  });
  EXPECT_NEAR(s.mean, 0.0, 0.01);
  EXPECT_NEAR(s.var, 1.0, 0.02);
  EXPECT_NEAR(s.skew, 0.0, 0.05);
}

TEST(rng_normal, mu_sigma_scaling) {
  rng g{102};
  const stats s = sample_stats(100000, [&] { return g.normal(3.0, 2.0); });
  EXPECT_NEAR(s.mean, 3.0, 0.03);
  EXPECT_NEAR(s.var, 4.0, 0.1);
}

TEST(rng_exponential, moments) {
  rng g{103};
  const stats s = sample_stats(200000, [&] { return g.exponential(2.0); });
  EXPECT_NEAR(s.mean, 0.5, 0.01);
  EXPECT_NEAR(s.var, 0.25, 0.01);
}

TEST(rng_gamma, moments_shape_above_one) {
  rng g{104};
  const stats s = sample_stats(200000, [&] { return g.gamma(4.5, 2.0); });
  EXPECT_NEAR(s.mean, 9.0, 0.15);  // shape*scale
  EXPECT_NEAR(s.var, 18.0, 0.8);   // shape*scale^2
}

TEST(rng_gamma, moments_shape_below_one) {
  rng g{105};
  const stats s = sample_stats(200000, [&] { return g.gamma(0.5, 1.0); });
  EXPECT_NEAR(s.mean, 0.5, 0.01);
  EXPECT_NEAR(s.var, 0.5, 0.03);
}

TEST(rng_sampling, deterministic_same_seed) {
  rng a{7}, b{7};
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(a.normal(), b.normal());
    EXPECT_EQ(a.gamma(2.5, 1.0), b.gamma(2.5, 1.0));
  }
}
