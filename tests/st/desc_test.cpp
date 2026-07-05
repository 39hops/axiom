#include <gtest/gtest.h>
#include <ax/st/desc.hpp>
#include <ax/st/rng.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

using namespace ax::st;
using ax::la::mat;

TEST(desc_moments, known_small_vector) {
  const std::vector<double> xs{2, 4, 4, 4, 5, 5, 7, 9};
  const moments m = describe(xs);
  EXPECT_EQ(m.n, 8u);
  EXPECT_NEAR(m.mean, 5.0, 1e-14);
  EXPECT_NEAR(m.var(), 32.0 / 7.0, 1e-13);  // sample variance
  EXPECT_NEAR(m.sd(), std::sqrt(32.0 / 7.0), 1e-13);
}

TEST(desc_moments, matches_naive_two_pass) {
  rng g{55};
  std::vector<double> xs(10000);
  for (double& x : xs) x = g.normal(2.0, 3.0);
  const moments m = describe(xs);
  // naive two-pass reference
  double s = 0;
  for (const double x : xs) s += x;
  const double mean = s / static_cast<double>(xs.size());
  double m2 = 0, m3 = 0, m4 = 0;
  for (const double x : xs) {
    const double d = x - mean;
    m2 += d * d;
    m3 += d * d * d;
    m4 += d * d * d * d;
  }
  const double n = static_cast<double>(xs.size());
  EXPECT_NEAR(m.mean, mean, 1e-12);
  EXPECT_NEAR(m.var(), m2 / (n - 1.0), 1e-9);
  EXPECT_NEAR(m.skewness(), (m3 / n) / std::pow(m2 / n, 1.5), 1e-9);
  EXPECT_NEAR(m.kurtosis(), n * m4 / (m2 * m2) - 3.0, 1e-9);
}

TEST(desc_moments, shifted_data_stable) {
  rng g{56};
  std::vector<double> raw(10000), shifted(10000);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    raw[i] = g.normal();
    shifted[i] = raw[i] + 1e9;
  }
  const moments a = describe(raw), b = describe(shifted);
  EXPECT_NEAR(b.var() / a.var(), 1.0, 1e-6);  // catastrophic-cancellation check
}

TEST(desc_moments, var_requires_two) {
  moments m;
  m.push(1.0);
  EXPECT_THROW(m.var(), std::domain_error);
  m.push(3.0);
  EXPECT_NEAR(m.var(), 2.0, 1e-14);
}

TEST(desc_quantile, r_type7) {
  const std::vector<double> xs{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  EXPECT_NEAR(quantile(xs, 0.25), 3.25, 1e-14);
  EXPECT_NEAR(quantile(xs, 0.5), 5.5, 1e-14);
  EXPECT_NEAR(quantile(xs, 0.0), 1.0, 1e-14);
  EXPECT_NEAR(quantile(xs, 1.0), 10.0, 1e-14);
  const std::vector<double> ps{0.25, 0.75};
  const std::vector<double> qs = quantiles(xs, ps);
  EXPECT_NEAR(qs[0], 3.25, 1e-14);
  EXPECT_NEAR(qs[1], 7.75, 1e-14);
  EXPECT_THROW(quantile(xs, -0.1), std::invalid_argument);
  EXPECT_THROW(quantile({}, 0.5), std::invalid_argument);
}

TEST(desc_cov, known_matrix) {
  // 4 observations, 2 variables
  mat x{{1, 2}, {2, 4}, {3, 5}, {4, 9}};
  const mat c = covariance(x);
  // hand: col0 mean 2.5, col1 mean 5; cov00 = sum d0^2/(n-1)
  const double d0[] = {-1.5, -0.5, 0.5, 1.5};
  const double d1[] = {-3.0, -1.0, 0.0, 4.0};
  double c00 = 0, c01 = 0, c11 = 0;
  for (int i = 0; i < 4; ++i) {
    c00 += d0[i] * d0[i];
    c01 += d0[i] * d1[i];
    c11 += d1[i] * d1[i];
  }
  c00 /= 3;
  c01 /= 3;
  c11 /= 3;
  EXPECT_NEAR(c(0, 0), c00, 1e-13);
  EXPECT_NEAR(c(0, 1), c01, 1e-13);
  EXPECT_NEAR(c(1, 0), c01, 1e-13);
  EXPECT_NEAR(c(1, 1), c11, 1e-13);
  const mat r = correlation(x);
  EXPECT_NEAR(r(0, 0), 1.0, 1e-13);
  EXPECT_NEAR(r(1, 1), 1.0, 1e-13);
  EXPECT_NEAR(r(0, 1), c01 / std::sqrt(c00 * c11), 1e-13);
  EXPECT_THROW(covariance(mat(1, 2)), std::invalid_argument);
}

TEST(desc_corr, perfect_linear) {
  mat x(5, 2);
  for (std::size_t i = 0; i < 5; ++i) {
    x(i, 0) = static_cast<double>(i);
    x(i, 1) = 3.0 * static_cast<double>(i) + 1.0;
  }
  EXPECT_NEAR(correlation(x)(0, 1), 1.0, 1e-12);
}
