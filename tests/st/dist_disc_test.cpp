#include <gtest/gtest.h>
#include <ax/st/dist.hpp>

#include <cmath>
#include <stdexcept>

using namespace ax::st;

namespace {
template <typename D>
void check_disc_sample_moments(const D& d, int n = 100000,
                               double rtol = 0.03) {
  rng g{77};
  double s1 = 0, s2 = 0;
  for (int i = 0; i < n; ++i) {
    const double x = static_cast<double>(d.sample(g));
    s1 += x;
    s2 += x * x;
  }
  const double mean = s1 / n;
  const double var = s2 / n - mean * mean;
  EXPECT_NEAR(mean, d.mean(), rtol * std::max(d.mean(), 1.0));
  EXPECT_NEAR(var, d.var(), 3.0 * rtol * std::max(d.var(), 1.0));
}

double binom_coef(int n, int k) {
  double r = 1.0;
  for (int i = 1; i <= k; ++i) r = r * (n - k + i) / i;
  return r;
}
}  // namespace

TEST(dist_binomial, pmf_cdf_quantile) {
  const binomial_dist d{10, 0.3};
  EXPECT_NEAR(d.pmf(3),
              binom_coef(10, 3) * std::pow(0.3, 3) * std::pow(0.7, 7), 1e-13);
  EXPECT_NEAR(d.pmf(0), std::pow(0.7, 10), 1e-14);
  // cdf(k) = sum of pmf
  double acc = 0.0;
  for (int k = 0; k <= 4; ++k) acc += d.pmf(k);
  EXPECT_NEAR(d.cdf(4), acc, 1e-12);
  EXPECT_EQ(d.cdf(-1), 0.0);
  EXPECT_EQ(d.cdf(10), 1.0);
  // quantile: smallest k with cdf(k) >= p
  for (const double p : {0.05, 0.3, 0.5, 0.95}) {
    const int k = d.quantile(p);
    EXPECT_GE(d.cdf(k), p);
    if (k > 0) EXPECT_LT(d.cdf(k - 1), p);
  }
  check_disc_sample_moments(d);  // mean 3, var 2.1
  EXPECT_THROW(binomial_dist(-1, 0.5), std::invalid_argument);
  EXPECT_THROW(binomial_dist(10, 1.5), std::invalid_argument);
}

TEST(dist_poisson, pmf_cdf_quantile) {
  const poisson_dist d{1.5};
  EXPECT_NEAR(d.pmf(2), std::exp(-1.5) * 1.5 * 1.5 / 2.0, 1e-14);
  EXPECT_NEAR(d.pmf(0), std::exp(-1.5), 1e-14);
  double acc = 0.0;
  for (int k = 0; k <= 3; ++k) acc += d.pmf(k);
  EXPECT_NEAR(d.cdf(3), acc, 1e-12);
  for (const double p : {0.05, 0.5, 0.95}) {
    const int k = d.quantile(p);
    EXPECT_GE(d.cdf(k), p);
    if (k > 0) EXPECT_LT(d.cdf(k - 1), p);
  }
  check_disc_sample_moments(d);
  EXPECT_THROW(poisson_dist(0.0), std::invalid_argument);
}

TEST(dist_poisson, large_lambda_sampling) {
  const poisson_dist d{80.0};
  check_disc_sample_moments(d, 100000, 0.02);
}

TEST(dist_negbinom, pmf_cdf_quantile) {
  const negbinom_dist d{3, 0.4};  // failures before 3rd success, p = 0.4
  EXPECT_NEAR(d.pmf(5),
              binom_coef(7, 5) * std::pow(0.4, 3) * std::pow(0.6, 5), 1e-13);
  EXPECT_NEAR(d.pmf(0), std::pow(0.4, 3), 1e-14);
  double acc = 0.0;
  for (int k = 0; k <= 6; ++k) acc += d.pmf(k);
  EXPECT_NEAR(d.cdf(6), acc, 1e-12);
  for (const double p : {0.1, 0.5, 0.9}) {
    const int k = d.quantile(p);
    EXPECT_GE(d.cdf(k), p);
    if (k > 0) EXPECT_LT(d.cdf(k - 1), p);
  }
  check_disc_sample_moments(d);  // mean r(1-p)/p = 4.5, var 11.25
  EXPECT_THROW(negbinom_dist(0, 0.5), std::invalid_argument);
  EXPECT_THROW(negbinom_dist(3, 0.0), std::invalid_argument);
}
