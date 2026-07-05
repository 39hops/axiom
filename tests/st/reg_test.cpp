// OLS regression. Hand oracle: simple regression on
// x = {1..5}, y = {2,4,5,4,5}: Sxx=10, Sxy=6 -> slope 0.6, intercept 2.2,
// RSS = 2.4, sigma2 = 0.8, SE(slope) = sqrt(0.08), SE(intercept) = sqrt(0.88),
// SST = 6 -> R^2 = 0.6, adj R^2 = 1 - 0.4*4/3.
#include <ax/la/decomp.hpp>
#include <ax/st/dist.hpp>
#include <ax/st/reg.hpp>
#include <ax/st/rng.hpp>

#include <gtest/gtest.h>

#include <cmath>

using namespace ax::st;
using ax::la::mat;
using ax::la::vec;

namespace {
mat column(std::initializer_list<double> xs) {
  mat m(xs.size(), 1);
  std::size_t i = 0;
  for (double x : xs) m(i++, 0) = x;
  return m;
}
}  // namespace

TEST(ols, simple_regression_hand_oracle) {
  mat x = column({1, 2, 3, 4, 5});
  vec y{2, 4, 5, 4, 5};
  auto r = ols(x, y);
  ASSERT_EQ(r.beta.size(), 2u);
  EXPECT_NEAR(r.beta[0], 2.2, 1e-12);  // intercept
  EXPECT_NEAR(r.beta[1], 0.6, 1e-12);  // slope
  EXPECT_NEAR(r.sigma2, 0.8, 1e-12);
  EXPECT_EQ(r.df_resid, 3u);
  EXPECT_NEAR(r.stderrs[0], std::sqrt(0.88), 1e-12);
  EXPECT_NEAR(r.stderrs[1], std::sqrt(0.08), 1e-12);
  EXPECT_NEAR(r.t_stats[1], 0.6 / std::sqrt(0.08), 1e-12);
  EXPECT_NEAR(r.r2, 0.6, 1e-12);
  EXPECT_NEAR(r.adj_r2, 1.0 - 0.4 * 4.0 / 3.0, 1e-12);
  // residuals: y - (2.2 + 0.6 x) = {-0.8, 0.6, 1.0, -0.6, -0.2}
  EXPECT_NEAR(r.residuals[2], 1.0, 1e-12);
  // p-value consistency with t(3)
  double tail = 1.0 - t_dist(3.0).cdf(r.t_stats[1]);
  EXPECT_NEAR(r.p_values[1], 2.0 * tail, 1e-12);
}

TEST(ols, exact_fit_r2_one) {
  mat x = column({1, 2, 3, 4});
  vec y{3, 5, 7, 9};  // y = 1 + 2x exactly
  auto r = ols(x, y);
  EXPECT_NEAR(r.beta[0], 1.0, 1e-10);
  EXPECT_NEAR(r.beta[1], 2.0, 1e-10);
  EXPECT_NEAR(r.r2, 1.0, 1e-10);
  EXPECT_NEAR(r.sigma2, 0.0, 1e-18);
  for (std::size_t i = 0; i < r.residuals.size(); ++i)
    EXPECT_NEAR(r.residuals[i], 0.0, 1e-10);
}

TEST(ols, multivariate_matches_normal_equations) {
  // 2 predictors + intercept; cross-check against inverse(X'X) X'y.
  mat x(6, 2);
  double xs1[] = {1, 2, 3, 4, 5, 6};
  double xs2[] = {2, 1, 4, 3, 6, 5};
  vec y{1.1, 1.9, 3.2, 3.8, 5.1, 5.9};
  for (std::size_t i = 0; i < 6; ++i) {
    x(i, 0) = xs1[i];
    x(i, 1) = xs2[i];
  }
  auto r = ols(x, y);
  // build design with intercept and solve normal equations independently
  mat d(6, 3);
  for (std::size_t i = 0; i < 6; ++i) {
    d(i, 0) = 1.0;
    d(i, 1) = xs1[i];
    d(i, 2) = xs2[i];
  }
  mat xtx(3, 3);
  vec xty(3);
  for (std::size_t a = 0; a < 3; ++a) {
    for (std::size_t b = 0; b < 3; ++b) {
      double s = 0;
      for (std::size_t i = 0; i < 6; ++i) s += d(i, a) * d(i, b);
      xtx(a, b) = s;
    }
    double s = 0;
    for (std::size_t i = 0; i < 6; ++i) s += d(i, a) * y[i];
    xty[a] = s;
  }
  vec beta_ne = ax::la::inverse(xtx) * xty;
  for (std::size_t j = 0; j < 3; ++j)
    EXPECT_NEAR(r.beta[j], beta_ne[j], 1e-10);
  // stderr consistency: cov = sigma2 * inverse(X'X)
  mat cov = ax::la::inverse(xtx);
  for (std::size_t j = 0; j < 3; ++j)
    EXPECT_NEAR(r.stderrs[j], std::sqrt(r.sigma2 * cov(j, j)), 1e-10);
}

TEST(ols, no_intercept) {
  mat x = column({1, 2, 3});
  vec y{2, 4, 6};  // y = 2x through origin
  auto r = ols(x, y, /*intercept=*/false);
  ASSERT_EQ(r.beta.size(), 1u);
  EXPECT_NEAR(r.beta[0], 2.0, 1e-12);
  EXPECT_NEAR(r.r2, 1.0, 1e-12);  // relative to zero, not mean
  EXPECT_EQ(r.df_resid, 2u);
}

TEST(ols, recovers_simulated_coefficients) {
  rng g{7};
  const std::size_t n = 200;
  mat x(n, 1);
  vec y(n);
  for (std::size_t i = 0; i < n; ++i) {
    x(i, 0) = g.uniform(0.0, 10.0);
    y[i] = 2.0 + 3.0 * x(i, 0) + g.normal(0.0, 1.0);
  }
  auto r = ols(x, y);
  EXPECT_LT(std::abs(r.beta[0] - 2.0), 3.0 * r.stderrs[0]);
  EXPECT_LT(std::abs(r.beta[1] - 3.0), 3.0 * r.stderrs[1]);
  EXPECT_LT(r.p_values[1], 1e-6);
  EXPECT_GT(r.r2, 0.95);
}

TEST(ols, rank_deficient_throws) {
  mat x(4, 2);
  for (std::size_t i = 0; i < 4; ++i) {
    x(i, 0) = static_cast<double>(i + 1);
    x(i, 1) = 2.0 * static_cast<double>(i + 1);  // duplicate direction
  }
  vec y{1, 2, 3, 4};
  EXPECT_THROW((void)ols(x, y), std::domain_error);
}

TEST(ols, size_mismatch_throws) {
  mat x = column({1, 2, 3});
  vec y{1, 2};
  EXPECT_THROW((void)ols(x, y), std::invalid_argument);
}
