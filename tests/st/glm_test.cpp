// GLM via IRLS. Oracles are closed forms: with a single binary predictor the
// logistic/Poisson MLE is exact (group log-odds / log-means), and null models
// have analytic intercepts. Score equations X'(y-mu)=0 (canonical link) give
// an independent convergence check.
#include <ax/st/reg.hpp>
#include <ax/st/rng.hpp>

#include <gtest/gtest.h>

#include <cmath>

using namespace ax::st;
using ax::la::mat;
using ax::la::vec;

TEST(glm_logistic, binary_predictor_closed_form) {
  // x=0: y {0,0,1} (odds 1/2); x=1: y {0,1,1,1} (odds 3)
  // beta0 = log(1/2), beta1 = log(3) - log(1/2) = log(6)
  // Fisher info diag: var(b0) = 1/(3*(1/3)*(2/3)) = 3/2
  //                   var(b1) = 3/2 + 1/(4*(3/4)*(1/4)) = 3/2 + 4/3 = 17/6
  mat x(7, 1);
  vec y(7);
  double xs[] = {0, 0, 0, 1, 1, 1, 1};
  double ys[] = {0, 0, 1, 0, 1, 1, 1};
  for (std::size_t i = 0; i < 7; ++i) {
    x(i, 0) = xs[i];
    y[i] = ys[i];
  }
  auto r = glm_fit(x, y, glm_family::logistic);
  ASSERT_TRUE(r.converged);
  EXPECT_NEAR(r.beta[0], std::log(0.5), 1e-8);
  EXPECT_NEAR(r.beta[1], std::log(6.0), 1e-8);
  EXPECT_NEAR(r.stderrs[0], std::sqrt(1.5), 1e-6);
  EXPECT_NEAR(r.stderrs[1], std::sqrt(17.0 / 6.0), 1e-6);
}

TEST(glm_logistic, null_model_intercept_is_logit_of_mean) {
  mat x(10, 0);  // no predictors, intercept only
  vec y{1, 0, 1, 1, 0, 1, 1, 1, 0, 1};  // mean 0.7
  auto r = glm_fit(x, y, glm_family::logistic);
  ASSERT_TRUE(r.converged);
  ASSERT_EQ(r.beta.size(), 1u);
  EXPECT_NEAR(r.beta[0], std::log(0.7 / 0.3), 1e-8);
}

TEST(glm_logistic, recovers_simulated_coefficients) {
  rng g{11};
  const std::size_t n = 500;
  mat x(n, 1);
  vec y(n);
  for (std::size_t i = 0; i < n; ++i) {
    x(i, 0) = g.normal();
    const double p = 1.0 / (1.0 + std::exp(-(-1.0 + 2.0 * x(i, 0))));
    y[i] = (g.next_double() < p) ? 1.0 : 0.0;
  }
  auto r = glm_fit(x, y, glm_family::logistic);
  ASSERT_TRUE(r.converged);
  EXPECT_LT(r.iters, 25u);
  EXPECT_LT(std::abs(r.beta[0] - (-1.0)), 3.0 * r.stderrs[0]);
  EXPECT_LT(std::abs(r.beta[1] - 2.0), 3.0 * r.stderrs[1]);
}

TEST(glm_logistic, score_equations_hold_at_fit) {
  // canonical link: X'(y - mu) = 0 at the MLE, including intercept column
  rng g{13};
  const std::size_t n = 80;
  mat x(n, 2);
  vec y(n);
  for (std::size_t i = 0; i < n; ++i) {
    x(i, 0) = g.normal();
    x(i, 1) = g.uniform(-1.0, 1.0);
    const double p =
        1.0 / (1.0 + std::exp(-(0.3 + 0.8 * x(i, 0) - 1.1 * x(i, 1))));
    y[i] = (g.next_double() < p) ? 1.0 : 0.0;
  }
  auto r = glm_fit(x, y, glm_family::logistic);
  ASSERT_TRUE(r.converged);
  double s0 = 0, s1 = 0, s2 = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double eta = r.beta[0] + r.beta[1] * x(i, 0) + r.beta[2] * x(i, 1);
    const double mu = 1.0 / (1.0 + std::exp(-eta));
    s0 += y[i] - mu;
    s1 += x(i, 0) * (y[i] - mu);
    s2 += x(i, 1) * (y[i] - mu);
  }
  EXPECT_NEAR(s0, 0.0, 1e-6);
  EXPECT_NEAR(s1, 0.0, 1e-6);
  EXPECT_NEAR(s2, 0.0, 1e-6);
}

TEST(glm_logistic, perfect_separation_flagged_not_thrown) {
  mat x(4, 1);
  vec y{0, 0, 1, 1};
  for (std::size_t i = 0; i < 4; ++i) x(i, 0) = static_cast<double>(i + 1);
  auto r = glm_fit(x, y, glm_family::logistic);
  EXPECT_FALSE(r.converged);
  for (std::size_t j = 0; j < r.beta.size(); ++j)
    EXPECT_TRUE(std::isfinite(r.beta[j]));
}

TEST(glm_logistic, invalid_response_throws) {
  mat x(3, 1);
  vec y{0, 2, 1};  // 2 not allowed
  for (std::size_t i = 0; i < 3; ++i) x(i, 0) = static_cast<double>(i);
  EXPECT_THROW((void)glm_fit(x, y, glm_family::logistic),
               std::invalid_argument);
}

TEST(glm_poisson, binary_predictor_closed_form) {
  // x=0: counts {1,2,3} mean 2; x=1: counts {4,6,8,6} mean 6
  // beta0 = log 2, beta1 = log 6 - log 2 = log 3
  // var(b0) = 1/(3*2) = 1/6; var(b1) = 1/6 + 1/(4*6) = 1/6 + 1/24 = 5/24
  mat x(7, 1);
  vec y{1, 2, 3, 4, 6, 8, 6};
  double xs[] = {0, 0, 0, 1, 1, 1, 1};
  for (std::size_t i = 0; i < 7; ++i) x(i, 0) = xs[i];
  auto r = glm_fit(x, y, glm_family::poisson);
  ASSERT_TRUE(r.converged);
  EXPECT_NEAR(r.beta[0], std::log(2.0), 1e-8);
  EXPECT_NEAR(r.beta[1], std::log(3.0), 1e-8);
  EXPECT_NEAR(r.stderrs[0], std::sqrt(1.0 / 6.0), 1e-6);
  EXPECT_NEAR(r.stderrs[1], std::sqrt(5.0 / 24.0), 1e-6);
}

TEST(glm_poisson, null_model_intercept_is_log_mean) {
  mat x(5, 0);
  vec y{2, 3, 5, 4, 6};  // mean 4
  auto r = glm_fit(x, y, glm_family::poisson);
  ASSERT_TRUE(r.converged);
  EXPECT_NEAR(r.beta[0], std::log(4.0), 1e-8);
}

TEST(glm_poisson, recovers_simulated_coefficients) {
  rng g{17};
  const std::size_t n = 400;
  mat x(n, 1);
  vec y(n);
  for (std::size_t i = 0; i < n; ++i) {
    x(i, 0) = g.uniform(0.0, 2.0);
    const double lambda = std::exp(0.5 + 0.7 * x(i, 0));
    // inverse-cdf poisson draw via cumulative sum
    double u = g.next_double(), c = std::exp(-lambda), acc = c;
    int k = 0;
    while (u > acc && k < 1000) {
      ++k;
      c *= lambda / k;
      acc += c;
    }
    y[i] = k;
  }
  auto r = glm_fit(x, y, glm_family::poisson);
  ASSERT_TRUE(r.converged);
  EXPECT_LT(std::abs(r.beta[0] - 0.5), 3.0 * r.stderrs[0]);
  EXPECT_LT(std::abs(r.beta[1] - 0.7), 3.0 * r.stderrs[1]);
}

TEST(glm_poisson, extreme_predictor_no_overflow) {
  // huge predictor values would overflow exp(eta) without the eta clamp;
  // fit must return finite coefficients (converged or not)
  mat x(4, 1);
  vec y{1, 2, 1000, 2000};
  double xs[] = {0.0, 1.0, 500.0, 1000.0};
  for (std::size_t i = 0; i < 4; ++i) x(i, 0) = xs[i];
  auto r = glm_fit(x, y, glm_family::poisson);
  for (std::size_t j = 0; j < r.beta.size(); ++j)
    EXPECT_TRUE(std::isfinite(r.beta[j]));
  EXPECT_TRUE(std::isfinite(r.deviance));
}

TEST(glm_poisson, negative_response_throws) {
  mat x(3, 1);
  vec y{1, -1, 2};
  for (std::size_t i = 0; i < 3; ++i) x(i, 0) = static_cast<double>(i);
  EXPECT_THROW((void)glm_fit(x, y, glm_family::poisson),
               std::invalid_argument);
}

TEST(glm_deviance, logistic_null_model_analytic) {
  // deviance of intercept-only binary model:
  // -2 [ k log p + (n-k) log(1-p) ], p = k/n
  mat x(10, 0);
  vec y{1, 0, 1, 1, 0, 1, 1, 1, 0, 1};  // k=7, n=10
  auto r = glm_fit(x, y, glm_family::logistic);
  const double p = 0.7;
  const double expected = -2.0 * (7.0 * std::log(p) + 3.0 * std::log(1 - p));
  EXPECT_NEAR(r.deviance, expected, 1e-8);
}
