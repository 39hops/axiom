// Time series. Fixed seeds -> deterministic; tolerances have ~3x slack over
// sampling error at the stated lengths.
#include <ax/st/rng.hpp>
#include <ax/st/ts.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <vector>

using namespace ax::st;

namespace {

std::vector<double> simulate_ar(std::span<const double> phi, std::size_t n,
                                std::uint64_t seed, std::size_t warmup = 500) {
  rng g{seed};
  const std::size_t p = phi.size();
  std::vector<double> x(warmup + n, 0.0);
  for (std::size_t t = p; t < x.size(); ++t) {
    double v = g.normal();
    for (std::size_t i = 0; i < p; ++i) v += phi[i] * x[t - 1 - i];
    x[t] = v;
  }
  return {x.begin() + static_cast<std::ptrdiff_t>(warmup), x.end()};
}

}  // namespace

TEST(acf_fn, white_noise_uncorrelated) {
  rng g{31};
  std::vector<double> x(2000);
  for (auto& v : x) v = g.normal();
  auto a = acf(x, 10);
  ASSERT_EQ(a.size(), 11u);
  EXPECT_DOUBLE_EQ(a[0], 1.0);
  for (std::size_t k = 1; k <= 10; ++k) EXPECT_LT(std::abs(a[k]), 0.05);
}

TEST(acf_fn, ar1_geometric_decay) {
  std::vector<double> phi{0.8};
  auto x = simulate_ar(phi, 20000, 33);
  auto a = acf(x, 5);
  for (std::size_t k = 1; k <= 5; ++k)
    EXPECT_NEAR(a[k], std::pow(0.8, static_cast<double>(k)), 0.08);
}

TEST(acf_fn, max_lag_too_large_throws) {
  std::vector<double> x{1, 2, 3};
  EXPECT_THROW((void)acf(x, 3), std::invalid_argument);
}

TEST(pacf_fn, ar2_cutoff_after_lag_two) {
  std::vector<double> phi{0.5, -0.3};
  auto x = simulate_ar(phi, 20000, 37);
  auto p = pacf(x, 8);
  ASSERT_EQ(p.size(), 8u);  // lags 1..8
  EXPECT_GT(std::abs(p[0]), 0.2);   // lag 1 nonzero
  EXPECT_GT(std::abs(p[1]), 0.15);  // lag 2 ~ -0.3
  for (std::size_t k = 2; k < 8; ++k) EXPECT_LT(std::abs(p[k]), 0.06);
}

TEST(pacf_fn, lag_one_equals_acf_lag_one) {
  std::vector<double> phi{0.6};
  auto x = simulate_ar(phi, 3000, 41);
  auto a = acf(x, 1);
  auto p = pacf(x, 1);
  EXPECT_DOUBLE_EQ(p[0], a[1]);
}

TEST(ar_fit_fn, recovers_ar2) {
  std::vector<double> phi{0.5, -0.3};
  auto x = simulate_ar(phi, 5000, 43);
  auto r = ar_fit(x, 2);
  ASSERT_EQ(r.phi.size(), 2u);
  EXPECT_NEAR(r.phi[0], 0.5, 0.05);
  EXPECT_NEAR(r.phi[1], -0.3, 0.05);
  EXPECT_NEAR(r.sigma2, 1.0, 0.1);
}

TEST(ar_fit_fn, mean_form_intercept) {
  // x_t = 10 + AR(1) noise: intercept = mean * (1 - phi)
  std::vector<double> phi{0.6};
  auto x = simulate_ar(phi, 5000, 47);
  for (auto& v : x) v += 10.0;
  auto r = ar_fit(x, 1);
  EXPECT_NEAR(r.phi[0], 0.6, 0.05);
  const double mean_est = r.intercept / (1.0 - r.phi[0]);
  EXPECT_NEAR(mean_est, 10.0, 0.2);
}

TEST(ar_fit_fn, p_too_large_throws) {
  std::vector<double> x{1, 2, 3};
  EXPECT_THROW((void)ar_fit(x, 3), std::invalid_argument);
}

TEST(arma_fit_fn, recovers_arma11) {
  // simulate ARMA(1,1): x_t = 0.6 x_{t-1} + e_t + 0.4 e_{t-1}
  rng g{53};
  const std::size_t warmup = 500, n = 4000;
  std::vector<double> x(warmup + n, 0.0);
  double e_prev = 0.0;
  for (std::size_t t = 1; t < x.size(); ++t) {
    const double e = g.normal();
    x[t] = 0.6 * x[t - 1] + e + 0.4 * e_prev;
    e_prev = e;
  }
  std::vector<double> xs(x.begin() + warmup, x.end());
  auto r = arma_fit(xs, 1, 1);
  ASSERT_TRUE(r.converged);
  ASSERT_EQ(r.phi.size(), 1u);
  ASSERT_EQ(r.theta.size(), 1u);
  EXPECT_NEAR(r.phi[0], 0.6, 0.1);
  EXPECT_NEAR(r.theta[0], 0.4, 0.1);
  EXPECT_NEAR(r.sigma2, 1.0, 0.15);
}

TEST(arma_fit_fn, pure_ar_agrees_with_ar_fit) {
  std::vector<double> phi{0.7};
  auto x = simulate_ar(phi, 5000, 59);
  auto yw = ar_fit(x, 1);
  auto css = arma_fit(x, 1, 0);
  ASSERT_TRUE(css.converged);
  EXPECT_NEAR(css.phi[0], yw.phi[0], 0.02);
}

TEST(periodogram_fn, sinusoid_peak_at_true_frequency) {
  const std::size_t n = 512;
  std::vector<double> x(n);
  for (std::size_t t = 0; t < n; ++t)
    x[t] = std::sin(2.0 * std::numbers::pi * 0.1 * static_cast<double>(t));
  auto r = periodogram(x);
  ASSERT_EQ(r.freq.size(), r.power.size());
  std::size_t imax = 0;
  for (std::size_t i = 1; i < r.power.size(); ++i)
    if (r.power[i] > r.power[imax]) imax = i;
  EXPECT_NEAR(r.freq[imax], 0.1, 1.0 / static_cast<double>(n));
}

TEST(periodogram_fn, white_noise_roughly_flat) {
  rng g{61};
  std::vector<double> x(1024);
  for (auto& v : x) v = g.normal();
  auto r = periodogram(x);
  double mean = 0.0, mx = 0.0;
  for (double p : r.power) {
    mean += p;
    mx = std::max(mx, p);
  }
  mean /= static_cast<double>(r.power.size());
  EXPECT_LT(mx / mean, 15.0);  // no dominant line in white noise
}

TEST(periodogram_fn, empty_throws) {
  std::vector<double> x;
  EXPECT_THROW((void)periodogram(x), std::invalid_argument);
}
