#include <gtest/gtest.h>
#include <ax/num/quad.hpp>

#include <cmath>

using ax::num::integrate;
using ax::num::integrate_ts;

namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;
}

TEST(quad_gk, polynomial_exact) {
  const auto r = integrate([](double x) { return x * x; }, 0.0, 1.0);
  EXPECT_NEAR(r.value, 1.0 / 3.0, 1e-13);
}

TEST(quad_gk, sine_over_period) {
  const auto r = integrate([](double x) { return std::sin(x); }, 0.0, kPi);
  EXPECT_NEAR(r.value, 2.0, 1e-12);
}

TEST(quad_gk, gaussian_wide_interval) {
  const auto r =
      integrate([](double x) { return std::exp(-x * x); }, -10.0, 10.0);
  EXPECT_NEAR(r.value, std::sqrt(kPi), 1e-10);
}

TEST(quad_gk, oscillatory) {
  const auto r =
      integrate([](double x) { return std::sin(10.0 * x); }, 0.0, 20.0);
  EXPECT_NEAR(r.value, (1.0 - std::cos(200.0)) / 10.0, 1e-9);
}

TEST(quad_gk, reversed_limits_flip_sign) {
  const auto r = integrate([](double x) { return x; }, 1.0, 0.0);
  EXPECT_NEAR(r.value, -0.5, 1e-13);
}

TEST(quad_gk, error_estimate_sane) {
  const auto r = integrate([](double x) { return std::cos(x); }, 0.0, 1.0);
  EXPECT_LT(r.error_est, 1e-10);
  EXPECT_GT(r.evals, 0);
}

TEST(quad_ts, inverse_sqrt_singularity) {
  const auto r =
      integrate_ts([](double x) { return 1.0 / std::sqrt(x); }, 0.0, 1.0);
  EXPECT_NEAR(r.value, 2.0, 1e-9);
}

TEST(quad_ts, log_singularity) {
  const auto r = integrate_ts([](double x) { return std::log(x); }, 0.0, 1.0);
  EXPECT_NEAR(r.value, -1.0, 1e-9);
}

TEST(quad_ts, smooth_agrees_with_gk) {
  const auto r =
      integrate_ts([](double x) { return std::exp(x); }, 0.0, 2.0);
  EXPECT_NEAR(r.value, std::exp(2.0) - 1.0, 1e-10);
}

TEST(quad_ts, both_endpoints_singular) {
  // ∫₀¹ 1/sqrt(x(1-x)) dx = pi. Tolerance limited by double representation
  // near x = 1: the integrand computes 1-x by subtraction, losing the
  // sub-eps tail (see quad.hpp note).
  const auto r = integrate_ts(
      [](double x) { return 1.0 / std::sqrt(x * (1.0 - x)); }, 0.0, 1.0);
  EXPECT_NEAR(r.value, kPi, 1e-7);
}
