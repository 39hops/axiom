#include <gtest/gtest.h>
#include <ax/num/root.hpp>

#include <cmath>
#include <stdexcept>

using ax::num::bisect;
using ax::num::brent;
using ax::num::newton;

namespace {
double cosx_minus_x(double x) { return std::cos(x) - x; }
double cubic(double x) { return x * x * x - 2.0 * x - 5.0; }
}  // namespace

TEST(root_bisect, cos_fixed_point) {
  const auto r = bisect(cosx_minus_x, 0.0, 1.0);
  EXPECT_TRUE(r.converged);
  EXPECT_NEAR(r.x, 0.7390851332151607, 1e-10);
  EXPECT_LT(std::abs(r.fx), 1e-10);
}

TEST(root_bisect, bracket_violation_throws) {
  EXPECT_THROW(bisect(cosx_minus_x, 2.0, 3.0), std::invalid_argument);
}

TEST(root_bisect, unconverged_flag) {
  const auto r = bisect(cosx_minus_x, 0.0, 1.0, 1e-12, 3);
  EXPECT_FALSE(r.converged);
}

TEST(root_brent, cos_fixed_point) {
  const auto r = brent(cosx_minus_x, 0.0, 1.0);
  EXPECT_TRUE(r.converged);
  EXPECT_NEAR(r.x, 0.7390851332151607, 1e-13);
}

TEST(root_brent, cubic) {
  const auto r = brent(cubic, 2.0, 3.0);
  EXPECT_TRUE(r.converged);
  EXPECT_LT(std::abs(cubic(r.x)), 1e-12);  // cross-check, no digit oracle
}

TEST(root_brent, root_at_endpoint) {
  const auto r = brent([](double x) { return x - 1.0; }, 1.0, 2.0);
  EXPECT_TRUE(r.converged);
  EXPECT_NEAR(r.x, 1.0, 1e-14);
}

TEST(root_newton, sqrt2) {
  const auto r = newton([](double x) { return x * x - 2.0; },
                        [](double x) { return 2.0 * x; }, 1.0);
  EXPECT_TRUE(r.converged);
  EXPECT_NEAR(r.x, std::sqrt(2.0), 1e-14);
  EXPECT_LT(r.iters, 10);
}

TEST(root_newton, cos_fixed_point) {
  const auto r = newton(cosx_minus_x,
                        [](double x) { return -std::sin(x) - 1.0; }, 0.5);
  EXPECT_TRUE(r.converged);
  EXPECT_NEAR(r.x, 0.7390851332151607, 1e-13);
}

TEST(root_newton, zero_derivative_throws) {
  EXPECT_THROW(newton([](double x) { return x * x + 1.0; },
                      [](double) { return 0.0; }, 0.0),
               std::runtime_error);
}
