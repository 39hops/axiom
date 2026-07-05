#include <gtest/gtest.h>
#include <ax/num/opt.hpp>

#include <cmath>

using ax::la::vec;
using ax::num::bfgs;
using ax::num::minimize;
using ax::num::nelder_mead;

namespace {
double rosenbrock(const vec& x) {
  const double a = 1.0 - x[0];
  const double b = x[1] - x[0] * x[0];
  return a * a + 100.0 * b * b;
}
vec rosenbrock_grad(const vec& x) {
  vec g(2);
  g[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
  g[1] = 200.0 * (x[1] - x[0] * x[0]);
  return g;
}
}  // namespace

TEST(opt_1d, parabola) {
  const auto r = minimize([](double x) { return (x - 2.0) * (x - 2.0) + 3.0; },
                          -10.0, 10.0);
  EXPECT_TRUE(r.converged);
  EXPECT_NEAR(r.x, 2.0, 1e-8);
  EXPECT_NEAR(r.fx, 3.0, 1e-12);
}

TEST(opt_1d, quartic_interior_min) {
  // f = x^4 - 3x^3 + 2, f' = 4x^3 - 9x^2 → interior min at x = 9/4
  const auto r =
      minimize([](double x) { return std::pow(x, 4) - 3.0 * std::pow(x, 3) + 2.0; },
               0.5, 5.0);
  EXPECT_TRUE(r.converged);
  EXPECT_NEAR(r.x, 2.25, 1e-7);
}

TEST(opt_1d, cosine) {
  const auto r = minimize([](double x) { return std::cos(x); }, 2.0, 5.0);
  EXPECT_NEAR(r.x, 3.141592653589793, 1e-7);
  EXPECT_NEAR(r.fx, -1.0, 1e-12);
}

TEST(opt_nm, rosenbrock) {
  vec x0(2);
  x0[0] = -1.2;
  x0[1] = 1.0;
  const auto r = nelder_mead(rosenbrock, x0);
  EXPECT_TRUE(r.converged);
  EXPECT_NEAR(r.x[0], 1.0, 1e-4);
  EXPECT_NEAR(r.x[1], 1.0, 1e-4);
}

TEST(opt_nm, sphere_far_start) {
  vec x0(3);
  x0[0] = 50.0;
  x0[1] = -30.0;
  x0[2] = 10.0;
  const auto r = nelder_mead(
      [](const vec& x) { return dot(x, x); }, x0);
  EXPECT_TRUE(r.converged);
  EXPECT_LT(r.x.norm(), 1e-4);
}

TEST(opt_nm, unconverged_flag) {
  vec x0(2);
  x0[0] = -1.2;
  x0[1] = 1.0;
  const auto r = nelder_mead(rosenbrock, x0, 1e-10, 5);
  EXPECT_FALSE(r.converged);
}

TEST(opt_bfgs, rosenbrock_analytic_gradient) {
  vec x0(2);
  x0[0] = -1.2;
  x0[1] = 1.0;
  const auto r = bfgs(rosenbrock, rosenbrock_grad, x0);
  EXPECT_TRUE(r.converged);
  EXPECT_NEAR(r.x[0], 1.0, 1e-5);
  EXPECT_NEAR(r.x[1], 1.0, 1e-5);
}

TEST(opt_bfgs, rosenbrock_numeric_gradient) {
  vec x0(2);
  x0[0] = -1.2;
  x0[1] = 1.0;
  const auto r = bfgs(rosenbrock, x0);
  EXPECT_TRUE(r.converged);
  EXPECT_NEAR(r.x[0], 1.0, 1e-5);
  EXPECT_NEAR(r.x[1], 1.0, 1e-5);
}

TEST(opt_bfgs, quadratic_bowl_5d) {
  // f = sum (i+1) x_i^2 — SPD diagonal
  const auto f = [](const vec& x) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
      s += static_cast<double>(i + 1) * x[i] * x[i];
    return s;
  };
  vec x0(5, 3.0);
  const auto r = bfgs(f, x0);
  EXPECT_TRUE(r.converged);
  EXPECT_LT(r.x.norm(), 1e-6);
  EXPECT_LT(r.fx, 1e-10);
}
