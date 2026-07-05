#include <gtest/gtest.h>
#include <ax/num/ode.hpp>

#include <cmath>

using ax::la::vec;
using ax::num::solve_ivp;

TEST(ode, exponential_growth) {
  const auto r = solve_ivp(
      [](double, const vec& y) {
        vec dy(1);
        dy[0] = y[0];
        return dy;
      },
      0.0, 2.0, vec(1, 1.0));
  EXPECT_TRUE(r.converged);
  const double got = r.y.back()[0];
  EXPECT_NEAR(got / std::exp(2.0), 1.0, 1e-7);
  EXPECT_NEAR(r.t.back(), 2.0, 1e-12);
}

TEST(ode, harmonic_oscillator_round_trip) {
  // y'' = -y as system; over 10*pi returns to start
  const double t1 = 10.0 * 3.141592653589793238462643;
  vec y0(2);
  y0[0] = 1.0;
  y0[1] = 0.0;
  const auto r = solve_ivp(
      [](double, const vec& y) {
        vec dy(2);
        dy[0] = y[1];
        dy[1] = -y[0];
        return dy;
      },
      0.0, t1, y0, 1e-10, 1e-12);
  EXPECT_TRUE(r.converged);
  EXPECT_NEAR(r.y.back()[0], 1.0, 1e-6);  // cos(10*pi) = 1
  EXPECT_NEAR(r.y.back()[1], 0.0, 1e-6);
  // energy drift small at every accepted step
  for (const vec& y : r.y) {
    const double e = y[0] * y[0] + y[1] * y[1];
    EXPECT_NEAR(e, 1.0, 1e-6);
  }
}

TEST(ode, gaussian_decay) {
  // y' = -2 t y, y(0)=1 → y(t) = exp(-t^2)
  const auto r = solve_ivp(
      [](double t, const vec& y) {
        vec dy(1);
        dy[0] = -2.0 * t * y[0];
        return dy;
      },
      0.0, 3.0, vec(1, 1.0));
  EXPECT_TRUE(r.converged);
  EXPECT_NEAR(r.y.back()[0], std::exp(-9.0), 1e-9);
}

TEST(ode, fast_decay_more_steps_still_correct) {
  const auto r = solve_ivp(
      [](double, const vec& y) {
        vec dy(1);
        dy[0] = -50.0 * y[0];
        return dy;
      },
      0.0, 1.0, vec(1, 1.0));
  EXPECT_TRUE(r.converged);
  EXPECT_NEAR(r.y.back()[0], std::exp(-50.0), 1e-10);
  EXPECT_GT(r.steps, 50);  // step size limited by stability
}

TEST(ode, times_monotone_and_aligned) {
  const auto r = solve_ivp(
      [](double, const vec& y) {
        vec dy(1);
        dy[0] = y[0];
        return dy;
      },
      0.0, 1.0, vec(1, 1.0));
  ASSERT_EQ(r.t.size(), r.y.size());
  for (std::size_t i = 1; i < r.t.size(); ++i) EXPECT_GT(r.t[i], r.t[i - 1]);
  EXPECT_EQ(r.t.front(), 0.0);
}
