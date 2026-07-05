// Integration. Oracle: diff(integrate(f)) numerically equals f at sample
// points (never trusts a copied antiderivative).
#include <ax/sym/calc.hpp>
#include <ax/sym/integrate.hpp>

#include <gtest/gtest.h>

#include <cmath>

using namespace ax::sym;

namespace {
const expr x = expr::symbol("x");

/** Verify d/dx integrate(f) == f on [lo, hi]. */
void check_roundtrip(const expr& f, double lo, double hi) {
  auto af = integrate(f, x);
  ASSERT_TRUE(af.has_value()) << "integrate returned nullopt";
  const expr d = diff(*af, x);
  for (int i = 0; i < 8; ++i) {
    const double t = lo + (hi - lo) * (i + 0.5) / 8.0;
    const double want = f.eval({{"x", t}});
    EXPECT_NEAR(d.eval({{"x", t}}), want, 1e-7 * (1.0 + std::abs(want)))
        << "at x = " << t;
  }
}
}  // namespace

// ---------- Task 3: table, linearity, power rule ----------

TEST(integrate_core, constant) {
  const expr f = expr::num(5);
  auto af = integrate(f, x);
  ASSERT_TRUE(af.has_value());
  EXPECT_TRUE(af->same(expr::num(5) * x));
}

TEST(integrate_core, power_rule) {
  check_roundtrip(x.pow(expr::num(3)), -2.0, 2.0);
  auto af = integrate(x.pow(expr::num(3)), x);
  const expr want =
      expr::num(ax::rational(ax::bigint(1), ax::bigint(4))) * x.pow(expr::num(4));
  EXPECT_TRUE(af->same(want));
}

TEST(integrate_core, polynomial_linearity) {
  const expr f =
      expr::num(3) * x.pow(expr::num(2)) + expr::num(2) * x + expr::num(1);
  check_roundtrip(f, -3.0, 3.0);
}

TEST(integrate_core, reciprocal_gives_log) {
  const expr f = x.pow(expr::num(-1));
  auto af = integrate(f, x);
  ASSERT_TRUE(af.has_value());
  EXPECT_TRUE(af->same(expr::fn("log", x)));
}

TEST(integrate_core, sin_of_linear) {
  const expr f = expr::fn("sin", expr::num(2) * x + expr::num(1));
  check_roundtrip(f, -2.0, 2.0);
}

TEST(integrate_core, exp_of_linear) {
  const expr f = expr::fn("exp", expr::num(3) * x);
  check_roundtrip(f, -1.0, 1.0);
}

TEST(integrate_core, linear_base_power) {
  const expr f = (expr::num(2) * x + expr::num(5)).pow(expr::num(7));
  check_roundtrip(f, -1.0, 1.0);
}

TEST(integrate_core, fractional_power) {
  const expr f = x.pow(expr::num(ax::rational(ax::bigint(1), ax::bigint(2))));
  check_roundtrip(f, 0.5, 4.0);
}

TEST(integrate_core, log_of_linear) {
  const expr f = expr::fn("log", x);
  check_roundtrip(f, 0.5, 5.0);
}

TEST(integrate_core, other_symbols_are_constants) {
  const expr y = expr::symbol("y");
  auto af = integrate(y, x);
  ASSERT_TRUE(af.has_value());
  EXPECT_TRUE(af->same(y * x));
}

TEST(integrate_core, honest_failure_gaussian) {
  const expr f = expr::fn("sin", x.pow(expr::num(2)));
  EXPECT_FALSE(integrate(f, x).has_value());
}
