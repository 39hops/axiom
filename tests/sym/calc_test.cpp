#include <gtest/gtest.h>
#include <ax/st/rng.hpp>
#include <ax/sym/calc.hpp>

#include <cmath>

using ax::sym::diff;
using ax::sym::expr;

namespace {
const expr x = expr::symbol("x");
const expr y = expr::symbol("y");

/** Numeric cross-check: eval(diff(f)) vs central difference of eval(f). */
void check_numeric(const expr& f, double lo = 0.3, double hi = 1.7) {
  const expr df = diff(f, x);
  ax::st::rng g{31};
  for (int i = 0; i < 5; ++i) {
    const double t = g.uniform(lo, hi);
    const double h = 1e-6;
    const double numeric =
        (f.eval({{"x", t + h}}) - f.eval({{"x", t - h}})) / (2.0 * h);
    EXPECT_NEAR(df.eval({{"x", t}}), numeric, 1e-5 * std::max(1.0, std::abs(numeric)))
        << "at x=" << t;
  }
}
}  // namespace

TEST(diff_basic, constants_and_symbol) {
  EXPECT_TRUE(diff(expr::num(7), x).same(expr::num(0)));
  EXPECT_TRUE(diff(x, x).same(expr::num(1)));
  EXPECT_TRUE(diff(x, y).same(expr::num(0)));
  EXPECT_TRUE(diff(y, x).same(expr::num(0)));
}

TEST(diff_basic, polynomial) {
  // d(x^3 + 2x)/dx = 3x^2 + 2
  const expr f = x.pow(expr::num(3)) + expr::num(2) * x;
  const expr expected = expr::num(3) * x.pow(expr::num(2)) + expr::num(2);
  EXPECT_TRUE(diff(f, x).same(expected));
}

TEST(diff_product, sin_cos) {
  // d(sin x cos x) = cos^2 - sin^2
  const expr f = expr::fn("sin", x) * expr::fn("cos", x);
  const expr expected = expr::fn("cos", x).pow(expr::num(2)) -
                        expr::fn("sin", x).pow(expr::num(2));
  EXPECT_TRUE(diff(f, x).same(expected));
}

TEST(diff_chain, exp_of_square) {
  // d(exp(x^2)) = 2x exp(x^2)
  const expr f = expr::fn("exp", x.pow(expr::num(2)));
  const expr expected = expr::num(2) * x * f;
  EXPECT_TRUE(diff(f, x).same(expected));
}

TEST(diff_chain, log) {
  EXPECT_TRUE(diff(expr::fn("log", x), x).same(x.pow(expr::num(-1))));
}

TEST(diff_numeric, cross_checks) {
  check_numeric(x.pow(expr::num(4)) - expr::num(3) * x);
  check_numeric(expr::fn("sin", x * x));
  check_numeric(expr::fn("exp", x) * expr::fn("cos", x));
  check_numeric(x.pow(expr::num(2)) / (x + expr::num(2)));
  check_numeric(expr::fn("sqrt", x + expr::num(1)));
  check_numeric(expr::fn("tan", x), 0.1, 1.2);
  // general power u^v, v non-constant: x^x
  check_numeric(x.pow(x), 0.5, 2.0);
}
