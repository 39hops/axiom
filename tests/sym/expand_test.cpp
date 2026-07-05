// Task 1: atan/asin/acos support + polynomial expansion.
#include <ax/sym/calc.hpp>
#include <ax/sym/expand.hpp>
#include <ax/sym/expr.hpp>
#include <ax/sym/print.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

using namespace ax::sym;

namespace {
const expr x = expr::symbol("x");
const expr y = expr::symbol("y");

double num_diff(const expr& e, double at) {
  const double h = 1e-6;
  return (e.eval({{"x", at + h}}) - e.eval({{"x", at - h}})) / (2 * h);
}
}  // namespace

// ---------- inverse trig ----------

TEST(inverse_trig, eval_matches_std) {
  EXPECT_NEAR(expr::fn("atan", x).eval({{"x", 1.0}}),
              std::numbers::pi / 4, 1e-15);
  EXPECT_NEAR(expr::fn("asin", x).eval({{"x", 0.5}}),
              std::numbers::pi / 6, 1e-15);
  EXPECT_NEAR(expr::fn("acos", x).eval({{"x", 0.5}}),
              std::numbers::pi / 3, 1e-15);
}

TEST(inverse_trig, atan_derivative_structural) {
  // d atan(x)/dx = 1/(1+x^2) == (1+x^2)^-1
  const expr d = diff(expr::fn("atan", x), x);
  const expr want = (expr::num(1) + x.pow(expr::num(2))).pow(expr::num(-1));
  EXPECT_TRUE(d.same(want));
}

TEST(inverse_trig, derivatives_numeric_crosscheck) {
  for (const char* f : {"atan", "asin", "acos"}) {
    const expr e = expr::fn(f, x);
    const expr d = diff(e, x);
    for (double at : {-0.7, -0.2, 0.1, 0.6}) {
      EXPECT_NEAR(d.eval({{"x", at}}), num_diff(e, at), 1e-6)
          << f << " at " << at;
    }
  }
}

TEST(inverse_trig, latex_names) {
  EXPECT_EQ(to_latex(expr::fn("atan", x)), "\\arctan\\left(x\\right)");
  EXPECT_EQ(to_latex(expr::fn("asin", x)), "\\arcsin\\left(x\\right)");
  EXPECT_EQ(to_latex(expr::fn("acos", x)), "\\arccos\\left(x\\right)");
}

// ---------- expand ----------

TEST(expand_fn, square_of_binomial) {
  const expr e = (x + expr::num(1)).pow(expr::num(2));
  const expr want =
      x.pow(expr::num(2)) + expr::num(2) * x + expr::num(1);
  EXPECT_TRUE(expand(e).same(want));
}

TEST(expand_fn, difference_of_squares) {
  const expr e = (x + y) * (x - y);
  const expr want = x.pow(expr::num(2)) - y.pow(expr::num(2));
  EXPECT_TRUE(expand(e).same(want));
}

TEST(expand_fn, cube_coefficients) {
  // (x+1)^3 = x^3 + 3x^2 + 3x + 1
  const expr e = (x + expr::num(1)).pow(expr::num(3));
  const expr want = x.pow(expr::num(3)) + expr::num(3) * x.pow(expr::num(2)) +
                    expr::num(3) * x + expr::num(1);
  EXPECT_TRUE(expand(e).same(want));
}

TEST(expand_fn, expands_inside_fn_args) {
  const expr e = expr::fn("sin", (x + expr::num(1)) * x);
  const expr want = expr::fn("sin", x.pow(expr::num(2)) + x);
  EXPECT_TRUE(expand(e).same(want));
}

TEST(expand_fn, already_expanded_fixed_point) {
  const expr e = x.pow(expr::num(2)) + expr::num(1);
  EXPECT_TRUE(expand(e).same(e));
}

TEST(expand_fn, distributes_constant_product) {
  // 2(x + y) = 2x + 2y  (canonicalization may already do this; expand must
  // return the distributed form either way)
  const expr e = expr::num(2) * (x + y);
  const expr want = expr::num(2) * x + expr::num(2) * y;
  EXPECT_TRUE(expand(e).same(want));
}

TEST(expand_fn, nested_product_of_sums) {
  // (x+1)(x+2)(x+3) = x^3 + 6x^2 + 11x + 6
  const expr e =
      (x + expr::num(1)) * (x + expr::num(2)) * (x + expr::num(3));
  const expr want = x.pow(expr::num(3)) + expr::num(6) * x.pow(expr::num(2)) +
                    expr::num(11) * x + expr::num(6);
  EXPECT_TRUE(expand(e).same(want));
}
