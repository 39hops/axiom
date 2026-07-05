#include <gtest/gtest.h>
#include <ax/sym/print.hpp>

using ax::bigint;
using ax::rational;
using ax::sym::expr;
using ax::sym::to_latex;
using ax::sym::to_string;

namespace {
const expr x = expr::symbol("x");
const expr y = expr::symbol("y");
const expr z = expr::symbol("z");
}  // namespace

TEST(print_text, atoms) {
  EXPECT_EQ(to_string(x), "x");
  EXPECT_EQ(to_string(expr::num(42)), "42");
  EXPECT_EQ(to_string(expr::num(rational(bigint(2), bigint(3)))), "2/3");
  EXPECT_EQ(to_string(expr::num(-5)), "-5");
}

TEST(print_text, polynomial) {
  const expr e = expr::num(2) * x + y.pow(expr::num(2)) - expr::num(3);
  EXPECT_EQ(to_string(e), "2*x + y^2 - 3");
}

TEST(print_text, parenthesization) {
  // canonical operand order puts the plain symbol first
  EXPECT_EQ(to_string((x + y) * z), "z*(x + y)");
  EXPECT_EQ(to_string(x.pow(y + expr::num(1))), "x^(y + 1)");
  EXPECT_EQ(to_string((x * y).pow(expr::num(2))), "(x*y)^2");
}

TEST(print_text, negative_terms_as_subtraction) {
  EXPECT_EQ(to_string(x - y), "x - y");
  EXPECT_EQ(to_string(-x + expr::num(1)), "-x + 1");
}

TEST(print_text, functions_and_division) {
  EXPECT_EQ(to_string(expr::fn("sin", x)), "sin(x)");
  EXPECT_EQ(to_string(x.pow(expr::num(-1))), "x^-1");
  EXPECT_EQ(to_string(expr::fn("exp", x * y)), "exp(x*y)");
}

TEST(print_latex, basics) {
  EXPECT_EQ(to_latex(x.pow(expr::num(2))), "x^{2}");
  EXPECT_EQ(to_latex(expr::num(rational(bigint(2), bigint(3)))),
            "\\frac{2}{3}");
  EXPECT_EQ(to_latex(expr::fn("sin", x)), "\\sin\\left(x\\right)");
  EXPECT_EQ(to_latex(expr::num(2) * x), "2 x");
}

TEST(print_latex, composite) {
  const expr e = expr::num(2) * x + y.pow(expr::num(2)) - expr::num(3);
  EXPECT_EQ(to_latex(e), "2 x + y^{2} - 3");
  EXPECT_EQ(to_latex((x + y) * z), "z \\left(x + y\\right)");
}

TEST(print_determinism, same_expr_same_string) {
  const expr a = (x + y).pow(expr::num(2)) * expr::fn("cos", z);
  const expr b = (expr::symbol("x") + expr::symbol("y"))
                     .pow(expr::num(2)) *
                 expr::fn("cos", expr::symbol("z"));
  EXPECT_EQ(to_string(a), to_string(b));
  EXPECT_FALSE(to_string(a).empty());
}
