#include <ax/sym/parse.hpp>

#include <ax/sym/calc.hpp>
#include <ax/sym/expr.hpp>
#include <ax/sym/print.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <numbers>
#include <string>
#include <vector>

namespace {

using ax::sym::diff;
using ax::sym::expr;
using ax::sym::parse;
using ax::sym::parse_error;
using ax::sym::simplify;
using ax::sym::to_string;

double ev(const expr& e, double x) {
  return e.eval({{"x", x},
                 {"pi", std::numbers::pi},
                 {"E", std::numbers::e}});
}

TEST(Parse, IntegerLiterals) {
  EXPECT_TRUE(parse("42").same(expr::num(42)));
  EXPECT_TRUE(parse("-7").same(expr::num(-7)));
  EXPECT_TRUE(parse("  0 ").same(expr::num(0)));
}

TEST(Parse, RationalViaDivision) {
  // sympy prints Rational(3,4) as 3/4; canonicalization folds it to a num.
  EXPECT_TRUE(parse("3/4").same(expr::num(ax::rational(ax::bigint(3),
                                                       ax::bigint(4)))));
}

TEST(Parse, DecimalFloatExact) {
  EXPECT_TRUE(parse("0.25").same(parse("1/4")));
  EXPECT_TRUE(parse("2.5").same(parse("5/2")));
  EXPECT_TRUE(parse("-0.125").same(parse("-1/8")));
}

TEST(Parse, Precedence) {
  EXPECT_DOUBLE_EQ(ev(parse("1 + 2*3"), 0.0), 7.0);
  EXPECT_DOUBLE_EQ(ev(parse("2**3**2"), 0.0), 512.0);  // right-assoc
  EXPECT_DOUBLE_EQ(ev(parse("-2**2"), 0.0), -4.0);     // -(2**2)
  EXPECT_DOUBLE_EQ(ev(parse("(-2)**2"), 0.0), 4.0);
  EXPECT_DOUBLE_EQ(ev(parse("10 - 3 - 2"), 0.0), 5.0);  // left-assoc
  EXPECT_DOUBLE_EQ(ev(parse("12/2/3"), 0.0), 2.0);      // left-assoc
  EXPECT_DOUBLE_EQ(ev(parse("2**-1"), 0.0), 0.5);       // unary after **
}

TEST(Parse, CaretIsPowToo) {
  // axiom's own printer emits ^; the parser accepts both spellings.
  EXPECT_TRUE(parse("x^2").same(parse("x**2")));
}

TEST(Parse, SymbolsAndFunctions) {
  const expr x = expr::symbol("x");
  EXPECT_TRUE(parse("x").same(x));
  EXPECT_TRUE(parse("sin(x)").same(expr::fn("sin", x)));
  const expr nested = parse("sin(log(x*(2*x - 3)))");
  const double xv = 2.0;
  EXPECT_NEAR(ev(nested, xv), std::sin(std::log(xv * (2 * xv - 3))), 1e-12);
}

TEST(Parse, PiAndE) {
  EXPECT_NEAR(ev(parse("sin(pi/2)"), 0.0), 1.0, 1e-12);
  EXPECT_NEAR(ev(parse("log(E)"), 0.0), 1.0, 1e-12);
}

TEST(Parse, LlmoptMonster) {
  const std::string s =
      "5*((3 - 4*x)*sin(log(x*(2*x - 3))) + "
      "(2*x - 3)*cos(log(x*(2*x - 3))))/(2*x - 3)";
  const expr e = parse(s);
  const double xv = 2.0;  // 2x-3 = 1 > 0, x(2x-3) = 2 > 0: in-domain
  const double inner = std::log(xv * (2 * xv - 3));
  const double want = 5.0 *
                      ((3 - 4 * xv) * std::sin(inner) +
                       (2 * xv - 3) * std::cos(inner)) /
                      (2 * xv - 3);
  EXPECT_NEAR(ev(e, xv), want, 1e-12);
}

TEST(Parse, RoundTripThroughPrinter) {
  const expr x = expr::symbol("x");
  const std::vector<expr> cases = {
      x.pow(expr::num(3)) + expr::num(2) * x,
      expr::fn("sin", x * x) * expr::fn("exp", -x),
      (x + expr::num(1)) / (x - expr::num(2)),
      x.pow(expr::num(ax::rational(ax::bigint(1), ax::bigint(2)))),
      diff(expr::fn("atan", x * x), x),
  };
  for (const expr& e : cases) {
    const expr back = parse(to_string(e));
    EXPECT_TRUE(back.same(e)) << to_string(e) << " -> " << to_string(back);
  }
}

TEST(Parse, ErrorsThrowWithOffset) {
  EXPECT_THROW(parse(""), parse_error);
  EXPECT_THROW(parse("foo(x)"), parse_error);  // unknown function
  EXPECT_THROW(parse("oo"), parse_error);
  EXPECT_THROW(parse("zoo"), parse_error);
  EXPECT_THROW(parse("nan"), parse_error);
  EXPECT_THROW(parse("I"), parse_error);
  EXPECT_THROW(parse("1 ++"), parse_error);
  EXPECT_THROW(parse("sin(x"), parse_error);
  EXPECT_THROW(parse("x + "), parse_error);
  EXPECT_THROW(parse("(x))"), parse_error);  // trailing garbage
  try {
    parse("x + $");
    FAIL() << "expected parse_error";
  } catch (const parse_error& pe) {
    EXPECT_EQ(pe.offset(), 4u);
  }
}

}  // namespace
