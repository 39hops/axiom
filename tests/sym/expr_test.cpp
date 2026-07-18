#include <gtest/gtest.h>
#include <ax/sym/expr.hpp>

#include <cmath>
#include <map>
#include <stdexcept>

using ax::rational;
using ax::sym::expr;
using ax::sym::simplify;

namespace {
const expr x = expr::symbol("x");
const expr y = expr::symbol("y");
const expr z = expr::symbol("z");
}  // namespace

TEST(expr_basic, num_and_symbol) {
  const expr two = expr::num(2);
  EXPECT_TRUE(two.is_num());
  EXPECT_EQ(two.value(), rational(ax::bigint(2)));
  EXPECT_TRUE(x.is_sym());
  EXPECT_EQ(x.name(), "x");
}

TEST(expr_canon, like_terms_collect) {
  EXPECT_TRUE((x + x).same(expr::num(2) * x));
  EXPECT_TRUE((x * x * x).same(x.pow(expr::num(3))));
  EXPECT_TRUE(((x + y) + (z + x)).same(expr::num(2) * x + y + z));
}

TEST(expr_canon, constant_folding) {
  EXPECT_TRUE((expr::num(2) + expr::num(3)).same(expr::num(5)));
  EXPECT_TRUE((expr::num(2) * expr::num(3)).same(expr::num(6)));
  const expr q = expr::num(2) / expr::num(3);
  EXPECT_TRUE(q.is_num());
  EXPECT_EQ(q.value(), rational(ax::bigint(2), ax::bigint(3)));
  EXPECT_TRUE(expr::num(2).pow(expr::num(10)).same(expr::num(1024)));
}

TEST(expr_canon, identities) {
  EXPECT_TRUE((x + expr::num(0)).same(x));
  EXPECT_TRUE((x * expr::num(1)).same(x));
  EXPECT_TRUE((x * expr::num(0)).same(expr::num(0)));
  EXPECT_TRUE(x.pow(expr::num(0)).same(expr::num(1)));
  EXPECT_TRUE(x.pow(expr::num(1)).same(x));
  EXPECT_TRUE((x - x).same(expr::num(0)));
  EXPECT_TRUE((x / x).same(expr::num(1)));
}

TEST(expr_canon, hash_consing_shares_nodes) {
  const expr a = x + y;
  const expr b = expr::symbol("x") + expr::symbol("y");
  EXPECT_TRUE(a.same(b));
  EXPECT_EQ(a.hash(), b.hash());
}

TEST(expr_subs, substitute_symbol) {
  const expr e = x.pow(expr::num(2)) + y;
  const expr s = e.subs(x, expr::num(3));
  EXPECT_TRUE(s.same(expr::num(9) + y));
  // subs with expression
  const expr t = e.subs(x, y);
  EXPECT_TRUE(t.same(y.pow(expr::num(2)) + y));
}

TEST(expr_eval, numeric) {
  const expr e = x.pow(expr::num(2)) + expr::num(2) * x + expr::num(1);
  EXPECT_NEAR(e.eval({{"x", 3.0}}), 16.0, 1e-12);
  const expr s = expr::fn("sin", x);
  EXPECT_NEAR(s.eval({{"x", 1.2}}), std::sin(1.2), 1e-14);
  const expr c = expr::fn("exp", x * y);
  EXPECT_NEAR(c.eval({{"x", 0.5}, {"y", 2.0}}), std::exp(1.0), 1e-14);
}

TEST(expr_eval, unbound_symbol_throws) {
  EXPECT_THROW(x.eval({}), std::logic_error);
}

TEST(expr_simplify, no_trig_identities_v1) {
  // sin^2 + cos^2 does NOT collapse in v1 — documents the non-goal
  const expr e = expr::fn("sin", x).pow(expr::num(2)) +
                 expr::fn("cos", x).pow(expr::num(2));
  EXPECT_FALSE(simplify(e).same(expr::num(1)));
}

TEST(expr_canon, mul_collects_powers_of_same_base) {
  EXPECT_TRUE((x.pow(expr::num(2)) * x.pow(expr::num(3)))
                  .same(x.pow(expr::num(5))));
  EXPECT_TRUE((x.pow(expr::num(2)) / x).same(x));
}

TEST(expr_canon, negation_and_subtraction) {
  const expr e = -(x + y);
  // -(x+y) = -1*x + -1*y
  EXPECT_TRUE(e.same(expr::num(-1) * x + expr::num(-1) * y));
  EXPECT_TRUE((expr::num(3) * x - expr::num(2) * x).same(x));
}

TEST(expr_canon, pow_of_pow_collapse_is_sound_over_reals) {
  // (x^2)^(1/2) is |x|, not x: even inner exponent + fractional outer must
  // NOT collapse via exponent multiplication (parity-audit soundness bug).
  const expr half = expr::num(ax::rational(ax::bigint(1), ax::bigint(2)));
  const expr x2_sqrt = x.pow(expr::num(2)).pow(half);
  EXPECT_FALSE(x2_sqrt.same(x));
  EXPECT_NEAR(x2_sqrt.eval({{"x", -3.0}}), 3.0, 1e-12);  // |−3|

  // Integer outer exponent always collapses: ((x^a)^n) = x^(an).
  EXPECT_TRUE(x.pow(half).pow(expr::num(4)).same(x.pow(expr::num(2))));
  // |inner| <= 1 collapses (principal-branch domain convention, as sympy):
  EXPECT_TRUE(x.pow(half).pow(half).same(
      x.pow(expr::num(ax::rational(ax::bigint(1), ax::bigint(4))))));
}
