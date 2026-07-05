#include <gtest/gtest.h>
#include <ax/sym/poly.hpp>

#include <stdexcept>
#include <vector>

using ax::bigint;
using ax::rational;
using ax::sym::expr;
using ax::sym::poly;

namespace {
rational q(long long n, long long d = 1) {
  return rational(bigint(n), bigint(d));
}
poly make(std::initializer_list<long long> cs) {
  std::vector<rational> v;
  for (const long long c : cs) v.push_back(q(c));
  return poly(std::move(v));
}
const expr x = expr::symbol("x");
}  // namespace

TEST(poly_basic, degree_and_zero) {
  EXPECT_EQ(poly().degree(), -1);
  EXPECT_EQ(make({5}).degree(), 0);
  EXPECT_EQ(make({1, 0, 3}).degree(), 2);
  EXPECT_EQ(make({1, 2, 0, 0}).degree(), 1);  // trailing zeros trimmed
}

TEST(poly_ring, identities) {
  const poly a = make({1, 2, 3});
  const poly b = make({-2, 0, 1});
  const poly c = make({4, -1});
  // distributive
  EXPECT_EQ((a + b) * c, a * c + b * c);
  // degree add
  EXPECT_EQ((a * b).degree(), a.degree() + b.degree());
  // a - a = 0
  EXPECT_EQ((a - a).degree(), -1);
}

TEST(poly_divmod, cubic_by_linear) {
  // (x^3 - 1) / (x - 1) = x^2 + x + 1, remainder 0
  const poly p = make({-1, 0, 0, 1});
  const poly d = make({-1, 1});
  const auto [quo, rem] = p.divmod(d);
  EXPECT_EQ(quo, make({1, 1, 1}));
  EXPECT_EQ(rem.degree(), -1);
  // with remainder: (x^2 + 1) / (x - 1) = x + 1 rem 2
  const auto [q2, r2] = make({1, 0, 1}).divmod(d);
  EXPECT_EQ(q2, make({1, 1}));
  EXPECT_EQ(r2, make({2}));
  EXPECT_THROW(p.divmod(poly()), std::domain_error);
}

TEST(poly_gcd, common_factor) {
  // gcd(x^2 - 1, x^3 - 1) = x - 1 (monic)
  const poly g = gcd(make({-1, 0, 1}), make({-1, 0, 0, 1}));
  EXPECT_EQ(g, make({-1, 1}));
  // coprime -> 1
  EXPECT_EQ(gcd(make({-1, 1}), make({1, 1})), make({1}));
}

TEST(poly_squarefree, strips_multiplicity) {
  // (x-1)^2 (x+2) = x^3 - 3x + 2
  const poly p = make({2, -3, 0, 1});
  const poly sf = p.square_free();
  // square-free part = (x-1)(x+2) = x^2 + x - 2 (monic-normalized)
  EXPECT_EQ(sf, make({-2, 1, 1}));
}

TEST(poly_roots, rational_roots_exact) {
  // 6x^3 - 5x^2 - 2x + 1 has roots 1, 1/3, -1/2
  const poly p = make({1, -2, -5, 6});
  std::vector<rational> roots = p.rational_roots();
  ASSERT_EQ(roots.size(), 3u);
  for (const rational& r : roots) EXPECT_TRUE(p.eval(r).is_zero());
  // no rational roots: x^2 + 1
  EXPECT_TRUE(make({1, 0, 1}).rational_roots().empty());
  // repeated root counted once: (x-1)^2
  EXPECT_EQ(make({1, -2, 1}).rational_roots().size(), 1u);
}

TEST(poly_expr, roundtrip) {
  // (x+1)^2 -> coeffs {1, 2, 1}
  const expr e = (x + expr::num(1)).pow(expr::num(2));
  const poly p = poly::from_expr(e, x);
  EXPECT_EQ(p, make({1, 2, 1}));
  // to_expr back: structural equality with expanded form
  EXPECT_TRUE(p.to_expr(x).same(x.pow(expr::num(2)) + expr::num(2) * x +
                                expr::num(1)));
  EXPECT_THROW(poly::from_expr(expr::fn("sin", x), x), std::invalid_argument);
  // other symbols are not polynomial in x
  EXPECT_THROW(
      poly::from_expr(x + expr::symbol("y"), x), std::invalid_argument);
}

TEST(poly_eval, exact) {
  const poly p = make({1, -2, -5, 6});
  EXPECT_TRUE(p.eval(q(1)).is_zero());
  EXPECT_TRUE(p.eval(q(1, 3)).is_zero());
  EXPECT_TRUE(p.eval(q(-1, 2)).is_zero());
  EXPECT_EQ(p.eval(q(2)), q(25));  // 6*8 - 5*4 - 2*2 + 1 = 25
}
