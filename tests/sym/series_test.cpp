/** Tests for ax::sym::series and the series ODE oracle. */
#include <ax/mathgen/ode.hpp>
#include <ax/mathgen/series_solve.hpp>
#include <ax/sym/series.hpp>
#include <ax/sym/series_oracle.hpp>

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace {

using ax::bigint;
using ax::rational;
using ax::sym::expr;
using ax::sym::series;

rational q(long long n, long long d = 1) {
  return rational(bigint(n), bigint(d));
}

const expr kX = expr::symbol("x");

TEST(Series, ConstructorPadsToOrder) {
  const series s({q(1), q(2)}, 4);
  EXPECT_EQ(s.order(), 4);
  EXPECT_EQ(s.coeff(1), q(2));
  EXPECT_EQ(s.coeff(3), q(0));
  EXPECT_THROW(s.coeff(4), std::out_of_range);
}

TEST(Series, AddMulTruncateToMinOrder) {
  const series a({q(1), q(1), q(1)}, 3);   // 1 + x + x^2
  const series b({q(1), q(-1)}, 2);        // 1 - x
  const series sum = a + b;
  EXPECT_EQ(sum.order(), 2);
  EXPECT_EQ(sum.coeff(0), q(2));
  EXPECT_EQ(sum.coeff(1), q(0));
  const series prod = a * b;               // (1+x+x^2)(1-x) = 1 + O(x^2)
  EXPECT_EQ(prod.order(), 2);
  EXPECT_EQ(prod.coeff(0), q(1));
  EXPECT_EQ(prod.coeff(1), q(0));
}

TEST(Series, DerivativeIntegrateAreTermwiseExact) {
  const series s({q(1), q(2), q(3)}, 3);  // 1 + 2x + 3x^2
  const series d = s.derivative();
  EXPECT_EQ(d.order(), 2);
  EXPECT_EQ(d.coeff(0), q(2));
  EXPECT_EQ(d.coeff(1), q(6));
  const series i = d.integrate();  // back, constant 0
  EXPECT_EQ(i.order(), 3);
  EXPECT_EQ(i.coeff(0), q(0));
  EXPECT_EQ(i.coeff(1), q(2));
  EXPECT_EQ(i.coeff(2), q(3));
}

TEST(Series, InverseOfOneMinusXIsGeometric) {
  const series g = series({q(1), q(-1)}, 5).inverse();
  for (int k = 0; k < 5; ++k) EXPECT_EQ(g.coeff(k), q(1));
  EXPECT_THROW(series({q(0), q(1)}, 3).inverse(), std::domain_error);
}

TEST(Series, ComposeRequiresZeroInnerConstant) {
  // exp(2x) via of_expr: coefficients 2^k / k!
  const series e = series::of_expr(
      expr::fn("exp", expr::num(2) * kX), kX, 5);
  EXPECT_EQ(e.coeff(0), q(1));
  EXPECT_EQ(e.coeff(1), q(2));
  EXPECT_EQ(e.coeff(2), q(2));
  EXPECT_EQ(e.coeff(3), q(4, 3));
  EXPECT_EQ(e.coeff(4), q(2, 3));
  EXPECT_THROW(series({q(1)}, 3).compose(series({q(1)}, 3)),
               std::domain_error);
}

TEST(Series, OfExprKnownMaclaurinCoefficients) {
  const series s = series::of_expr(expr::fn("sin", kX), kX, 6);
  EXPECT_EQ(s.coeff(1), q(1));
  EXPECT_EQ(s.coeff(3), q(-1, 6));
  EXPECT_EQ(s.coeff(5), q(1, 120));
  const series c = series::of_expr(expr::fn("cos", kX), kX, 5);
  EXPECT_EQ(c.coeff(0), q(1));
  EXPECT_EQ(c.coeff(2), q(-1, 2));
  EXPECT_EQ(c.coeff(4), q(1, 24));
  const series l = series::of_expr(
      expr::fn("log", expr::num(1) + kX), kX, 5);
  EXPECT_EQ(l.coeff(1), q(1));
  EXPECT_EQ(l.coeff(2), q(-1, 2));
  EXPECT_EQ(l.coeff(4), q(-1, 4));
  const series r = series::of_expr(
      expr::fn("sqrt", expr::num(1) + kX), kX, 4);
  EXPECT_EQ(r.coeff(0), q(1));
  EXPECT_EQ(r.coeff(1), q(1, 2));
  EXPECT_EQ(r.coeff(2), q(-1, 8));
  EXPECT_EQ(r.coeff(3), q(1, 16));
  // 1/(1+x) through a negative integer power
  const series inv = series::of_expr(
      (expr::num(1) + kX).pow(expr::num(-1)), kX, 4);
  EXPECT_EQ(inv.coeff(2), q(1));
  EXPECT_EQ(inv.coeff(3), q(-1));
}

TEST(Series, OfExprRejectsOutOfScopeShapes) {
  EXPECT_THROW(series::of_expr(expr::symbol("t"), kX, 3),
               std::domain_error);
  EXPECT_THROW(series::of_expr(expr::fn("exp", expr::num(1) + kX), kX, 3),
               std::domain_error);
  EXPECT_THROW(series::of_expr(expr::fn("erf", kX), kX, 3),
               std::domain_error);
}

// ---------------------------------------------------------------- oracle

using ax::sym::check_odesol_series;
using ax::sym::series_verdict;

expr y_of_x() { return expr::fn("y", kX); }
expr d1_y() { return expr::fn("Derivative", std::vector<expr>{y_of_x(), kX}); }
expr make_eq(const expr& l, const expr& r) {
  return expr::fn("Eq", std::vector<expr>{l, r});
}

TEST(SeriesOracle, TrueSolutionIsEquivalentToOrder) {
  // y' - y = 0, y = exp(x)
  const expr eq = make_eq(d1_y() - y_of_x(), expr::num(0));
  const series cand = series::of_expr(expr::fn("exp", kX), kX, 8);
  const auto r = check_odesol_series(eq, cand, kX);
  EXPECT_EQ(r.v, series_verdict::equivalent_to_order);
  EXPECT_EQ(r.order, 7);  // residual carries the derivative's order
}

TEST(SeriesOracle, PerturbedCoefficientIsExactWitness) {
  std::vector<rational> c = series::of_expr(expr::fn("exp", kX), kX, 8)
                                .coeffs();
  c[5] = c[5] + q(1, 7);
  const expr eq = make_eq(d1_y() - y_of_x(), expr::num(0));
  const auto r = check_odesol_series(eq, series(std::move(c), 8), kX);
  EXPECT_EQ(r.v, series_verdict::not_equivalent);
  EXPECT_EQ(r.order, 4);  // first divergence: [x^4] of y' picks up a_5
}

TEST(SeriesOracle, UnsupportedEquationIsUndecided) {
  const expr eq = make_eq(d1_y() - expr::fn("erf", kX), expr::num(0));
  const auto r = check_odesol_series(eq, series({q(0)}, 4), kX);
  EXPECT_EQ(r.v, series_verdict::undecided_beyond_order);
}

// ----------------------------------------------------------- recurrence

TEST(SeriesSolve, MatchesDrawnSolutionAcrossFamilies) {
  const int order = 8;
  for (int level = 1; level <= 3; ++level)
    for (long long seed = 0; seed < 10; ++seed) {
      for (int fam = 0; fam < 3; ++fam) {
        const ax::mathgen::ode_problem p =
            fam == 0 ? ax::mathgen::make_linear_first_order(level, seed)
            : fam == 1 ? ax::mathgen::make_second_order_cc(level, seed)
                       : ax::mathgen::make_separable_growth(level, seed);
        const auto sol = ax::mathgen::series_solve(p, order);
        EXPECT_EQ(sol.y, series::of_expr(p.sol, kX, order))
            << p.family << " L" << level << " seed " << seed;
        EXPECT_EQ(static_cast<int>(sol.steps.size()),
                  order - sol.ode_order);
        const auto chk = check_odesol_series(p.eq, sol.y, kX);
        EXPECT_EQ(chk.v, series_verdict::equivalent_to_order)
            << p.family << " L" << level << " seed " << seed;
      }
    }
}

}  // namespace
