// Polynomial solving. Oracle: substitute every root back, |p(root)| small.
#include <ax/sym/solve.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>

using namespace ax::sym;

namespace {
const expr x = expr::symbol("x");

double residual(const expr& p, const expr& root) {
  return std::abs(p.subs(x, root).eval());
}

// complex roots are checked against double coefficients via Horner
double poly_at(std::span<const double> c, std::complex<double> z,
               std::complex<double>* out = nullptr) {
  std::complex<double> acc{0.0, 0.0};
  for (std::size_t i = c.size(); i-- > 0;) acc = acc * z + c[i];
  if (out) *out = acc;
  return std::abs(acc);
}
}  // namespace

TEST(solve_poly_t, linear) {
  const expr p = expr::num(2) * x + expr::num(3);
  auto r = solve_poly(p, x);
  ASSERT_EQ(r.exact.size(), 1u);
  EXPECT_TRUE(r.complete_exact);
  EXPECT_NEAR(r.exact[0].eval(), -1.5, 1e-15);
  EXPECT_LT(residual(p, r.exact[0]), 1e-12);
}

TEST(solve_poly_t, quadratic_rational_roots) {
  const expr p = x.pow(expr::num(2)) - expr::num(5) * x + expr::num(6);
  auto r = solve_poly(p, x);
  ASSERT_EQ(r.exact.size(), 2u);
  EXPECT_TRUE(r.complete_exact);
  std::vector<double> got{r.exact[0].eval(), r.exact[1].eval()};
  std::sort(got.begin(), got.end());
  EXPECT_NEAR(got[0], 2.0, 1e-12);
  EXPECT_NEAR(got[1], 3.0, 1e-12);
}

TEST(solve_poly_t, quadratic_irrational_roots) {
  const expr p = x.pow(expr::num(2)) - expr::num(2);
  auto r = solve_poly(p, x);
  ASSERT_EQ(r.exact.size(), 2u);
  EXPECT_TRUE(r.complete_exact);
  for (const expr& root : r.exact) EXPECT_LT(residual(p, root), 1e-12);
  std::vector<double> got{r.exact[0].eval(), r.exact[1].eval()};
  std::sort(got.begin(), got.end());
  EXPECT_NEAR(got[0], -std::sqrt(2.0), 1e-12);
  EXPECT_NEAR(got[1], std::sqrt(2.0), 1e-12);
}

TEST(solve_poly_t, quadratic_complex_pair) {
  const expr p = x.pow(expr::num(2)) + expr::num(1);
  auto r = solve_poly(p, x);
  EXPECT_TRUE(r.exact.empty());
  EXPECT_FALSE(r.complete_exact);
  ASSERT_EQ(r.approx.size(), 2u);
  double coeffs[] = {1.0, 0.0, 1.0};
  for (auto z : r.approx) EXPECT_LT(poly_at(coeffs, z), 1e-9);
}

TEST(solve_poly_t, cubic_all_rational) {
  // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
  const expr p = x.pow(expr::num(3)) - expr::num(6) * x.pow(expr::num(2)) +
                 expr::num(11) * x - expr::num(6);
  auto r = solve_poly(p, x);
  ASSERT_EQ(r.exact.size(), 3u);
  EXPECT_TRUE(r.complete_exact);
  for (const expr& root : r.exact) EXPECT_LT(residual(p, root), 1e-12);
}

TEST(solve_poly_t, cubic_cardano_one_real) {
  // x^3 + 2x + 1: no rational root, D > 0 -> one real (Cardano radical),
  // complex pair numeric
  const expr p = x.pow(expr::num(3)) + expr::num(2) * x + expr::num(1);
  auto r = solve_poly(p, x);
  ASSERT_EQ(r.exact.size(), 1u);
  EXPECT_FALSE(r.complete_exact);
  EXPECT_LT(residual(p, r.exact[0]), 1e-9);
  ASSERT_EQ(r.approx.size(), 2u);
  double coeffs[] = {1.0, 2.0, 0.0, 1.0};
  for (auto z : r.approx) EXPECT_LT(poly_at(coeffs, z), 1e-9);
}

TEST(solve_poly_t, cubic_casus_irreducibilis) {
  // x^3 - 3x + 1: three real irrational roots (trig form)
  const expr p = x.pow(expr::num(3)) - expr::num(3) * x + expr::num(1);
  auto r = solve_poly(p, x);
  ASSERT_EQ(r.exact.size(), 3u);
  EXPECT_TRUE(r.complete_exact);
  for (const expr& root : r.exact) EXPECT_LT(residual(p, root), 1e-9);
}

TEST(solve_poly_t, quartic_biquadratic) {
  // x^4 - 5x^2 + 4 = (x^2-1)(x^2-4): roots +-1, +-2 (rational deflation)
  const expr p = x.pow(expr::num(4)) - expr::num(5) * x.pow(expr::num(2)) +
                 expr::num(4);
  auto r = solve_poly(p, x);
  ASSERT_EQ(r.exact.size(), 4u);
  EXPECT_TRUE(r.complete_exact);
  for (const expr& root : r.exact) EXPECT_LT(residual(p, root), 1e-12);
}

TEST(solve_poly_t, quartic_biquadratic_irrational) {
  // x^4 - 4x^2 + 1: roots +-sqrt(2 +- sqrt(3)), no rational roots
  const expr p = x.pow(expr::num(4)) - expr::num(4) * x.pow(expr::num(2)) +
                 expr::num(1);
  auto r = solve_poly(p, x);
  ASSERT_EQ(r.exact.size(), 4u);
  EXPECT_TRUE(r.complete_exact);
  for (const expr& root : r.exact) EXPECT_LT(residual(p, root), 1e-9);
}

TEST(solve_poly_t, sextic_numeric_only) {
  const expr p = x.pow(expr::num(6)) - x - expr::num(1);
  auto r = solve_poly(p, x);
  EXPECT_FALSE(r.complete_exact);
  EXPECT_EQ(r.exact.size() + r.approx.size(), 6u);
  double coeffs[] = {-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  for (auto z : r.approx) EXPECT_LT(poly_at(coeffs, z), 1e-8);
}

TEST(solve_poly_t, repeated_root_multiplicity) {
  // (x-1)^2 (x+2) = x^3 - 3x + 2: root 1 listed twice, -2 once
  const expr p = x.pow(expr::num(3)) - expr::num(3) * x + expr::num(2);
  auto r = solve_poly(p, x);
  ASSERT_EQ(r.exact.size(), 3u);
  EXPECT_TRUE(r.complete_exact);
  int ones = 0;
  for (const expr& root : r.exact)
    if (std::abs(root.eval() - 1.0) < 1e-12) ++ones;
  EXPECT_EQ(ones, 2);
}

TEST(solve_poly_t, non_polynomial_throws) {
  const expr p = expr::fn("sin", x) - expr::num(1);
  EXPECT_THROW((void)solve_poly(p, x), std::invalid_argument);
}

TEST(durand_kerner_t, converges_on_cubic) {
  // (x-1)(x-2)(x-3): coeffs lowest-first {-6, 11, -6, 1}
  double coeffs[] = {-6.0, 11.0, -6.0, 1.0};
  auto roots = durand_kerner(coeffs);
  ASSERT_EQ(roots.size(), 3u);
  std::vector<double> re;
  for (auto z : roots) {
    EXPECT_LT(std::abs(z.imag()), 1e-9);
    re.push_back(z.real());
  }
  std::sort(re.begin(), re.end());
  EXPECT_NEAR(re[0], 1.0, 1e-9);
  EXPECT_NEAR(re[1], 2.0, 1e-9);
  EXPECT_NEAR(re[2], 3.0, 1e-9);
}
