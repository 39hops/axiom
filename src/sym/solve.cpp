#include <ax/sym/poly.hpp>
#include <ax/sym/solve.hpp>

#include <cmath>
#include <stdexcept>

namespace ax::sym {

namespace {

const rational kzero{};

double to_double(const rational& q) { return expr::num(q).eval(); }

expr half_pow(const rational& num_den) {  // sqrt of a rational as expr
  return expr::num(num_den).pow(
      expr::num(rational(bigint(1), bigint(2))));
}

/** Real cube root of e (whose numeric value is v): sign-aware x^(1/3). */
expr cbrt_signed(const expr& e, double v) {
  const expr third = expr::num(rational(bigint(1), bigint(3)));
  if (v >= 0.0) return e.pow(third);
  return -((-e).pow(third));
}

/** Deflate all rational roots (with multiplicity) out of p; roots are
    appended as exprs. */
poly deflate_rational_roots(poly p, std::vector<expr>& out) {
  std::vector<rational> roots;
  try {
    roots = p.rational_roots();
  } catch (const std::overflow_error&) {
    return p;  // coefficients too large for the rational-root scan
  }
  for (const rational& r : roots) {
    const poly factor({-r, rational(bigint(1))});  // (x - r)
    while (p.degree() >= 1) {
      auto [q, rem] = p.divmod(factor);
      if (!(rem == poly{})) break;
      out.push_back(expr::num(r));
      p = q;
    }
  }
  return p;
}

/** Exact quadratic a x^2 + b x + c. Real roots appended to exact; a complex
    pair goes to approx. */
void solve_quadratic(const rational& a, const rational& b, const rational& c,
                     std::vector<expr>& exact,
                     std::vector<std::complex<double>>& approx) {
  const rational disc = b * b - rational(bigint(4)) * a * c;
  const rational inv2a = rational(bigint(1)) / (rational(bigint(2)) * a);
  if (disc > kzero) {
    const expr s = half_pow(disc);
    exact.push_back((expr::num(-b) + s) * expr::num(inv2a));
    exact.push_back((expr::num(-b) - s) * expr::num(inv2a));
  } else if (disc == kzero) {
    exact.push_back(expr::num(-b * inv2a));
    exact.push_back(expr::num(-b * inv2a));
  } else {
    const double re = to_double(-b * inv2a);
    const double im =
        std::sqrt(-to_double(disc)) * std::abs(to_double(inv2a));
    approx.emplace_back(re, im);
    approx.emplace_back(re, -im);
  }
}

/** Cubic with no rational roots, monic-normalized coefficients
    x^3 + r2 x^2 + r1 x + r0. */
void solve_cubic(const rational& r2, const rational& r1, const rational& r0,
                 std::vector<expr>& exact,
                 std::vector<std::complex<double>>& approx) {
  // depress: x = t - r2/3
  const rational third(bigint(1), bigint(3));
  const rational p = r1 - r2 * r2 * third;
  const rational q = r2 * r2 * r2 * rational(bigint(2), bigint(27)) -
                     r2 * r1 * third + r0;
  const expr shift = expr::num(-r2 * third);
  const rational d = q * q * rational(bigint(1), bigint(4)) +
                     p * p * p * rational(bigint(1), bigint(27));

  auto push_complex_pair_numeric = [&] {
    // remaining two roots of the cubic are the nonreal ones
    const double c3[] = {to_double(r0), to_double(r1), to_double(r2), 1.0};
    for (auto z : durand_kerner(c3))
      if (std::abs(z.imag()) > 1e-8) approx.push_back(z);
  };

  if (d > kzero || (d == kzero && p == kzero)) {
    // one real root: t = cbrt(-q/2 + sqrt(D)) + cbrt(-q/2 - sqrt(D))
    const rational mq2 = -q * rational(bigint(1), bigint(2));
    const expr sqrt_d = half_pow(d);
    const expr u_e = expr::num(mq2) + sqrt_d;
    const expr v_e = expr::num(mq2) - sqrt_d;
    const double sd = std::sqrt(to_double(d));
    const expr t = cbrt_signed(u_e, to_double(mq2) + sd) +
                   cbrt_signed(v_e, to_double(mq2) - sd);
    exact.push_back(shift + t);
    if (d > kzero) push_complex_pair_numeric();
    // d == 0 && p == 0: triple root, but then q == 0 and root was rational
  } else if (d == kzero) {
    // multiple real roots — all rational for rational coefficients, so the
    // deflation step should have caught them; fall back to numeric anyway
    const double c3[] = {to_double(r0), to_double(r1), to_double(r2), 1.0};
    for (auto z : durand_kerner(c3)) approx.push_back(z);
  } else {
    // three distinct real roots (casus irreducibilis): trig form
    // t_k = 2 sqrt(-p/3) cos( (acos(3q/(2p) sqrt(-3/p)) - 2 pi k) / 3 )
    const rational m2 = -p * third;             // (-p/3) > 0
    const rational a3q2p = rational(bigint(3)) * q /
                           (rational(bigint(2)) * p);  // 3q/(2p)
    const rational m3p = rational(bigint(-3)) / p;     // -3/p > 0
    const expr amp = expr::num(rational(bigint(2))) * half_pow(m2);
    const expr theta =
        expr::fn("acos", expr::num(a3q2p) * half_pow(m3p)) * expr::num(third);
    const expr pi_e = expr::fn("acos", expr::num(-1));
    const rational two_thirds(bigint(2), bigint(3));
    for (int k = 0; k < 3; ++k) {
      const expr angle =
          theta - expr::num(rational(bigint(k)) * two_thirds) * pi_e;
      exact.push_back(shift + amp * expr::fn("cos", angle));
    }
  }
}

/** Biquadratic x^4 + r2 x^2 + r0 (odd coefficients zero): substitute
    z = x^2. Returns false when not biquadratic. */
bool solve_biquadratic(const rational& r3, const rational& r2,
                       const rational& r1, const rational& r0,
                       std::vector<expr>& exact,
                       std::vector<std::complex<double>>& approx) {
  if (!(r3 == kzero) || !(r1 == kzero)) return false;
  std::vector<expr> zroots;
  std::vector<std::complex<double>> zapprox;
  solve_quadratic(rational(bigint(1)), r2, r0, zroots, zapprox);
  const expr half = expr::num(rational(bigint(1), bigint(2)));
  for (const expr& zr : zroots) {
    const double v = zr.eval();
    if (v > 0.0) {
      exact.push_back(zr.pow(half));
      exact.push_back(-(zr.pow(half)));
    } else if (v == 0.0) {
      exact.push_back(expr::num(0));
      exact.push_back(expr::num(0));
    } else {  // x^2 = negative real -> pure imaginary pair
      approx.emplace_back(0.0, std::sqrt(-v));
      approx.emplace_back(0.0, -std::sqrt(-v));
    }
  }
  for (auto z : zapprox) {
    // x^2 = z (complex): x = +-sqrt(z)
    const auto s = std::sqrt(z);
    approx.push_back(s);
    approx.push_back(-s);
  }
  return true;
}

}  // namespace

std::vector<std::complex<double>> durand_kerner(std::span<const double> coeffs,
                                                double tol, int max_iter) {
  if (coeffs.size() < 2 || coeffs.back() == 0.0)
    throw std::invalid_argument("durand_kerner: need degree >= 1");
  const std::size_t n = coeffs.size() - 1;
  std::vector<std::complex<double>> a(coeffs.begin(), coeffs.end());
  for (auto& c : a) c /= coeffs.back();  // monic
  auto eval = [&](std::complex<double> z) {
    std::complex<double> acc{0.0, 0.0};
    for (std::size_t i = a.size(); i-- > 0;) acc = acc * z + a[i];
    return acc;
  };
  std::vector<std::complex<double>> z(n);
  const std::complex<double> seed{0.4, 0.9};
  std::complex<double> t{1.0, 0.0};
  for (std::size_t k = 0; k < n; ++k) {
    t *= seed;
    z[k] = t;
  }
  for (int it = 0; it < max_iter; ++it) {
    double worst = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
      std::complex<double> den{1.0, 0.0};
      for (std::size_t j = 0; j < n; ++j)
        if (j != k) den *= z[k] - z[j];
      const auto corr = eval(z[k]) / den;
      z[k] -= corr;
      worst = std::max(worst, std::abs(corr));
    }
    if (worst < tol) break;
  }
  return z;
}

poly_roots solve_poly(const expr& equation_lhs, const expr& x) {
  poly p = poly::from_expr(equation_lhs, x);  // throws invalid_argument
  if (p.degree() < 1)
    throw std::invalid_argument("solve_poly: constant polynomial");

  poly_roots out;
  p = deflate_rational_roots(std::move(p), out.exact);

  const int deg = p.degree();
  if (deg >= 1) {
    auto c = [&](int k) { return p.coeff(static_cast<std::size_t>(k)); };
    const rational lead = c(deg);
    auto monic = [&](int k) { return c(k) / lead; };
    switch (deg) {
      case 1:
        out.exact.push_back(expr::num(-monic(0)));
        break;
      case 2:
        solve_quadratic(c(2), c(1), c(0), out.exact, out.approx);
        break;
      case 3:
        solve_cubic(monic(2), monic(1), monic(0), out.exact, out.approx);
        break;
      case 4:
        if (solve_biquadratic(monic(3), monic(2), monic(1), monic(0),
                              out.exact, out.approx))
          break;
        [[fallthrough]];
      default: {
        std::vector<double> dc(static_cast<std::size_t>(deg) + 1);
        for (int k = 0; k <= deg; ++k)
          dc[static_cast<std::size_t>(k)] = to_double(c(k));
        for (auto z : durand_kerner(dc)) out.approx.push_back(z);
        break;
      }
    }
  }
  out.complete_exact = out.approx.empty();
  return out;
}

}  // namespace ax::sym
