#include <ax/sym/poly.hpp>
#include <ax/sym/solve.hpp>

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace ax::sym {

namespace {

const rational kzero{};

constexpr long long kmax_poly_exponent = 512;  ///< from_expr degree guard

/** Every numeric integer exponent in e is within kmax_poly_exponent —
    keeps poly::from_expr's O(n) power expansion bounded. */
bool exponents_bounded(const expr& e) {
  if (e.is_pow() && e.args()[1].is_num()) {
    const rational& n = e.args()[1].value();
    if (n.den() == bigint(1) && abs(n.num()) > bigint(kmax_poly_exponent))
      return false;
  }
  for (const expr& a : e.args())
    if (!exponents_bounded(a)) return false;
  return true;
}

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
    // remaining two roots of the cubic are the nonreal ones: take the two
    // with the largest |imag| so the count is right even if the filter
    // threshold would mis-classify a near-real iterate
    const double c3[] = {to_double(r0), to_double(r1), to_double(r2), 1.0};
    auto zs = durand_kerner(c3);
    std::sort(zs.begin(), zs.end(), [](const auto& a, const auto& b) {
      return std::abs(a.imag()) > std::abs(b.imag());
    });
    approx.push_back(zs[0]);
    approx.push_back(zs[1]);
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

namespace {

bool depends_on(const expr& e, const expr& x) {
  if (e.same(x)) return true;
  for (const expr& a : e.args())
    if (depends_on(a, x)) return true;
  return false;
}

/** Decompose e as a*x + b with x-free a, b (a structurally nonzero). */
bool split_linear(const expr& e, const expr& x, expr& a, expr& b) {
  a = expr::num(0);
  b = expr::num(0);
  auto add_term = [&](const expr& t) -> bool {
    if (!depends_on(t, x)) {
      b = b + t;
      return true;
    }
    if (t.same(x)) {
      a = a + expr::num(1);
      return true;
    }
    if (t.is_mul()) {
      expr coef = expr::num(1);
      int x_count = 0;
      for (const expr& f : t.args()) {
        if (f.same(x)) {
          ++x_count;
        } else if (depends_on(f, x)) {
          return false;  // nonlinear in x
        } else {
          coef = coef * f;
        }
      }
      if (x_count != 1) return false;
      a = a + coef;
      return true;
    }
    return false;
  };
  if (e.is_add()) {
    for (const expr& t : e.args())
      if (!add_term(t)) return false;
  } else if (!add_term(e)) {
    return false;
  }
  return !a.same(expr::num(0));
}

/** Principal-value inverse of f(name) applied to rhs; nullopt when the
    inverse does not apply (domain violation on a numeric rhs, or unknown
    function). */
std::optional<expr> invert_fn(const std::string& name, const expr& rhs) {
  double v = 0.0;
  bool numeric = false;
  if (rhs.is_num()) {
    numeric = true;
    v = rhs.eval();
  }
  if (name == "exp") {
    if (numeric && v <= 0.0) return std::nullopt;
    return expr::fn("log", rhs);
  }
  if (name == "log") return expr::fn("exp", rhs);
  if (name == "sqrt") {
    if (numeric && v < 0.0) return std::nullopt;
    return rhs.pow(expr::num(2));
  }
  if (name == "sin") {
    if (numeric && std::abs(v) > 1.0) return std::nullopt;
    return expr::fn("asin", rhs);
  }
  if (name == "cos") {
    if (numeric && std::abs(v) > 1.0) return std::nullopt;
    return expr::fn("acos", rhs);
  }
  if (name == "tan") return expr::fn("atan", rhs);
  return std::nullopt;
}

bool contains_symbol(const expr& e) {
  if (e.is_sym()) return true;
  for (const expr& a : e.args())
    if (contains_symbol(a)) return true;
  return false;
}

/** Numeric verification of a candidate root. Candidates with free symbols
    are kept (structural construction is the guarantee); fully numeric
    candidates must evaluate finite and satisfy the equation. */
bool root_verifies(const expr& lhs, const expr& rhs, const expr& x,
                   const expr& root) {
  const expr le = lhs.subs(x, root), re = rhs.subs(x, root);
  if (contains_symbol(le) || contains_symbol(re)) return true;
  try {
    const double l = le.eval();
    const double r = re.eval();
    if (!std::isfinite(l) || !std::isfinite(r)) return false;
    return std::abs(l - r) <= 1e-8 * (1.0 + std::abs(r));
  } catch (const std::logic_error&) {
    return false;  // numeric candidate that cannot evaluate: reject
  }
}

}  // namespace

std::vector<expr> solve(const expr& lhs, const expr& rhs, const expr& x) {
  if (!x.is_sym()) throw std::invalid_argument("solve: x must be a symbol");
  const expr e = lhs - rhs;
  std::vector<expr> out;

  // polynomial route (rational coefficients)
  try {
    auto pr = solve_poly(e, x);
    for (const expr& r : pr.exact)
      if (root_verifies(lhs, rhs, x, r)) out.push_back(r);
    return out;
  } catch (const std::invalid_argument&) {
  }

  // linear with symbolic coefficients: a x + b == 0
  {
    expr a = expr::num(0), b = expr::num(0);
    if (split_linear(e, x, a, b)) {
      const expr r = -(b / a);
      if (root_verifies(lhs, rhs, x, r)) out.push_back(r);
      return out;
    }
  }

  // single-function isolation: c_f * f(u) + k == 0  ->  f(u) == -k/c_f
  {
    expr fn_term = expr::num(0), k = expr::num(0);
    bool found = false, bad = false;
    auto scan_term = [&](const expr& t) {
      if (!depends_on(t, x)) {
        k = k + t;
        return;
      }
      if (found) {
        bad = true;  // two x-dependent terms
        return;
      }
      found = true;
      fn_term = t;
    };
    if (e.is_add())
      for (const expr& t : e.args()) scan_term(t);
    else
      scan_term(e);
    if (found && !bad) {
      expr cf = expr::num(1), f = fn_term;
      if (fn_term.is_mul()) {
        expr rest = expr::num(1);
        int dep_count = 0;
        for (const expr& g : fn_term.args()) {
          if (depends_on(g, x)) {
            ++dep_count;
            f = g;
          } else {
            rest = rest * g;
          }
        }
        if (dep_count != 1) return out;
        cf = rest;
      }
      if (f.is_fn() && depends_on(f.args()[0], x)) {
        const expr target = -(k / cf);
        if (auto inner_rhs = invert_fn(f.name(), target)) {
          for (const expr& r : solve(f.args()[0], *inner_rhs, x))
            if (root_verifies(lhs, rhs, x, r)) out.push_back(r);
        }
      }
    }
  }
  return out;
}

std::vector<expr> solve_linear_system(const std::vector<std::vector<expr>>& a,
                                      const std::vector<expr>& b) {
  const std::size_t n = a.size();
  if (n == 0) throw std::invalid_argument("solve_linear_system: empty");
  if (b.size() != n)
    throw std::invalid_argument("solve_linear_system: shape mismatch");
  for (const auto& row : a)
    if (row.size() != n)
      throw std::invalid_argument("solve_linear_system: shape mismatch");

  const expr zero = expr::num(0);
  std::vector<std::vector<expr>> m(n);
  for (std::size_t i = 0; i < n; ++i) {
    m[i] = a[i];
    m[i].push_back(b[i]);
  }
  for (std::size_t col = 0; col < n; ++col) {
    std::size_t piv = col;
    while (piv < n && m[piv][col].same(zero)) ++piv;
    if (piv == n)
      throw std::domain_error("solve_linear_system: singular (zero pivot)");
    std::swap(m[col], m[piv]);
    for (std::size_t r = col + 1; r < n; ++r) {
      if (m[r][col].same(zero)) continue;
      const expr f = m[r][col] / m[col][col];
      for (std::size_t k = col; k <= n; ++k)
        m[r][k] = m[r][k] - f * m[col][k];
    }
  }
  std::vector<expr> sol(n, zero);
  for (std::size_t i = n; i-- > 0;) {
    expr s = m[i][n];
    for (std::size_t k = i + 1; k < n; ++k) s = s - m[i][k] * sol[k];
    sol[i] = s / m[i][i];
  }
  return sol;
}

poly_roots solve_poly(const expr& equation_lhs, const expr& x) {
  if (!exponents_bounded(equation_lhs))
    throw std::invalid_argument("solve_poly: exponent too large");
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
