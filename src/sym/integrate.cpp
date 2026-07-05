#include <ax/sym/calc.hpp>
#include <ax/sym/integrate.hpp>
#include <ax/sym/poly.hpp>

#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ax::sym {

namespace {

constexpr int kmax_parts_depth = 3;  ///< integration-by-parts recursion cap

bool depends_on(const expr& e, const expr& x) {
  if (e.same(x)) return true;
  for (const expr& a : e.args())
    if (depends_on(a, x)) return true;
  return false;
}

/** e == a*x + b with rational a != 0? */
std::optional<rational> linear_coefficient(const expr& e, const expr& x) {
  auto term_coeff = [&](const expr& t) -> std::optional<rational> {
    if (t.same(x)) return rational(bigint(1));
    if (t.is_mul()) {
      const auto fs = t.args();
      if (fs.size() == 2 && fs[0].is_num() && fs[1].same(x))
        return fs[0].value();
    }
    return std::nullopt;
  };
  if (auto c = term_coeff(e)) return c;
  if (e.is_add()) {
    std::optional<rational> coeff;
    for (const expr& t : e.args()) {
      if (!depends_on(t, x)) continue;
      if (coeff) return std::nullopt;
      coeff = term_coeff(t);
      if (!coeff) return std::nullopt;
    }
    return coeff;
  }
  return std::nullopt;
}

expr inv(const rational& a) { return expr::num(rational(bigint(1)) / a); }

expr rat(long long n, long long d = 1) {
  return expr::num(rational(bigint(n), bigint(d)));
}

/** Table antiderivative of fn(name, u), du absorbed by the caller. */
std::optional<expr> fn_table(const std::string& name, const expr& u) {
  if (name == "sin") return -(expr::fn("cos", u));
  if (name == "cos") return expr::fn("sin", u);
  if (name == "tan") return -(expr::fn("log", expr::fn("cos", u)));
  if (name == "exp") return expr::fn("exp", u);
  if (name == "log") return u * expr::fn("log", u) - u;
  if (name == "sqrt")
    return rat(2, 3) * u.pow(rat(3, 2));
  return std::nullopt;
}

/** Antiderivative of u^n d u. */
std::optional<expr> pow_table(const expr& u, const rational& n) {
  if (n == rational(bigint(-1))) return expr::fn("log", u);
  const rational n1 = n + rational(bigint(1));
  return u.pow(expr::num(n1)) * expr::num(rational(bigint(1)) / n1);
}

std::optional<expr> integrate_impl(const expr& e, const expr& x, int depth);

// ---------------------------------------------------------------- core rules

std::optional<expr> integrate_pow_core(const expr& base, const expr& ex,
                                       const expr& x) {
  if (depends_on(ex, x) || !ex.is_num()) return std::nullopt;
  const auto a = linear_coefficient(base, x);
  if (!a) return std::nullopt;
  auto t = pow_table(base, ex.value());
  if (!t) return std::nullopt;
  return *t * inv(*a);
}

std::optional<expr> integrate_core(const expr& e, const expr& x, int depth) {
  if (!depends_on(e, x)) return e * x;
  if (e.same(x)) return rat(1, 2) * x.pow(expr::num(2));
  if (e.is_add()) {
    expr sum = expr::num(0);
    for (const expr& t : e.args()) {
      auto part = integrate_impl(t, x, depth);
      if (!part) return std::nullopt;
      sum = sum + *part;
    }
    return sum;
  }
  if (e.is_mul()) {
    // pull x-free factors out; a single dependent factor recurses
    expr c = expr::num(1);
    std::vector<expr> dep;
    for (const expr& f : e.args()) {
      if (depends_on(f, x))
        dep.push_back(f);
      else
        c = c * f;
    }
    if (dep.size() == 1) {
      auto part = integrate_impl(dep[0], x, depth);
      if (part) return c * *part;
    }
    return std::nullopt;  // products of dependents: heuristics
  }
  if (e.is_pow()) return integrate_pow_core(e.args()[0], e.args()[1], x);
  if (e.is_fn()) {
    const expr& u = e.args()[0];
    if (const auto a = linear_coefficient(u, x)) {
      auto t = fn_table(e.name(), u);
      if (t) return *t * inv(*a);
    }
    return std::nullopt;
  }
  return std::nullopt;
}

// ------------------------------------------------------------ u-substitution

/** c with h == c * d (c x-free), matched structurally or via exact
    polynomial division. */
std::optional<expr> constant_ratio(const expr& h, const expr& d,
                                   const expr& x) {
  if (h.same(d)) return expr::num(1);
  auto split_num = [](const expr& e) -> std::pair<rational, expr> {
    if (e.is_mul()) {
      const auto fs = e.args();
      if (fs[0].is_num()) {
        expr rest = expr::num(1);
        for (std::size_t i = 1; i < fs.size(); ++i) rest = rest * fs[i];
        return {fs[0].value(), rest};
      }
    }
    return {rational(bigint(1)), e};
  };
  auto [ch, rh] = split_num(h);
  auto [cd, rd] = split_num(d);
  if (rh.same(rd)) return expr::num(ch / cd);
  try {
    const poly ph = poly::from_expr(h, x);
    const poly pd = poly::from_expr(d, x);
    if (pd.degree() >= 0) {
      auto [q, rem] = ph.divmod(pd);
      if (rem == poly{} && q.degree() == 0)
        return expr::num(q.coeff(0));
    }
  } catch (const std::invalid_argument&) {
  }
  return std::nullopt;
}

std::optional<expr> integrate_usub(const expr& e, const expr& x) {
  if (!e.is_mul()) return std::nullopt;
  std::vector<expr> dep;
  expr cfree = expr::num(1);
  for (const expr& f : e.args()) {
    if (depends_on(f, x))
      dep.push_back(f);
    else
      cfree = cfree * f;
  }
  if (dep.size() < 2) return std::nullopt;
  for (std::size_t i = 0; i < dep.size(); ++i) {
    // candidate composite factor: fn(g) or g^n with non-trivial inner g
    const expr& f = dep[i];
    expr g = expr::num(0);
    std::optional<expr> table;
    if (f.is_fn() && depends_on(f.args()[0], x)) {
      g = f.args()[0];
      table = fn_table(f.name(), g);
    } else if (f.is_pow() && f.args()[1].is_num() &&
               depends_on(f.args()[0], x)) {
      g = f.args()[0];
      table = pow_table(g, f.args()[1].value());
    }
    if (!table) continue;
    expr h = expr::num(1);
    for (std::size_t j = 0; j < dep.size(); ++j)
      if (j != i) h = h * dep[j];
    const expr dg = diff(g, x);
    if (const auto c = constant_ratio(h, dg, x)) return cfree * *c * *table;
  }
  return std::nullopt;
}

// ------------------------------------------------------- partial fractions

/** Gaussian elimination over rationals; a is n x (n+1) augmented. */
std::optional<std::vector<rational>> gauss_rational(
    std::vector<std::vector<rational>> a) {
  const std::size_t n = a.size();
  for (std::size_t col = 0; col < n; ++col) {
    std::size_t piv = col;
    while (piv < n && a[piv][col].is_zero()) ++piv;
    if (piv == n) return std::nullopt;
    std::swap(a[col], a[piv]);
    for (std::size_t r = 0; r < n; ++r) {
      if (r == col || a[r][col].is_zero()) continue;
      const rational f = a[r][col] / a[col][col];
      for (std::size_t k = col; k <= n; ++k)
        a[r][k] = a[r][k] - f * a[col][k];
    }
  }
  std::vector<rational> sol(n);
  for (std::size_t i = 0; i < n; ++i) sol[i] = a[i][n] / a[i][i];
  return sol;
}

struct linear_factor {
  rational root;
  int mult;
};

/** Split e (product form) into numerator and denominator polynomials in x. */
bool as_rational_function(const expr& e, const expr& x, poly& p, poly& q) {
  expr num_e = expr::num(1), den_e = expr::num(1);
  auto classify = [&](const expr& f) {
    if (f.is_pow() && f.args()[1].is_num()) {
      const rational& n = f.args()[1].value();
      if (n.den() == bigint(1) && n.num().is_negative()) {
        den_e = den_e * f.args()[0].pow(expr::num(-n));
        return;
      }
    }
    num_e = num_e * f;
  };
  if (e.is_mul())
    for (const expr& f : e.args()) classify(f);
  else
    classify(e);
  try {
    p = poly::from_expr(num_e, x);
    q = poly::from_expr(den_e, x);
  } catch (const std::invalid_argument&) {
    return false;
  }
  return q.degree() >= 1;
}

std::optional<expr> integrate_rational_fn(const expr& e, const expr& x) {
  poly p, q;
  if (!as_rational_function(e, x, p, q)) return std::nullopt;

  expr result = expr::num(0);
  auto [s, r] = p.divmod(q);
  if (s.degree() >= 0) {
    // polynomial part, integrated term by term
    for (int k = 0; k <= s.degree(); ++k) {
      const rational& c = s.coeff(static_cast<std::size_t>(k));
      if (c.is_zero()) continue;
      const rational kk(bigint(k + 1));
      result = result + expr::num(c / kk) * x.pow(expr::num(kk));
    }
  }
  if (r == poly{}) return result;

  // factor q: rational roots with multiplicity, then at most one
  // irreducible quadratic
  std::vector<linear_factor> lin;
  poly rest = q;
  std::vector<rational> roots;
  try {
    roots = q.rational_roots();
  } catch (const std::overflow_error&) {
    return std::nullopt;
  }
  for (const rational& root : roots) {
    const poly factor({-root, rational(bigint(1))});
    int m = 0;
    while (rest.degree() >= 1) {
      auto [qq, rem] = rest.divmod(factor);
      if (!(rem == poly{})) break;
      rest = qq;
      ++m;
    }
    if (m > 0) lin.push_back({root, m});
  }
  bool has_quad = false;
  rational qb, qc;  // monic x^2 + qb x + qc
  if (rest.degree() == 2) {
    const rational l2 = rest.coeff(2);
    qb = rest.coeff(1) / l2;
    qc = rest.coeff(0) / l2;
    const rational disc = qb * qb - rational(bigint(4)) * qc;
    if (!(disc < rational{})) return std::nullopt;  // real irrational roots
    has_quad = true;
  } else if (rest.degree() > 2) {
    return std::nullopt;
  }

  // unknowns: sum of multiplicities (+2 for the quadratic) == deg q
  const int n_unknown = q.degree();
  // basis polynomials: q / (x-root)^k, and {1, x} * q / quad
  std::vector<poly> basis;
  for (const auto& lf : lin) {
    poly acc = q;
    const poly factor({-lf.root, rational(bigint(1))});
    for (int k = 1; k <= lf.mult; ++k) {
      acc = acc.divmod(factor).first;
      basis.push_back(acc);
    }
  }
  if (has_quad) {
    poly qdivquad = q.divmod(rest).first;
    basis.push_back(qdivquad);                                // constant term
    basis.push_back(qdivquad * poly({rational{}, rational(bigint(1))}));  // x
  }
  if (static_cast<int>(basis.size()) != n_unknown) return std::nullopt;

  // match coefficients of r == sum u_j basis_j (degrees < deg q)
  std::vector<std::vector<rational>> aug(
      static_cast<std::size_t>(n_unknown),
      std::vector<rational>(static_cast<std::size_t>(n_unknown) + 1));
  for (int row = 0; row < n_unknown; ++row) {
    for (int col = 0; col < n_unknown; ++col)
      aug[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] =
          basis[static_cast<std::size_t>(col)].coeff(
              static_cast<std::size_t>(row));
    aug[static_cast<std::size_t>(row)][static_cast<std::size_t>(n_unknown)] =
        r.coeff(static_cast<std::size_t>(row));
  }
  const auto sol = gauss_rational(std::move(aug));
  if (!sol) return std::nullopt;

  // integrate each simple term
  std::size_t idx = 0;
  for (const auto& lf : lin) {
    const expr lin_e = x - expr::num(lf.root);
    for (int k = 1; k <= lf.mult; ++k, ++idx) {
      const rational& a = (*sol)[idx];
      if (a.is_zero()) continue;
      if (k == 1) {
        result = result + expr::num(a) * expr::fn("log", lin_e);
      } else {
        const rational e1(bigint(1 - k));
        result =
            result + expr::num(a / e1) * lin_e.pow(expr::num(e1));
      }
    }
  }
  if (has_quad) {
    // basis push order was {1, x}: sol[idx] = C, sol[idx+1] = B
    const rational ccoef = (*sol)[idx];
    const rational bcoef = (*sol)[idx + 1];
    const expr quad_e =
        x.pow(expr::num(2)) + expr::num(qb) * x + expr::num(qc);
    const rational half(bigint(1), bigint(2));
    const rational m2 = qc - qb * qb * rational(bigint(1), bigint(4));
    // (B x + C)/(x^2+qb x+qc): B/2 log(quad) + (C - B qb/2)/m atan((x+qb/2)/m)
    if (!bcoef.is_zero())
      result =
          result + expr::num(bcoef * half) * expr::fn("log", quad_e);
    const rational catan = ccoef - bcoef * qb * half;
    if (!catan.is_zero()) {
      const expr m = expr::num(m2).pow(expr::num(half));
      const expr im = expr::num(m2).pow(expr::num(-half));
      result = result + expr::num(catan) * im *
                            expr::fn("atan", (x + expr::num(qb * half)) * im);
    }
  }
  return result;
}

// ------------------------------------------------------ integration by parts

/** LIATE priority: lower = better u. */
int liate_priority(const expr& f, const expr& x) {
  if (f.is_fn()) {
    const std::string& n = f.name();
    if (n == "log" || n == "atan" || n == "asin" || n == "acos") return 0;
    return 2;  // trig / exp: prefer as dv
  }
  try {
    (void)poly::from_expr(f, x);
    return 1;  // algebraic
  } catch (const std::invalid_argument&) {
  }
  return 2;
}

std::optional<expr> integrate_parts(const expr& e, const expr& x, int depth) {
  if (depth <= 0) return std::nullopt;
  expr u = expr::num(0), dv = expr::num(0);
  expr cfree = expr::num(1);
  if (e.is_fn()) {
    if (liate_priority(e, x) != 0) return std::nullopt;
    u = e;
    dv = expr::num(1);
  } else if (e.is_mul()) {
    std::vector<expr> dep;
    for (const expr& f : e.args()) {
      if (depends_on(f, x))
        dep.push_back(f);
      else
        cfree = cfree * f;
    }
    if (dep.size() != 2) return std::nullopt;
    const int p0 = liate_priority(dep[0], x), p1 = liate_priority(dep[1], x);
    if (p0 == p1) return std::nullopt;
    u = (p0 < p1) ? dep[0] : dep[1];
    dv = (p0 < p1) ? dep[1] : dep[0];
  } else {
    return std::nullopt;
  }
  auto v = integrate_impl(dv, x, depth - 1);
  if (!v) return std::nullopt;
  const expr du = diff(u, x);
  auto rest = integrate_impl(*v * du, x, depth - 1);
  if (!rest) return std::nullopt;
  return cfree * (u * *v - *rest);
}

// ----------------------------------------------------------------- dispatch

/** Numeric spot check d(candidate)/dx == e at a few points; accepts when no
    sample point is evaluable (domain restrictions). */
bool verify(const expr& e, const expr& anti, const expr& x) {
  const expr d = diff(anti, x);
  int checked = 0;
  for (double t : {0.31, 1.72, 2.94, -0.57, 4.13}) {
    double want, got;
    try {
      want = e.eval({{x.name(), t}});
      got = d.eval({{x.name(), t}});
    } catch (const std::logic_error&) {
      continue;
    }
    if (!std::isfinite(want) || !std::isfinite(got)) continue;
    ++checked;
    if (std::abs(got - want) > 1e-6 * (1.0 + std::abs(want))) return false;
  }
  return true;
}

std::optional<expr> integrate_impl(const expr& e, const expr& x, int depth) {
  if (auto r = integrate_core(e, x, depth)) return r;
  if (auto r = integrate_usub(e, x)) return r;
  if (auto r = integrate_rational_fn(e, x)) return r;
  if (auto r = integrate_parts(e, x, depth)) return r;
  return std::nullopt;
}

}  // namespace

std::optional<expr> integrate(const expr& e, const expr& x) {
  if (!x.is_sym())
    throw std::invalid_argument("integrate: x must be a symbol");
  auto r = integrate_impl(e, x, kmax_parts_depth);
  if (r && !verify(e, *r, x)) return std::nullopt;  // heuristic safety net
  return r;
}

}  // namespace ax::sym
