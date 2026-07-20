#include <ax/search/search.hpp>

#include <ax/sym/calc.hpp>
#include <ax/sym/count_ops.hpp>
#include <ax/sym/expand.hpp>
#include <ax/sym/oracle.hpp>
#include <ax/sym/poly.hpp>
#include <ax/sym/print_sstr.hpp>

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

/** Rules tranche 2 (Phase C task 4): i_apart, i_log_power,
    i_transcend_div, i_inverse_trig, i_sqrt_basis — ports of the L5/L8
    autopsy rules from llmopt/search/rules.py. Candidates only; the edge
    verifier owns soundness (bad fires are pruned, never emitted). */

namespace ax::search {

namespace tranche2 {

using sym::expr;
using sym::poly;

const ax::rational kZero{};
const ax::rational kOneR{ax::bigint(1)};

bool contains(const expr& e, const expr& target) {
  if (e.same(target)) return true;
  for (const expr& a : e.args())
    if (contains(a, target)) return true;
  return false;
}

std::optional<std::pair<expr, expr>> unpack_i(const expr& node) {
  if (!node.is_fn() || node.name() != "Integral" ||
      node.args().size() != 2 || !node.args()[1].is_sym())
    return std::nullopt;
  return std::make_pair(node.args()[0], node.args()[1]);
}

/** Is e a rational function of x (numbers, x, + * /, integer powers)?
    Other symbols count as constants, like sympy is_rational_function. */
bool is_rational_in(const expr& e, const expr& x) {
  switch (e.k()) {
    case sym::kind::num:
    case sym::kind::sym:
      return true;
    case sym::kind::fn:
      return !contains(e, x);
    case sym::kind::add:
    case sym::kind::mul: {
      for (const expr& a : e.args())
        if (!is_rational_in(a, x)) return false;
      return true;
    }
    case sym::kind::pow: {
      if (!is_rational_in(e.args()[0], x)) return false;
      if (!contains(e.args()[0], x) && !contains(e.args()[1], x))
        return true;
      return e.args()[1].is_num() &&
             e.args()[1].value().den() == ax::bigint(1);
    }
  }
  return false;
}

/** Split canonical(f) into numerator/denominator by negative exponents
    (the sympy fraction(together(f)) analogue). */
std::pair<expr, expr> fraction_of(const expr& f, const expr& x) {
  const expr c = sym::canonical(f, x);
  expr num = expr::num(1);
  expr den = expr::num(1);
  const auto add_factor = [&](const expr& g) {
    if (g.is_num()) {
      num = num * expr::num(ax::rational(g.value().num(), ax::bigint(1)));
      if (!(g.value().den() == ax::bigint(1)))
        den = den * expr::num(ax::rational(g.value().den(), ax::bigint(1)));
      return;
    }
    if (g.is_pow() && g.args()[1].is_num() && g.args()[1].value() < kZero) {
      const auto flipped = -g.args()[1].value();
      den = den * (flipped == kOneR ? g.args()[0]
                                    : g.args()[0].pow(expr::num(flipped)));
      return;
    }
    num = num * g;
  };
  if (c.is_mul())
    for (const expr& g : c.args()) add_factor(g);
  else
    add_factor(c);
  return {num, den};
}

/** ∫ of a plain polynomial (term-wise power rule); nullopt if not poly. */
std::optional<expr> integrate_poly(const poly& p, const expr& x) {
  expr out = expr::num(0);
  for (int k = 0; k <= p.degree(); ++k) {
    const ax::rational& c = p.coeff(static_cast<std::size_t>(k));
    if (c == kZero) continue;
    const ax::rational c1 = c / ax::rational(ax::bigint(k + 1));
    out = out + expr::num(c1) * x.pow(expr::num(k + 1));
  }
  return out;
}

/** Whole-node substitution of a log atom by a placeholder symbol. */
expr replace_log(const expr& e, const expr& target, const expr& repl) {
  if (e.same(target)) return repl;
  switch (e.k()) {
    case sym::kind::num:
    case sym::kind::sym:
      return e;
    case sym::kind::fn: {
      std::vector<expr> mapped;
      for (const expr& a : e.args())
        mapped.push_back(replace_log(a, target, repl));
      return expr::fn(e.name(), std::move(mapped));
    }
    case sym::kind::add: {
      expr out = expr::num(0);
      for (const expr& t : e.args())
        out = out + replace_log(t, target, repl);
      return out;
    }
    case sym::kind::mul: {
      expr out = expr::num(1);
      for (const expr& f2 : e.args())
        out = out * replace_log(f2, target, repl);
      return out;
    }
    case sym::kind::pow:
      return replace_log(e.args()[0], target, repl)
          .pow(replace_log(e.args()[1], target, repl));
  }
  return e;
}

/** Exact Gaussian elimination for M c = rhs over rationals; free
    variables are set to 0 (sympy solve + subs(c->0) semantics).
    Returns false when inconsistent. */
bool solve_rational_linear(std::vector<std::vector<ax::rational>> m,
                           std::vector<ax::rational> rhs,
                           std::vector<ax::rational>& out) {
  const std::size_t rows = m.size();
  const std::size_t cols = rows ? m[0].size() : 0;
  std::vector<std::size_t> pivot_col;
  std::size_t r = 0;
  for (std::size_t c = 0; c < cols && r < rows; ++c) {
    std::size_t p = r;
    while (p < rows && m[p][c] == kZero) ++p;
    if (p == rows) continue;
    std::swap(m[p], m[r]);
    std::swap(rhs[p], rhs[r]);
    const ax::rational inv = kOneR / m[r][c];
    for (std::size_t j = c; j < cols; ++j) m[r][j] = m[r][j] * inv;
    rhs[r] = rhs[r] * inv;
    for (std::size_t i = 0; i < rows; ++i) {
      if (i == r || m[i][c] == kZero) continue;
      const ax::rational f = m[i][c];
      for (std::size_t j = c; j < cols; ++j)
        m[i][j] = m[i][j] - f * m[r][j];
      rhs[i] = rhs[i] - f * rhs[r];
    }
    pivot_col.push_back(c);
    ++r;
  }
  for (std::size_t i = r; i < rows; ++i)
    if (!(rhs[i] == kZero)) return false;  // inconsistent
  out.assign(cols, kZero);
  for (std::size_t i = 0; i < pivot_col.size(); ++i)
    out[pivot_col[i]] = rhs[i];
  return true;
}

poly poly_diff(const poly& p) { return p.derivative(); }

poly poly_scale(const poly& p, const ax::rational& c) {
  return p * poly({c});
}

poly x_power(std::size_t j) {
  std::vector<ax::rational> cs(j + 1, kZero);
  cs[j] = kOneR;
  return poly(std::move(cs));
}

std::vector<ax::rational> coeff_vector(const poly& p, std::size_t n) {
  std::vector<ax::rational> v(n, kZero);
  for (int k = 0; k <= p.degree() && static_cast<std::size_t>(k) < n; ++k)
    v[static_cast<std::size_t>(k)] = p.coeff(static_cast<std::size_t>(k));
  return v;
}

// ----------------------------------------------------------- i_apart

std::vector<expr> i_apart(const expr& node) {
  const auto un = unpack_i(node);
  if (!un) return {};
  const auto& [f, x] = *un;
  if (!is_rational_in(f, x)) return {};
  const auto [nume, dene] = fraction_of(f, x);
  poly pn, pd;
  try {
    pn = poly::from_expr(sym::expand(nume), x);
    pd = poly::from_expr(sym::expand(dene), x);
  } catch (const std::exception&) {
    return {};
  }
  if (pd.degree() < 1) return {};
  auto [quo, rem] = pn.divmod(pd);
  // factor den by distinct rational roots; v1 requires full linear split
  std::vector<ax::rational> roots;
  try {
    roots = pd.square_free().rational_roots();
  } catch (const std::exception&) {
    return {};
  }
  if (static_cast<int>(roots.size()) != pd.degree()) return {};
  std::sort(roots.begin(), roots.end(),
            [](const ax::rational& a, const ax::rational& b) {
              return a < b;
            });
  // distinct linear roots: rem/pd = sum rem(r)/pd'(r) / (x - r)
  const poly pdp = pd.derivative();
  expr out = expr::num(0);
  if (quo.degree() >= 0)
    for (int k = 0; k <= quo.degree(); ++k) {
      const auto& c = quo.coeff(static_cast<std::size_t>(k));
      if (!(c == kZero)) out = out + expr::num(c) * x.pow(expr::num(k));
    }
  for (const ax::rational& rt : roots) {
    const ax::rational a = rem.eval(rt) / pdp.eval(rt);
    if (a == kZero) continue;
    out = out + expr::num(a) / (x - expr::num(rt));
  }
  const expr g = out;
  if (g.same(f)) return {};
  return {expr::integral(g, x)};
}

// ------------------------------------------------------- i_log_power

std::vector<expr> i_log_power(const expr& node) {
  const auto un = unpack_i(node);
  if (!un) return {};
  const auto& [f, x] = *un;
  // c * x**n * log(k*x)**m ; n rational, m positive int
  expr c = expr::num(1);
  ax::rational n = kZero;
  std::optional<expr> L;
  long long m = 0;
  std::vector<expr> factors;
  if (f.is_mul())
    for (const expr& a : f.args()) factors.push_back(a);
  else
    factors.push_back(f);
  for (const expr& a : factors) {
    expr base = a;
    ax::rational e = kOneR;
    if (a.is_pow() && a.args()[1].is_num()) {
      base = a.args()[0];
      e = a.args()[1].value();
    }
    if (!contains(a, x)) {
      c = c * a;
    } else if (base.is_fn() && base.name() == "log") {
      if (L || !(e.den() == ax::bigint(1)) || e < kZero) return {};
      // log arg must be slope*x exactly (no constant term)
      const expr slope = sym::diff(base.args()[0], x);
      if (contains(slope, x) ||
          !base.args()[0].same(slope * x))
        return {};
      L = base;
      const std::string ms = e.num().to_string();
      if (ms.size() > 2) return {};
      m = std::stoll(ms);
      if (m <= 0) return {};
    } else if (base.same(x)) {
      n = n + e;
    } else {
      return {};
    }
  }
  if (!L) return {};
  if (n == ax::rational(ax::bigint(-1))) {
    const expr m1 = expr::num(m + 1);
    return {c * L->pow(m1) / m1};
  }
  // x^(n+1) * sum_j (-1)^(m-j) (m!/j!) L^j / (n+1)^(m-j+1)
  ax::rational fact = kOneR;
  for (long long i = 2; i <= m; ++i) fact = fact * ax::rational(ax::bigint(i));
  const ax::rational n1 = n + kOneR;
  expr s = expr::num(0);
  ax::rational jfact = kOneR;
  for (long long j = 0; j <= m; ++j) {
    if (j >= 2) jfact = jfact * ax::rational(ax::bigint(j));
    ax::rational coef = fact / jfact;
    if ((m - j) % 2 == 1) coef = -coef;
    ax::rational denpow = kOneR;
    for (long long i = 0; i < m - j + 1; ++i) denpow = denpow * n1;
    s = s + expr::num(coef / denpow) * L->pow(expr::num(j));
  }
  return {c * x.pow(expr::num(n1)) * s};
}

// ---------------------------------------------------- i_transcend_div

std::vector<expr> i_transcend_div(const expr& node) {
  const auto un = unpack_i(node);
  if (!un) return {};
  const auto& [f, x] = *un;
  const auto [nume, dene] = fraction_of(f, x);
  poly pd;
  try {
    pd = poly::from_expr(sym::expand(dene), x);
  } catch (const std::exception&) {
    return {};
  }
  if (pd.degree() < 1) return {};
  // group expanded numerator terms by transcendental monomial
  const expr en = sym::expand(nume);
  std::vector<expr> terms;
  if (en.is_add())
    for (const expr& t : en.args()) terms.push_back(t);
  else
    terms.push_back(en);
  std::vector<std::pair<expr, expr>> groups;  // (trans, rational coeff sum)
  for (const expr& t : terms) {
    expr rat = expr::num(1);
    expr trans = expr::num(1);
    std::vector<expr> fs;
    if (t.is_mul())
      for (const expr& a : t.args()) fs.push_back(a);
    else
      fs.push_back(t);
    for (const expr& a : fs)
      (is_rational_in(a, x) ? rat : trans) = (is_rational_in(a, x) ? rat : trans) * a;
    bool found = false;
    for (auto& [tr, sum] : groups)
      if (tr.same(trans)) {
        sum = sum + rat;
        found = true;
      }
    if (!found) groups.emplace_back(trans, rat);
  }
  if (groups.size() < 2) return {};
  expr out = expr::num(0);
  bool changed = false;
  for (const auto& [trans, coeff] : groups) {
    const auto [rn_e, rd_e] = fraction_of(coeff / dene, x);
    poly rn, rd;
    try {
      rn = poly::from_expr(sym::expand(rn_e), x);
      rd = poly::from_expr(sym::expand(rd_e), x);
    } catch (const std::exception&) {
      return {};
    }
    if (rd.degree() < 0) return {};
    auto [q, r] = rn.divmod(rd);
    if (q.degree() >= 0) changed = true;
    expr qe = expr::num(0);
    for (int k = 0; k <= q.degree(); ++k) {
      const auto& cq = q.coeff(static_cast<std::size_t>(k));
      if (!(cq == kZero)) qe = qe + expr::num(cq) * x.pow(expr::num(k));
    }
    expr re = expr::num(0);
    for (int k = 0; k <= r.degree(); ++k) {
      const auto& cr = r.coeff(static_cast<std::size_t>(k));
      if (!(cr == kZero)) re = re + expr::num(cr) * x.pow(expr::num(k));
    }
    out = out + trans * qe + trans * re / rd_e;
  }
  if (!changed || out.same(f)) return {};
  return {expr::integral(out, x)};
}

// --------------------------------------------------- i_inverse_trig

/** sqrt(r) as an expr, exact when r is a perfect square of a rational
    (sympy auto-evaluates sqrt(4) -> 2; an unevaluated sqrt(4) atom
    leaves the verifier structurally blind and the edge UNDECIDED —
    measured on the whole L5 inverse-trig family). Double gives the
    root guess; the check is exact bigint arithmetic, so an out-of-range
    guess only falls back to the symbolic atom (coverage, not
    soundness). */
expr sqrt_of(const ax::rational& r) {
  const auto exact_root = [](const ax::bigint& n) -> std::optional<ax::bigint> {
    double d = 0.0;
    try {
      d = std::stod(n.to_string());
    } catch (const std::exception&) {
      return std::nullopt;
    }
    if (!(d >= 0) || d > 9e18) return std::nullopt;
    const auto g = static_cast<long long>(std::sqrt(d));
    for (long long c : {g - 1, g, g + 1}) {
      if (c < 0) continue;
      const ax::bigint cb{c};
      if (cb * cb == n) return cb;
    }
    return std::nullopt;
  };
  if (!(r < ax::rational{})) {
    const auto pn = exact_root(r.num());
    const auto pd = exact_root(r.den());
    if (pn && pd) return expr::num(ax::rational(*pn, *pd));
  }
  return expr::fn("sqrt", expr::num(r));
}

std::vector<expr> i_inverse_trig(const expr& node) {
  const auto un = unpack_i(node);
  if (!un) return {};
  const auto& [f, x] = *un;
  auto [nume, dene] = fraction_of(f, x);
  expr poly_part = expr::num(0);
  if (contains(nume, x)) {
    poly pn, pd;
    try {
      pn = poly::from_expr(sym::expand(nume), x);
      pd = poly::from_expr(sym::expand(dene), x);
    } catch (const std::exception&) {
      return {};
    }
    auto [quo, rem] = pn.divmod(pd);
    if (rem.degree() > 0 || quo.degree() < 0) return {};
    const auto pp = integrate_poly(quo, x);
    if (!pp) return {};
    poly_part = *pp;
    nume = rem.degree() < 0 ? expr::num(0) : expr::num(rem.coeff(0));
    if (nume.is_num() && nume.value() == kZero) return {};
  }
  // rational form: den = a*x^2 + b, a>0, b>0
  {
    poly pd;
    bool ok = true;
    try {
      pd = poly::from_expr(sym::expand(dene), x);
    } catch (const std::exception&) {
      ok = false;
    }
    if (ok && pd.degree() == 2 && pd.coeff(1) == kZero &&
        kZero < pd.coeff(2) && kZero < pd.coeff(0)) {
      const ax::rational a = pd.coeff(2);
      const ax::rational b = pd.coeff(0);
      return {poly_part +
              nume / sqrt_of(a * b) *
                  expr::fn("atan", x * sqrt_of(a / b))};
    }
  }
  // sqrt form: den = co * sqrt(b - a*x^2), a>0, b>0
  {
    expr co = expr::num(1);
    std::optional<expr> radicand;
    std::vector<expr> fs;
    if (dene.is_mul())
      for (const expr& g : dene.args()) fs.push_back(g);
    else
      fs.push_back(dene);
    for (const expr& g : fs) {
      const bool half_pow =
          g.is_pow() && g.args()[1].is_num() &&
          g.args()[1].value() == ax::rational(ax::bigint(1), ax::bigint(2));
      if (g.is_fn() && g.name() == "sqrt" && contains(g, x)) {
        if (radicand) return {};
        radicand = g.args()[0];
      } else if (half_pow && contains(g, x)) {
        if (radicand) return {};
        radicand = g.args()[0];
      } else {
        co = co * g;
      }
    }
    if (radicand && !contains(co, x)) {
      poly q;
      try {
        q = poly::from_expr(sym::expand(*radicand), x);
      } catch (const std::exception&) {
        return {};
      }
      if (q.degree() == 2 && q.coeff(1) == kZero) {
        const ax::rational a = -q.coeff(2);
        const ax::rational b = q.coeff(0);
        if (kZero < a && kZero < b)
          return {nume / (co * sqrt_of(a)) *
                  expr::fn("asin", x * sqrt_of(a / b))};
      }
    }
  }
  return {};
}

// ----------------------------------------------------- i_sqrt_basis

/** Solve for polynomial A with 2A'P + AP' = 2h; nullopt if no solution. */
std::optional<poly> sqrt_basis_solve(const poly& h, const poly& P,
                                     int degA) {
  const poly Pp = P.derivative();
  const std::size_t nrows =
      static_cast<std::size_t>(std::max(h.degree(),
                                        degA - 1 + P.degree()) + 2);
  std::vector<std::vector<ax::rational>> m(
      nrows, std::vector<ax::rational>(static_cast<std::size_t>(degA) + 1,
                                       kZero));
  for (int j = 0; j <= degA; ++j) {
    // column j: coeffs of 2*(x^j)'*P + x^j*P'
    poly col = poly_scale(x_power(static_cast<std::size_t>(j)) * Pp, kOneR);
    if (j > 0)
      col = col + poly_scale(x_power(static_cast<std::size_t>(j - 1)) * P,
                             ax::rational(ax::bigint(2 * j)));
    const auto v = coeff_vector(col, nrows);
    for (std::size_t i = 0; i < nrows; ++i)
      m[i][static_cast<std::size_t>(j)] = v[i];
  }
  const auto rhs = coeff_vector(poly_scale(h, ax::rational(ax::bigint(2))),
                                nrows);
  std::vector<ax::rational> sol;
  if (!solve_rational_linear(std::move(m), rhs, sol)) return std::nullopt;
  return poly(std::move(sol));
}

std::vector<expr> i_sqrt_basis(const expr& node) {
  const auto un = unpack_i(node);
  if (!un) return {};
  const auto& [f, x] = *un;
  if (sym::count_ops(f) > 200) return {};  // coarse; budget guards
  // collect sqrt-of-poly radicands (fn sqrt and pow ±1/2 forms)
  std::vector<expr> bases;
  const std::function<void(const expr&)> walk = [&](const expr& e) {
    const bool half =
        e.is_pow() && e.args()[1].is_num() &&
        (e.args()[1].value() == ax::rational(ax::bigint(1), ax::bigint(2)) ||
         e.args()[1].value() == ax::rational(ax::bigint(-1), ax::bigint(2)));
    if ((e.is_fn() && e.name() == "sqrt") || half) {
      const expr b = e.args()[0];
      if (contains(b, x)) {
        try {
          (void)poly::from_expr(sym::expand(b), x);
          bases.push_back(b);
        } catch (const std::exception&) {
        }
      }
    }
    for (const expr& a : e.args()) walk(a);
  };
  walk(f);
  if (bases.empty()) return {};
  const expr P_e = *std::max_element(
      bases.begin(), bases.end(), [](const expr& a, const expr& b) {
        return sym::count_ops(a) < sym::count_ops(b);
      });
  poly P;
  try {
    P = poly::from_expr(sym::expand(P_e), x);
  } catch (const std::exception&) {
    return {};
  }
  if (P.degree() < 1) return {};
  // h = cancel(expand(f*sqrt(P))): canonical merges the radicals
  const expr h_e =
      sym::canonical(f * expr::fn("sqrt", P_e), x);
  poly h;
  bool h_is_poly = true;
  try {
    h = poly::from_expr(sym::expand(h_e), x);
  } catch (const std::exception&) {
    h_is_poly = false;
  }
  if (!h_is_poly) {
    // sqrt*log combination branch (tranche 4; the L8 autopsy family):
    // ansatz (A(x) + B(x)*log(q)) * sqrt(P). Clearing the d/dx residual
    // by 2*sqrt(P)*q leaves a poly identity in (x, u=log(q)) with
    // deg_u <= 1, which decouples into two rational linear systems:
    //   u^1:  q*(2B'P + BP')          = h2_1        (solve B)
    //   u^0:  q*(2A'P + AP') + 2BPq'  = h2_0        (then solve A)
    std::vector<expr> logs;
    const std::function<void(const expr&)> wl = [&](const expr& q) {
      if (q.is_fn() && q.name() == "log" && contains(q.args()[0], x)) {
        bool dup = false;
        for (const expr& g : logs) dup = dup || g.same(q);
        if (!dup) logs.push_back(q);
      }
      for (const expr& a2 : q.args()) wl(a2);
    };
    wl(f);
    if (logs.size() != 1) return {};
    const expr& Lg = logs[0];
    poly q;
    try {
      q = poly::from_expr(sym::expand(Lg.args()[0]), x);
    } catch (const std::exception&) {
      return {};
    }
    if (q.degree() < 1) return {};
    const expr u = expr::symbol("ulog_");
    expr h2_e = expr::num(0);
    try {
      h2_e = sym::canonical(
          f * expr::num(2) * expr::fn("sqrt", P_e) * Lg.args()[0], x);
    } catch (const std::exception&) {
      return {};
    }
    // split by u-degree: part0 = h2|_{u=0}; part1 = (h2 - part0)|_{u=1}
    // (fraction_of first: canonical can leave a residual poly
    // denominator; exact-divide the parts through it)
    const auto [h2n, h2d] = fraction_of(h2_e, x);
    const expr h2ph = sym::expand(replace_log(h2n, Lg, u));
    const expr part0_e = h2ph.subs(u, expr::num(0));
    const expr part1_e = sym::expand(h2ph - part0_e).subs(u, expr::num(1));
    poly h0, h1, hden;
    try {
      h0 = poly::from_expr(sym::expand(part0_e), x);
      h1 = poly::from_expr(sym::expand(part1_e), x);
      hden = poly::from_expr(sym::expand(h2d), x);
    } catch (const std::exception&) {
      return {};  // deg_u > 1, non-poly residue, or radical denominator
    }
    if (hden.degree() >= 0 && !(hden.degree() == 0 &&
                                hden.coeff(0) == kOneR)) {
      auto [q0, r0] = h0.divmod(hden);
      auto [q1, r1] = h1.divmod(hden);
      if (r0.degree() >= 0 || r1.degree() >= 0) return {};
      h0 = q0;
      h1 = q1;
    }
    const int hdeg = std::max(h0.degree(), h1.degree());
    const int degA = std::max(hdeg - P.degree() + 1, 0) + 1;
    if (degA > 8) return {};
    const poly Pp = P.derivative();
    const poly qp = q.derivative();
    const std::size_t nrows = static_cast<std::size_t>(
        std::max({h0.degree(), h1.degree(),
                  degA + P.degree() + q.degree()}) + 2);
    // shared column shape: col_j = q*(2*(x^j)'*P + x^j*P')
    const auto make_cols = [&] {
      std::vector<std::vector<ax::rational>> m(
          nrows, std::vector<ax::rational>(
                     static_cast<std::size_t>(degA) + 1, kZero));
      for (int j = 0; j <= degA; ++j) {
        poly col = x_power(static_cast<std::size_t>(j)) * Pp;
        if (j > 0)
          col = col + poly_scale(
                          x_power(static_cast<std::size_t>(j - 1)) * P,
                          ax::rational(ax::bigint(2 * j)));
        col = col * q;
        const auto v = coeff_vector(col, nrows);
        for (std::size_t i = 0; i < nrows; ++i)
          m[i][static_cast<std::size_t>(j)] = v[i];
      }
      return m;
    };
    // u^1: solve B
    std::vector<ax::rational> solB;
    if (!solve_rational_linear(make_cols(), coeff_vector(h1, nrows), solB))
      return {};
    poly B{std::vector<ax::rational>(solB)};
    // u^0: rhs = h0 - 2*B*P*q'; solve A
    const poly corr = poly_scale(B * P * qp, ax::rational(ax::bigint(2)));
    const poly rhs0 = h0 + poly_scale(corr, ax::rational(ax::bigint(-1)));
    std::vector<ax::rational> solA;
    if (!solve_rational_linear(make_cols(), coeff_vector(rhs0, nrows), solA))
      return {};
    poly A{std::vector<ax::rational>(solA)};
    expr a_e = expr::num(0);
    for (int k2 = 0; k2 <= A.degree(); ++k2) {
      const auto& c = A.coeff(static_cast<std::size_t>(k2));
      if (!(c == kZero)) a_e = a_e + expr::num(c) * x.pow(expr::num(k2));
    }
    expr b_e = expr::num(0);
    for (int k2 = 0; k2 <= B.degree(); ++k2) {
      const auto& c = B.coeff(static_cast<std::size_t>(k2));
      if (!(c == kZero)) b_e = b_e + expr::num(c) * x.pow(expr::num(k2));
    }
    const expr ans = (a_e + b_e * Lg) * expr::fn("sqrt", P_e);
    if (ans.is_num() && ans.value() == kZero) return {};
    deriv_trace_push("ansatz (A(x) + B(x)*log(" +
                     sym::to_sstr(Lg.args()[0]) + "))*sqrt(" +
                     sym::to_sstr(P_e) +
                     ") with polynomial A, B; differentiate, clear by"
                     " 2*sqrt(P)*q, match coefficients");
    return {ans};
  }
  const int degA = std::max(h.degree() - P.degree() + 1, 0) + 1;
  if (degA > 8) return {};
  const auto A = sqrt_basis_solve(h, P, degA);
  if (!A) return {};
  expr a_e = expr::num(0);
  for (int k = 0; k <= A->degree(); ++k) {
    const auto& c = A->coeff(static_cast<std::size_t>(k));
    if (!(c == kZero)) a_e = a_e + expr::num(c) * x.pow(expr::num(k));
  }
  if (a_e.is_num() && a_e.value() == kZero) return {};
  deriv_trace_push("ansatz A(x)*sqrt(" + sym::to_sstr(P_e) +
                   ") with polynomial A;"
                   " match 2*A'*P + A*P' = 2*f*sqrt(P) in the poly ring");
  return {a_e * expr::fn("sqrt", P_e)};
}

}  // namespace tranche2

void add_tranche2(rule_set& r) {
  r.integral.emplace_back("i_apart", tranche2::i_apart);
  r.integral.emplace_back("i_log_power", tranche2::i_log_power);
  r.integral.emplace_back("i_transcend_div", tranche2::i_transcend_div);
  r.integral.emplace_back("i_inverse_trig", tranche2::i_inverse_trig);
  r.integral.emplace_back("i_sqrt_basis", tranche2::i_sqrt_basis);
}

}  // namespace ax::search
