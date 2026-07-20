#include <ax/sym/oracle.hpp>

#include <ax/sym/budget.hpp>
#include <ax/sym/calc.hpp>
#include <ax/sym/count_ops.hpp>
#include <ax/sym/expand.hpp>
#include <ax/sym/poly.hpp>

#include <array>
#include <functional>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <map>
#include <numbers>
#include <set>
#include <iostream>
#include <string>

namespace ax::sym {

namespace {

const rational kZero{};
const rational kOne{bigint(1)};

/** Numerator/denominator pair; both canonical exprs, den never literal 0. */
struct ratio {
  expr num;
  expr den;
};

/** Integer exponents beyond this are kept opaque (matches the guard the
    poly-conversion layer already applies to huge powers). */
constexpr long long kMaxExpandedExponent = 64;

ratio as_ratio(const expr& e, const expr& x);

/** Rebuild e with canonicalized sub-parts, then split into num/den. */
ratio ratio_pow(const expr& base, const expr& ex, const expr& x) {
  if (ex.is_num() && ex.value().den() == bigint(1)) {
    const bigint& n = ex.value().num();
    const std::string ns = n.to_string();
    // Small integer exponent: distribute over the base's num/den.
    if (ns.size() <= 2 ||
        (ns.size() <= 3 && ns[0] == '-')) {  // fits easily in long long
      const long long k = std::strtoll(ns.c_str(), nullptr, 10);
      // blowup guard: distributing base^k through the ratio machinery
      // feeds expand(); bound k by the base's additive width up front
      const double bw = static_cast<double>(
          base.is_add() ? base.args().size() : 1);
      double est = 1.0;
      for (long long i = 1; i <= std::llabs(k) && est < 1e9; ++i)
        est *= (bw + static_cast<double>(std::llabs(k) - i)) /
               static_cast<double>(i);
      if (k != 0 && std::llabs(k) <= kMaxExpandedExponent &&
          est <= 3000.0) {
        ratio b = as_ratio(base, x);
        const expr kk = expr::num(std::llabs(k));
        if (k > 0) return {b.num.pow(kk), b.den.pow(kk)};
        return {b.den.pow(kk), b.num.pow(kk)};
      }
      if (k == 0) return {expr::num(1), expr::num(1)};
    }
  }
  // Fractional negative exponent (e.g. u^(-1/2) from 1/sqrt(u)): route the
  // positive power into the denominator so sqrt shapes share a common
  // denominator and cancel via pow merging.
  if (ex.is_num() && !(ex.value().den() == bigint(1)) &&
      ex.value() < rational{}) {
    return {expr::num(1), canonical(base, x).pow(expr::num(-ex.value()))};
  }
  // Opaque power: canonicalize both parts, keep as an atom.
  return {canonical(base, x).pow(canonical(ex, x)), expr::num(1)};
}

ratio as_ratio(const expr& e, const expr& x) {
  check_work_budget();
  switch (e.k()) {
    case kind::num:
    case kind::sym:
      return {e, expr::num(1)};
    case kind::fn: {
      // sqrt(u) -> u^(1/2): lets expr's pow merging do the algebra
      // (sqrt(u)*sqrt(u) -> u, sqrt(u)/u -> u^(-1/2), ...). Sound because
      // sqrt defined implies u >= 0.
      if (e.name() == "sqrt")
        return {canonical(e.args()[0], x)
                    .pow(expr::num(rational(bigint(1), bigint(2)))),
                expr::num(1)};
      // rebuild ALL args (carriers are n-ary; rebuilding unary silently
      // truncated Subs to one arg -> span assertion downstream)
      std::vector<expr> mapped;
      mapped.reserve(e.args().size());
      for (const expr& a : e.args()) mapped.push_back(canonical(a, x));
      return {expr::fn(e.name(), std::move(mapped)), expr::num(1)};
    }
    case kind::pow:
      return ratio_pow(e.args()[0], e.args()[1], x);
    case kind::mul: {
      ratio r{expr::num(1), expr::num(1)};
      for (const expr& f : e.args()) {
        ratio rf = as_ratio(f, x);
        r.num = r.num * rf.num;
        r.den = r.den * rf.den;
      }
      return r;
    }
    case kind::add: {
      ratio r{expr::num(0), expr::num(1)};
      for (const expr& t : e.args()) {
        ratio rt = as_ratio(t, x);
        r.num = r.num * rt.den + rt.num * r.den;
        r.den = r.den * rt.den;
      }
      return r;
    }
  }
  return {e, expr::num(1)};  // unreachable
}

/** Reduce num/den by polynomial gcd when both are polynomials in x;
    denominator made monic. Returns true on success. */
bool poly_reduce(expr& num, expr& den, const expr& x) {
  try {
    poly pn = poly::from_expr(num, x);
    poly pd = poly::from_expr(den, x);
    if (pd.degree() < 0) return false;  // structurally zero denominator
    poly g = gcd(pn, pd);
    if (g.degree() > 0) {
      pn = pn.divmod(g).first;
      pd = pd.divmod(g).first;
    }
    const rational lc = pd.coeff(static_cast<std::size_t>(pd.degree()));
    if (!(lc == kOne)) {
      const poly scale({rational(lc.den(), lc.num())});
      pn = pn * scale;
      pd = pd * scale;
    }
    num = pn.to_expr(x);
    den = pd.to_expr(x);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

// --------------------------------------------------------------- sampling

/** Collect free symbol names. */
void free_symbols(const expr& e, std::set<std::string>& out) {
  if (e.is_sym()) {
    out.insert(e.name());
    return;
  }
  for (const expr& a : e.args()) free_symbols(a, out);
}

/** Deterministic parameter binding for symbols other than x: irrational-ish
    values chosen to dodge coincidental algebraic relations. */
std::map<std::string, double> parameter_env(const expr& a, const expr& b,
                                            const std::string& xname) {
  std::set<std::string> syms;
  free_symbols(a, syms);
  free_symbols(b, syms);
  std::map<std::string, double> env;
  env["pi"] = std::numbers::pi;
  env["E"] = std::numbers::e;
  double v = 1.0;
  for (const std::string& s : syms) {
    if (s == xname || s == "pi" || s == "E") continue;
    v += 0.31830988618;  // 1/pi increments: distinct, non-integer
    env[s] = v;
  }
  return env;
}

constexpr std::array<double, 14> kSamplePoints = {
    0.37, 1.71, 2.93, 5.41, 0.123, 7.77, 12.3, -0.37, -1.71, -2.93, -5.41,
    -0.123, -7.77, -12.3};

constexpr int kMinValidPoints = 3;
constexpr int kMinWitnesses = 2;

bool eval_at(const expr& e, std::map<std::string, double>& env, double& out) {
  try {
    out = e.eval(env);
  } catch (const std::exception&) {
    return false;
  }
  return std::isfinite(out);
}

}  // namespace

/** @param use_trial run the pre-expand factored-form cancellation.
    The trial and the plain path both return correct canonical forms,
    but DIFFERENT ones; canonical() below runs both and keeps the
    smaller, so a trial that helps L4 cannot cost L7 a beam path. */
expr canonical_impl(const expr& e, const expr& x, bool use_trial) {
  ratio r = as_ratio(e, x);
  // factored-form cancellation BEFORE expand: hash-consing makes shared
  // factors (incl. radical sums like 2*sqrt(x)+sqrt(5)) pointer-equal
  // across num/den, and expand destroys that structure. Divide the
  // factored numerator by each denominator factor; accept only a clean
  // result (no negative powers / no inverted-Add wrappers survive).
  // short-circuit: the trial's unique value over the end-stage factor
  // division is Add-factor cancellation (expand destroys those); skip
  // the whole attempt when the denominator has no Add factor. (Perf:
  // the unconditional trial taxed every canonical call and pushed 7
  // borderline L7 roots over the search deadline.)
  const auto has_add_factor = [](const expr& d) {
    if (d.is_add()) return true;
    if (!d.is_mul()) return false;
    for (const expr& f : d.args())
      if (f.is_add() || (f.is_pow() && f.args()[0].is_add())) return true;
    return false;
  };
  if (use_trial && !(r.den.is_num()) && has_add_factor(r.den)) {
    const std::function<bool(const expr&)> dirty = [&](const expr& q) {
      if (q.is_pow() && q.args()[1].is_num() && q.args()[1].value() < kZero)
        return true;
      for (const expr& a : q.args())
        if (dirty(a)) return true;
      return false;
    };
    // scalar-multiple adds: make_mul distributes numerics into Add
    // factors, so the numerator may hold c*f where the denominator
    // holds f (18*sqrt(x)+9*sqrt(5) vs 2*sqrt(x)+sqrt(5)). Detect via
    // the first-term ratio and cancel to the constant.
    const auto divide_factor = [&](expr t, const expr& f) {
      if (f.is_add() && t.is_mul()) {
        for (const expr& g0 : t.args()) {
          if (!g0.is_add()) continue;
          // the numerator factor may hold nested products (24*sqrt(x)*
          // (5*x - 2) + 6*sqrt(5)) where the denominator is distributed;
          // expand it for the term-count/ratio match, but divide out the
          // ORIGINAL factor g0 so the quotient stays factored.
          const expr g = g0.args().size() == f.args().size()
                             ? g0
                             : expand(g0);
          if (!g.is_add() || g.args().size() != f.args().size()) continue;
          // term ratio g0/fi, dividing factor-wise (a bare g0/fi with a
          // Mul fi builds pow(mul,-1) and never cancels). Add ordering
          // is not scale-invariant (135*x**2 + 9*cos(x) + 27 leads with
          // the cos term while 15*x**2 + cos(x) + 3 leads with x**2), so
          // g0 is ratioed against EVERY term of f, not just f0.
          for (const expr& fi : f.args()) {
            expr c = g.args()[0];
            if (fi.is_mul())
              for (const expr& h : fi.args()) c = c / h;
            else
              c = c / fi;
            if (!c.is_num()) continue;
            if (g.same(expr::num(c.value()) * f))
              return (t / g0) * c;
          }
        }
      }
      return t / f;
    };
    expr trial = r.num;
    if (r.den.is_mul())
      for (const expr& f : r.den.args()) trial = divide_factor(trial, f);
    else
      trial = divide_factor(trial, r.den);
    if (!dirty(trial)) r = {trial, expr::num(1)};
  }
  expr num = expand(r.num);
  expr den = expand(r.den);
  if (num.is_num() && num.value() == kZero) return expr::num(0);
  if (den.is_num()) return num / den;
  if (!poly_reduce(num, den, x)) {
    // opaque factors (fn nodes, radicals) block whole-expression poly
    // reduction; cancel the polynomial PART of the numerator against the
    // denominator instead: num = P(x)*rest -> gcd-reduce P/den.
    expr pprod = expr::num(1);
    expr opaque = expr::num(1);
    if (num.is_mul()) {
      for (const expr& f : num.args()) {
        bool is_poly = true;
        try {
          (void)poly::from_expr(f, x);
        } catch (const std::exception&) {
          is_poly = false;
        }
        if (is_poly)
          pprod = pprod * f;
        else
          opaque = opaque * f;
      }
      if (!(opaque.is_num() && opaque.value() == kOne)) {
        expr pn = expand(pprod);
        expr pd = den;
        if (poly_reduce(pn, pd, x)) {
          num = pn * opaque;
          den = pd;
        }
      }
    }
  }
  // shared-opaque-factor cancellation (the radicand-tax fix): a
  // non-numeric factor present in EVERY additive term of both numerator
  // and denominator divides through — e.g.
  // (a*sqrt(P) + b*sqrt(P)) / (q1*sqrt(P) + q2*sqrt(P)) -> (a+b)/(q1+q2).
  {
    const auto factor_list = [](const expr& t) {
      std::vector<expr> fs;
      if (t.is_mul())
        for (const expr& a : t.args()) fs.push_back(a);
      else
        fs.push_back(t);
      return fs;
    };
    const auto has_factor = [&](const expr& t, const expr& f) {
      for (const expr& a : factor_list(t)) {
        if (a.same(f)) return true;
        if (a.is_pow() && a.args()[0].same(f) && a.args()[1].is_num() &&
            kZero < a.args()[1].value())
          return true;
      }
      return false;
    };
    const auto terms_of = [](const expr& e2) {
      std::vector<expr> ts;
      if (e2.is_add())
        for (const expr& t : e2.args()) ts.push_back(t);
      else
        ts.push_back(e2);
      return ts;
    };
    bool progress = true;
    while (progress && !den.is_num()) {
      progress = false;
      // candidate factors: non-numeric factors of the first num term
      for (const expr& f : factor_list(terms_of(num).front())) {
        const expr base =
            f.is_pow() && f.args()[1].is_num() && kZero < f.args()[1].value()
                ? f.args()[0]
                : f;
        if (base.is_num()) continue;
        bool all = true;
        for (const expr& t : terms_of(num)) all = all && has_factor(t, base);
        for (const expr& t : terms_of(den)) all = all && has_factor(t, base);
        if (!all) continue;
        expr new_num = expr::num(0);
        for (const expr& t : terms_of(num)) new_num = new_num + t / base;
        expr new_den = expr::num(0);
        for (const expr& t : terms_of(den)) new_den = new_den + t / base;
        num = new_num;
        den = new_den;
        progress = true;
        break;
      }
    }
    if (den.is_num()) return num / den;
    // the cancel may have re-enabled full polynomial reduction
    poly_reduce(num, den, x);
  }
  // per-opaque-group exact division (sympy cancel on mixed-opaque
  // sums): group the expanded numerator's terms by their opaque
  // monomial, and when every group's polynomial coefficient divides the
  // poly denominator exactly, replace the fraction by the quotient sum
  // — (p1(x)*atan + p2(x)*log)/q(x) with q | p1, q | p2. Without this,
  // ansatz rules see phantom 1/q monomials and their linear systems go
  // inconsistent (the L7 atan*log root cause).
  if (num.is_add()) {
    poly pd_g;
    bool den_poly = true;
    try {
      pd_g = poly::from_expr(expand(den), x);
    } catch (const std::exception&) {
      den_poly = false;
    }
    if (den_poly && pd_g.degree() >= 1) {
      std::vector<std::pair<expr, expr>> groups;  // opaque -> poly sum
      bool shape_ok = true;
      for (const expr& t : num.args()) {
        expr pprod = expr::num(1);
        expr opaque = expr::num(1);
        const auto push = [&](const expr& f) {
          try {
            (void)poly::from_expr(expand(f), x);
            pprod = pprod * f;
          } catch (const std::exception&) {
            opaque = opaque * f;
          }
        };
        if (t.is_mul())
          for (const expr& f : t.args()) push(f);
        else
          push(t);
        bool found = false;
        for (auto& [op, ps] : groups)
          if (op.same(opaque)) {
            ps = ps + pprod;
            found = true;
          }
        if (!found) groups.emplace_back(opaque, pprod);
        (void)shape_ok;
      }
      expr new_num = expr::num(0);
      bool all_divide = !groups.empty();
      for (const auto& [op, ps] : groups) {
        poly pg;
        try {
          pg = poly::from_expr(expand(ps), x);
        } catch (const std::exception&) {
          all_divide = false;
          break;
        }
        auto [q, r] = pg.divmod(pd_g);
        if (r.degree() >= 0) {
          all_divide = false;
          break;
        }
        new_num = new_num + q.to_expr(x) * op;
      }
      if (all_divide) return new_num;
    }
  }
  // divide factor-wise so shared factors cancel via pow merging
  // (num/mul{a,b} as pow(mul,-1) would block a*x/(2*x)-style cancels)
  if (den.is_mul()) {
    expr out = num;
    for (const expr& f : den.args()) out = out / f;
    return out;
  }
  return num / den;
}

expr canonical(const expr& e, const expr& x) {
  const expr plain = canonical_impl(e, x, false);
  // The trial path is a different-but-equal canonical form, not a better
  // one: measured, enabling it unconditionally bought L4 (+11, factored
  // radical sums survive) and cost L7 (-7, changed forms changed HCE and
  // rerouted beams). Run both and keep the smaller — ties go to the
  // plain path, so the search's historical trajectories are preserved.
  const expr trial = canonical_impl(e, x, true);
  if (trial.same(plain)) return plain;
  return count_ops(trial) < count_ops(plain) ? trial : plain;
}

verdict equivalent(const expr& a, const expr& b, const expr& x) {
  const expr c = canonical(a - b, x);
  if (c.is_num() && c.value() == kZero) return verdict::equivalent;

  std::map<std::string, double> env = parameter_env(a, b, x.name());
  int valid = 0;
  int witnesses = 0;
  for (const double p : kSamplePoints) {
    env[x.name()] = p;
    double va = 0.0;
    double vb = 0.0;
    if (!eval_at(a, env, va) || !eval_at(b, env, vb)) continue;
    ++valid;
    const double scale = std::max(std::abs(va), std::abs(vb));
    if (std::abs(va - vb) > 1e-9 + 1e-9 * scale) {
      // Confirm at a nearby point to rule out a conditioning fluke.
      env[x.name()] = p + 1e-3;
      double wa = 0.0;
      double wb = 0.0;
      if (eval_at(a, env, wa) && eval_at(b, env, wb) &&
          std::abs(wa - wb) >
              1e-9 + 1e-9 * std::max(std::abs(wa), std::abs(wb)))
        ++witnesses;
    }
  }
  if (witnesses >= kMinWitnesses && valid >= kMinValidPoints)
    return verdict::not_equivalent;
  return verdict::undecided;
}

verdict equivalent_mod_const(const expr& candidate, const expr& integrand,
                             const expr& x) {
  return equivalent(diff(candidate, x), integrand, x);
}

}  // namespace ax::sym
