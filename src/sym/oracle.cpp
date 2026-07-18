#include <ax/sym/oracle.hpp>

#include <ax/sym/calc.hpp>
#include <ax/sym/expand.hpp>
#include <ax/sym/poly.hpp>

#include <array>
#include <cmath>
#include <cstdlib>
#include <map>
#include <numbers>
#include <set>
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
      if (k != 0 && std::llabs(k) <= kMaxExpandedExponent) {
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
  switch (e.k()) {
    case kind::num:
    case kind::sym:
      return {e, expr::num(1)};
    case kind::fn: {
      const expr arg = canonical(e.args()[0], x);
      // sqrt(u) -> u^(1/2): lets expr's pow merging do the algebra
      // (sqrt(u)*sqrt(u) -> u, sqrt(u)/u -> u^(-1/2), ...). Sound because
      // sqrt defined implies u >= 0.
      if (e.name() == "sqrt")
        return {arg.pow(expr::num(rational(bigint(1), bigint(2)))),
                expr::num(1)};
      return {expr::fn(e.name(), arg), expr::num(1)};
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

expr canonical(const expr& e, const expr& x) {
  ratio r = as_ratio(e, x);
  expr num = expand(r.num);
  expr den = expand(r.den);
  if (num.is_num() && num.value() == kZero) return expr::num(0);
  if (den.is_num()) return num / den;
  poly_reduce(num, den, x);
  return num / den;
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
