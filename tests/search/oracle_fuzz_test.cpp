/** @file oracle_fuzz_test.cpp Fuzz-the-oracle CI (relay 2026-07-27-0
    ask 2). Property-based fuzz of verify_edge: random states x random
    rewrites, symbolic-verdict vs INDEPENDENT-numeric cross-check.

    The independent evaluator shares nothing with the symbolic layer
    under test: Integral atoms evaluate by adaptive G7-K15 quadrature
    from a fixed base point, Derivative atoms by central finite
    difference, Subs by environment substitution — so a bug in diff/
    canonical/equivalent cannot cancel out of both sides. The value
    cache is persistent (a verifier bug would fossilize into cached
    labels), which is why this runs as a standing CI node, not a
    one-off.

    Properties:
      1. ACCEPT-soundness: every edge verify_edge accepts must be
         numerically equal (mod an additive constant on integral
         edges).
      2. REJECT-completeness on corruptions: an x-dependent
         perturbation of an accepted child that the numerics confirm
         non-equivalent must be rejected.
    Deterministic seeds; minimum-coverage floors keep the test from
    passing by skipping. */
#include <ax/search/search.hpp>

#include <ax/num/quad.hpp>
#include <ax/sym/parse.hpp>
#include <ax/sym/print_sstr.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

using ax::search::default_rules;
using ax::search::state;
using ax::search::successor_options;
using ax::search::successors;
using ax::search::verify_edge;
using ax::sym::expr;

const expr X = expr::symbol("x");

bool has_carrier(const expr& e) {
  if (e.is_fn() && (e.name() == "Integral" || e.name() == "Derivative" ||
                    e.name() == "Subs"))
    return true;
  for (const expr& a : e.args())
    if (has_carrier(a)) return true;
  return false;
}

int integral_count(const expr& e) {
  int n = e.is_fn() && e.name() == "Integral" ? 1 : 0;
  for (const expr& a : e.args()) n += integral_count(a);
  return n;
}

/** Independent numeric evaluation. Throws std::exception on domain
    trouble (log of negatives, unbound symbols); callers skip those
    sample points. base = lower quadrature limit for every Integral
    atom — antiderivative values differ from the symbolic ones by a
    constant per atom, which the mod-const comparison absorbs. */
double nval(const expr& e, std::map<std::string, double> env,
            double base) {
  if (!has_carrier(e)) return e.eval(env);
  switch (e.k()) {
    case ax::sym::kind::add: {
      double s = 0;
      for (const expr& a : e.args()) s += nval(a, env, base);
      return s;
    }
    case ax::sym::kind::mul: {
      double p = 1;
      for (const expr& a : e.args()) p *= nval(a, env, base);
      return p;
    }
    case ax::sym::kind::pow:
      return std::pow(nval(e.args()[0], env, base),
                      nval(e.args()[1], env, base));
    case ax::sym::kind::fn: {
      if (e.name() == "Integral") {
        if (e.args().size() != 2 || !e.args()[1].is_sym())
          throw std::runtime_error("multi-limit integral");
        const std::string v = e.args()[1].name();
        const expr f = e.args()[0];
        const double upper = env.count(v) ? env.at(v) : throw
          std::runtime_error("unbound integration variable");
        const auto r = ax::num::integrate(
            [&](double tt) {
              auto env2 = env;
              env2[v] = tt;
              return nval(f, env2, base);
            },
            base, upper, 1e-9, 1e-9, 30);
        return r.value;
      }
      if (e.name() == "Derivative") {
        if (e.args().size() != 2 || !e.args()[1].is_sym())
          throw std::runtime_error("multi-limit derivative");
        const std::string v = e.args()[1].name();
        const double at = env.at(v);
        const double h = 1e-5 * (std::abs(at) + 1.0);
        auto lo = env, hi = env;
        lo[v] = at - h;
        hi[v] = at + h;
        return (nval(e.args()[0], hi, base) -
                nval(e.args()[0], lo, base)) /
               (2 * h);
      }
      if (e.name() == "Subs") {
        auto env2 = env;
        env2[e.args()[1].name()] = nval(e.args()[2], env, base);
        return nval(e.args()[0], env2, base);
      }
      // scalar fn with carrier inside the argument
      const double a = nval(e.args()[0], env, base);
      if (e.name() == "sin") return std::sin(a);
      if (e.name() == "cos") return std::cos(a);
      if (e.name() == "tan") return std::tan(a);
      if (e.name() == "exp") return std::exp(a);
      if (e.name() == "log") return std::log(a);
      if (e.name() == "sqrt") return std::sqrt(a);
      throw std::runtime_error("unknown fn " + e.name());
    }
    default:
      return e.eval(env);
  }
}

/** Numeric verdict for parent ~ child, mod an additive constant when
    the pair carries Integral atoms. Returns +1 equal, -1 different,
    0 inconclusive (too few finite sample points). */
int numeric_verdict(const expr& parent, const expr& child) {
  static const double pts[] = {0.7, 1.3, 2.1, 0.45, 3.3};
  const bool mod_const =
      integral_count(parent) + integral_count(child) > 0;
  std::vector<double> diffs;
  for (const double p : pts) {
    try {
      const double a = nval(parent, {{"x", p}}, 0.31);
      const double b = nval(child, {{"x", p}}, 0.31);
      if (!std::isfinite(a) || !std::isfinite(b)) continue;
      if (std::abs(a) > 1e8 || std::abs(b) > 1e8) continue;
      diffs.push_back(a - b);
    } catch (const std::exception&) {
      continue;
    }
  }
  if (diffs.size() < 2) return 0;
  double lo = diffs[0], hi = diffs[0], amax = 0;
  for (const double d : diffs) {
    lo = std::min(lo, d);
    hi = std::max(hi, d);
    amax = std::max(amax, std::abs(d));
  }
  const double spread = hi - lo;
  const double scale = mod_const ? std::max(1.0, amax) : 1.0;
  if (mod_const) {
    if (spread < 1e-4 * scale) return +1;
    if (spread > 1e-2 * scale) return -1;
  } else {
    if (amax < 1e-5) return +1;
    if (amax > 1e-2) return -1;
  }
  return 0;  // gray zone: numerics not decisive enough to convict
}

/** Random integrand grammar: the L1-L4-ish family the farm feeds the
    oracle (polys, trig/exp/log atoms of linear arguments, products,
    sums). Deterministic under the caller's engine. */
expr gen_atom(std::mt19937& rng) {
  std::uniform_int_distribution<int> pick(0, 7);
  std::uniform_int_distribution<int> coef(-9, 9);
  std::uniform_int_distribution<int> deg(1, 4);
  switch (pick(rng)) {
    case 0: return expr::num(coef(rng) == 0 ? 3 : coef(rng));
    case 1: return X;
    case 2: return X.pow(expr::num(deg(rng)));
    case 3: return expr::fn("sin", X);
    case 4: return expr::fn("cos", X);
    case 5: return expr::fn("exp", X);
    case 6: return X.pow(expr::num(-1));
    default: return expr::fn("log", X);
  }
}

expr gen_integrand(std::mt19937& rng, int depth) {
  std::uniform_int_distribution<int> pick(0, 2);
  std::uniform_int_distribution<int> coef(-9, 9);
  if (depth <= 0) return gen_atom(rng);
  switch (pick(rng)) {
    case 0: {  // c1*a1 + c2*a2
      const int c1 = coef(rng), c2 = coef(rng);
      return expr::num(c1 == 0 ? 2 : c1) * gen_integrand(rng, depth - 1) +
             expr::num(c2 == 0 ? -5 : c2) * gen_integrand(rng, depth - 1);
    }
    case 1:  // product
      return gen_integrand(rng, depth - 1) * gen_atom(rng);
    default:  // scaled atom
      return expr::num(coef(rng) == 0 ? 7 : coef(rng)) * gen_atom(rng);
  }
}

struct fuzz_stats {
  int accepted_checked = 0;
  int accept_violations = 0;
  int corrupted_checked = 0;
  int reject_violations = 0;
  std::vector<std::string> failures;
};

void fuzz_roots(unsigned seed, int n_roots, fuzz_stats& st) {
  std::mt19937 rng(seed);
  const auto& rules = default_rules();
  successor_options sopt;
  sopt.use_macros = true;
  std::uniform_int_distribution<int> depth(0, 2);
  std::uniform_int_distribution<int> mut(0, 2);
  for (int i = 0; i < n_roots; ++i) {
    const expr root = expr::integral(gen_integrand(rng, depth(rng)), X);
    state s{root};
    for (const auto& [rule, child] : successors(s, rules, sopt)) {
      if (integral_count(child.e) > 3) continue;  // nested-quad wall
      // Property 1: accepted edges are numerically true
      const int nv = numeric_verdict(root, child.e);
      if (nv != 0) ++st.accepted_checked;
      if (nv < 0) {
        ++st.accept_violations;
        st.failures.push_back("ACCEPT-UNSOUND " + rule + " :: " +
                              ax::sym::to_sstr(root) + " -> " +
                              ax::sym::to_sstr(child.e));
        continue;
      }
      // Property 2: an x-dependent corruption that numerics convict
      // must be rejected by the oracle
      expr bad = child.e;
      switch (mut(rng)) {
        case 0: bad = bad + X * X; break;
        case 1: bad = bad * expr::num(ax::rational(3, 2)); break;
        default: bad = bad + expr::fn("sin", X); break;
      }
      if (bad.same(child.e)) continue;
      if (numeric_verdict(root, bad) < 0) {
        ++st.corrupted_checked;
        if (verify_edge(root, bad, rules.external)) {
          ++st.reject_violations;
          st.failures.push_back("REJECT-MISSED " + rule + " :: " +
                                ax::sym::to_sstr(root) + " -> " +
                                ax::sym::to_sstr(bad));
        }
      }
    }
  }
}

TEST(OracleFuzz, AcceptSoundAndCorruptionReject) {
  fuzz_stats st;
  for (const unsigned seed : {20260727u, 31415926u, 27182818u})
    fuzz_roots(seed, 40, st);
  std::string report;
  for (const auto& f : st.failures) report += f + "\n";
  EXPECT_EQ(st.accept_violations, 0) << report;
  EXPECT_EQ(st.reject_violations, 0) << report;
  // coverage floors: the test must not pass by skipping (deterministic
  // seeds make these counts stable; adjust deliberately if the grammar
  // or rule set changes)
  EXPECT_GE(st.accepted_checked, 150) << "accept coverage collapsed";
  EXPECT_GE(st.corrupted_checked, 80) << "corruption coverage collapsed";
}

TEST(OracleFuzz, HandwrittenTrapEdges) {
  // regression pins for known-dangerous shapes: each pair is WRONG and
  // must stay rejected (mod-const trap: the corruption differs by an
  // x-dependent term that vanishes at small sample grids)
  const auto& ext = default_rules().external;
  const auto p1 = ax::sym::parse("Integral(2*x, x)");
  EXPECT_FALSE(verify_edge(p1, ax::sym::parse("x**2 + x"), ext));
  EXPECT_FALSE(verify_edge(p1, ax::sym::parse("x**2 + sin(x)"), ext));
  EXPECT_FALSE(
      verify_edge(p1, ax::sym::parse("Integral(2*x + 1/1000000, x)"), ext));
  // and the true ones stay accepted (mod additive constant)
  EXPECT_TRUE(verify_edge(p1, ax::sym::parse("x**2"), ext));
  EXPECT_TRUE(verify_edge(p1, ax::sym::parse("x**2 + 7"), ext));
}

}  // namespace
