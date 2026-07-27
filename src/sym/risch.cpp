/** @file risch.cpp Magic boards r2 certificates (see risch.hpp).

    The math: for p non-constant polynomial and q rational, Liouville's
    theorem gives
      Integral(q * exp(p)) elementary  <=>  exists rational y with
                                            y' + p'*y = q
    and poles of y are constrained to poles of q (order bookkeeping on
    y' vs p'*y shows a pole of y outside q's poles would surface in q).
    With q polynomial, y is polynomial; with q Laurent (single pole at
    0) and p linear, y is Laurent. Both solves are triangular with a
    finite obstruction check, hence decisions, not heuristics.
    Trigonometric integrands go through exp(i*p) over Q(i). */
#include <ax/sym/risch.hpp>

#include <ax/sym/calc.hpp>
#include <ax/sym/expand.hpp>

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ax::sym {

namespace {

// ------------------------------------------------------------ smallint
std::optional<long long> as_small_int(const rational& q) {
  if (!(q.den() == ax::bigint(1))) return std::nullopt;
  const std::string s = q.num().to_string();
  if (s.size() > 12) return std::nullopt;
  return std::stoll(s);
}

// ------------------------------------------------- complex rationals
struct qi {
  rational re, im;
  bool is_zero() const { return re.is_zero() && im.is_zero(); }
};
qi operator+(const qi& a, const qi& b) { return {a.re + b.re, a.im + b.im}; }
qi operator-(const qi& a, const qi& b) { return {a.re - b.re, a.im - b.im}; }
qi operator*(const qi& a, const qi& b) {
  return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
qi operator*(const rational& a, const qi& b) { return {a * b.re, a * b.im}; }
qi operator/(const qi& a, const qi& b) {
  const rational n = b.re * b.re + b.im * b.im;  // != 0 for b != 0
  return {(a.re * b.re + a.im * b.im) / n, (a.im * b.re - a.re * b.im) / n};
}

using laurent = std::map<long long, qi>;  // degree -> coefficient

void put(laurent& L, long long k, const qi& c) {
  if (c.is_zero()) return;
  auto [it, fresh] = L.try_emplace(k, c);
  if (!fresh) {
    it->second = it->second + c;
    if (it->second.is_zero()) L.erase(it);
  }
}

/** Laurent coefficients of an EXPANDED expression in x; nullopt when
    any term is not c * x^k (other symbols, functions, non-integer
    exponents). */
std::optional<laurent> laurent_of(const expr& e, const expr& x) {
  laurent L;
  const auto term = [&](const expr& t) -> bool {
    rational c{ax::bigint(1)};
    long long k = 0;
    const auto factor = [&](const expr& f) -> bool {
      if (f.is_num()) {
        c = c * f.value();
        return true;
      }
      if (f.same(x)) {
        k += 1;
        return true;
      }
      if (f.is_pow() && f.args()[0].same(x) && f.args()[1].is_num()) {
        const auto n = as_small_int(f.args()[1].value());
        if (!n) return false;
        k += *n;
        return true;
      }
      return false;
    };
    if (t.is_mul()) {
      for (const expr& f : t.args())
        if (!factor(f)) return false;
    } else if (!factor(t)) {
      return false;
    }
    put(L, k, qi{c, rational{}});
    return true;
  };
  if (e.is_add()) {
    for (const expr& t : e.args())
      if (!term(t)) return std::nullopt;
  } else if (!term(e)) {
    return std::nullopt;
  }
  return L;
}

long long deg(const laurent& L) { return L.empty() ? 0 : L.rbegin()->first; }
long long ord(const laurent& L) { return L.empty() ? 0 : L.begin()->first; }
qi at(const laurent& L, long long k) {
  const auto it = L.find(k);
  return it == L.end() ? qi{} : it->second;
}

/** Does y' + P*y = Q admit a rational solution?  P polynomial (deg
    d1 >= 1 requires Q polynomial; d1 == 0 i.e. P constant allows a
    single pole at 0), Q Laurent. Triangular solve + obstruction
    check: a 'false' is a proof. */
bool ode_solvable(const laurent& P, const laurent& Q) {
  if (Q.empty()) return true;  // y = 0
  const long long d1 = deg(P);
  const qi Plead = at(P, d1);

  if (d1 >= 1) {
    if (ord(Q) < 0) return true;  // outside the certified family: treat
                                  // as solvable so caller stays silent
    laurent y;
    // unknowns y_k, k = degQ-d1 .. 0, resolved from the top equation
    // t = k + d1 (P's leading term dominates y' when d1 >= 1)
    for (long long k = deg(Q) - d1; k >= 0; --k) {
      const long long t = k + d1;
      qi rhs = at(Q, t) - (rational(ax::bigint(t + 1)) * at(y, t + 1));
      for (long long j = 0; j < d1; ++j)
        rhs = rhs - at(P, j) * at(y, t - j);
      put(y, k, rhs / Plead);
    }
    // residual equations t = d1-1 .. 0 are the obstruction
    for (long long t = d1 - 1; t >= 0; --t) {
      qi lhs = rational(ax::bigint(t + 1)) * at(y, t + 1);
      for (long long j = 0; j <= d1; ++j)
        lhs = lhs + at(P, j) * at(y, t - j);
      if (!(lhs - at(Q, t)).is_zero()) return false;
    }
    return true;
  }

  // d1 == 0: P = c (nonzero), Q Laurent with pole only at 0
  const qi c = Plead;
  if (c.is_zero()) return true;  // pure integration: Laurent + log terms
  laurent y;
  const long long mneg = ord(Q);
  if (mneg < 0) {
    if (mneg == -1) {
      // y would need a residue term: equation at x^-1 is
      // Q_{-1} = 0*y_0 + c*y_{-1} with y_{-1} forced 0 by pole order
      if (!at(Q, -1).is_zero()) return false;  // the Ei obstruction
    } else {
      // pole order M = -mneg - 1; bottom equation fixes y_{mneg+1}
      put(y, mneg + 1, at(Q, mneg) / qi{rational(ax::bigint(mneg + 1)),
                                        rational{}});
      for (long long t = mneg + 1; t <= -2; ++t)
        put(y, t + 1,
            (at(Q, t) - c * at(y, t)) /
                qi{rational(ax::bigint(t + 1)), rational{}});
      // obstruction at x^-1: Q_{-1} = 0*y_0 + c*y_{-1}
      if (!(at(Q, -1) - c * at(y, -1)).is_zero()) return false;
    }
  }
  // polynomial side always solves top-down: y_t = (Q_t-(t+1)y_{t+1})/c
  return true;
}

// ----------------------------------------------------- shape detection
struct shape {
  expr core;      // the transcendental atom (exp/sin/cos/log node)
  expr arg;       // its argument (the monomial key for wave grouping)
  expr var;       // integration variable
  laurent P;      // ODE coefficient p' (or i*p', or n+1 for log family)
  laurent Q;      // right-hand side COMPONENT in the e^{i*arg} (resp.
                  // e^{arg}) monomial: q_exp / q_cos / -i*q_sin — the
                  // relative i matters when same-argument atoms sum
  enum family { exp_k, trig_k, log_k };
  family kind = exp_k;
  std::string tag;
  shape(expr c, expr a, expr v)
      : core(std::move(c)), arg(std::move(a)), var(std::move(v)) {}
};

std::optional<std::pair<expr, expr>> unpack_i(const expr& node) {
  if (!node.is_fn() || node.name() != "Integral" ||
      node.args().size() != 2 || !node.args()[1].is_sym())
    return std::nullopt;
  return std::make_pair(node.args()[0], node.args()[1]);
}

bool contains(const expr& e, const expr& target) {
  if (e.same(target)) return true;
  for (const expr& a : e.args())
    if (contains(a, target)) return true;
  return false;
}

int count_occ(const expr& e, const expr& target) {
  if (e.same(target)) return 1;
  int n = 0;
  for (const expr& a : e.args()) n += count_occ(a, target);
  return n;
}

/** Classify f = q * T(p) with exactly one transcendental factor T. */
std::optional<shape> detect(const expr& f, const expr& x) {
  std::vector<expr> factors;
  if (f.is_mul())
    for (const expr& g : f.args()) factors.push_back(g);
  else
    factors.push_back(f);

  std::optional<expr> trans;      // exp/sin/cos atom
  std::optional<expr> logpow;     // log(x)^m factor (pow or bare)
  expr q = expr::num(1);
  for (const expr& g : factors) {
    const bool is_trans = g.is_fn() && g.args().size() == 1 &&
                          (g.name() == "exp" || g.name() == "sin" ||
                           g.name() == "cos") &&
                          contains(g.args()[0], x);
    const bool is_logp =
        (g.is_fn() && g.name() == "log" && g.args().size() == 1 &&
         g.args()[0].same(x)) ||
        (g.is_pow() && g.args()[0].is_fn() &&
         g.args()[0].name() == "log" &&
         g.args()[0].args()[0].same(x) && g.args()[1].is_num());
    if (is_trans) {
      if (trans || logpow) return std::nullopt;  // one transcendental only
      trans = g;
    } else if (is_logp) {
      if (trans || logpow) return std::nullopt;
      logpow = g;
    } else {
      q = q * g;
    }
  }

  if (trans) {
    const expr& p = trans->args()[0];
    const auto pl = laurent_of(expand(p), x);
    if (!pl || pl->empty() || ord(*pl) < 0 || deg(*pl) < 1)
      return std::nullopt;
    const auto ql = laurent_of(expand(q), x);
    if (!ql) return std::nullopt;
    if (!ql->empty() && ord(*ql) < 0 && deg(*pl) >= 2)
      return std::nullopt;  // Laurent q only certified for linear p
    if (!ql->empty() && ord(*ql) < 0 && ord(*ql) < -64) return std::nullopt;
    shape s(*trans, p, x);
    // P = p' (exp) or i*p' (trig)
    const bool is_exp = trans->name() == "exp";
    s.kind = is_exp ? shape::exp_k : shape::trig_k;
    for (const auto& [k, c] : *pl) {
      if (k == 0) continue;
      const rational kk = rational(ax::bigint(k)) * c.re;
      put(s.P, k - 1, is_exp ? qi{kk, rational{}} : qi{rational{}, kk});
    }
    if (s.P.empty()) return std::nullopt;  // constant p
    // component in the e^{i*arg} monomial: cos -> q, sin -> -i*q
    const qi comp = trans->name() == "sin"
                        ? qi{rational{}, -rational(ax::bigint(1))}
                        : qi{rational(ax::bigint(1)), rational{}};
    for (const auto& [k, c] : *ql) put(s.Q, k, comp * c);
    const bool lin = deg(*pl) == 1;
    s.tag = std::string("risch-") + (is_exp ? "exp" : "trig") +
            (lin ? "-laurent" : "-poly");
    return s;
  }

  if (logpow) {
    // c0 * x^n * log(x)^m  --(u = log x)-->  c0 * e^{(n+1)u} * u^m
    long long m = 1;
    if (logpow->is_pow()) {
      const auto mm = as_small_int(logpow->args()[1].value());
      if (!mm) return std::nullopt;
      m = *mm;
    }
    const auto ql = laurent_of(expand(q), x);
    if (!ql || ql->size() > 1) return std::nullopt;  // q must be c0*x^n
    const long long n = ql->empty() ? 0 : ql->begin()->first;
    const qi c0 = ql->empty() ? qi{rational(ax::bigint(1)), rational{}}
                              : ql->begin()->second;
    if (n + 1 == 0) return std::nullopt;  // pure du integral: elementary
    shape s(logpow->is_pow() ? logpow->args()[0] : *logpow,
            logpow->is_pow() ? logpow->args()[0] : *logpow, x);
    s.kind = shape::log_k;
    put(s.P, 0, qi{rational(ax::bigint(n + 1)), rational{}});
    put(s.Q, m, c0);
    s.tag = "risch-log";
    return s;
  }

  return std::nullopt;
}

}  // namespace

board_cert risch_certify(const expr& integral_node) {
  const auto u = unpack_i(integral_node);
  if (!u) return {};
  const auto s = detect(u->first, u->second);
  if (!s) return {};
  if (ode_solvable(s->P, s->Q)) return {};  // provably elementary
  return {true, s->tag};
}

board_cert dead_state(const expr& state) {
  // Linear decomposition: state = sum of c_j * Integral(...) terms +
  // context. Same-argument atoms share the e^{i*arg} monomial (sin and
  // cos are NOT independent — measured on stuck_states_p1:
  // Integral(-8*(2x+1)**2*sin(u)) + Integral(16*cos(u)) is elementary
  // as a SUM while both atoms certify individually), so certificates
  // combine per argument group: one ODE with the coefficient-weighted
  // component sum. Non-linear contexts and constant-shifted argument
  // pairs make the group undecided, never masked.
  struct occurrence {
    expr atom;
    shape sh;
    rational coeff;
  };
  std::vector<expr> terms;
  if (state.is_add())
    for (const expr& t : state.args()) terms.push_back(t);
  else
    terms.push_back(state);

  std::vector<occurrence> occs;
  for (const expr& term : terms) {
    rational c{ax::bigint(1)};
    std::optional<expr> atom;
    bool clean = true;
    if (term.is_fn() && term.name() == "Integral") {
      atom = term;
    } else if (term.is_mul()) {
      for (const expr& f : term.args()) {
        if (f.is_num()) c = c * f.value();
        else if (f.is_fn() && f.name() == "Integral" && !atom) atom = f;
        else clean = false;  // non-numeric cofactor: not a linear term
      }
    }
    if (!atom || !clean) continue;
    const auto u = unpack_i(*atom);
    if (!u) continue;
    auto sh = detect(u->first, u->second);
    if (!sh) continue;
    occs.push_back({*atom, std::move(*sh), c});
  }

  for (std::size_t i = 0; i < occs.size(); ++i) {
    const auto& base = occs[i];
    // group same (family, argument, variable) occurrences
    const auto in_group = [&](const occurrence& oc) -> int {
      // 1 = same monomial group, 0 = independent, -1 = taints the group
      if (oc.sh.kind != base.sh.kind) return 0;
      if (!oc.sh.var.same(base.sh.var)) return 0;
      if (base.sh.kind == shape::log_k) {
        // same tower monomial e^{(n+1)u} iff P matches
        const qi d = at(oc.sh.P, 0) - at(base.sh.P, 0);
        return d.is_zero() ? 1 : 0;
      }
      const expr d = oc.sh.arg - base.sh.arg;
      if (!d.is_num()) return 0;  // different monomial: independent
      if (d.value().is_zero()) return 1;
      return -1;  // constant-shifted twin: e^{ic} phase not rational
    };

    laurent Qsum;
    int group_core_occ = 0;
    bool tainted = false;
    bool unique = true;
    for (const auto& oc : occs) {
      const int g = in_group(oc);
      if (g == 0) continue;
      if (g < 0) {
        tainted = true;
        break;
      }
      for (const auto& [k, q] : oc.sh.Q) put(Qsum, k, oc.coeff * q);
      group_core_occ +=
          count_occ(oc.atom, oc.sh.core) +
          (base.sh.kind == shape::trig_k
               ? count_occ(oc.atom,
                           expr::fn(oc.sh.core.name() == "sin" ? "cos"
                                                               : "sin",
                                    oc.sh.arg))
               : 0);
      if (count_occ(state, oc.atom) != 1) unique = false;
    }
    if (tainted || !unique) continue;
    // the certified monomial must not appear anywhere outside the
    // group's atoms (else a value-preserving move could recombine)
    int state_core_occ = 0;
    if (base.sh.kind == shape::exp_k)
      state_core_occ = count_occ(state, expr::fn("exp", base.sh.arg));
    else if (base.sh.kind == shape::trig_k)
      state_core_occ = count_occ(state, expr::fn("sin", base.sh.arg)) +
                       count_occ(state, expr::fn("cos", base.sh.arg));
    else
      state_core_occ = count_occ(state, base.sh.core);
    if (state_core_occ != group_core_occ) continue;
    if (!ode_solvable(base.sh.P, Qsum)) return {true, base.sh.tag};
  }
  return {};
}

std::vector<bool> dead_state_mask(const std::vector<expr>& states) {
  std::vector<bool> mask;
  mask.reserve(states.size());
  for (const expr& s : states) mask.push_back(dead_state(s).dead);
  return mask;
}

}  // namespace ax::sym
