#include <ax/search/search.hpp>

#include <ax/sym/calc.hpp>
#include <ax/sym/count_ops.hpp>
#include <ax/sym/expand.hpp>
#include <ax/sym/integrate.hpp>
#include <ax/sym/oracle.hpp>
#include <ax/sym/poly.hpp>

#include <algorithm>
#include <map>
#include <optional>
#include <vector>

/** Rules tranche 3 (Phase C task 5): the autopsy rules — i_cyclic,
    i_unprod, i_ansatz_exp, i_linear_basis. The euler rewrite is NOT
    ported (it needs complex-valued atoms; real-valued expr cannot carry
    exp(I*x) — external-slot territory, documented in the plan). */

namespace ax::search {

// shared with rules2.cpp
namespace tranche2 {
bool contains(const sym::expr& e, const sym::expr& target);
bool is_rational_in(const sym::expr& e, const sym::expr& x);
bool solve_rational_linear(std::vector<std::vector<ax::rational>> m,
                           std::vector<ax::rational> rhs,
                           std::vector<ax::rational>& out);
}  // namespace tranche2

namespace tranche3 {

using sym::expr;
using sym::poly;
using tranche2::contains;
using tranche2::is_rational_in;
using tranche2::solve_rational_linear;

const ax::rational kZero{};
const ax::rational kOneR{ax::bigint(1)};

std::optional<std::pair<expr, expr>> unpack_i(const expr& node) {
  if (!node.is_fn() || node.name() != "Integral" ||
      node.args().size() != 2 || !node.args()[1].is_sym())
    return std::nullopt;
  return std::make_pair(node.args()[0], node.args()[1]);
}

/** Slope of e if linear in x (slope x-free), else nullopt. */
std::optional<expr> linear_coeff(const expr& e, const expr& x) {
  const expr d = sym::diff(e, x);
  if (contains(d, x)) return std::nullopt;
  return d;
}

std::vector<expr> factors_of(const expr& e) {
  std::vector<expr> out;
  if (e.is_mul())
    for (const expr& f : e.args()) out.push_back(f);
  else
    out.push_back(e);
  return out;
}

std::vector<expr> terms_of(const expr& e) {
  std::vector<expr> out;
  if (e.is_add())
    for (const expr& t : e.args()) out.push_back(t);
  else
    out.push_back(e);
  return out;
}

bool is_zero_num(const expr& e) {
  return e.is_num() && e.value() == kZero;
}

/** Collect fn atoms with one of the given names (whole nodes, deduped,
    deterministic tree order). */
void collect_fns(const expr& e, std::initializer_list<const char*> names,
                 std::vector<expr>& out) {
  if (e.is_fn())
    for (const char* n : names)
      if (e.name() == n) {
        bool dup = false;
        for (const expr& u : out) dup = dup || u.same(e);
        if (!dup) out.push_back(e);
      }
  for (const expr& a : e.args()) collect_fns(a, names, out);
}

expr replace_atom(const expr& e, const expr& target, const expr& repl) {
  if (e.same(target)) return repl;
  switch (e.k()) {
    case sym::kind::num:
    case sym::kind::sym:
      return e;
    case sym::kind::fn: {
      std::vector<expr> mapped;
      for (const expr& a : e.args())
        mapped.push_back(replace_atom(a, target, repl));
      return expr::fn(e.name(), std::move(mapped));
    }
    case sym::kind::add: {
      expr out = expr::num(0);
      for (const expr& t : e.args()) out = out + replace_atom(t, target, repl);
      return out;
    }
    case sym::kind::mul: {
      expr out = expr::num(1);
      for (const expr& f : e.args()) out = out * replace_atom(f, target, repl);
      return out;
    }
    case sym::kind::pow:
      return replace_atom(e.args()[0], target, repl)
          .pow(replace_atom(e.args()[1], target, repl));
  }
  return e;
}

// ------------------------------------------------------------ i_cyclic

std::vector<expr> i_cyclic(const expr& node) {
  const auto un = unpack_i(node);
  if (!un) return {};
  const auto& [f, x] = *un;
  if (!f.is_mul()) return {};
  std::vector<expr> ex, tr;
  ax::rational coeff = kOneR;
  for (const expr& a : f.args()) {
    if (a.is_num()) coeff = coeff * a.value();
    else if (a.is_fn() && a.name() == "exp") ex.push_back(a);
    else if (a.is_fn() && (a.name() == "sin" || a.name() == "cos"))
      tr.push_back(a);
    else return {};
  }
  if (ex.size() != 1 || tr.size() != 1) return {};
  const auto a = linear_coeff(ex[0].args()[0], x);
  const auto b = linear_coeff(tr[0].args()[0], x);
  if (!a || !b || is_zero_num(*a) || is_zero_num(*b)) return {};
  const expr& v = tr[0].args()[0];
  const expr num = tr[0].name() == "sin"
                       ? *a * expr::fn("sin", v) - *b * expr::fn("cos", v)
                       : *a * expr::fn("cos", v) + *b * expr::fn("sin", v);
  return {expr::num(coeff) * ex[0] * num /
          (a->pow(expr::num(2)) + b->pow(expr::num(2)))};
}

// ------------------------------------------------------------ i_unprod

std::vector<expr> i_unprod(const expr& node) {
  const auto un = unpack_i(node);
  if (!un) return {};
  const auto& [f, x] = *un;
  if (!f.is_add()) return {};
  if (sym::count_ops(f) > 200) return {};  // coarse gate; work budget is the guarantee
  std::vector<expr> out;
  std::vector<expr> seen;
  const auto emit = [&](const expr& A) -> bool {
    for (const expr& s : seen)
      if (s.same(A)) return false;
    seen.push_back(A);
    const expr resid = sym::expand(f - sym::diff(A, x));
    if (sym::count_ops(resid) >= sym::count_ops(f)) return false;
    out.push_back(is_zero_num(resid) ? A : A + expr::integral(resid, x));
    return out.size() >= 6;
  };
  const auto table = [](const std::string& n,
                        const expr& v) -> std::optional<expr> {
    if (n == "sin") return -expr::fn("cos", v);
    if (n == "cos") return expr::fn("sin", v);
    if (n == "exp") return expr::fn("exp", v);
    return std::nullopt;
  };
  const auto bad_cof = [&x](const expr& c) {
    std::vector<expr> fns;
    collect_fns(c, {"sin", "cos", "exp", "log", "Integral"}, fns);
    return !fns.empty();
  };
  for (const expr& t : terms_of(f)) {
    std::vector<expr> fns;
    collect_fns(t, {"sin", "cos", "exp"}, fns);
    // guess family 1: t = cof * u' * h(u) -> A = cof * H(u)
    for (const expr& fn : fns) {
      const expr& v = fn.args()[0];
      const expr dv = sym::diff(v, x);
      if (is_zero_num(dv)) continue;
      const expr cof = sym::canonical(t / (dv * fn), x);
      if (bad_cof(cof)) continue;
      const auto H = table(fn.name(), v);
      if (!H) continue;
      if (emit(cof * *H)) return out;
    }
    // guess family 2: t = cof * h(u) is the f'*H half -> A = (∫cof)*h(u)
    for (const expr& fn : fns) {
      const expr cof = sym::canonical(t / fn, x);
      if (bad_cof(cof) || !is_rational_in(cof, x)) continue;
      // skip plain polynomials (family 1 territory)
      bool is_poly = true;
      try {
        (void)poly::from_expr(sym::expand(cof), x);
      } catch (const std::exception&) {
        is_poly = false;
      }
      if (is_poly) continue;
      const auto F = sym::integrate(cof, x);  // ax's own Phase-7 integrator
      if (!F) continue;
      std::vector<expr> logs;
      collect_fns(*F, {"log"}, logs);
      if (logs.empty()) continue;
      if (emit(*F * fn)) return out;
    }
  }
  return out;
}

// -------------------------------------------------------- i_ansatz_exp

std::vector<expr> i_ansatz_exp(const expr& node) {
  const auto un = unpack_i(node);
  if (!un) return {};
  const auto& [f0, x] = *un;
  if (sym::count_ops(f0) > 200) return {};  // coarse gate; work budget is the guarantee
  const expr f = sym::expand(f0);
  // group terms by their exp-product signature
  struct group {
    expr sig;
    std::vector<expr> terms;
  };
  std::vector<group> groups;
  std::vector<expr> rest;
  for (const expr& t : terms_of(f)) {
    std::vector<expr> exps;
    collect_fns(t, {"exp"}, exps);
    std::vector<expr> pexps;
    for (const expr& e : exps) {
      try {
        (void)poly::from_expr(sym::expand(e.args()[0]), x);
        pexps.push_back(e);
      } catch (const std::exception&) {
      }
    }
    if (pexps.empty()) {
      rest.push_back(t);
      continue;
    }
    expr sig = expr::num(1);
    for (const expr& e : pexps) sig = sig * e;
    bool found = false;
    for (auto& g : groups)
      if (g.sig.same(sig)) {
        g.terms.push_back(t);
        found = true;
      }
    if (!found) groups.push_back({sig, {t}});
  }
  std::vector<expr> solved;
  for (const auto& g : groups) {
    expr sum = expr::num(0);
    for (const expr& t : g.terms) sum = sum + t;
    // w = combined exponent; P = g / exp-product
    std::vector<expr> exps;
    collect_fns(g.sig, {"exp"}, exps);
    expr w = expr::num(0);
    for (const expr& e : exps) w = w + e.args()[0];
    poly wp, Pp;
    bool ok = true;
    try {
      wp = poly::from_expr(sym::expand(w), x);
      Pp = poly::from_expr(sym::expand(sym::canonical(sum / g.sig, x)), x);
    } catch (const std::exception&) {
      ok = false;
    }
    if (!ok || wp.degree() < 2) {
      for (const expr& t : g.terms) rest.push_back(t);
      continue;
    }
    // solve Q' + Q*w' = P for polynomial Q
    const poly dw = wp.derivative();
    const int degQ = std::max(Pp.degree() - dw.degree(), 0);
    const std::size_t nrows =
        static_cast<std::size_t>(std::max(Pp.degree(),
                                          degQ + dw.degree()) + 2);
    std::vector<std::vector<ax::rational>> m(
        nrows,
        std::vector<ax::rational>(static_cast<std::size_t>(degQ) + 1, kZero));
    for (int j = 0; j <= degQ; ++j) {
      std::vector<ax::rational> cj(static_cast<std::size_t>(j) + 1, kZero);
      cj[static_cast<std::size_t>(j)] = kOneR;
      poly xj(std::move(cj));
      poly col = xj * dw;  // Q*w' part
      if (j > 0) {
        std::vector<ax::rational> cd(static_cast<std::size_t>(j), kZero);
        cd[static_cast<std::size_t>(j - 1)] = ax::rational(ax::bigint(j));
        col = col + poly(std::move(cd));  // Q' part
      }
      for (int i = 0; i <= col.degree() &&
                      static_cast<std::size_t>(i) < nrows; ++i)
        m[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
            col.coeff(static_cast<std::size_t>(i));
    }
    std::vector<ax::rational> rhs(nrows, kZero);
    for (int i = 0; i <= Pp.degree() &&
                    static_cast<std::size_t>(i) < nrows; ++i)
      rhs[static_cast<std::size_t>(i)] = Pp.coeff(static_cast<std::size_t>(i));
    std::vector<ax::rational> sol;
    if (!solve_rational_linear(std::move(m), rhs, sol)) {
      for (const expr& t : g.terms) rest.push_back(t);
      continue;
    }
    expr Q = expr::num(0);
    for (std::size_t j = 0; j < sol.size(); ++j)
      if (!(sol[j] == kZero))
        Q = Q + expr::num(sol[j]) * x.pow(expr::num(static_cast<long long>(j)));
    solved.push_back(Q * g.sig);
  }
  if (solved.empty()) return {};
  expr out = expr::num(0);
  for (const expr& s : solved) out = out + s;
  expr tail = expr::num(0);
  for (const expr& t : rest) tail = tail + t;
  if (!is_zero_num(tail)) out = out + expr::integral(tail, x);
  return {out};
}

// ------------------------------------------------------ i_linear_basis

std::vector<expr> i_linear_basis(const expr& node) {
  const auto un = unpack_i(node);
  if (!un) return {};
  const auto& [f0, x] = *un;
  // coarse size gate only: the cooperative work budget is the real
  // guarantee now (llmopt attribution: the old 60-op gate was clipping
  // linear_basis solves on exactly the L7/L8 states it serves)
  if (sym::count_ops(f0) > 200) return {};
  expr f = sym::expand(f0);
  // Laurent tail: c*x^-n split off analytically
  expr laurent = expr::num(0);
  {
    std::vector<expr> kept;
    for (const expr& t : terms_of(f)) {
      ax::rational c = kOneR;
      expr rest_t = t;
      if (t.is_mul() && t.args()[0].is_num()) {
        c = t.args()[0].value();
        rest_t = expr::num(1);
        for (std::size_t i = 1; i < t.args().size(); ++i)
          rest_t = rest_t * t.args()[i];
      }
      const bool neg_xpow =
          rest_t.is_pow() && rest_t.args()[0].same(x) &&
          rest_t.args()[1].is_num() &&
          rest_t.args()[1].value() < kZero &&
          rest_t.args()[1].value().den() == ax::bigint(1);
      if (neg_xpow) {
        const ax::rational n = -rest_t.args()[1].value();
        if (n == kOneR) {
          laurent = laurent + expr::num(c) * expr::fn("log", x);
        } else {
          const ax::rational e1 = kOneR - n;
          laurent = laurent + expr::num(c / e1) * x.pow(expr::num(e1));
        }
      } else {
        kept.push_back(t);
      }
    }
    f = expr::num(0);
    for (const expr& t : kept) f = f + t;
  }
  if (is_zero_num(f))
    return is_zero_num(laurent) ? std::vector<expr>{}
                                : std::vector<expr>{laurent};

  const auto is_poly_in_x = [&x](const expr& e) {
    try {
      (void)poly::from_expr(sym::expand(e), x);
      return true;
    } catch (const std::exception&) {
      return false;
    }
  };
  // trig args: polynomial or (const +) log(poly)
  std::vector<expr> trig_fns;
  collect_fns(f, {"sin", "cos"}, trig_fns);
  std::vector<expr> args;
  std::vector<expr> trig_log_inners;
  for (const expr& fn : trig_fns) {
    const expr& v = fn.args()[0];
    if (!contains(v, x)) continue;
    bool dup = false;
    for (const expr& a : args) dup = dup || a.same(v);
    if (dup) continue;
    bool admissible = is_poly_in_x(v);
    if (!admissible) {
      // (const +) log(poly): find the x-dependent part
      std::vector<expr> logs_in;
      collect_fns(v, {"log"}, logs_in);
      if (logs_in.size() == 1 && contains(logs_in[0], x) &&
          is_poly_in_x(logs_in[0].args()[0])) {
        const expr vr = v - logs_in[0];
        if (!contains(vr, x)) {
          admissible = true;
          trig_log_inners.push_back(logs_in[0].args()[0]);
        }
      }
    }
    if (admissible) args.push_back(v);
  }
  std::vector<expr> trig;
  for (const expr& v : args) {
    trig.push_back(expr::fn("sin", v));
    trig.push_back(expr::fn("cos", v));
  }
  std::vector<expr> all_exps;
  collect_fns(f, {"exp"}, all_exps);
  std::vector<expr> exps;
  for (const expr& e : all_exps)
    if (contains(e.args()[0], x) && is_poly_in_x(e.args()[0]))
      exps.push_back(e);
  std::vector<expr> all_logs;
  collect_fns(f, {"log"}, all_logs);
  std::vector<expr> logs;
  for (const expr& g : all_logs)
    if (contains(g.args()[0], x) && is_poly_in_x(g.args()[0]))
      logs.push_back(g);
  std::vector<expr> all_atans;
  collect_fns(f, {"atan"}, all_atans);
  std::vector<expr> atans;
  for (const expr& g : all_atans)
    if (contains(g.args()[0], x) && is_poly_in_x(g.args()[0]))
      atans.push_back(g);

  std::vector<expr> gens;
  for (const auto& v : trig) gens.push_back(v);
  for (const auto& v : exps) gens.push_back(v);
  for (const auto& v : logs) gens.push_back(v);
  for (const auto& v : atans) gens.push_back(v);
  if (gens.empty() || gens.size() > 8) return {};

  // basis monomials
  std::optional<expr> ep;
  if (!exps.empty()) {
    expr p = expr::num(1);
    for (const expr& e : exps) p = p * e;
    ep = p;
  }
  std::vector<expr> mons;
  for (const expr& v : args) {
    const expr s_ = expr::fn("sin", v);
    const expr c_ = expr::fn("cos", v);
    int mv = 1;
    for (const expr& t : terms_of(f)) {
      int tot = 0;
      for (const expr& b : {s_, c_}) {
        int e = contains(t, b) ? 1 : 0;
        // find explicit powers b^k
        const std::function<void(const expr&)> scan = [&](const expr& q) {
          if (q.is_pow() && q.args()[0].same(b) && q.args()[1].is_num() &&
              q.args()[1].value().den() == ax::bigint(1) &&
              kZero < q.args()[1].value()) {
            const std::string ks = q.args()[1].value().num().to_string();
            if (ks.size() <= 1) e = std::max(e, std::stoi(ks));
          }
          for (const expr& a2 : q.args()) scan(a2);
        };
        scan(t);
        tot += e;
      }
      mv = std::max(mv, tot);
    }
    mv = std::min(mv + 1, 5);
    for (int a2 = 0; a2 <= mv; ++a2)
      for (int b2 = 0; b2 <= mv - a2; ++b2)
        if (a2 + b2 >= 1)
          mons.push_back(s_.pow(expr::num(a2)) * c_.pow(expr::num(b2)));
  }
  if (ep) {
    mons.push_back(*ep);
    for (const expr& t : trig) mons.push_back(*ep * t);
  }
  for (const expr& L : logs) {
    mons.push_back(L);
    for (const expr& t : trig) mons.push_back(L * t);
    if (ep) mons.push_back(L * *ep);
  }
  for (const expr& A : atans) {
    mons.push_back(A);
    for (const expr& L : logs) mons.push_back(A * L);
  }
  mons.push_back(expr::num(1));

  // clear multiplier for log/atan/trig-log denominators
  expr clear = expr::num(1);
  if (!logs.empty() || !atans.empty() || !trig_log_inners.empty()) {
    clear = clear * x;
    std::vector<expr> log_args;
    for (const expr& g : logs) {
      clear = clear * g.args()[0];
      log_args.push_back(g.args()[0]);
    }
    for (const expr& g : atans)
      clear = clear * (expr::num(1) + g.args()[0].pow(expr::num(2)));
    for (const expr& p : trig_log_inners) {
      bool dup = false;
      for (const expr& la : log_args) dup = dup || la.same(p);
      if (!dup) clear = clear * p;
    }
  }
  // degree bound from the cleared integrand (placeholder-substituted)
  std::vector<expr> ph;
  for (std::size_t i = 0; i < gens.size(); ++i)
    ph.push_back(expr::symbol("g" + std::to_string(i) + "_"));
  const auto placehold = [&](expr e) {
    for (std::size_t i = 0; i < gens.size(); ++i)
      e = replace_atom(e, gens[i], ph[i]);
    return e;
  };
  // size the ansatz: x-degree of the cleared integrand
  int deg = 1;
  {
    const expr pf = sym::expand(placehold(
        sym::canonical(f * clear, x)));
    const std::function<void(const expr&, int)> xdeg = [&](const expr& q,
                                                           int acc) {
      if (q.same(x)) deg = std::max(deg, acc + 1);
      if (q.is_pow() && q.args()[0].same(x) && q.args()[1].is_num() &&
          q.args()[1].value().den() == ax::bigint(1) &&
          kZero < q.args()[1].value()) {
        const std::string ks = q.args()[1].value().num().to_string();
        if (ks.size() <= 1) deg = std::max(deg, acc + std::stoi(ks));
      }
      for (const expr& a2 : q.args()) xdeg(a2, acc);
    };
    xdeg(pf, 0);
  }
  const std::size_t nc = static_cast<std::size_t>(deg + 2) * mons.size();
  if (deg > 8 || nc > 200) return {};

  // Column-wise system construction (the wedge lesson, final face):
  // differentiate each basis element separately — hundreds of SMALL
  // canonicals instead of one giant candidate expression whose expand
  // is unbounded. No unknown symbols ever enter an expr.
  struct mono_row {
    expr mono;
    std::size_t idx;
  };
  std::vector<mono_row> rows_ix;
  std::vector<std::vector<ax::rational>> M;
  std::vector<ax::rational> rhs;
  const auto row_of = [&](const expr& mono) {
    for (const auto& r : rows_ix)
      if (r.mono.same(mono)) return r.idx;
    rows_ix.push_back({mono, rows_ix.size()});
    M.emplace_back(nc, kZero);
    rhs.push_back(kZero);
    return rows_ix.back().idx;
  };
  // decompose an expanded, placeholder-substituted expr into
  // (monomial -> rational) contributions
  const auto add_terms = [&](const expr& e, std::size_t col,
                             bool to_rhs) -> bool {
    for (const expr& t : terms_of(e)) {
      if (is_zero_num(t)) continue;
      ax::rational coeff = kOneR;
      expr mono = expr::num(1);
      for (const expr& fac : factors_of(t)) {
        if (fac.is_num())
          coeff = coeff * fac.value();
        else
          mono = mono * fac;
      }
      const std::size_t r = row_of(mono);
      if (to_rhs)
        rhs[r] = rhs[r] + coeff;
      else
        M[r][col] = M[r][col] + coeff;
    }
    return true;
  };
  bool bad = false;
  for (std::size_t i = 0; i < mons.size() && !bad; ++i)
    for (int j = 0; j <= deg + 1 && !bad; ++j) {
      const std::size_t col =
          i * static_cast<std::size_t>(deg + 2) + static_cast<std::size_t>(j);
      try {
        const expr basis_el = x.pow(expr::num(j)) * mons[i];
        const expr dcol = sym::expand(placehold(
            sym::canonical(sym::diff(basis_el, x) * clear, x)));
        add_terms(dcol, col, false);
      } catch (const std::exception&) {
        bad = true;
      }
    }
  if (bad) return {};
  try {
    const expr frhs =
        sym::expand(placehold(sym::canonical(f * clear, x)));
    add_terms(frhs, 0, true);
  } catch (const std::exception&) {
    return {};
  }
  std::vector<ax::rational> sol;
  if (!solve_rational_linear(std::move(M), rhs, sol)) return {};
  expr a = laurent;
  for (std::size_t i = 0; i < mons.size(); ++i)
    for (int j = 0; j <= deg + 1; ++j) {
      const ax::rational& c =
          sol[i * static_cast<std::size_t>(deg + 2) +
              static_cast<std::size_t>(j)];
      if (!(c == kZero))
        a = a + expr::num(c) * x.pow(expr::num(j)) * mons[i];
    }
  if (is_zero_num(a)) return {};
  return {a};
}

}  // namespace tranche3

void add_tranche3(rule_set& r) {
  r.integral.emplace_back("i_cyclic", tranche3::i_cyclic);
  r.integral.emplace_back("i_unprod", tranche3::i_unprod);
  r.integral.emplace_back("i_ansatz_exp", tranche3::i_ansatz_exp);
  r.integral.emplace_back("i_linear_basis", tranche3::i_linear_basis);
}

}  // namespace ax::search
