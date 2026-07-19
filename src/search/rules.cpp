#include <ax/search/search.hpp>

#include <ax/sym/calc.hpp>
#include <ax/sym/expand.hpp>
#include <ax/sym/oracle.hpp>

#include <optional>
#include <utility>
#include <vector>

/** Rules tranche 1 (Phase C task 3): line-for-line port of
    llmopt/search/rules.py CORE_RULES + MACRO_RULES and the L1-L4
    integral closers (i_const, i_power, i_sum, i_const_factor, i_table,
    i_usub, i_parts). Candidate enumeration is deterministic in tree
    order (llmopt: matching sympy's chain choice is not required; run
    determinism and oracle-validity are). */

namespace ax::search {

namespace {

using sym::expr;

const expr kU = expr::symbol("u_");  // reserved substitution symbol

bool contains(const expr& e, const expr& target) {
  if (e.same(target)) return true;
  for (const expr& a : e.args())
    if (contains(a, target)) return true;
  return false;
}

/** Structural subtree substitution (sympy .subs(g, U) for our rules). */
expr replace_subtree(const expr& e, const expr& target, const expr& repl) {
  if (e.same(target)) return repl;
  switch (e.k()) {
    case sym::kind::num:
    case sym::kind::sym:
      return e;
    case sym::kind::fn: {
      std::vector<expr> mapped;
      mapped.reserve(e.args().size());
      for (const expr& a : e.args())
        mapped.push_back(replace_subtree(a, target, repl));
      return expr::fn(e.name(), std::move(mapped));
    }
    case sym::kind::add: {
      expr out = expr::num(0);
      for (const expr& t : e.args())
        out = out + replace_subtree(t, target, repl);
      return out;
    }
    case sym::kind::mul: {
      expr out = expr::num(1);
      for (const expr& f : e.args())
        out = out * replace_subtree(f, target, repl);
      return out;
    }
    case sym::kind::pow:
      return replace_subtree(e.args()[0], target, repl)
          .pow(replace_subtree(e.args()[1], target, repl));
  }
  return e;
}

/** (f, x) for first-order single-variable Derivative carriers. */
std::optional<std::pair<expr, expr>> unpack_d(const expr& node) {
  if (!node.is_fn() || node.name() != "Derivative" ||
      node.args().size() != 2 || !node.args()[1].is_sym())
    return std::nullopt;
  return std::make_pair(node.args()[0], node.args()[1]);
}

/** (f, x) for single-variable indefinite Integral carriers. */
std::optional<std::pair<expr, expr>> unpack_i(const expr& node) {
  if (!node.is_fn() || node.name() != "Integral" ||
      node.args().size() != 2 || !node.args()[1].is_sym())
    return std::nullopt;
  return std::make_pair(node.args()[0], node.args()[1]);
}

expr mul_except(std::span<const expr> args, std::size_t skip) {
  expr out = expr::num(1);
  for (std::size_t j = 0; j < args.size(); ++j)
    if (j != skip) out = out * args[j];
  return out;
}

// -------------------------------------------------------- diff rules

std::vector<expr> d_const(const expr& node) {
  if (const auto u = unpack_d(node); u && !contains(u->first, u->second))
    return {expr::num(0)};
  return {};
}

std::vector<expr> d_x(const expr& node) {
  if (const auto u = unpack_d(node); u && u->first.same(u->second))
    return {expr::num(1)};
  return {};
}

std::vector<expr> d_sum(const expr& node) {
  const auto u = unpack_d(node);
  if (!u || !u->first.is_add()) return {};
  expr out = expr::num(0);
  for (const expr& t : u->first.args())
    out = out + expr::derivative(t, u->second);
  return {out};
}

std::vector<expr> d_product(const expr& node) {
  const auto u = unpack_d(node);
  if (!u || !u->first.is_mul() || !contains(u->first, u->second)) return {};
  std::vector<expr> out;
  const auto args = u->first.args();
  for (std::size_t i = 0; i < args.size(); ++i) {
    const expr rest = mul_except(args, i);
    out.push_back(expr::derivative(args[i], u->second) * rest +
                  args[i] * expr::derivative(rest, u->second));
  }
  return out;
}

std::vector<expr> d_power(const expr& node) {
  const auto u = unpack_d(node);
  if (!u) return {};
  const auto& [f, x] = *u;
  if (!f.is_pow() || contains(f.args()[1], x) || !contains(f.args()[0], x))
    return {};
  const expr& base = f.args()[0];
  const expr& n = f.args()[1];
  return {n * base.pow(n - expr::num(1)) * expr::derivative(base, x)};
}

std::vector<expr> d_chain_table(const expr& node) {
  const auto u = unpack_d(node);
  if (!u) return {};
  const auto& [f, x] = *u;
  if (!f.is_fn() || f.args().size() != 1) return {};
  const expr& inner = f.args()[0];
  const std::string& n = f.name();
  expr outer = expr::num(0);
  if (n == "sin") outer = expr::fn("cos", inner);
  else if (n == "cos") outer = -expr::fn("sin", inner);
  else if (n == "tan")
    outer = expr::num(1) / expr::fn("cos", inner).pow(expr::num(2));
  else if (n == "exp") outer = expr::fn("exp", inner);
  else if (n == "log") outer = expr::num(1) / inner;
  else if (n == "sqrt")  // sympy sqrt is Pow (d_power covers it); axiom's
    outer = expr::num(1) /  // fn form takes the table route instead
            (expr::num(2) * expr::fn("sqrt", inner));
  else return {};
  return {outer * expr::derivative(inner, x)};
}

std::vector<expr> d_quotient(const expr& node) {
  const auto u = unpack_d(node);
  if (!u) return {};
  const auto& [f, x] = *u;
  // as_numer_denom: split mul factors by negative numeric exponents
  expr num = expr::num(1);
  expr den = expr::num(1);
  const auto add_factor = [&](const expr& g) {
    if (g.is_pow() && g.args()[1].is_num() &&
        g.args()[1].value() < ax::rational{}) {
      const auto flipped = -g.args()[1].value();
      den = den * (flipped == ax::rational(ax::bigint(1))
                       ? g.args()[0]
                       : g.args()[0].pow(expr::num(flipped)));
    } else if (g.is_num()) {
      num = num * expr::num(ax::rational(g.value().num(), ax::bigint(1)));
      if (!(g.value().den() == ax::bigint(1)))
        den = den * expr::num(ax::rational(g.value().den(), ax::bigint(1)));
    } else {
      num = num * g;
    }
  };
  if (f.is_mul())
    for (const expr& g : f.args()) add_factor(g);
  else
    add_factor(f);
  if ((den.is_num() && den.value() == ax::rational(ax::bigint(1))) ||
      !contains(den, x))
    return {};
  return {(expr::derivative(num, x) * den - num * expr::derivative(den, x)) /
          den.pow(expr::num(2))};
}

std::vector<expr> d_const_factor(const expr& node) {
  const auto u = unpack_d(node);
  if (!u || !u->first.is_mul()) return {};
  const auto& [f, x] = *u;
  expr cpart = expr::num(1);
  expr rest = expr::num(1);
  for (const expr& a : f.args())
    (contains(a, x) ? rest : cpart) = (contains(a, x) ? rest : cpart) * a;
  const expr one = expr::num(1);
  if (cpart.same(one) || rest.same(one)) return {};
  return {cpart * expr::derivative(rest, x)};
}

// ---------------------------------------------------- integral rules

std::vector<expr> i_const(const expr& node) {
  if (const auto u = unpack_i(node); u && !contains(u->first, u->second))
    return {u->first * u->second};
  return {};
}

std::vector<expr> i_power(const expr& node) {
  const auto u = unpack_i(node);
  if (!u) return {};
  const auto& [f, x] = *u;
  if (f.same(x)) return {x.pow(expr::num(2)) / expr::num(2)};
  if (!f.is_pow() || !f.args()[0].same(x) || contains(f.args()[1], x))
    return {};
  const expr& n = f.args()[1];
  if (n.is_num() && n.value() == ax::rational(ax::bigint(-1)))
    return {expr::fn("log", x)};
  const expr n1 = n + expr::num(1);
  return {x.pow(n1) / n1};
}

std::vector<expr> i_sum(const expr& node) {
  const auto u = unpack_i(node);
  if (!u || !u->first.is_add()) return {};
  expr out = expr::num(0);
  for (const expr& t : u->first.args())
    out = out + expr::integral(t, u->second);
  return {out};
}

std::vector<expr> i_const_factor(const expr& node) {
  const auto u = unpack_i(node);
  if (!u || !u->first.is_mul()) return {};
  const auto& [f, x] = *u;
  expr cpart = expr::num(1);
  expr rest = expr::num(1);
  for (const expr& a : f.args())
    (contains(a, x) ? rest : cpart) = (contains(a, x) ? rest : cpart) * a;
  const expr one = expr::num(1);
  if (cpart.same(one) || rest.same(one)) return {};
  return {cpart * expr::integral(rest, x)};
}

std::vector<expr> i_table(const expr& node) {
  const auto u = unpack_i(node);
  if (!u) return {};
  const auto& [f, x] = *u;
  if (f.is_fn() && f.args().size() == 1 && f.args()[0].same(x)) {
    if (f.name() == "sin") return {-expr::fn("cos", x)};
    if (f.name() == "cos") return {expr::fn("sin", x)};
    if (f.name() == "exp") return {expr::fn("exp", x)};
    if (f.name() == "log")  // the invisible-*1 by-parts case
      return {x * expr::fn("log", x) - x};
  }
  return {};
}

/** u-candidates: fn-atom arguments + pow bases, deterministic tree
    order, deduped by handle, must contain x and differ from x. */
void usub_candidates(const expr& f, const expr& x, std::vector<expr>& out) {
  if (f.is_fn()) out.push_back(f.args()[0]);
  if (f.is_pow()) out.push_back(f.args()[0]);
  for (const expr& a : f.args()) usub_candidates(a, x, out);
}

std::vector<expr> i_usub(const expr& node) {
  const auto u = unpack_i(node);
  if (!u) return {};
  const auto& [f, x] = *u;
  std::vector<expr> cands;
  usub_candidates(f, x, cands);
  std::vector<expr> uniq;
  for (const expr& g : cands) {
    if (!contains(g, x) || g.same(x)) continue;
    bool dup = false;
    for (const expr& q : uniq) dup = dup || q.same(g);
    if (!dup) uniq.push_back(g);
  }
  std::vector<expr> out;
  for (const expr& g : uniq) {
    const expr dg = sym::diff(g, x);
    if (dg.is_num() && dg.value() == ax::rational{}) continue;
    // cancel + simplify(doit=False) ~ canonical rational normalization
    const expr q = replace_subtree(sym::canonical(f / dg, x), g, kU);
    if (contains(q, x) || !contains(q, kU)) continue;
    out.push_back(expr::subs_carrier(expr::integral(q, kU), kU, g));
  }
  return out;
}

std::vector<expr> i_parts(const expr& node) {
  const auto u = unpack_i(node);
  if (!u || !u->first.is_mul()) return {};
  const auto& [f, x] = *u;
  std::vector<expr> out;
  const auto args = f.args();
  for (std::size_t i = 0; i < args.size(); ++i) {
    const expr du = sym::diff(args[i], x);
    if (du.is_num() && du.value() == ax::rational{}) continue;
    const expr dv = mul_except(args, i);
    const expr v = expr::integral(dv, x);
    out.push_back(args[i] * v - expr::integral(v * du, x));
  }
  return out;
}

}  // namespace

void add_tranche2(rule_set& r);  // rules2.cpp
void add_tranche3(rule_set& r);  // rules3.cpp

const rule_set& default_rules() {
  static const rule_set rs = [] {
    rule_set r;
    r.core = {
        {"d_const", d_const},   {"d_x", d_x},
        {"d_sum", d_sum},       {"d_product", d_product},
        {"d_power", d_power},   {"d_chain_table", d_chain_table},
    };
    r.macros = {
        {"d_quotient", d_quotient},
        {"d_const_factor", d_const_factor},
    };
    r.integral = {
        {"i_const", i_const},
        {"i_power", i_power},
        {"i_sum", i_sum},
        {"i_const_factor", i_const_factor},
        {"i_table", i_table},
        {"i_usub", i_usub},
        {"i_parts", i_parts},
    };
    add_tranche2(r);
    add_tranche3(r);
    r.algebra.emplace_back(
        "cancel", [](const expr& e) -> std::optional<expr> {
          const expr n = sym::canonical(e, expr::symbol("x"));
          if (n.same(e)) return std::nullopt;
          return n;
        });
    r.algebra.emplace_back(
        "expand", [](const expr& e) -> std::optional<expr> {
          const expr n = sym::expand(e);
          if (n.same(e)) return std::nullopt;
          return n;
        });
    r.algebra.emplace_back(
        "subs_eval", [](const expr& e) -> std::optional<expr> {
          const expr n = subs_eval_pass(e);
          if (n.same(e)) return std::nullopt;
          return n;
        });
    return r;
  }();
  return rs;
}

}  // namespace ax::search
