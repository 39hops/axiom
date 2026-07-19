#include <ax/search/search.hpp>

#include <ax/sym/budget.hpp>
#include <ax/sym/calc.hpp>
#include <ax/sym/count_ops.hpp>
#include <ax/sym/oracle.hpp>

#include <chrono>
#include <iostream>
#include <set>

namespace ax::search {

namespace {

bool is_carrier_name(const std::string& n) {
  return n == "Integral" || n == "Derivative" || n == "Subs";
}

bool is_carrier(const expr& e) {
  return e.is_fn() && is_carrier_name(e.name());
}

void walk_carriers(const expr& e, int& count) {
  if (is_carrier(e)) {
    ++count;
    return;  // sympy atoms(): outermost carriers count once
  }
  for (const expr& a : e.args()) walk_carriers(a, count);
}

void free_symbols(const expr& e, std::set<std::string>& out) {
  if (e.is_sym()) {
    const std::string& n = e.name();
    if (n != "pi" && n != "E") out.insert(n);
    return;
  }
  // Subs binds its variable (sympy semantics); indefinite Integral and
  // Derivative do NOT — Integral(f, x).free_symbols contains x because
  // the antiderivative depends on it. Binding it here made every pure
  // integral edge free-symbol-empty and rejected (measured: L1 60->1).
  if (e.is_fn() && e.name() == "Subs") {
    std::set<std::string> inner;
    free_symbols(e.args()[0], inner);
    if (e.args()[1].is_sym()) inner.erase(e.args()[1].name());
    out.insert(inner.begin(), inner.end());
    free_symbols(e.args()[2], out);
    return;
  }
  for (const expr& a : e.args()) free_symbols(a, out);
}

bool has_integral_or_subs(const expr& e) {
  if (e.is_fn() && (e.name() == "Integral" || e.name() == "Subs"))
    return true;
  for (const expr& a : e.args())
    if (has_integral_or_subs(a)) return true;
  return false;
}

bool has_integral(const expr& e) {
  if (e.is_fn() && e.name() == "Integral") return true;
  for (const expr& a : e.args())
    if (has_integral(a)) return true;
  return false;
}

}  // namespace

namespace {
thread_local long long size_rejects_ = 0;
}  // namespace

long long verify_size_reject_count() { return size_rejects_; }

bool is_solved(const expr& e) {
  if (is_carrier(e)) return false;
  for (const expr& a : e.args())
    if (!is_solved(a)) return false;
  return true;
}

int unsolved_count(const expr& e) {
  int n = 0;
  walk_carriers(e, n);
  return n;
}

double hce(const state& s) {
  return 100.0 * unsolved_count(s.e) +
         static_cast<double>(sym::count_ops(s.e)) + 0.1 * s.plies;
}

expr doit_no_integrals(const expr& e) {
  switch (e.k()) {
    case sym::kind::num:
    case sym::kind::sym:
      return e;
    case sym::kind::fn: {
      if (e.name() == "Derivative") {
        expr f = doit_no_integrals(e.args()[0]);
        // repeated diff per limit; a Derivative of an Integral resolves
        // through diff's peel rule, never through an integrator
        for (std::size_t i = 1; i < e.args().size(); ++i)
          f = sym::diff(f, e.args()[i]);
        return f;
      }
      if (e.name() == "Subs") {
        const expr inner = doit_no_integrals(e.args()[0]);
        if (is_solved(inner))
          return inner.subs(e.args()[1], e.args()[2]);
        return expr::fn("Subs", std::vector<expr>{inner, e.args()[1],
                                                  e.args()[2]});
      }
      std::vector<expr> mapped;
      mapped.reserve(e.args().size());
      for (const expr& a : e.args()) mapped.push_back(doit_no_integrals(a));
      return expr::fn(e.name(), std::move(mapped));
    }
    case sym::kind::add: {
      expr out = expr::num(0);
      for (const expr& t : e.args()) out = out + doit_no_integrals(t);
      return out;
    }
    case sym::kind::mul: {
      expr out = expr::num(1);
      for (const expr& f : e.args()) out = out * doit_no_integrals(f);
      return out;
    }
    case sym::kind::pow:
      return doit_no_integrals(e.args()[0])
          .pow(doit_no_integrals(e.args()[1]));
  }
  return e;
}

bool verify_edge(const expr& parent, const expr& child,
                 const external_slots& ext) {
  using sym::verdict;
  // size-gate (the native RULE_WALL analogue for verification: no
  // preemption, so bound by refusing oversized checks — a rejected
  // legal edge costs coverage, never soundness). The q-l5-23 wedge was
  // a single canonical() on a log^2 monster child inside this call.
  if (sym::count_ops(parent) + sym::count_ops(child) > 400) {
    ++size_rejects_;
    return false;
  }
  const auto decide = [&](const expr& a, const expr& b,
                          const expr& var) -> bool {
    const verdict v = sym::equivalent(a, b, var);
    if (v == verdict::equivalent) return true;
    if (v == verdict::not_equivalent) return false;
    // UNDECIDED: external slot may decide; absent slot rejects
    return ext.equivalence ? ext.equivalence(a, b) : false;
  };

  const auto t0 = std::chrono::steady_clock::now();
  struct slow_report {
    std::chrono::steady_clock::time_point t0;
    const expr& p;
    ~slow_report() {
      const double dt = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
      if (dt > 2.0)
        std::cerr << "[slow-verify] " << sym::count_ops(p) << " ops "
                  << dt << "s" << std::endl;
    }
  } reporter{t0, parent};
  try {
    // the native RULE_WALL: all sym work below is budget-polled;
    // expiry lands in the catch as a conservative edge rejection
    sym::work_budget_scope budget(std::chrono::milliseconds(3000));
    if (has_integral(parent)) {
      // equality mod constant: d/dv of the difference vanishes for every
      // free symbol. Structurally-equal Integral atoms cancel in the
      // subtraction; a difference whose derivative still carries
      // Integral/Subs is rejected (the _is_zero carrier rule:
      // incompleteness prunes a legal move, never admits an illegal one).
      const expr d = parent - child;
      std::set<std::string> frees;
      free_symbols(d, frees);
      if (frees.empty()) return is_solved(d) && decide(d, expr::num(0), expr::symbol("x"));
      for (const std::string& v : frees) {
        const expr var = expr::symbol(v);
        const expr dd = doit_no_integrals(sym::diff(d, var));
        if (has_integral_or_subs(dd)) return false;
        if (!decide(dd, expr::num(0), var)) return false;
      }
      return true;
    }
    const expr a = doit_no_integrals(parent);
    const expr b = doit_no_integrals(child);
    if (has_integral_or_subs(a) || has_integral_or_subs(b)) return false;
    std::set<std::string> frees;
    free_symbols(a, frees);
    free_symbols(b, frees);
    const expr var =
        frees.empty() ? expr::symbol("x") : expr::symbol(*frees.begin());
    return decide(a, b, var);
  } catch (const std::exception&) {
    return false;  // a crashing verify rejects the edge (safe direction)
  }
}

}  // namespace ax::search
