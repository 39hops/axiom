#include <ax/search/search.hpp>

#include <ax/sym/budget.hpp>
#include <ax/sym/calc.hpp>
#include <ax/sym/count_ops.hpp>

#include <algorithm>
#include <iostream>
#include <chrono>
#include <map>
#include <unordered_set>

namespace ax::search {

namespace {

/** Replace every occurrence of target (a subtree) with repl, rebuilding
    through the canonicalizing factories (our trees are canonical, so the
    rebuild is identity away from the replacement site). */
expr replace_node(const expr& e, const expr& target, const expr& repl) {
  if (e.same(target)) return repl;
  switch (e.k()) {
    case sym::kind::num:
    case sym::kind::sym:
      return e;
    case sym::kind::fn: {
      std::vector<expr> mapped;
      mapped.reserve(e.args().size());
      bool changed = false;
      for (const expr& a : e.args()) {
        mapped.push_back(replace_node(a, target, repl));
        changed = changed || !mapped.back().same(a);
      }
      if (!changed) return e;
      return expr::fn(e.name(), std::move(mapped));
    }
    case sym::kind::add: {
      expr out = expr::num(0);
      bool changed = false;
      for (const expr& t : e.args()) {
        const expr m = replace_node(t, target, repl);
        changed = changed || !m.same(t);
        out = out + m;
      }
      return changed ? out : e;
    }
    case sym::kind::mul: {
      expr out = expr::num(1);
      bool changed = false;
      for (const expr& f : e.args()) {
        const expr m = replace_node(f, target, repl);
        changed = changed || !m.same(f);
        out = out * m;
      }
      return changed ? out : e;
    }
    case sym::kind::pow: {
      const expr b = replace_node(e.args()[0], target, repl);
      const expr x = replace_node(e.args()[1], target, repl);
      if (b.same(e.args()[0]) && x.same(e.args()[1])) return e;
      return b.pow(x);
    }
  }
  return e;
}

void collect_atoms(const expr& e, const std::string& name,
                   std::vector<expr>& out) {
  if (e.is_fn() && e.name() == name) {
    out.push_back(e);
    // sympy atoms() also finds nested carriers inside; carriers rarely
    // nest same-kind, but Integral(Integral(...)...) from i_parts does:
    for (const expr& a : e.args()) collect_atoms(a, name, out);
    return;
  }
  for (const expr& a : e.args()) collect_atoms(a, name, out);
}

std::vector<expr> sorted_atoms(const expr& e, const std::string& name) {
  std::vector<expr> atoms;
  collect_atoms(e, name, atoms);
  // dedupe (hash-consing: same() duplicates are pointer-equal)
  std::vector<expr> uniq;
  for (const expr& a : atoms) {
    bool dup = false;
    for (const expr& u : uniq) dup = dup || u.same(a);
    if (!dup) uniq.push_back(a);
  }
  std::stable_sort(uniq.begin(), uniq.end(),
                   [](const expr& a, const expr& b) {
                     return sym::count_ops(a) < sym::count_ops(b);
                   });
  return uniq;
}

struct expr_key_hash {
  std::size_t operator()(const expr& e) const { return e.hash(); }
};
struct expr_key_eq {
  bool operator()(const expr& a, const expr& b) const { return a.same(b); }
};

/** Per-(rule, node) memo. Rules are pure; nodes are hash-consed. */
using memo_key = std::pair<const void*, std::size_t>;

}  // namespace

namespace {
thread_local bool clear_rule_cache_flag = false;
}
void successors_cache_clear() { clear_rule_cache_flag = true; }

std::vector<std::pair<std::string, state>> successors(
    const state& s, const rule_set& rules, const successor_options& opt) {
  std::vector<std::pair<std::string, state>> out;
  std::unordered_set<expr, expr_key_hash, expr_key_eq> seen;
  seen.insert(s.e);

  static thread_local std::map<std::pair<const std::string*, std::size_t>,
                               std::vector<expr>>
      rule_cache;
  if (rule_cache.size() > 200'000 || clear_rule_cache_flag) {
    rule_cache.clear();
    clear_rule_cache_flag = false;
  }

  const auto want = [&](const std::string& name) {
    if (opt.only_rules != nullptr &&
        std::find(opt.only_rules->begin(), opt.only_rules->end(), name) ==
            opt.only_rules->end())
      return false;
    if (opt.move_filter && !opt.move_filter(name)) return false;
    return true;
  };

  const auto emit = [&](const std::string& name, const expr& child_expr) {
    if (seen.count(child_expr)) return;
    if (opt.deadline &&
        std::chrono::steady_clock::now() > *opt.deadline)
      return;  // expired: stop paying verification for new children
    const bool pay_oracle =
        opt.verify_p >= 1.0 ||
        static_cast<double>(child_expr.hash() % 1000) <
            opt.verify_p * 1000.0;
    if (pay_oracle &&
        !verify_edge(s.e, child_expr, rules.external))
      return;
    seen.insert(child_expr);
    state child{child_expr, s.plies + 1, s.history};
    child.history.push_back(name);
    out.emplace_back(name, std::move(child));
  };

  const auto expired = [&] {
    return opt.deadline &&
           std::chrono::steady_clock::now() > *opt.deadline;
  };
  const auto fire = [&](const rule& r, const expr& node) {
    const auto key = std::make_pair(&r.first, node.hash());
    auto it = rule_cache.find(key);
    if (it == rule_cache.end()) {
      std::vector<expr> rewrites;
      // persistent no-fire memo (Arc 3): skip a fire this mask has seen
      // produce nothing; proposal-side only, fingerprint-guarded
      if (fire_mask_check(r.first, node))
        return rule_cache.emplace(key, std::move(rewrites)).first->second;
      bool aborted = false;
      const auto t0 = std::chrono::steady_clock::now();
      try {
        sym::work_budget_scope budget(std::chrono::milliseconds(8000));
        rewrites = r.second(node);
      } catch (const std::exception&) {
        // a crashing or budget-expired rule costs one move, never the
        // search (work_expired derives runtime_error and lands here)
        aborted = true;  // NOT a proven no-fire: never mask it
      }
      const double dt = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
      if (dt > 2.0)  // slow-fire observability (the wedge-hunt lesson)
        std::cerr << "[slow-fire] " << r.first << " "
                  << sym::count_ops(node) << " ops " << dt << "s"
                  << std::endl;
      if (rewrites.empty() && !aborted) fire_mask_record(r.first, node);
      it = rule_cache.emplace(key, std::move(rewrites)).first;
    }
    return it->second;
  };

  // Derivative-node rules (core + macros)
  std::vector<const rule*> drules;
  for (const rule& r : rules.core) drules.push_back(&r);
  if (opt.use_macros)
    for (const rule& r : rules.macros) drules.push_back(&r);
  for (const expr& node : sorted_atoms(s.e, "Derivative")) {
    for (const rule* r : drules) {
      if (expired()) return out;
      if (!want(r->first)) continue;
      for (const expr& rw : fire(*r, node))
        emit(r->first, replace_node(s.e, node, rw));
    }
  }

  // Integral-node rules with multi-limit peeling
  std::vector<const rule*> irules;
  for (const rule& r : rules.integral) irules.push_back(&r);
  for (const rule& r : rules.external.int_rules) irules.push_back(&r);
  for (const expr& node : sorted_atoms(s.e, "Integral")) {
    const bool nested = node.args().size() > 2;
    const expr inner =
        nested ? expr::integral(node.args()[0], node.args()[1]) : node;
    for (const rule* r : irules) {
      if (expired()) return out;
      if (!want(r->first)) continue;
      for (const expr& rw : fire(*r, inner)) {
        expr new_node = rw;
        if (nested) {
          std::vector<expr> a;
          a.push_back(rw);
          for (std::size_t i = 2; i < node.args().size(); ++i)
            a.push_back(node.args()[i]);
          new_node = expr::fn("Integral", std::move(a));
        }
        emit(r->first, replace_node(s.e, node, new_node));
      }
    }
  }

  // whole-expression algebra moves
  std::vector<const algebra_move*> amoves;
  for (const algebra_move& m : rules.algebra) amoves.push_back(&m);
  for (const algebra_move& m : rules.external.algebra) amoves.push_back(&m);
  for (const algebra_move* m : amoves) {
    if (expired()) return out;
    if (!want(m->first)) continue;
    std::optional<expr> next;
    try {
      next = m->second(s.e);
    } catch (const std::exception&) {
      continue;
    }
    if (next) emit(m->first, *next);
  }

  return out;
}

}  // namespace ax::search
