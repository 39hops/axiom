#include <ax/search/search.hpp>

#include <ax/sym/calc.hpp>
#include <ax/sym/expand.hpp>

#include <algorithm>
#include <unordered_set>

namespace ax::search {

namespace {

expr doit_subs_only(const expr& e);

struct expr_key_hash {
  std::size_t operator()(const expr& e) const { return e.hash(); }
};
struct expr_key_eq {
  bool operator()(const expr& a, const expr& b) const { return a.same(b); }
};

/** subs_eval algebra move: back-substitute solved Subs carriers. */
std::optional<expr> subs_eval(const expr& e) {
  const expr next = doit_subs_only(e);
  if (next.same(e)) return std::nullopt;
  return next;
}

expr doit_subs_only(const expr& e) {
  switch (e.k()) {
    case sym::kind::num:
    case sym::kind::sym:
      return e;
    case sym::kind::fn: {
      if (e.name() == "Subs" && is_solved(e.args()[0]))
        return e.args()[0].subs(e.args()[1], e.args()[2]);
      std::vector<expr> mapped;
      mapped.reserve(e.args().size());
      for (const expr& a : e.args()) mapped.push_back(doit_subs_only(a));
      return expr::fn(e.name(), std::move(mapped));
    }
    case sym::kind::add: {
      expr out = expr::num(0);
      for (const expr& t : e.args()) out = out + doit_subs_only(t);
      return out;
    }
    case sym::kind::mul: {
      expr out = expr::num(1);
      for (const expr& f : e.args()) out = out * doit_subs_only(f);
      return out;
    }
    case sym::kind::pow:
      return doit_subs_only(e.args()[0]).pow(doit_subs_only(e.args()[1]));
  }
  return e;
}

}  // namespace

const rule_set& default_rules() {
  static const rule_set rs = [] {
    rule_set r;
    // Starter algebra moves; rule tranches C3-C5 fill core/macros/integral.
    r.algebra.emplace_back("expand", [](const expr& e) -> std::optional<expr> {
      const expr n = sym::expand(e);
      if (n.same(e)) return std::nullopt;
      return n;
    });
    r.algebra.emplace_back("subs_eval", subs_eval);
    return r;
  }();
  return rs;
}

bool replay_verify(const expr& root, const std::vector<std::string>& history,
                   const rule_set& rules) {
  // Backtracking walk over same-name children with verify_p = 1
  // (the labels-are-not-unique lesson).
  successor_options opt;
  opt.use_macros = true;
  opt.verify_p = 1.0;
  std::function<bool(const state&, std::size_t)> walk =
      [&](const state& cur, std::size_t i) -> bool {
    if (i == history.size()) return true;
    for (auto& [name, child] : successors(cur, rules, opt))
      if (name == history[i] && walk(child, i + 1)) return true;
    return false;
  };
  return walk(state{root}, 0);
}

search_result beam_search(const expr& root, const rule_set& rules,
                          const beam_options& opt) {
  const auto eval_fn =
      opt.eval_fn ? opt.eval_fn : [](const state& s) { return hce(s); };

  state root_state{root};
  if (is_solved(root_state)) return {true, root_state, 1};

  std::vector<state> beam{root_state};
  std::optional<state> best_solved;
  std::unordered_set<expr, expr_key_hash, expr_key_eq> visited{root};
  long long nodes = 1;
  bool budget_hit = false;

  successor_options sopt;
  sopt.use_macros = opt.use_macros;
  sopt.verify_p = opt.verify_p;

  for (int ply = 0; ply < opt.max_plies && !budget_hit; ++ply) {
    std::vector<state> candidates;
    for (const state& s : beam) {
      auto kids = successors(s, rules, sopt);
      if (opt.proposer) kids = opt.proposer(s, std::move(kids));
      if (opt.propose_k) {
        const int kk = std::max(1, *opt.propose_k);
        // never guillotine a terminal: solved kids survive any cut
        std::vector<std::pair<std::string, state>> kept(
            kids.begin(),
            kids.begin() + std::min<std::size_t>(kids.size(),
                                                 static_cast<std::size_t>(kk)));
        for (std::size_t i = static_cast<std::size_t>(kk); i < kids.size();
             ++i)
          if (is_solved(kids[i].second)) kept.push_back(kids[i]);
        kids = std::move(kept);
      }
      for (auto& [name, child] : kids) {
        if (visited.count(child.e)) continue;
        visited.insert(child.e);
        if (opt.state_filter && !opt.state_filter(child)) continue;
        if (opt.trace != nullptr) opt.trace->push_back(child);
        ++nodes;
        if (opt.max_nodes && nodes >= *opt.max_nodes) {
          candidates.push_back(child);
          budget_hit = true;
          break;
        }
        if (is_solved(child) &&
            (!best_solved || eval_fn(child) < eval_fn(*best_solved)))
          best_solved = child;
        candidates.push_back(child);
      }
      if (budget_hit) break;
    }
    if (budget_hit || candidates.empty()) break;
    std::stable_sort(candidates.begin(), candidates.end(),
                     [&](const state& a, const state& b) {
                       return eval_fn(a) < eval_fn(b);
                     });
    beam.assign(candidates.begin(),
                candidates.begin() +
                    std::min<std::size_t>(candidates.size(),
                                          static_cast<std::size_t>(
                                              std::max(1, opt.width))));
    if (best_solved && eval_fn(*best_solved) <= eval_fn(beam.front())) break;
    if (opt.ply_hook && !best_solved && opt.ply_hook(ply, beam, nodes))
      break;
  }

  if (best_solved) {
    if (opt.verify_p < 1.0 &&
        !replay_verify(root, best_solved->history, rules)) {
      const state& fallback =
          *std::min_element(beam.begin(), beam.end(),
                            [&](const state& a, const state& b) {
                              return eval_fn(a) < eval_fn(b);
                            });
      return {false, fallback, nodes, true};
    }
    return {true, *best_solved, nodes};
  }
  const state& fallback =
      *std::min_element(beam.begin(), beam.end(),
                        [&](const state& a, const state& b) {
                          return eval_fn(a) < eval_fn(b);
                        });
  return {false, fallback, nodes};
}

}  // namespace ax::search
