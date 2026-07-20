/** @file emit.cpp Phase D chain emission: replay_chain + annotate
    (docs/plans/2026-07-20-phase-d-chain-emission.md).

    Ports llmopt scripts/expert_iter_steps.py: the replay walk that
    records the state sequence, and annotate()'s hints (rule-fire
    syndrome on the largest Integral node) + think (ansatz rules'
    verbalized derivation via the DERIV_TRACE analogue). */
#include <ax/search/search.hpp>

#include <ax/sym/count_ops.hpp>

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace ax::search {

namespace {

using sym::expr;

// thread-local DERIV_TRACE analogue: armed by annotate() around an
// ansatz re-fire; rules push their verbalized derivation when armed.
thread_local bool trace_armed = false;
thread_local std::optional<std::string> trace_msg;

/** Largest Integral node by count_ops (llmopt: max(atoms(Integral),
    key=count_ops); ties keep the first found in tree order). */
std::optional<expr> largest_integral(const expr& e) {
  std::optional<expr> best;
  long long best_ops = -1;
  const std::function<void(const expr&)> walk = [&](const expr& q) {
    if (q.is_fn() && q.name() == "Integral") {
      const long long ops = sym::count_ops(q);
      if (ops > best_ops) {
        best_ops = ops;
        best = q;
      }
    }
    for (const expr& a : q.args()) walk(a);
  };
  walk(e);
  return best;
}

}  // namespace

void deriv_trace_arm() {
  trace_armed = true;
  trace_msg.reset();
}

std::optional<std::string> deriv_trace_take() {
  trace_armed = false;
  auto out = std::move(trace_msg);
  trace_msg.reset();
  return out;
}

void deriv_trace_push(const std::string& msg) {
  if (trace_armed) trace_msg = msg;
}

std::optional<std::vector<state>> replay_chain(
    const expr& root, const std::vector<std::string>& history,
    const rule_set& rules,
    std::optional<std::chrono::steady_clock::time_point> deadline) {
  successor_options opt;
  opt.use_macros = true;
  opt.verify_p = 1.0;
  opt.deadline = deadline;
  std::vector<state> chain;
  const std::function<bool(const state&, std::size_t)> walk =
      [&](const state& cur, std::size_t i) -> bool {
    if (i == history.size()) {
      chain.push_back(cur);
      return true;
    }
    for (auto& [name, child] : successors(cur, rules, opt))
      if (name == history[i]) {
        chain.push_back(cur);
        if (walk(child, i + 1)) return true;
        chain.pop_back();
      }
    return false;
  };
  if (!walk(state{root}, 0)) return std::nullopt;
  return chain;
}

annotation annotate(const expr& cur, const rule_set& rules,
                    const std::string& fired_rule) {
  annotation out;
  const auto node = largest_integral(cur);
  if (!node) return out;
  for (const auto& [name, fn] : rules.integral)
    if (!fn(*node).empty()) out.hints.push_back(name);
  if (fired_rule == "i_linear_basis" || fired_rule == "i_sqrt_basis") {
    for (const auto& [name, fn] : rules.integral)
      if (name == fired_rule) {
        deriv_trace_arm();
        (void)fn(*node);
        out.think = deriv_trace_take();
        break;
      }
  }
  return out;
}

}  // namespace ax::search
