#pragma once
/** @file inverse.hpp S7: inverse-move enumeration (relay 2026-07-27-0).

    For a state t, enumerate the legal PREDECESSOR set
      {(rule, p) : t in successors(p)}.

    Two-stage design: (1) candidate predecessors are proposed by per-rule
    inverse constructors where a closed-form un-application exists
    (i_const / i_power / i_sum / i_const_factor / i_table / i_parts /
    i_usub, the d_* patterns, cancel<->expand near-involution) plus a
    generic diff-trick fallback (any closed antiderivative F at a site
    proposes Integral(d F/dx, x) — covers every single-shot integral
    closer including the tranche-2/3 ansatz rules); (2) every candidate
    is settled by running the FORWARD engine: successors(p) with
    verify_p = 1 must contain t. Stage 2 makes the returned set sound by
    construction — each (p, t) edge is oracle-verified through the same
    verify_edge the search pays — so stage-1 incompleteness costs recall,
    never a wrong predecessor. */
#include <ax/search/search.hpp>

namespace ax::search {

struct predecessor {
  std::string rule;  // bare rule name (llmopt contract)
  expr p;
};

struct inverse_options {
  /** Hard wall, checked between candidate verifications (candidates are
      finite and individually budget-scoped, so between-candidate checks
      bound the wall without preemption). Expired -> return what's found. */
  std::optional<std::chrono::steady_clock::time_point> deadline;
  /** Cap on candidate predecessors sent to the forward-verify funnel. */
  std::size_t max_candidates = 160;
  bool use_macros = true;
};

/** Enumerate verified predecessors of t. Deterministic order (candidate
    proposal order is tree order; dedupe by hash-consed handle). */
std::vector<predecessor> predecessors(const expr& t, const rule_set& rules,
                                      const inverse_options& opt = {});

}  // namespace ax::search
