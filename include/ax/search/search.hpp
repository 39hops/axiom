#pragma once
/** @file search.hpp Derivation-search chassis (Phase C task 2).

    Port of llmopt/search/derivation.py: State, successors (rule moves
    targeting one carrier node + whole-expression algebra moves), native
    three-valued verify_edge, beam_search with the full hook surface, and
    the replay_verify soundness boundary.

    Differences kept deliberately (docs/plans/2026-07-18-phase-c-solver.md):
    no signal timeboxes (native rules are deterministic; external bridge
    rules keep Python-side walls), verify_p defaults to 1 (native oracle is
    cheap), and verification is three-valued — UNDECIDED rejects the edge
    unless an external equivalence slot decides it (absent slot = reject:
    conservative-sound). */
#include <ax/sym/expr.hpp>

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace ax::search {

using sym::expr;

// ------------------------------------------------------------------ state

struct state {
  expr e;
  int plies = 0;
  std::vector<std::string> history;  // bare rule names (llmopt contract)
};

/** No unsolved carriers (Integral/Derivative/Subs) anywhere. */
bool is_solved(const expr& e);
inline bool is_solved(const state& s) { return is_solved(s.e); }

/** Count of unsolved carrier atoms. */
int unsolved_count(const expr& e);

/** Hand-crafted evaluation v0, sympy-number-exact:
    100*unsolved + count_ops + 0.1*plies. Lower is better. */
double hce(const state& s);

// ------------------------------------------------------------------ rules

/** A rule maps one carrier node to candidate rewrites (empty = no fire).
    Pure functions of the node; memoized per (rule, node) by successors. */
using rule_fn = std::function<std::vector<expr>(const expr& node)>;
using rule = std::pair<std::string, rule_fn>;

/** Whole-expression algebra move (nullopt = no change / not applicable). */
using algebra_fn = std::function<std::optional<expr>(const expr&)>;
using algebra_move = std::pair<std::string, algebra_fn>;

/** External slots (bridge-gated): sympy-side fallback rules and the
    UNDECIDED-equivalence decider for the hybrid config. All optional;
    absence costs coverage, never soundness. */
struct external_slots {
  std::vector<rule> int_rules;  // e.g. i_heurisch over the bridge
  std::vector<algebra_move> algebra;  // e.g. factor/trigsimp
  /** Decide an UNDECIDED edge: return true iff externally verified
      equivalent. Absent -> UNDECIDED rejects. */
  std::function<bool(const expr& parent, const expr& child)> equivalence;
};

/** The rule tables (filled by tranches C3-C5; C2 ships the chassis and
    a starter set used by its tests). */
struct rule_set {
  std::vector<rule> core;      // Derivative-node rules
  std::vector<rule> macros;    // promoted macros (d_const_factor, ...)
  std::vector<rule> integral;  // Integral-node rules
  std::vector<algebra_move> algebra;
  external_slots external;
};

/** The default native rule set (grows with tranches). */
const rule_set& default_rules();

// ------------------------------------------------------------- verification

/** Resolve Derivative carriers natively (repeated diff per limit) and
    solved Subs carriers; Integrals stay unevaluated — the exact analogue
    of sympy doit(integrals=False). */
expr doit_no_integrals(const expr& e);

/** Oracle check that a move preserves value. Integral edges verify
    modulo an additive constant by differentiating the difference per
    free symbol; Derivative edges resolve carriers then require
    equivalence. Three-valued: only a structural/witness-clean EQUIVALENT
    accepts; NOT_EQUIVALENT and UNDECIDED reject (external.equivalence,
    when present, may decide UNDECIDED). */
bool verify_edge(const expr& parent, const expr& child,
                 const external_slots& ext);

// -------------------------------------------------------------- successors

struct successor_options {
  bool use_macros = false;
  double verify_p = 1.0;  // deterministic sampling in the child key
  const std::vector<std::string>* only_rules = nullptr;
  std::function<bool(const std::string& rule_name)> move_filter;
};

/** Legal, non-identity, verified successor states (bare-name labels). */
std::vector<std::pair<std::string, state>> successors(
    const state& s, const rule_set& rules, const successor_options& opt = {});

// ------------------------------------------------------------------- beam

struct search_result {
  bool solved = false;
  state best;
  long long nodes = 0;
  bool corrupted = false;  // sampled-mode winning path failed full replay
};

struct beam_options {
  int width = 8;
  int max_plies = 12;
  std::optional<long long> max_nodes;
  bool use_macros = false;
  double verify_p = 1.0;
  std::function<double(const state&)> eval_fn;  // default hce
  /** Reorder (and optionally truncate via propose_k) the children. */
  std::function<std::vector<std::pair<std::string, state>>(
      const state&, std::vector<std::pair<std::string, state>>)>
      proposer;
  std::optional<int> propose_k;  // never guillotines a solved kid
  std::function<bool(const state&)> state_filter;  // false = prune
  std::function<bool(int ply, const std::vector<state>& beam,
                     long long nodes)>
      ply_hook;  // true = abort (regret gate)
  std::vector<state>* trace = nullptr;
};

/** Re-verify a winning path (bare names, backtracking over same-name
    siblings) with verify_p = 1. The soundness boundary. */
bool replay_verify(const expr& root, const std::vector<std::string>& history,
                   const rule_set& rules);

search_result beam_search(const expr& root, const rule_set& rules,
                          const beam_options& opt = {});

}  // namespace ax::search
