# axiom Phase C Implementation Plan (solver kernel: llmopt/search in C++)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. TDD: RED before implementation on every task.

**Goal:** Port llmopt's derivation-search engine (`llmopt/search/derivation.py`, `rules.py`, the `engine.py` markov configuration) to axiom. Qualification per llmopt's `docs/superpowers/specs/2026-07-18-axiom-backend.md`: the C++ engine must re-derive the oracle-signed chain corpus (oracle-valid chains required, same rule sequence NOT required) via llmopt's chain-replay harness, then match solve-rate per level on fresh seeds with every emitted chain verified by BOTH oracles.

**Division of labor:** sympy-calling rules stay Python-side as *gated fallback rules over the bridge* — `i_heurisch` (sympy.integrate as an op-capped leaf closer) and the heavyweight algebra moves axiom does not implement natively in v1 (`factor`, `trigsimp`, `powsimp`). The C++ `successors()` exposes **external rule slots**: a registry of named callbacks; when llmopt embeds the kernel via pybind these are served from Python, and pure-native runs simply lack them (an absent rule costs coverage, never soundness). Neural components (syndrome policy, dispatcher, LLM entropy-k) also stay llmopt-side — the kernel's `proposer`/`propose_k`/`state_filter`/`ply_hook` hooks mirror `beam_search`'s exactly so they can be driven across the bridge later; the native default brain is the Markov prior (the zero-NN 316/360 configuration).

**Key differences the C++ engine legitimately keeps (documented, not hidden):**
- **No fork walls / signal timeboxes.** Python's `_timeboxed` + 90s fork walls exist because sympy can live-lock (llmopt pathology #10). Native rules are deterministic and fast; per-rule budgets become plain `steady_clock` checks in the successor loop, and external (bridge) rules keep their Python-side boxes. Retiring walls on axiom-decided paths is an explicit Phase C payoff.
- **verify_p defaults to 1.** Sampled verification (spec O2) amortized sympy's per-edge oracle cost; the native oracle is ~µs-scale, so every edge pays it. The `verify_p`/`replay_verify`/`corrupted` machinery is still ported (the hybrid config wants it, and the soundness boundary must exist for external rules).
- **Verification is three-valued.** `verify_edge` accepts only proof-or-witness-clean edges: EQUIVALENT accepts; NOT_EQUIVALENT and UNDECIDED both reject (conservative-sound, same contract as `_is_zero`'s carrier rejection — incompleteness prunes a legal move, never admits an illegal one).

**Tech Stack:** C++23 STL, GoogleTest, ax::sym (expr/oracle/parse/print_sstr), ax::la (for `i_linear_basis`), ax::pyrand+mathgen (fresh-seed races).

**File map:**
- `include/ax/sym/expr.hpp` + `src/sym/expr.cpp` — n-ary fn nodes; `Integral`/`Derivative`/`Subs` carriers
- `src/sym/{calc,parse,print_sstr,oracle}.cpp` — carrier support (diff-of-Integral, sstr round-trip, count_ops, opaque-atom handling)
- `include/ax/search/{state,rules,beam,engine}.hpp`, `src/search/*.cpp`
- Tests `tests/search/*`, fixtures from llmopt (chain corpus TSV/JSONL, count_ops fixture)

---

### Task C1: carrier nodes + count_ops parity

Generalize `expr` fn nodes to n-ary (`expr::fn(name, vector<expr>)`); add factories `integral(f, x)`, `derivative(f, x)`, `subs(e, x, r)` printing/parsing sympy sstr forms (`Integral(f, x)`, `Derivative(f, x)`, `Subs(e, x, r)`); `diff(Integral(f, x), x) == f` (the verify identity); eval throws (carriers are never numeric — the oracle's sampler then skips, keeping carrier states UNDECIDED territory); canonicalization treats carriers as opaque atoms. `count_ops(e)` matching `sympy.count_ops` (fixture-gated: generated over mathgen L1–L8 + carrier-wrapped shapes). HCE = `100*unsolved + count_ops + 0.1*plies` cross-checked against sympy values on the same fixture.
Commit `feat(sym): Integral/Derivative/Subs carrier nodes + sympy count_ops parity`.

### Task C2: chassis — State/successors/verify_edge/beam

`ax::search`: `state{expr, plies, history}`, key = hash-cons handle (pointer identity replaces srepr strings — free dedup); `successors()` iterating Derivative nodes (CORE+MACRO rules), Integral nodes with **multi-limit peeling** (nested `Integral(f, x, x)` from by-parts: rules see the innermost limit), algebra moves, per-(rule,node) memo cache; `verify_edge` native (Integral edges: differentiate the difference per free symbol, structural Integral-atom cancellation, reject surviving carriers; Derivative edges: exact equivalence); `beam_search` with the full hook surface (`proposer`, `propose_k` + never-guillotine-a-solved-kid, `state_filter`, `expand_rules` gate with full-expansion fallback, `ply_hook`, `max_nodes`, `trace`), `replay_verify` with same-label backtracking. Tests: hand-built 2–3-ply derivations verified end-to-end; a wrong-rewrite injection must be rejected by verify_edge; beam determinism (same input → same chain).
Commit `feat(search): derivation-search chassis - State, successors, verify_edge, beam`.

### Task C3: rules tranche 1 (the L1–L4 closers)

Diff: `d_const, d_x, d_sum, d_product, d_power, d_chain_table, d_quotient` + macro `d_const_factor`. Integrals: `i_const, i_power, i_sum, i_const_factor, i_table, i_usub` (Subs carriers + `subs_eval` back-substitution move), `i_parts` (stepwise, inner integrals unevaluated). Algebra moves: `expand`, `cancel`, `together` (via oracle's ratio machinery), `subs_eval`. Gate: replay the L1–L4 slice of the chain corpus.
Commit `feat(search): rules tranche 1 - diff core, table/linearity/u-sub/by-parts`.

### Task C4: rules tranche 2 (rational + special forms)

`i_apart` (partial fractions — reuse `ax::sym` integrate machinery), `i_log_power, i_transcend_div, i_inverse_trig, i_sqrt_basis`. Gate: L5–L6 chain slice.
Commit `feat(search): rules tranche 2 - apart, log-power, inverse-trig, sqrt basis`.

### Task C5: rules tranche 3 (the autopsy rules) + euler

`i_cyclic` (I = f − I algebra), `i_unprod` (reverse product rule), `i_ansatz_exp` (undetermined coefficients), `i_linear_basis` (meet-in-the-middle over answer shapes as one `ax::la` solve — subsumes the others; the ladder's L3 30/30 rule), `euler` rewrite (trig→complex exponentials — needs `exp` of symbolic-complex arguments handled as opaque atoms; scope to what the corpus chains use). Gate: L7–L8 chain slice.
Commit `feat(search): rules tranche 3 - cyclic, unprod, ansatz, linear basis, euler`.

### Task C6: engine facade + qualification

`ax::search::solve()`: Markov-prior proposer (bigram/unigram table loaded from a TSV export of `checkpoints/markov_prior.json` — llmopt exports; unseen-rule trial mass = 0.5·median, the measured rule), width 3, propose_k 3, max_plies 24, macros on, external-rule registry (heurisch slot), optional verify_p<1 path with replay boundary + corrupted counter. Qualification (llmopt owns adjudication): (a) chain-replay harness over the full oracle-signed corpus — target: every corpus root re-derived to an oracle-valid chain; (b) fresh-seed per-level race vs the sympy engine at equal node budgets, every emitted chain verified by both oracles; (c) throughput measurement (the 10–50x farm claim is measured here, not promised).
Commit `feat(search): solve() facade - markov brain, external rule slots, qualification`.

---

## Self-review notes

- Soundness inheritance: verify-at-the-boundary is preserved (replay_verify), UNDECIDED never counts as valid, external-rule absence degrades coverage not correctness.
- No invented rule semantics: each rule ports line-for-line from `rules.py` with its fixture/chain gate; where axiom lacks a primitive (factor/trigsimp), the move is an external slot, not an approximation.
- The corpus, not the spec, decides rule priority: if chain replay shows a tranche-2 rule dominating early, reorder.
- Perf discipline: correctness first; the throughput claim is measured in C6 against the same benchmarks llmopt publishes.
