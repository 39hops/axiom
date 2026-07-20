# axiom Phase D Implementation Plan (chain emission: farm-shard rows)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. TDD: RED before implementation on every task.

**Goal:** The farm swap needs the native engine to emit *training rows*, not just verdicts. For each solved root, emit the winning chain as (cur, nxt) step pairs in llmopt's shard schema (`scripts/farm_v22.py` writer, `scripts/expert_iter_steps.py::_chain_worker` semantics). Acceptance: 100-root emission run whose rows llmopt diffs for shape and oracle-verifies pair-by-pair against a sympy-farmed sample. This is the last artifact before farming L9.

**Row schema (from farm_v22.py, verbatim):**

```json
{"cur": "<sstr>", "nxt": "<sstr>", "level": <int>,
 "source": "axiom-chain" | "axiom-oneply",
 "hints": ["<rule-name>", ...], "think": "<verbalized derivation>" | null}
```

- `cur`/`nxt` — sympy sstr of consecutive states along the winning chain, obtained by **replaying the history** with same-label backtracking (llmopt keeps replay verbatim because labels are not unique across siblings; ours is `replay_verify`'s walk, extended to record the state sequence).
- `source` — llmopt's runs stamp `v22-chain`/`v22-oneply`; native rows stamp `axiom-chain`/`axiom-oneply` so provenance survives shard mixing. One-ply = chain of exactly one (cur, nxt) pair.
- `hints` — the **rule-fire syndrome** at cur: the names of every integral rule that fires (returns non-empty) on the *largest* Integral node of cur (by count_ops, matching `annotate`'s `max(atoms(Integral), key=count_ops)`). This is the engine's sensory organ verbalized; order = rule-registry order.
- `think` — only for ansatz rules (`i_linear_basis`, `i_sqrt_basis`): the rule's verbalized internal derivation, matching `_trace` message conventions in llmopt's rules.py:
  - `i_sqrt_basis` poly branch: `"ansatz A(x)*sqrt(P) with polynomial A; match 2*A'*P + A*P' = 2*f*sqrt(P) in the poly ring"` (P spelled via sstr).
  - `i_sqrt_basis` log branch: `"ansatz (A(x) + B(x)*log(q))*sqrt(P) with polynomial A, B; differentiate, clear by 2*sqrt(P)*q, match coefficients"`.
  - `i_linear_basis`: `"ansatz sum of c_ij * x^j * m_i over basis {m1, m2, ...}; differentiate and equate coefficients"` — first 8 basis monomials by sstr, `", ..."` beyond.
  - All other rules: `null`.

**What is NOT ported:** the in-language token filter (`MathTokenizer` roundtrip) and the one-ply ration cap stay llmopt-side — they are corpus-composition policy, applied at shard-assembly time, not emission semantics. Native emission is complete and unfiltered; llmopt cuts prefixes.

**Tech stack:** existing `ax::search` chassis (`replay_verify` walk, `default_rules()` registry), `ax::sym::to_sstr`, JSONL writer from the gate tool. New tool `axiom-chain-emit`.

---

### Task D1: replay walk that records the state chain

`ax::search::replay_chain(root, history, rules) -> optional<vector<state>>` — same-label backtracking walk (replay_verify's structure), returning the full state sequence root..answer instead of a bool. RED: fixture test replays a known 3-ply history and asserts the state list matches the beam's; a corrupted history returns nullopt.
Commit `feat(search): replay_chain - state-sequence replay with same-label backtracking`.

### Task D2: syndrome + think annotation

`ax::search::annotate(cur, rules) -> {hints, think}`: hints = names of integral rules firing on the largest Integral node of cur (count_ops tiebreak: first in tree order); think = verbalized derivation for ansatz rules, message text matching llmopt's `_trace` strings byte-for-byte (fixture-gated against strings lifted from llmopt's rules.py — they are conventions, not decoration; the model trains on them). Native ansatz rules gain an optional trace out-param (no global mutable state — thread-local slot like the work budget).
RED: fixtures for each `_trace` branch; a non-ansatz rule yields null think.
Commit `feat(search): chain annotation - rule-fire syndrome hints + ansatz think traces`.

### Task D3: emit tool + 100-root gate artifact

`tools/chain_emit_main.cpp` -> `axiom-chain-emit <roots.jsonl> <out.jsonl> [budget] [plies] [width] [prior.tsv]`: for each root, solve with the markov3 config; on solve, replay_chain + annotate every step; write rows. Every (cur, nxt) pair re-verified through the oracle (diff-back mod const on Integral edges — emission is a farm artifact, soundness bar identical to the gate's). Dedup (cur, nxt) within the run. Emit `data/qual/phaseD_sample100.jsonl` over the first 100 solved qualification roots and hand to llmopt for shape-diff + dual-oracle adjudication.
Commit `feat(tools): axiom-chain-emit - farm-shard chain emission (Phase D)`.

---

## Self-review notes

- Soundness: emitted pairs pass the same edge verification as search; a row that fails diff-back is dropped and counted, never written. UNDECIDED drops the row (conservative).
- Schema fidelity beats elegance: `hints` order, `think` phrasing, and sstr spelling are all training-data conventions — fixture-gate them against llmopt's strings, don't paraphrase.
- Provenance: `source: axiom-*` keeps native rows distinguishable in mixed shards forever.
- Python bridge: once the tool qualifies, the same emit path can ship over pybind (`solve(..., emit_chain=True)`) — deferred until llmopt asks.
