# axiom as llmopt's native verification oracle — design spec

## Purpose

llmopt (github.com/39hops/llmopt) is an oracle-verified derivation-search
engine over calculus rewrites. Its farm profile shows the binding cost is
sympy's Python-object churn inside the verification loop:
differentiate / simplify / cancel called on every candidate edge. llmopt
**never trusts `integrate`** — it verifies by differentiation and
equivalence-mod-constant. That verification loop is exactly the shape of a
fast native CAS kernel, and it is the one axiom is designated to replace.

axiom's job is *not* to out-integrate sympy. It is to answer, fast and
soundly, three questions:

1. `diff(expr, x)` — symbolic derivative (already complete, Phase 5).
2. `canonical(expr)` — canonicalize/cancel far enough that equal things
   usually become identical.
3. `equivalent(a, b)` — a three-valued verdict: `EQUIVALENT`,
   `NOT_EQUIVALENT`, or `UNDECIDED`.

## Soundness contract (non-negotiable)

- `EQUIVALENT` is emitted only on a *structural proof*: the canonical forms
  are the same hash-consed node. Never on numeric agreement alone.
- `NOT_EQUIVALENT` is emitted only on a *numeric witness*: a sample point in
  both domains where the values differ beyond tolerance (with interval
  cross-checks to rule out conditioning artifacts, see plan).
- Everything else is `UNDECIDED`. axiom never guesses "valid". Incomplete
  is documented, not papered over.

llmopt additionally verifies integrals as equality-mod-constant:
`equivalent_mod_const(d/dx candidate, integrand)` reduces to
`equivalent(diff(candidate, x), integrand)` — differentiation first, never
integration of the difference (llmopt's own hard-won rule; sympy pathology
#2 in their RESULTS).

## Input surface: sympy `sstr` text

llmopt's dump files carry expressions in sympy's `sstr` print format, e.g.

```
5*((3 - 4*x)*sin(log(x*(2*x - 3))) + (2*x - 3)*cos(log(x*(2*x - 3))))/(2*x - 3)
```

Grammar (the "expression zoo": compositions of polynomials, rationals,
sin/cos/tan, exp/log, atan/asin/acos, sqrt, integer and rational powers):

```
expr    := term  (('+' | '-') term)*
term    := unary (('*' | '/') unary)*
unary   := ('-' | '+') unary | power
power   := atom ['**' unary]              # right-associative
atom    := INT | FLOAT | NAME | NAME '(' expr ')' | '(' expr ')'
NAME    := [A-Za-z_][A-Za-z0-9_]*         # symbols and known fn names
```

- Constants: `pi`, `E` map to dedicated symbols with exact-aware printing
  and correct numeric evaluation; `oo`, `zoo`, `nan`, `I` are *rejected*
  (parse error) — real-valued expr cannot represent them, and a farm row
  containing them must surface as `PARSE_ERROR`, not a silent wrong answer.
- `FLOAT` decimals are converted exactly to rational (base-10 expansion).
- Unknown function names are a parse error (fail fast; the zoo is closed).

The parser is a deliverable with its own test suite, including round-trip
`parse(to_string(e)) same e` property tests and the monster nesting shapes
from llmopt's farm.

## Parity protocol: dump-file JSONL harness

axiom is **not the oracle of record** until it has been cross-checked
against sympy on ~10⁵ real farm rows. The audit interface is file-based
(the Python bridge comes only after parity):

- Input: one JSON object per line (JSONL). Tasks:
  - `{"id": <str>, "task": "diff",  "var": "x", "expr": <sstr>}`
  - `{"id": <str>, "task": "equiv", "var": "x", "lhs": <sstr>, "rhs": <sstr>}`
  - `{"id": <str>, "task": "equiv_mod_const", "var": "x", "candidate": <sstr>, "integrand": <sstr>}`
- Output: one JSON object per line, same `id` order:
  - diff: `{"id":..., "status": "ok", "result": <sstr-compatible text>}`
  - equiv: `{"id":..., "status": "ok", "verdict": "EQUIVALENT" | "NOT_EQUIVALENT" | "UNDECIDED"}`
  - any failure: `{"id":..., "status": "error", "error": <message>}` —
    a crashed row is a row-level error record, never a lost line.

Disagreements with sympy are *findings* to be reported upstream, not bugs
to silently paper over: the harness output is designed for diffing, and
UNDECIDED rows are expected (they cost llmopt a sympy fallback, not
correctness). Success criterion for oracle-of-record status: zero rows
where axiom says EQUIVALENT and sympy says unequal (soundness), and an
UNDECIDED rate low enough that the fallback tax is small (target < 5%,
measured not promised).

JSON handling is a minimal in-repo STL-only reader/writer for this flat
schema (strings/objects one level deep) — not a general JSON library.

## Out of scope (this phase)

- pybind11 bridge (explicitly *after* the parity run passes).
- Beating sympy's `simplify` in general; canonical() targets the zoo.
- Complex-valued or extended-real expressions.
- Modifying the llmopt repo.

## Phase renumbering

This work is Phase 8 (llmopt oracle). The proof kernel (prf) and CUDA
backend from the 2026-07-05 design spec shift to Phases 9 and 10.
