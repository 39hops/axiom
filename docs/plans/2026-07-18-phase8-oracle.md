# axiom Phase 8 Implementation Plan (llmopt oracle: parser, equivalence, parity harness)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. TDD: RED before implementation on every task.

**Goal:** Make ax::sym usable as llmopt's native verification oracle: a parser for sympy `sstr` text, a canonicalizer (`canonical`) strong enough for the llmopt expression zoo, a sound three-valued `equivalent` / `equivalent_mod_const`, and a JSONL dump-file parity harness. See `docs/specs/2026-07-18-llmopt-oracle.md` for the contract.

**Architecture:** Sound-by-construction verdicts: EQUIVALENT only from structural identity of canonical forms (hash-cons pointer equality); NOT_EQUIVALENT only from a robust numeric witness; otherwise UNDECIDED. The harness is a standalone executable (`axiom-oracle`) with row-isolated error handling — one bad row emits an error record, never kills the run.

**Tech Stack:** C++23 STL, GoogleTest, existing ax::sym (expr, calc, expand, poly, print).

**File map:**
- `include/ax/sym/parse.hpp` + `src/sym/parse.cpp` — sstr parser
- `include/ax/sym/oracle.hpp` + `src/sym/oracle.cpp` — canonical, equivalent, equivalent_mod_const
- `src/sym/expr.cpp`, `print.cpp`, `calc.cpp` — pi/E constant support as needed
- `tools/oracle_main.cpp` + `src/sym/jsonl.{hpp→include}/…` — harness executable + minimal JSONL io
- Tests: `tests/sym/parse_test.cpp`, `oracle_test.cpp`, `jsonl_test.cpp`

---

### Task 1: sympy-sstr parser

```cpp
// include/ax/sym/parse.hpp
/** Parse sympy sstr text into an expr. Throws parse_error (derives
    std::runtime_error, carries byte offset) on malformed input, unknown
    function names, or unrepresentable atoms (oo, zoo, nan, I). */
expr parse(std::string_view src);
```

Recursive descent per the spec grammar: `+ -` < `* /` < unary `-` < `**` (right-assoc; `-x**2` is `-(x**2)`, `2**-x` legal). Known fns: sin cos tan exp log sqrt atan asin acos. `pi`/`E` → reserved symbols (eval binds them; printers emit `pi`/`E`). Decimal floats → exact rational. Whitespace-insensitive.

Tests: literals (`42`, `-7`, `3/4` parses as div → rational num, `0.25` == 1/4); precedence (`1+2*3`=7 by eval, `2**3**2`=512, `-x**2` vs `(-x)**2`, `a-b-c` left-assoc, `6/2/3` = 1); functions incl. nesting `sin(log(x*(2*x-3)))`; the full monster from the spec parses and evals at x=2 to a finite cross-checked value (computed with std::sin/log inline in the test, not hardcoded); round-trip property: for a family of randomly built exprs, `parse(to_string(e)).same(e)` after simplify; errors: `foo(x)`, `oo`, `1++`, `sin(x`, empty string throw parse_error with sensible offset.
Commit `feat(sym): sympy-sstr expression parser`.

---

### Task 2: canonical() — zoo-targeted canonicalization

```cpp
// include/ax/sym/oracle.hpp
/** Canonical form for equivalence checking: expand-and-collect polynomial
    parts, rational-function normalization p/q with poly gcd cancellation
    (univariate over rationals via ax::sym::poly), recursive canonicalization
    of fn arguments and exponents. Idempotent: canonical(canonical(e))
    same canonical(e). */
expr canonical(const expr& e, const expr& x);
```

Strategy: bottom-up. Rational-function subtrees in x (from_expr succeeds on numerator/denominator after expand) → divmod/gcd-cancelled p/q. Non-poly factors (fn nodes, irrational powers) treated as opaque atoms multiplied back in; their arguments canonicalized recursively. No trig identities in v1 — sin²+cos² stays UNDECIDED territory, documented.

Tests: `(x**2-1)/(x-1)` → `x+1` structurally; llmopt-shaped cancel `((2*x-3)*f)/(2*x-3)` with f a fn factor → f; diff-of-quotient recombination: canonical(diff((x**2+1)/(x-1))) equals canonical of the hand-expanded form; idempotence property on random exprs; expand-collect `(x+1)**2 - x**2 - 2*x - 1` → 0; leaves `sin(x)**2 + cos(x)**2` alone (documented incompleteness).
Commit `feat(sym): canonical form - expand, collect, rational cancellation`.

---

### Task 3: equivalent / equivalent_mod_const

```cpp
enum class verdict { equivalent, not_equivalent, undecided };

/** Sound three-valued equivalence of a and b as functions of x.
    equivalent: canonical(a-b) is literal 0 (structural proof).
    not_equivalent: a numeric witness point where both sides evaluate
    finitely and differ by more than atol+rtol*scale, confirmed at a
    second nearby point (guards conditioning flukes).
    undecided: everything else. */
verdict equivalent(const expr& a, const expr& b, const expr& x);

/** Integral-step check: diff(candidate,x) equivalent to integrand. */
verdict equivalent_mod_const(const expr& candidate, const expr& integrand,
                             const expr& x);
```

Witness sampling: deterministic point set (e.g. ±{0.37, 1.71, 2.93, 5.41,…} plus small offsets), skip points where either side throws/returns non-finite (domain holes), require ≥ 2 disagreeing points for NOT_EQUIVALENT and ≥ 6 valid points sampled overall before allowing UNDECIDED→NOT_EQUIVALENT escalation; if fewer than 3 valid points exist, verdict is undecided. Symbols other than x are bound to fixed deterministic values (documented: equivalence is then modulo those parameters — the zoo is univariate).

Tests: proof path `equivalent(2*x+2, 2*(x+1))` = equivalent; witness path `x**2` vs `x**3` = not_equivalent; incompleteness honest: `sin(x)**2+cos(x)**2` vs `1` = undecided (numerics agree, no structural proof — MUST NOT be equivalent); domain-hole handling `log(x)` vs `log(-x)` (disjoint domains → undecided, not crash); mod-const: candidate `x**2/2 + 7` vs integrand `x` → equivalent; wrong candidate `x**2` vs `x` → not_equivalent; llmopt monster: equivalent_mod_const of `5*x*sin(log(x*(2*x-3)))/…`-family pair from the spec example verified against its true integrand computed by ax diff in-test.
Commit `feat(sym): three-valued equivalence oracle - EQUIVALENT/NOT_EQUIVALENT/UNDECIDED`.

---

### Task 4: JSONL parity harness

Minimal STL-only JSONL io (flat objects, string values, full string escaping both directions) + `tools/oracle_main.cpp` → target `axiom-oracle`:

```
axiom-oracle <in.jsonl> <out.jsonl>   # or - for stdin/stdout
```

Per spec protocol: tasks `diff`, `equiv`, `equiv_mod_const`; every input row produces exactly one output row; any exception in a row → `{"id":…, "status":"error", "error":…}`; unknown task → error row; malformed JSON line → error row with line number as id fallback. diff results printed with ax::sym to_string (parseable back by Task 1 parser — round-trip tested).

Tests: jsonl unit tests (escape round-trip incl. `"`, `\`, newline; missing field; garbage line); end-to-end: write a temp in.jsonl with all three tasks + one poisoned row, run the harness entry function (factored `int run_oracle(std::istream&, std::ostream&)` so tests don't spawn processes), assert verdicts and the error row; ordering preserved; 1k-row synthetic throughput smoke (generated exprs, no timing assert, just completes).
Commit `feat(sym): axiom-oracle JSONL parity harness`.

---

### Task 5: docs + parity handoff

README module table + status; spec cross-links; a `docs/specs` note on how llmopt runs the parity audit (dump ~10⁵ farm rows → run both oracles → diff verdicts; axiom-EQUIVALENT vs sympy-unequal rows are soundness bugs, UNDECIDED rate is the fallback tax metric). Commit `docs: Phase 8 oracle status and parity protocol`.

---

## Self-review notes

- Soundness invariant enforced by construction and by tests that assert UNDECIDED where numerics agree without proof (the sin²+cos² sentinel).
- No hardcoded oracle constants: every numeric expectation in tests computed in-test via std math or ax cross-modules.
- Build discipline: `ninja -j4` max (WSL compute job sharing the machine).
- pybind11 deferred until the parity run passes (spec).
