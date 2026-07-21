# 2026-07-21 — Truncated power-series tranche (llmopt relay, Artin-GO)

Goal: diversify L9b chains beyond ansatz methods with a formal truncated
power-series module — exact ℚ, no floats, no sympy in the hot path — and
open series-space rewrites (erf/Liouville jailbreak later).

## Rungs

1. `ax::sym::series` — dense truncated series over `rational`.
   Invariant: `coeffs.size() == order` (N coefficients a_0..a_{N-1},
   truncation O(x^N)). Ops: add/sub (order = min), Cauchy product
   (O(N^2)), termwise derivative (order-1) / integrate (order+1, const 0),
   reciprocal (c_0 != 0), integer pow, composition (inner constant term
   must be 0, Horner). `of_expr(e, x, N)`: structural Maclaurin —
   num / x / add / mul / integer pow / exp / sin / cos / log(1+u) /
   sqrt(1+u) with binomial(1/2); anything else throws std::domain_error
   (mapped to UNDECIDED downstream). Immutable value type like `poly`.

2. Oracle `check_odesol_series(eq, candidate, x)` — substitute the
   candidate series for y(x) and Derivative(y,x..n) carriers in the Eq,
   expand every other subexpression via of_expr, compute the exact
   residual series. Verdicts (order replaces the IC-check as keystone):
   - any nonzero residual coefficient → NOT_EQUIVALENT (exact
     arithmetic: a nonzero coefficient is a structural witness);
   - all zero through the natural truncation M → EQUIVALENT_TO_ORDER
     with order M (residual is O(x^M));
   - expansion failure → UNDECIDED_BEYOND_ORDER. Never guesses valid.

3. `ax::mathgen::series_solve(ode_problem, N)` — derive the coefficient
   recurrence structurally from the Eq carrier: move rhs left, collect
   Add terms by Derivative order i with expr coefficients c_i(x); pure
   terms form q(x); top order k needs numeric constant c_k. Since
   i < k, j >= 0 gives index n-j+i < n+k, the unknown a_{n+k} appears
   exactly once with factor c_k[0]*(n+k)!/n! — solve linearly, one exact
   rational step per coefficient. ICs seed a_0 = y0 (and a_1 = yp0 for
   k = 2). Each step is a chain row.

4. `axiom-series-sample <out.jsonl> [seeds_per_cell] [order]` — farm-gate
   emitter. One row per coefficient step:
   {family, level, seed, order, ode_order, n, a_n, cur, nxt, verdict,
    residual_order}. `a_n` exact-rational text is the byte-exact
   adjudication contract vs sympy series(); cur/nxt are partial-sum sstr
   + textual " + O(x**m)" marker (training-row shape, not byte-contract).
   Every problem double-checked before writing: recurrence coefficients
   == of_expr(p.sol) coefficients, and rung-2 oracle on the built series.
   Failing rows written with honest verdict, never dropped.

## Second item (separate investigation, not this tranche's code)

Relay claims the L7 pre-expand cancellation short-circuit costs 7-8
solves; session memory records 82e079b measured null twice and 757b7e8
(best-of-both canonical) as the real fix. Needs fresh measurement before
touching canonical; any canonical change re-runs the 74k audit.
