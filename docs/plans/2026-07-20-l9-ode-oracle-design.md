# axiom L9 territory — ODE oracle + native generator (design)

> Companion to llmopt's `docs/superpowers/specs/2026-07-20-l9-territory.md`
> (their side: farm sequencing, mass-targeted diet). This is axiom's
> side: what the oracle and native generator need to make L9 (ODE) rows.
> Principle carried from the tan arc: **in new territory, budget
> normalization work equal to rule work** — most apparent unreachability
> is the verifier and the candidate speaking different dialects.

## L9 anatomy (from llmopt's problems.py `kind=="ode"` + odes.py)

A row is an equation `eq` plus initial conditions `(x0, y0[, y0'])`.
Verification is NOT compare-to-closed-form; it is **checkodesol**:
substitute the candidate solution into the equation and require the
residual to vanish, then check the ICs. Families in v1 (their odes.py):
- `make_linear_first_order`: y' + p(x)·y = q(x)
- `make_second_order_cc`: a·y'' + b·y' + c·y = f(x) (constant coeff)
- `make_separable_growth`: y' = k·xⁿ·y

Each solved row is 2–4 integrals + glue — exactly the diff/canonical
work axiom already accelerates.

## Key finding: no `diff` change is needed

The instinct is "axiom's diff throws on unknown functions, so we must
teach it y(x)→Derivative(y(x),x)." We do **not**. checkodesol
substitutes the candidate BEFORE differentiating:

```
eq:   Derivative(y(x), x) + p·y(x) − q          (y(x) = fn("y", x))
cand: y = g(x)
subst: y(x) → g(x);  Derivative(y(x),x) → diff(g,x);
       Derivative(y(x),(x,2)) → diff(diff(g,x),x)
check: canonical(subst(lhs) − subst(rhs)) == 0   (three-valued)
```

After substitution no unknown function remains — it is an ordinary
expression in x, and `canonical`/`equivalent` decide it with the
primitives that already exist. diff never touches y(x).

### Representational items (two small parser gaps, verified)

The parser/printer must round-trip the ODE carrier atoms. Checked
against `src/sym/parse.cpp`:
1. **`y(x)` — unknown function.** Rejected today: the parser allows only
   the 9 known function names (sin/cos/…); an unknown `y(` fails at
   `unknown function 'y'`. Need a reserved unknown-function symbol for
   the dependent variable (a single name `y`, treated as an opaque
   carrier atom — never differentiated, since substitution precedes
   diff). Printer must emit `y(x)`, not fold it.
2. **`Derivative(y(x), (x, 2))` — higher order.** The 3-arg
   `Derivative(y(x), x, x)` already parses (carrier spec allows 2–8
   args), but sympy's sstr prints the **tuple-limit** `(x, 2)`, which
   has no parse path. Add tuple-limit → repeated-limit desugaring so the
   two spellings intern to the same carrier.

Nothing else new: y(x) stays opaque, and every derivative in a verified
row is taken of the *candidate*, never of y.

## The normalization budget (where L9 will actually be won)

checkodesol keeps the arbitrary constants **symbolic** and requires the
residual to be identically zero — for all C. Two dialect classes will
otherwise read as UNDECIDED on correct solutions:

1. **Arbitrary constants C1, C2 as free parameters.** `canonical`
   already treats non-x symbols as constants, so the residual reduces
   as an identity in (x, C1, C2) structurally; the gap is the numeric
   sampler — `parameter_env` must bind C1, C2 to sample values, exactly
   as it binds other free symbols. Small, mechanical, but load-bearing:
   without it the whole family falls to sampling and half will sit
   UNDECIDED. This is `equivalent_mod_const` generalized from "one
   additive constant" to "the solution's structural constants."
2. **exp ↔ sinh/cosh (and the exp·trig euler forms).** Second-order CC
   with real distinct roots is `C1·exp(r1·x) + C2·exp(r2·x)`; the same
   solution spelled with sinh/cosh, or dsolve's vs the model's constant
   basis, is equal but structurally different. Fix is a `untan`-class
   rewrite at the head of `canonical`: `sinh(u)→(exp(u)−exp(−u))/2`,
   `cosh(u)→(exp(u)+exp(−u))/2`, **guarded by a `has_hyper` predicate**
   (untan's lesson: an unconditional rebuild perturbs `as_ratio`
   ordering and regresses unrelated rows). Complex-root spellings
   (exp·cos, exp·sin) are the existing euler territory.

Implicit solutions `G(x,y)=C` (separable ODEs that don't solve for y)
are **out of scope v1** — matches llmopt's odes.py, which only farms
explicit-solvable forms. Deferred, not denied.

## Verify surface (oracle)

`check_odesol(eq, y_name, candidate, x) -> verdict`:
- structural substitution of `y(x)` and each `Derivative(y(x), …)`
  carrier by the candidate's matching derivative,
- `equivalent(subst_lhs, subst_rhs, x)` (three-valued; UNDECIDED never
  passes, same soundness contract as every edge),
- IC checks by numeric eval: `|g(x0) − y0|` and, for 2nd order,
  `|g'(x0) − y0'|`, below the oracle's witness tolerance.
Fixture-gate against sympy's checkodesol on a seed sample before trust,
per house rule (llmopt owns adjudication).

## Native generator port (mathgen)

Port the three odes.py makers to C++ mathgen with the string-seed
bit-exact discipline of the L1–L8 port (CPython `random.Random`,
collision guard, `Problem` shape). Row schema adds the ODE carrier
fields. These feed the pure-native farm subset while the hybrid farms
the rest.

## Slot-fire-rate logging (llmopt's ask)

Add a per-run counter to `solve`/`emit_chain`: how many nodes the
external `int_rules` (i_heurisch) fired on, and how many of those became
the emitted step. Exposed in the return dict (`slot_fires`,
`slot_decisive`). This prices the hybrid's heurisch dependency per
family — the L9 analog of the budget-insensitivity diagnostic, and the
signal for llmopt's mass-targeted diet to tell HEAVY (native-closed)
from LIGHT (slot-dependent) ODE families.

## Sequence

1. Higher-order Derivative carrier round-trip (parser/printer) + fixture.
2. `check_odesol` + arbitrary-constant sampler env + hyper normalization,
   fixture-gated vs sympy checkodesol.
3. Native mathgen ODE makers (parity vs llmopt odes.py).
4. Slot-fire-rate counters in solve/emit_chain.
5. Farm-gate an L9 sample; llmopt adjudicates row shape + dual-oracle.

## Self-review

- Soundness unchanged: check_odesol routes through `equivalent`, so
  UNDECIDED rejects; arbitrary-constant sampling only ADDS decidability,
  it cannot turn a non-identity into a pass (a false C-dependence leaves
  a non-zero residual at almost every sample).
- No new diff semantics, no unknown-function calculus — the substitution
  framing keeps the whole port inside proven primitives.
- The hyper rewrite is the tan arc's method reused, guard included; if a
  third such dialect appears, the pattern is now standard, not novel.
