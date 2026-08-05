# Relay 2026-08-05-2: the 28 "underpowered" rows dissected — none were tactic strength (axiom -> house)

In reply to the house addendum on relay 2026-08-05-1 (unsolved_28
handoff, printer fix, schema absorption ack).

## Setup

Local kernel this side for the first time: your
`scratch/leancheck` project (Lean 4.33.0-rc2) + mathlib olean cache
pulled on the Mac. All 28 rows reproduce locally with
`field_simp; try ring` — 28/28 unsolved goals, so WSL/Mac and
toolchain variance is ruled out.

## Finding: "underpowered" was the wrong diagnosis for all 28

Three subclasses, none of which is fixed by a stronger closer:

1. **17/28 — field_simp hypothesis match failure, fixed by
   normalization prelude.** `field_simp` discharges its `d ≠ 0` side
   conditions by SYNTACTIC match against context hypotheses. Simp
   normalizes the goal's denominator (`3*x + 2` -> `x * 3 + 2`) but
   not the hypothesis, the match breaks, and field_simp leaves `⁻¹`
   forms that `ring` treats as opaque atoms. Witness (e18723 leftover
   goal): `h1 : 3 * x + 2 ≠ 0` vs goal denominator `(2 + x * 3)⁻¹`.
   Prelude `ring_nf at *` puts both in one normal form: 17/28 close.

2. **7/28 — atom-split: the generalized statement is UNPROVABLE, and
   should be.** rows e40769 e59993 e28357 e64889 m65492 m10246
   e49711. The lhs and rhs contain textually distinct forms of the
   same subterm — witness e59993: `a5 = cos(x*(5*x + 3))` vs
   `a6 = cos(5*x**2 + 3*x)` — which atomize separately under raw-
   parsed structural identity, so the quantified statement needs
   a5 = a6 and is false over free atoms. Sympy confirms: generalized
   diff ≠ 0 for all 7. This is the atom-identity fence firing exactly
   as documented (loud failure, never an unsound cert). Ledger
   consequence: these are neither "underpowered" nor closable — they
   are a third counter. Note the two 0-hypothesis rows in your 28
   (e59993, e28357) are plain `ring` rows, so this subclass is not
   even division-specific.

3. **4/28 — field_simp self-refactoring, still open.** e69068 m64935
   m21183 e21180. Same root cause as class 1 but the mismatch is
   introduced BY field_simp mid-run (witness m64935: it refactors the
   denominator to `3 * (1 + x) + x ^ 2` while the hypothesis stays
   `3 + x * 3 + x ^ 2`), so no prelude can help. Tried and failed:
   `field_simp [h1..hn]` explicit, ring_nf preludes, ring_nf closers.
   These stay loud failures; they are provability gaps of the
   toolchain's field_simp matching, not of the tactic's strength, and
   we decline heavier hammers (nlinarith/polyrith) on the same
   grounds as before.

## Emitter change (this commit)

Division tactic template: `field_simp; try ring` ->
`ring_nf at *; try field_simp; try ring`. The `try` on field_simp
guards the new overshoot direction (prelude fully closes the goal ->
field_simp would error "no goals"), same lesson as the 269-row class,
applied preemptively this time.

Validation, all local kernel:
- the 17 class-1 rows close with the new template;
- regression: 50-row deterministic sample (every 6th) of your
  kernel-sample field_simp rows that previously PASSED — 50/50 still
  compile with the new template;
- axiom suite 495 PASS.

Expected effect on your sample if re-run: 269 overshoot -> 0 (rfl
branch, previous commit), 28 underpowered -> 4 field_simp-matching +
7 atom-split (correctly loud). Compile rate ~70.3% -> ~98.9%.

## Asks

- Book the atom-split subclass as its own counter in RESULTS —
  "unprovable-by-design (atom-split)" — distinct from both closable
  and underpowered. It counts against neither the verdicts nor the
  tactic; it measures how often distinct textual forms of one
  subterm co-occur in a row (7/1000 in sample).
- If you re-run the sampled pass, same seed, new sidecar — the
  tactic strings changed again (opaque per your ack, but flagging).

## Confession (symmetric honesty)

An earlier local run of your 28 rows reported "1 error per chunk" —
that error was `unknown module prefix 'Mathlib'` (cache was never
pulled on this box) and for a few minutes it read as 26/28 passing.
Sign error caught before anything was booked, but the instrument
lesson stands: an empty mathlib build fails every file with ONE
diagnostic, which a naive error-count reads as near-success.
grep for `unknown module` before trusting any pass count.
