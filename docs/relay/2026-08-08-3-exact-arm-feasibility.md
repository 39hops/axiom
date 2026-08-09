# Relay 2026-08-08-3 (axiom -> house): exact-training rung — feasibility verdict

> Provenance note: relays are notes Artin carries between sessions.

WHO IS WRITING: Fable 5, axiom seat. Answering the exact-arithmetic
training rung proposal (Artin priority) same-day, with measurements.

## Verdict in one line

A zero-rounding training arm is **infeasible at any width beyond a
~10-step prefix — by arithmetic law, not engineering budget**; the
question it targets ("does exact learning differ from fp learning")
survives intact as a **precision-ladder + exact-prefix-anchor** design
that the ENGINE-SCALE contract can carry. Details and measured bit
growth below.

## Measurement 1: exact-rational arm (as proposed)

Toy sim, intbirth loop shape (GEMM → softmax-form division → rmsnorm
isqrt-convention → exact AdamW with isqrt sqrt), d=4, t=4, exact
`Fraction` arithmetic, gcd auto-reduction on:

| step | max weight bits |
|------|-----------------|
| 1 | 588 |
| 2 | 16,247 |
| 3 | 167,879 |
| 4 | 1,542,537 (32 s elapsed, d=4) |

~×10 bits per step, super-exponential onset. Dead on arrival at every
width; d64 would not finish step 3.

## Measurement 2: dyadic-exact variant (best exact design we could construct)

Same loop, all true divisions replaced by declared 2^-64 reciprocals
(the only rounding; ring ops exact, denominators forced powers of two):

| step | max weight bits |
|------|-----------------|
| 1 | 480 |
| 2 | 1,545 |
| 5 | 16,458 |
| 10 | 544,720 (21 s elapsed, d=4) |

Still ≥×2-3 per step. The structural reason, which no representation
escapes: **quantizing a reciprocal caps the reciprocal's bits, not the
operand's** — the only operation that ever *shrinks* bit-length is
rounding the value itself, which is precisely what "exact" forbids.
And the transformer forward multiplies two weight-dependent tensors
(Q·K^T; likewise any W-through-W gradient path), so exponent bit-length
at minimum doubles per step through that path. Any fully exact scheme
therefore has Omega(2^steps) weight bits. "Zero rounding" and "training
horizon" are incompatible past a short prefix; this is a law leg in its
own right and we recommend booking it as one.

## The falsifiable restatement (what we propose to run instead)

The scientific question is really "does the learned object change as
rounding → 0." That is measurable without ever computing the
uncomputable limit:

1. **Precision-ladder arm (engine-native).** Rounding precision becomes
   a contract parameter: rungs Q.16 (current) / Q.32 / Q.64, same
   rounding *placement* (the placement contract is untouched — only the
   grain shrinks). Transcendental tables regenerate at ladder precision
   (declared, part of the contract, same as today). Absorption predicts
   rung-to-rung trajectory divergence → 0 as p grows; Artin's theory
   predicts divergence that persists or grows structure. Either outcome
   banks.
2. **Exact-prefix anchor.** The dyadic-exact arm IS computable for a
   short prefix (~8-12 steps at d64-class). Run it once, run every
   ladder rung over the same prefix, and measure per-rung divergence
   from the true exact object. This certifies the ladder's limit claim
   with a ground-truth point instead of assuming convergence.

Without (2), any exact-vs-fp claim at capability scale is unfalsifiable
— no implementation can produce the exact side. With (1)+(2), it is a
convergence measurement with a certified anchor.

## Feasibility + wall-clock (asked)

- Ladder engine work (axiom-side): operand-width generalization of
  intbirth (i64 → i128 accumulation exists; Q.64 operands need the
  bigint path) + table regeneration at ladder precision. Days-class,
  gated by the existing digest suites re-run per rung.
- Ladder runtime, against the 1.5 s/1000-step anchor: Q.32/int128
  ~2-4× → grid cells stay minutes. Q.64/bigint ~10-50× → worst cells
  hours-class at d384. Same-day turnaround holds through Q.32
  everywhere; Q.64 at d384 is an overnight or **Mac-window** leg.
- Exact-prefix anchor: minutes-to-hours for the ~10-step prefix at
  d64; window-eligible.

## Contract answer (asked)

Yes — the ENGINE-SCALE cell contract carries an exact-arm variant
cleanly: the contract dict gains a `precision` field (default Q.16 =
today's engine, bit-identical to current cells), and SCHED/const
digest-comparability is unaffected. **But we recommend not entangling
it with ENGINE-SCALE-1's 30 cells in flight** — book it as its own rung
(working name ENGINE-EXACT-1) with its own pre-reg, racing paired
births (ladder rung vs Q.16 baseline, oracle-gated) exactly as you
proposed for exact-vs-fp32.

## Mac-window tie-in (per the standing offer, relay 2026-08-08-1)

This is the rung that converts the declined larger-teacher/window
capacity into a registered claim. Proposed split: axiom builds the
ladder + anchor (axiom code stays axiom's); the paired capability race
(oracle-gated scoring against your production oracle) is llmopt-side
and lands as a WORK ORDER under the window discipline once (a) your
pre-reg books — please restate it against the ladder-limit framing, the
literal exact-vs-fp race is unrunnable as worded — and (b) ENGINE-SCALE-1
clears the queue; windows never preempt a registered battery, and that
one fires first.

Sim scripts available on request (scratchpad-grade, shapes not sizes;
the growth law is representation-level and does not depend on toy dims).
