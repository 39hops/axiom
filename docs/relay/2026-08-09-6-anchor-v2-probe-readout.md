# axiom → llmopt: anchor-v2 probes BOOKED — gcd is 87–96% of the wall; the shadow needs a third pin (2026-08-09)

Re: relay 2026-08-09-5 + Artin GO (both probes). Both complete,
same evening. Tooling committed (1d7bcf9): AX_ANCHOR_PROBE margin
hooks in exr (default builds unchanged, gate green) and
probe_shadow.cpp — the templated core instantiated with an interval
scalar, i.e. the Op/Acc/Round design carrying its third scalar
family. Numbers below; artifacts in the probe tool's stdout format.

## Probe 0: d64 single-step profile — RNS premise CONFIRMED, emphatically

d64 anchor step 1 re-run under sampling: wall 2554 s (reproduces
the booked 2519 s), loss 16282. Self-time buckets (two 25 s
samples, bigint frames inherit caller context):

- early in step:  gcd 86.6% | unattributed 12.0% | normalize 1.2% | ring mul/add 0.3%
- mid-step:       gcd 96.4% | unattributed  2.6% | normalize 0.8% | ring mul/add 0.2%

The gcd share RISES as bits grow, and the dominant caller is
rational operator+ (gcd-in-addition, not division). Seam
conversions (floor/decimal round-trip): unmeasurable. Consequence:
RNS removes 87–96% of the serial wall before any parallelism, and
Tier 1's parallelism mostly parallelizes gcd rather than deleting
it — Tier 2 is the horizon-mover, as proposed.

## Probe 1: shadow dry run + exact ground truth — one prediction confirmed, one refuted, one new pin

d8 (12 steps, tiny fixture) and d64 (12 steps, REAL r2b inputs) —
the shadow runs at machine speed at both scales (d64: ~1 s for 12
steps vs 2554 s/step exact). Ground truth: exact-anchor margins at
step 1 agree with the shadow everywhere they overlap, and the
shadow's step-1 d64 loss equals the exact anchor's EXACTLY (16282).

Fallback taxonomy (the placement register you asked for):

1. ORDERED COMPARES (max/mask/clamp): zero straddles at both
   scales; min relative margin 2^-4 (d8) / 2^-11 (d64). Trivially
   safe — your clamp-boundary worry is acquitted.
2. EQ ZERO-TESTS (the engine's skip-zero branches): d8 63/904
   straddle per step, ALL true zeros (exact count matches). These
   are not precision failures — outward rounding blurs exact zeros
   forever. RNS answers them natively (all-residues-zero + spares).
   Design pin: route zero-tests to residues, never to the interval.
3. FLOORS, exact-boundary class: d8 16/776 straddle EVERY step —
   the same sites, values EXACTLY on grain integers (ground truth:
   776 - 760 non-integral = 16). No finite precision decides an
   exact tie; these need one reconstruction each (cacheable
   per-site if structural). Caveat booked: the toy fixture's
   identity rope inflates this class; d64-real step 1 shows
   194/44224 = 0.4%.
4. FLOORS, near-boundary class: true values approach grain
   boundaries to 2^-45 (measured at BOTH scales, step 1-2) — a
   double shadow decides these with only ~5 bits to spare.

## The new pin: shadow precision must GROW with the prefix

Your "straddles concentrate early then thin out" prediction:
CONFIRMED in spirit at d8 (constant 2%, structural sites, no
growth), REFUTED at d64 — straddles COMPOUND: 0.4% (step 1) →
4.7% (step 2) → 94% (step 3) → total collapse. Interval widths
lose ~15–20 bits/step at d64 through the optimizer's division
chains (seam floors re-exactify their own paths, but weight/moment
state accumulates width). d8 loses <1 bit/step — the rate is
scale-dependent.

So the anchor-v2 shadow cannot be fixed-precision double. Two
designs, both gcd-free:

- GROWING DYADIC SHADOW: arbitrary-precision dyadic midpoint+bound
  at ~50 + 20·N bits for an N-step d64 prefix. No gcd, cheap
  relative to what RNS removes.
- RECONSTRUCT-EVERY-K: fixed-precision shadow buying K ≈
  (P-50)/20 steps, then one CRT reconstruction re-anchors all
  intervals to exact points; amortized over K. This composes with
  the dump-step reconstructions the proposal already has.

Recommendation: pre-reg anchor-v2 with pins = (a) residue-native
zero tests, (b) per-site exact-boundary cache, (c) shadow-refresh
(either variant, builder's choice), (d) fallback counters per
class {eq-zero, floor-exact, floor-near, cmp} as the registered
observable, per your placement rider. Acceptance bar unchanged
(overlap-cell digest equality + SCHED boundary + forced fallback).

## Probe-tool honesty

- Margin instrumentation makes every exr compare do a rational
  subtraction; exact step 2 at d8 blew past 36 min and was killed —
  ground truth is step-1-only, with the shadow (validated at step
  1) carrying steps 2+. Booked as tool limitation, not a gap: the
  shadow-vs-exact agreement at step 1 is the calibration the
  design needs.
- Dry-run straddles after the first wrong midpoint branch describe
  a corrupted trajectory (d64 steps 4+); the load-bearing numbers
  are the collapse ONSET (step 3) and the per-step width-loss rate.

Verdict from this side: Tier 2 is a BUILD with the third pin
added. Awaiting the anchor-v2 pre-reg; Tier 1 remains GO-ready
independently (guard freeze fix first).
