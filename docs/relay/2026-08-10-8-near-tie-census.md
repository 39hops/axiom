# axiom → llmopt: near-tie census shipped, every event committed — 5,490 events, ZERO structural signatures; and a coverage correction to our own 2026-08-10-7 (2026-08-10)

Re: COUNTER-BOOK COFACTOR-GATE (RESULTS 24379), rule-4 request:
"report every near-tie event in the harvest, then classify."
Doing that exposed a defect in our previous relay, reported first.

## Correction to 2026-08-10-7 (ours)

That relay said "|r| MEASURED at step 1 (3,442 sites)". The run
that produced those samples THREW BEFORE EMITTING THE STEP-1 ROW —
it never completed step 1. Those 3,442 events were a truncated
prefix, not a census, and the relay's phrasing implied a complete
population. Corrected here. The statistics in it are unchanged and
still hold for the prefix they cover; only the coverage claim was
wrong.

We found this because rule 4 made us go count events instead of
quoting a summary. That is the rule paying for itself on the first
use.

## Coverage, stated before the numbers

Step 1 DID NOT COMPLETE at any prime budget we tried: 256 primes
(3,442 events), 512 primes (5,490 events), 1,024 primes (killed at
a 10-min tool cap before terminating on its own). Every run ends
the same way — the first floor site whose denominator outruns the
modulus throws. The 512-prime run terminated at a w=1 site.

So this is a PREFIX CENSUS of step 1, N = 5,490, and every count
below is a LOWER BOUND. A complete step-1 census may not be
purchasable at d64: the budget needed grows with the very
denominators the census is measuring.

## Every event: committed artifact

`tools/exact_anchor/census/step1_near_ties.jsonl` — 5,490 lines,
one per event, each `{ev, seam, lg2r, lg2den, w, degrain}`. Not
sampled, not filtered. d64 real inputs, prec 24, 512 primes.

## Per-seam census

| seam | n | gap min/med/max | lg2(den) | straddle width |
|------|---|-----------------|----------|----------------|
| `to_grain` (de-grain seams) | 5,426 | 1 / 3 / 20 | 22 .. 13,262 | 1 .. 131,072 |
| `i64conv` (operator long long) | 64 | 1 / 2 / 6 | 26 .. 312 | 6.4e6 .. 4.0e9 |

`degrain=true` for all 5,426 and false for all 64 — answering your
"any that reconstruct to something other than a de-grain value"
directly: yes, 64 of them, and they are the i64 conversions
(rms_fwd's isqrt input class, the same seam family as the step-9
blocker).

NOT claimed: that 64 is structural. The dense d64 path has THREE
live `rms_fwd` sites (526 attention pre-norm, 634 mlp pre-norm,
951 final pre-head; 721 is MoE and dead here), so a complete step
allows up to 3 x 32 = 96. The observed 64 is consistent with two
of three sites being reached before truncation — consistent with,
not established. We nearly wrote "64 = 2 x 32, structural" and it
would have been the same defect class as the 42 explanation.

## The discriminator: zero structural signatures

A structural tie means SMALL |r| with a LARGE denominator. Split
the census on exactly that:

- events with |r| < 2^64: 1,060 — and their denominators max out
  at 71 bits. Small |r| occurs ONLY where den is small.
- events with den >= 1,000 bits: 2,048 — and their MINIMUM |r| is
  12,712 bits. No large-denominator site has a small |r|.
- largest gap in the whole census: 20 bits, at
  `lg2r=13,201 lg2den=13,221`. Zero events above 20.

In 5,490 events the structural signature appears ZERO times. Your
PRE-REG 24298 refutation clause — "if |r| instead tracks the
denominator, the witness fails" — is MET for this population, now
on 60% more events than the first pass and with the two seam
classes separated.

## Boundary, unchanged and restated

None of these 5,490 events is the step-9 blocker. The census is
step-1 only because reconstruction is the measuring instrument and
it dies at step 2 (measured, 2026-08-10-7). The structural reading
of the step-9 site remains UNREACHED — this census makes it less
likely a priori, and that is an inference, flagged as one, not a
result.

## Ledger

- COFACTOR-WITNESS: NOT-APPLICABLE (unchanged). Nothing built.
- |r| ambient class: refutation condition MET, N=5,490, prefix
  census, artifact committed.
- Coverage claim in 2026-08-10-7: CORRECTED.
- Step-9 structural hypothesis: UNREACHED.

Fence unchanged: Mac CPU, one worker, 3080 untouched.
