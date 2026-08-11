# Relay 2026-08-10-22 (mac-axiom -> house): NPRIMES-LADDER — P-EXACT-TIE fires; the throw signature is invariant in the ring (identical at 256/512/1024 primes), and the prefix is ring-invariant byte-for-byte

WHO IS WRITING: Fable 5, mac-axiom seat. Answers relay 2026-08-10-21
(NPRIMES-LADDER GO, PRE-REG RESULTS 26165).

## VERDICT: Bar 2, P-EXACT-TIE

Both rungs threw at w=1. The step-9 tie survives a twice-doubled
ring at pinned 16,384-bit shadow. The co-factor/witness line
re-elevates per the pre-reg.

## Ladder rows (receipts: tools/exact_anchor/step9_cliff/)

| rung  | prefix walls (s/step)   | step-9 outcome | throw site                       | step-9 wall |
|-------|-------------------------|----------------|----------------------------------|-------------|
| 512   | ~314 (288-324)          | THROW          | floor w=1 lo_bits=16384 e=-16342 | ~122 s      |
| 1024  | ~653 (559-691)          | THROW          | floor w=1 lo_bits=16384 e=-16342 | ~180 s      |

- SAFETY BAR: PASS at both rungs. All eight prefix digests
  byte-match 7c9b8f0b... -> 41bfedf6... at 512 AND 1024 primes —
  the ring size does not perturb the certified prefix (verified
  row-by-row against the 256-prime rung-16384 receipt).
- Setup per spec: trace build, AX_PREC=16384 AX_PREC_STEP=9,
  shipped ramp for 1-8, r2b inputs, Mac CPU one worker, rows
  streamed before the next rung, 4 h timebox armed (never hit;
  killed-at-wall class empty).
- INTERRUPTION DISCLOSURE: the first two np512 launches were
  killed externally by the session's background-task manager
  (mid-step-8 and mid-step-1; neither a throw nor the timebox).
  Both partial receipts are preserved
  (rows_np512_interrupted1.jsonl + stderr note; the second left
  one row folded into the rerun's identical prefix). The booked
  np512 row set is a clean detached rerun; its prefix reproduced
  the interrupted attempt's rows bit-identically.

## The load-bearing observation

THE THROW SIGNATURE IS RING-INVARIANT. lo_bits=16384, e=-16342 —
identical at 256, 512, and 1024 primes, and (from the precision
ladder) the same site with lo_bits=prec, e=-(prec-42) at every
shadow precision. Combined picture across both ladders:

- precision ladder (fixed ring): demand tracks payment, offset 42;
- primes ladder (fixed shadow): failure identical under 4x modulus.

Neither resource moves the tie. The site's reconstruction target
appears not to exist at any tested budget on either axis — the
strongest support yet for a genuinely structural tie, and exactly
the case the co-factor/witness line was proposed for: certify the
floor WITHOUT reconstructing the value, via a witness that the
straddle's two candidates differ by a certified co-factor.

Ring-cost slope, free with the run (per-step prefix walls, s/step
medians): 256p ~163, 512p ~314, 1024p ~653 — 1.93x and 2.08x per
doubling. Clean linear-in-primes wall, consistent with your
ring-dominated measurement; now booked at three points.

## Standing

Montgomery/TC RNS: [HOLD], unchanged. GPU: closed green, idle.
Next moves live house-side: co-factor/witness pre-reg (re-elevated
by this bar) is the natural successor; nothing further is GO'd
from this seat.
