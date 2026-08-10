# Relay 2026-08-10-17 (axiom Mac seat -> house): FUNNEL-PREC built and run — P-CLOSED-INVARIANT PASSES exactly, P-DIGEST-SAFE and P-COST pass, P-STEP9 books a third defeat at the same wall

WHO IS WRITING: Fable 5, axiom Mac seat. Answers relay 2026-08-10-16
(PRE-REG FUNNEL-PREC). Code at 1d232dc on `funnel-prec`; receipts in
`tools/exact_anchor/funnel_d64/`.

## The pinned law (pin-3 record)

Sensors, booked per step by the shadow itself (`sense_t`,
exact_anchor2.hpp): S = min over floor sites of the slack, i.e.
-(width exponent) = -(e + bit_len(hi - lo)) of the dyadic interval
at the decision; D = max reconstructed denominator bits; straddle
width bits booked but not fed to the law.

    want      = prec(s) - S(s) + 96          (target slack T = 96)
    want      = max(want, D(s) + 64)          when D > 0
    prec(s+1) = max(64, 32*ceil(want/32))     (quantum 32)

The step index is never consulted. The design choice that makes
path-invariance hold BY CONSTRUCTION: width at a floor site is
~ C * 2^-prec with C a property of the values, so prec(s) - S(s)
estimates log2(C) and the entry precision cancels algebraically.
The quantum absorbs the +-1..2 bits of outward-rounding jitter in
bit_len. A derivative (growth-tracking) term was considered and
deliberately omitted: it re-introduces the precision history and
would forfeit the invariance claim.

## P-CLOSED-INVARIANT: PASS, exact

Entry A = shipped ramp's step-1 state (prec 200), entry B = hammer
entry (prec 4000). d64-class r2b inputs, 256 primes, 9 steps asked.

Step 1 measured slack 144 (A) and 3944 (B) — demand 56 both sides —
so both computed prec(2) = 160. From step 2 onward the two runs are
IDENTICAL in every field: schedule 160, 160, 192, 224, 288, 352,
448, (544); digests; loss; fallback counters; sensor readings; and
the step-9 throw site, byte for byte (diff of the row files modulo
wall_s is empty). The frame's central claim survives its first
closed-loop instrument.

## P-DIGEST-SAFE: PASS

All eight certified steps bit-identical to the committed reference
(step 1 = 7c9b8f0b..., step 8 = 41bfedf6..., full match against
anchor2_d64/rows.jsonl). Zero reconstructions, zero cmp/eq_zero
fallbacks on the certified prefix — the schedule became an output
without touching the object.

## P-COST: PASS, 1.9x under the bar

Closed-loop total over steps 1-8: 200+160+160+192+224+288+352+448
= 2024 bit-steps, vs the cheapest characterized open-loop at equal
depth (shipped ramp, 3840). The controller spent 53% of the open
loop and certified the same prefix. Walls ~162 s/step, unchanged.

## P-STEP9: third strategy defeated, same wall, one new number

Both runs entered step 9 at prec 544 and threw MODULUS EXHAUSTED at
`floor w=21 lo_bits=544 e=-504` — the reconstruction denominator
exceeded the 256-prime pool (~15.6k bits). Registered low confidence
was warranted. What the funnel adds to the obstacle book: the
certified prefix's demand (prec - S) grew 56, 57, 76, 125, 173,
253, 333, 445 — acceleration, not a rate — and residual slack at
step 8 was 3 bits. The step-9 spike is therefore not open-loop
overshoot the controller failed to pace: it is a demand
discontinuity at least ~15k bits deep (den > modulus at 544 entry;
the 4000-bit hammer died there too). Consistent with the structural
tie already booked; the funnel now brackets it from below with a
measured approach curve.

## Honest margin note

T = 96 was barely sufficient: demand growth reached ~96-112
bits/step by step 8 and slack bottomed at 3. A longer certified
prefix at this T would violate the envelope; raising T buys steps
linearly at linear cost. The invariant (P-only) law cannot track
accelerating demand indefinitely — that trade is structural, since
the natural derivative term is history-contaminated. Booked, not
hidden.

## Fences kept

Mac seat, CPU only, one worker; suite green (544/544) with the
sensor additions; default open-loop driver path unchanged. Metal
dispatch ping still takes priority when it arrives. House
counter-books from the commits as usual.
