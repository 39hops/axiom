# Relay 2026-08-11-1 (mac-axiom -> house): LEDGER CENSUS — P-LEDGER-BOUNDED fires on the distinct ledger (22,971 bits at step 9, flat per-step mass, no compounding); the all-events bracket blows the bound linearly; gate reading is house's call

WHO IS WRITING: Fable 5, mac-axiom seat. Answers relay 2026-08-11-0
(COFACTOR-WITNESS-2 arm 1, LEDGER CENSUS).

## Instrument (attested)

- Trace-only ledger in Exact2::div (exact_anchor2.hpp) + per-step
  ledger rows in run_anchor2.cpp, everything under
  #ifdef AX_ANCHOR2_TRACE.
- ATTESTATION: default-build OBJECT byte-identical pre/post edits
  (shasum 78c458fb... both sides). Note for the standard: BINARY
  hashes are non-attestable on macOS — a same-source double build
  differs by ~1,800 bytes (LC_UUID + stabs timestamps; control run
  receipt in the relay commit). Object-level comparison is the
  reproducible form of the COFACTOR-GATE attestation on this
  platform.
- D' is booked gcd-free as a bit-length ledger, BRACKETED both
  ways, registered before the run:
    dp_bits_distinct — sum of bit lengths over DISTINCT non-dyadic
      divisor values per step (lower ledger: the product a witness
      would actually clear denominators with);
    dp_bits_all — every non-dyadic division event (upper ledger:
      no scalar path passes more divisors than there are events).
  Dyadic (power-of-two point) divisors exempt per the pre-reg;
  wide-shadow divisors counted separately (ZERO occurred — every
  divisor on the path was a point integer).
- Config: 256 primes, shipped ramp 1-8, step 9 at 16,384
  (AX_PREC_STEP=9), r2b inputs, Mac CPU one worker. Prefix digests
  byte-match the certified ladder (safety standard carried over);
  the step-9 throw reproduced at the same site, and the partial
  step-9 ledger row was captured through the throw (the driver
  prints the ledger before rethrowing — the bar quantity survives
  the death of the step).

## The curves (receipts: step9_cliff/{rows,stderr}_census.*)

step | nondyadic evs | distinct | max bits | distinct cum | all cum
-----|---------------|----------|----------|--------------|--------
1    | 171,360       | 152      | 21       |  2,441       |  2.14M
2    | 237,216       | 175      | 21       |  5,253       |  4.82M
3    | 237,216       | 175      | 22       |  8,094       |  7.98M
4    | 237,184       | 170      | 22       | 10,858       | 11.25M
5    | 237,152       | 164      | 23       | 13,529       | 14.68M
6    | 237,120       | 161      | 23       | 16,164       | 18.21M
7    | 237,056       | 157      | 23       | 18,763       | 21.87M
8    | 237,056       | 156      | 26       | 21,332       | 25.63M
9*   |   5,696       |  92      | 21       | 22,971       | 25.73M
(* partial, through the throw)

## Reading (bar language quoted; the gate is yours to call)

- P-LEDGER-BOUNDED, on the DISTINCT ledger: FIRES. 22,971 <=
  32,768. Stronger than the bar asked: per-step distinct mass is
  FLAT (~2.4-2.8k bits/step, ~150-175 distinct divisor values,
  max single divisor 26 bits). There is NO compounding on this
  ledger — the c-compounding fear does not materialize in distinct
  divisor VALUES; the engine reuses the same isqrt outputs /
  softmax sums / AdamW denominators across lanes.
- The ALL-EVENTS bracket blows the bound (25.7M bits) but grows
  LINEARLY (~3.5M bits/step), not exponentially — the compounding
  is lane-multiplicity, not value growth.
- Honest scope on the fired bar: 22,971 bits is an upper bound on
  the product of ALL distinct divisors across all steps; any
  single scalar path's true denominator divides a product of far
  fewer (bounded divisions per path per step), so the witness-side
  number is likely SMALLER still. But note the ring arithmetic:
  256 primes x 61 bits ~ 15.6k-bit modulus — an in-ring sign test
  on a 22,971-bit ledger needs the 512-prime ring (~31.2k bits),
  which the NPRIMES ladder already priced at ~2x wall. Affordable,
  but the ring choice belongs in the arm-3 pre-reg.

## Standing

Arm 3 (P-SIGN-DECIDES) NOT built — awaiting the gate reading
house-side per the fence. Montgomery/TC RNS [HOLD]; GPU closed.
