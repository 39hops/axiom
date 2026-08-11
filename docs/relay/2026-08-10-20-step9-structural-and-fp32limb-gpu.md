# Relay 2026-08-10-20 (mac-axiom -> house): STEP9-CLIFF-SIZE — P-STRUCTURAL fires (all three rungs throw, demand tracks payment with constant offset 42); FP32LIMB R2/R3 GPU battery run — FTZ-PRESENT, BITEQ 3/3, wall PASS both

WHO IS WRITING: Fable 5, mac-axiom seat. Answers relay 2026-08-10-19
(STEP9-CLIFF-SIZE GO, PRE-REG RESULTS 25887) and the separate Metal
window ping (relay -13/-16 precedence, --gpu-ok authorized).

## Part 1 — STEP9-CLIFF-SIZE: P-STRUCTURAL

VERDICT: Bar 2, P-STRUCTURAL, fires. All three rungs threw. The
structural-tie hypothesis survives 2^18 bits — but see the mechanism
note: the honest wording is structural WITH RESPECT TO SHADOW
PRECISION AT FIXED RING.

### Setup (per spec)

- run_anchor2, d64-class r2b inputs (tools/int_adamw/r2b_{tables,init}
  .bin), 256 primes, TRACE BUILD (-DAX_ANCHOR2_TRACE), Mac CPU, one
  worker throughout (GPU battery ran on the GPU device in parallel).
- Steps 1-8: SHIPPED RAMP (the "say which" choice). Digests
  byte-identical to the booked ladder on every rung
  (7c9b8f0b... step 1 through 41bfedf6... step 8).
- INSTRUMENT NOTE (one modification, committed): the trace build's
  AX_PREC override applied to ALL steps; the pre-reg needs the ramp
  for 1-8 and the rung price at step 9 only. Added an AX_PREC_STEP
  env gate to run_anchor2.cpp (default 1 preserves old semantics).
  Verified live on the tiny fixture before any rung fired: step 1
  prec 200 (ramp), step 2 took the override. Method note applied:
  the knob was confirmed via the printed prec field.

### Ladder rows (receipts: tools/exact_anchor/step9_cliff/)

| rung (AX_PREC) | outcome | throw site                          | step-9 wall |
|----------------|---------|-------------------------------------|-------------|
| 16,384         | THROW   | floor w=1 lo_bits=16384 e=-16342    | ~103 s      |
| 65,536         | THROW   | floor w=1 lo_bits=65536 e=-65494    | ~383 s      |
| 262,144        | THROW   | floor w=1 lo_bits=262144 e=-262102  | ~1,868 s    |

All rungs streamed (rows + stderr with full backtrace) before the
next fired. No rung hit the 4 h timebox; killed class empty. Step-9
attempt walls are file-mtime deltas (step-8 row -> throw), +/-1 s,
booked as attempt rows in each rung's JSONL.

### Two observations the bar wording doesn't capture

1. THE DEMAND TRACKS THE PAYMENT EXACTLY. Across all four known
   attempts (4,000 / 16,384 / 65,536 / 262,144 bits):
   lo_bits = prec and e = -(prec - 42). Constant offset 42. The
   shadow fills whatever precision is paid and still leaves a
   width-1 straddle. There is no fixed finite demand being
   approached; the ">=15k bits" sensor reading was the shadow's
   shared exponent, consistent with the -17 retraction.
2. MECHANISM (exact_anchor2.hpp:257-299): the shadow SUCCEEDS in
   narrowing the floor to fh-fl = 1; the exact-boundary integer
   test fails; the throw is reconstruct_rat exhausting the fixed
   256-prime RNS modulus. The ladder raises shadow precision, but
   the binding resource at this site is the RING. Your wall
   corollary ("the only speed lever is the ring") extends: at this
   site the ring is also the feasibility ceiling. The untested knob
   is prime count — a 512/1024-prime arm at modest AX_PREC would
   separate "ring too small" from "tie exact in any modulus", and
   is the natural next pre-reg alongside the re-elevated
   co-factor/witness line.
3. Wall scaling, for the record: ~3.7-4.9x per 4x precision on the
   failing attempts — the "weak shadow term" caveat was right; it
   does not stay weak at step-9 demands.

## Part 2 — FP32LIMB R2/R3 GPU battery (window ping honored)

Order run as authorized: ftz -> biteq -> wall, all with --gpu-ok,
Apple M3 Pro, fast-math OFF (MTLMathModeSafe + fastMathEnabled=NO).
Receipts: tools/fp32limb/receipts/{r2_ftz,r2_biteq,r3_wall}.log.

- ftz: FTZ-PRESENT. mul_half/add_self/sub_makes all FLUSHED;
  mul_ident PRESERVED. Range restriction REQUIRED, per pre-reg.
- biteq (P-KERNEL-BITEQ): BIT-IDENTICAL vs the R1 oracle, seeds
  1/2/3, n=64, bad=0 each. FIRES.
- wall (R3): cpu_fp64 median 0.3062 s; fp32limb 0.0366 s
  (0.120x, PASS<=1.07); intacc 0.0011 s (0.004x, PASS<=1.07).
  Both instantiations clear the bar by an order of magnitude-plus.

## HOLDs unchanged

Montgomery/TC RNS: wsl-axiom, [HOLD], needs its own pre-reg +
Artin GO. Nothing further is GO'd by this relay; the 512-prime
step-9 arm suggested above is a PROPOSAL, not a start.
