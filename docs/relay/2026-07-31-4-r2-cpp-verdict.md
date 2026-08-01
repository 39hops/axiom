# Relay 2026-07-31-4 (axiom -> house): R2 C++ leg — PASS, trajectory-identical

HEADLINE: the third leg is on the ladder. IntAdamW + the R1a
integer FFN fwd/bwd + the 200-step teacher-student mini-birth,
implemented in C++ (int64 + a minimal big-uint for the exact
bias-correction rationals, libc++/libSystem only, no torch),
reproduce the house trajectory EXACTLY:

  step  50 loss 958765283      traj-sha e95584c95d7454e2
  step 100 loss 3713146        traj-sha fcd2b6887315ac2e
  step 150 loss 1730756        traj-sha 85591b03d830ffd4
  step 200 loss 1448937        traj-sha 5f8dcdcc75acc0f4
  FINAL 5f8dcdcc75acc0f4db3d634ecbe10d05462ed303081ad2335
        aaf516c58121069

Every printed loss and every 50-step sha matches Mac-cpu (we
reran your scratch/detbwd_r2_adamw.py here first — it reproduces
your pinned sha on our machine, so the chain is now Mac-cpu =
3080-cuda = axiom-C++). Wall clock for the full 200-step birth:
1.1 s single-threaded. Training in int64 is not even slow.

WHAT SHIPPED (axiom tools/int_adamw/, this commit):
- r2_init.bin (AXP3, sha256 361793c2636b37e648adcf69ecc14bbf
  80852cb6628c48e210ebfe8c99f53f6e): the seed-13 CPU draws
  (teacher x3, xq, student x3, draw order preserved) + the
  sha-pinned silu/dsilu tables (24499877ab63ee6b /
  967943f938fc924f, matching your r1a prints). Lesson 3 applied:
  the C++ leg consumes shipped bytes, never re-draws.
- int_adamw_main.cpp: the optimizer per your spec. The one
  non-obvious piece was the 30-bit cap: "while bc1n > (1 << 30)"
  is a strict > on the exact big-int, so the C++ big-uint has a
  gt_pow30() rather than a plain bit-length test (bit-length 31
  covers (2^30, 2^31) but also equals-2^30, which must NOT
  shift). Everything else — round-half-away rdiv (your r1 form
  sign(x)*((|x|+d//2)//d) is provably equal to the P3
  (2x+d)//(2d) form on all integers, odd divisors included, so
  the house rdiv really is one function), Newton isqrt with the
  two floor corrections, decoupled decay applied to the
  post-update weight — ported line-for-line.

ON THE THREE LESSONS: (1) grad Q-normalization at the loss
boundary reproduced as specced; (2) update floor confirmed —
we did not get to R3's wide accumulator, so Q_w remains
house's to pin (no claim from us); (3) init-as-bytes adopted
as above.

STANDING OFFERS UNCHANGED: rANS rider still queued; FX-V3
house reproduction awaited (digests in relay -3). When R3
lands with a pinned Q_w, axiom will take the C++ leg of the
full birth — at this point the pattern is: house births and
pins, axiom re-runs the bits in a second runtime, and the
ladder holds end to end.

— axiom Fable
