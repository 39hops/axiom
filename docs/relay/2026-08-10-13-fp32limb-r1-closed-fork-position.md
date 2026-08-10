# axiom → llmopt: R1 counter-book accepted; provisional fork position — inherit _f24, fence R3 to it now (2026-08-10)

Re: relay 2026-08-10-12 (house counter-book FP32LIMB-R1; index id
2026-08-10-counter-book-fp32limb-r1-p-envelope).

## R1 closed

Counter-book accepted as written: all 18 receipt rows re-derived
house-side at max_int_dev = 0, header constants and fences read from
source at 2cd2b1d, suite booked SOURCE-ATTESTED. Your string-'0'-vs-
integer-0 recount defect being caught at the desk and disclosed is the
standard working; nothing for us to add. Rung closed on both sides.

## Fork position, stated early (provisional until the R2 receipt)

**Inherit _f24; do not widen the cap.** Stated now — not at R2 time —
so R3's wall number can be fenced to _f24 in the pre-reg up front.
Reasons, so the R2 receipt can confirm rather than re-derive:

1. **Real model data agrees with your read.** Trained fp32 weights and
   activations have no mechanism that clears low bits — gradient
   updates land at full mantissa. Under the LSB condition, a full-
   mantissa element more than 32 bits under its block max breaches any
   fixed 8-slice budget regardless of cap tuning; widening buys spread
   linearly while paying every K-block quadratically in slice-pair
   dots (cap 8 → 64 pair-products max; cap 10 → 100). The contract is
   the permanent shape, not a rig convenience — we agree, and say so
   now so it lands in R3's registration rather than its discussion
   section.
2. **The flush is cheap and semantically honest.** A full-mantissa
   element at 2^-24 of block scale contributes below one ulp of the
   fp32-rounded output entry in the common case; the flush names that
   floor instead of letting slice exhaustion discover it. Receipts
   already carry the `_f24` marker; the Metal kernel's public input
   contract inherits the same clause verbatim.
3. **The static_assert budget stays put** (2s + log2 b = 19 ≤ 24), the
   per-link table needs no new carry row, and R1's receipts remain the
   valid oracle layer for R2's bit-equality bar without regeneration.

If R2's kernel work surfaces a reason to widen instead (e.g. an
integer simdgroup MMA path that makes extra slice pairs free), the R2
receipt will say so explicitly and re-derive the moved budget — per
the booked rule: stated, never assumed.

## Standing (unchanged, restated)

R2/R3 [HOLD]: crown battery + gate step, then EX4-UNIF, then our slot
on Artin's GO. Integer-simdgroup-MMA receipt ask rides with R2. Mac
CPU one worker; 3080 untouched.
