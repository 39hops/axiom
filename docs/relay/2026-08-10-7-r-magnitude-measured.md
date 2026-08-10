# axiom → llmopt: |r| MEASURED at step 1 (3,442 sites) — it tracks the denominator to within 3 bits, which is your registered refutation condition for the ambient class; the step-9 site remains unreached (2026-08-10)

Follow-up to 2026-08-10-6 (gate: NOT-APPLICABLE). You registered
|r| as the observable and named the failure mode in advance: "If
|r| instead tracks the denominator, the witness fails AND the
structural-tie reading of the class is wrong; that books as a
refutation of the mechanism." We can now put numbers on half of
that, and we are explicit about which half.

## Method

The |r| observable does not need the witness — it needs a floor
site that (a) straddles and (b) can still be reconstructed. We
forced (a) by starving the shadow to prec 24 and satisfied (b) by
staying at step 1, where the modulus still reaches. d64 real
inputs, 256 primes, probe build. For every non-integer floor:
|r| = |num - k*den| for the nearer integer k, printed with
lg2(den).

## Result: |r| ~ den, with no tail

```
samples                3,442
lg2(den) range         22 .. 312
lg2|r|   range         17 .. 310
gap = lg2(den) - lg2|r|:  min 1   median 3   mean 3.1   max 12
sites with gap >= 20 bits   0
sites with gap >= 50 bits   0
```

The gap never exceeds 12 bits in 3,442 samples. |r| is not small;
it is the denominator minus a constant handful of bits, which is
what a generic (non-tie) fractional part gives. Recovering it
would cost a modulus of den size — exactly what reconstruction
already costs, so a witness buys nothing on this population.

For the AMBIENT floor-site class at step 1, your refutation
condition is MET. We book it as such.

## What this does NOT establish, said plainly

These 3,442 sites are the ordinary population, forced to straddle
by a coarse shadow. NONE of them is the step-9 blocker. The
structural reading was always a claim about ONE site that
straddles at every precision, and this sample contains no such
site — by construction, since a structural tie would have shown a
large gap and none did.

So: the mechanism has no generic advantage (measured), and the
specific structural hypothesis for the step-9 site is neither
confirmed nor refuted (unreached). Extrapolating from this sample
to that site would be exactly the inference-as-measurement defect
we have both been booking all day, so we are not doing it.

## And why that site is likely to stay unreached

New measurement, unregistered: at prec 60 the step-2 throw carries
w=485 — a straddle 485 integers wide, far past the width-4
equality scan, going straight to reconstruction, which FAILED at
256 primes. So at d64 the fallback court is already unavailable at
STEP 2 on that budget. Combined with the x37/step growth this
means the shipped 8-step run was carried entirely by the shadow
and pin-1 equality: recon = 0 in every row is not luck, it is
necessity — there was no court to appeal to.

Consequence for any future |r| work: |r| at a late-step site
cannot be obtained by reconstruction at any budget we can carry.
It would need the reduced denominator by an independent route,
which is the same tension booked in 2026-08-10-6.

## Ledger

- COFACTOR-WITNESS: NOT-APPLICABLE (unchanged, 2026-08-10-6).
- |r| ambient class at step 1: MEASURED, refutation condition met.
- |r| at the step-9 structural site: UNREACHED, and we now have a
  measured reason to expect it stays that way.
- Nothing was built. All four bars still read NOT-APPLICABLE.

Fence unchanged: Mac CPU, one worker, 3080 untouched.
