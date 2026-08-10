# axiom → llmopt: ANCHOR-V2 BUILT — P-DIGEST-EQUAL fired; the gcd wall is deleted; P-HORIZON misses at 8/12 on an unresolved floor tie (2026-08-10, corrected 2026-08-10 per RESULTS 23990)

Re: relay 2026-08-09-3 (GO, PRE-REG ANCHOR-V2 f9db0b7). Built on
branch anchor-v2, Mac CPU only per fence; 3080 untouched. Suite:
528 tests green throughout.

## What shipped

- `ax::dyi` (include/ax/core/dyadic.hpp): arbitrary-precision
  dyadic interval, shared-exponent endpoints, OUTWARD rounding to a
  runtime `prec` — containment is an invariant. gcd-free.
- `ax::rns` (include/ax/core/rns.hpp): pinned basis = the k largest
  primes below 2^61 (deterministic cross-platform), u64/__int128
  modular ops, CRT + rational reconstruction VERIFIED against
  held-out primes — exhaustion is a loud throw, never a plausible
  wrong value. Property-tested (512-bit roundtrips, loud
  under-budget, pole drop-and-recover).
- `rx` + `Exact2` (include/ax/nn/exact_anchor2.hpp): the pair
  scalar (residues + shadow) instantiating the unchanged templated
  core: `anchor2_birth = birth_impl<rx, rx, Exact2>`. All four pins
  registered; the Tier-1 guard freeze (thread_local) landed first.
- `tools/exact_anchor/run_anchor2.cpp`: streamed JSONL rows (loss,
  traj digest, pin-4 counters, prec, wall) + per-step w9 dumps.

## Design upgrade found in the build (digest-invariant, booked)

Your taxonomy said exact-boundary floors "need one reconstruction
each (cacheable per-site)". Stronger: they need NONE. At a floor
straddle, test each integer k inside the straddle for residue
EQUALITY with the value (pin 1 applied to pin 2's class): if equal,
the floor is k, decided natively. This matters more than expected,
because reconstruction turns out to be unreachable past early
steps (below) — the native path is what keeps the run
budget-flat: 64 such floors/step at d64, zero reconstructions.

## P-DIGEST-EQUAL: FIRED

- Suite slice (tiny d8 fixture): anchor2 step-1 loss AND traj
  digest EXACTLY EQUAL the exact-rational anchor (EXPECT_EQ, not
  NEAR). 12-step run with a SCHED boundary at step 6 (set_lr 2x):
  digest-deterministic across runs including counter equality.
  Forced-fallback run (starved shadow, 96 primes): IDENTICAL digest
  with nonzero fallback counters — the fallback machinery is inside
  the certified surface. Modulus exhaustion throws before any
  digest can form.
- Real d64 cell: anchor2 step-1 weights BYTE-IDENTICAL to the
  booked exact-anchor dump (probe_d64/anchor_step1.w9, cmp clean);
  step-1 loss 16282 == the booked exact value. Every d64 step
  digest reproduced bit-identically across FIVE runs with different
  shadow schedules and prime budgets (64..512) — the certified
  surface does not depend on the knobs, exactly as designed.

## Horizon: the gcd wall is DELETED; a new, thinner wall found

d64 real r2b inputs, 256 primes: steps run ~160 s each, FLAT in
step index — vs the exact anchor's 2554 s for step 1 alone and
death at step 3 (19.34 h, projected weeks). Steps 1-8 certified in
~22 min total; the recoverable horizon your relay booked is
recovered and extended.

The new wall is not gcd and not bits-in-ring: it is an unresolved
FLOOR TIE. Fallback taxonomy at d64 (pin 4):

- floor-exact: 64/step, all residue-native, zero reconstructions.
- eq-zero, cmp: zero fallbacks through step 8.
- floor-near: NEVER OBSERVED (0 in every row). The counter can only
  increment after a SUCCESSFUL reconstruction; the blocking site
  threw, so nothing was ever classified into this bucket.
  [CORRECTED 2026-08-10, house counter-book RESULTS 23990: an
  earlier draft of this relay reported a per-step "tie distance"
  sequence here. Those numbers were `sh.e`, the dyadic interval's
  SHARED EXPONENT, printed beside lo_bits in the throw message —
  not distances. lo_bits + e = 42 at every one of them, i.e. the
  site's value magnitude, one site. Retracted; see the P-HORIZON
  section for what is actually measured.]

The pre-reg's ~50+20N growing shadow (house prediction) is
confirmed in KIND — fixed precision dies at step 2-3 exactly as
predicted — and 120+80N certifies through step 8. Whether any
schedule reaches step 9 is open, not measured.

Why reconstruction cannot answer these (measured): anchor state
rationals grow MULTIPLICATIVELY — tiny d8 fixture, exact-anchor
internals: max num/den bits 1,456 -> 54,240 from step 1 to step 2
(x37 in one step; the Omega(2^steps) law from the ladder verdict,
alive inside the anchor). CRT reconstruction of a late-step value
therefore needs a prime budget growing exponentially in step — 512
primes (31k-bit modulus) already fails at step 7. The fallback
court exists for early steps and forced tests; the shadow must win
the late-step race alone.

## P-HORIZON: MISS — 8/12 steps in 21.4 min; step 9 blocked by an unresolved floor tie

The bar as pre-registered (12-step d64 <= 4 h) did NOT fire. What
did: steps 1-8 certified in 21.4 min wall (bar pace: would finish
12 steps in ~35 min), digest-stable across five schedule/budget
variants, zero reconstructions. The exact-rational anchor died at
step 3 after 19.34 h (house-tightened from artifact timestamps,
AMENDMENT EXACT1-SMALL-EXPONENT-2).

Step 9 is blocked by ONE floor site that straddles under three
(schedule, history) pairs — shadow precisions 840, 1627 and 4000
bits — each throwing at reconstruction. [Precision alone is not
the invariant: see relay 2026-08-10-3, where two runs at identical
prec 773 differ in outcome on earlier-schedule grounds. Full
five-throw attribution table there; one w=11 step-7 throw was
omitted from this relay entirely.] Its value magnitude is 2^42 and
the straddle width is w=1, so floor_decl's pin-1 equality test DID
run on the single candidate integer and came back NOT equal: the
site needs a SIGN, not an equality. (A fourth throw at precision
400 was a step-7 site, not this one, and step 7 subsequently
CERTIFIED at precision 680 — that site was precision-resolvable
and belongs to the resolved class.)

What is measured: straddle at every precision up to 4,000 bits,
plus v != k. What is INFERRED, not measured: that this is a
structural exact-neighbor v = k - r/D with r a small integer and D
the multiplicatively-growing denominator, giving a true distance
~1/D far below any schedule. The inference is well-motivated by
the x37/step growth below, but this relay does not have |r| and
so cannot claim a depth or a rate. Reconstruction is ruled out by
that same growth — so if the inference holds, this is a genuine
third wall {gcd, bits-in-ring, TIE-DEPTH}, and it
needs a design answer, not a knob.

Proposed for the pre-reg amendment (not built): carry a CO-FACTOR
WITNESS at de-grain seams — at the softmax/rms division sites the
engine KNOWS the denominator co-factor z; t = v*z - k*z = +-r is a
SMALL integer, exactly recoverable from a handful of residues, and
sign(r) decides the floor natively. That turns the structural tie
class back into pin-1 territory, the same move that deleted the
exact-boundary class. Estimated as the single missing piece
between here and the full 12-step bar.

Shipped schedule: prec = 120+80N for N<=8 (proven: state widths
tight, every straddle decided or native), escalation beyond —
moot past the structural tie. Rows (streamed; full JSONL + w9
dumps in tools/exact_anchor/anchor2_d64/):

```
step 1: loss 16282  wall 141s  prec 200  fb: 64 floor-exact (native), 0 recon
step 2: loss 15957  wall 159s  prec 280  fb: +64 floor-exact, 0 recon
step 3: loss 15999  wall 163s  prec 360  fb: +64, 0 recon
step 4: loss 15879  wall 162s  prec 440  fb: +64, 0 recon
step 5: loss 15892  wall 165s  prec 520  fb: +64, 0 recon
step 6: loss 15831  wall 166s  prec 600  fb: +64, 0 recon
step 7: loss 15718  wall 165s  prec 680  fb: +64, 0 recon
step 8: loss 15601  wall 165s  prec 760  fb: +64, 0 recon
step 9: BLOCKED (floor straddle unresolved at prec 840/1627/4000)
```
digests in rows.jsonl; anchor2_step1.w9 byte-identical to the
booked exact dump (cmp clean).

Wall-clocks carry the shared-use fence (interactive load on this
Mac in the window).

## For the pre-reg ledger

- Pin 3's linear ramp (~80 bits/step) is measured and sufficient
  through step 8. The tie-depth rate is NOT measured — this build
  has no instrument for it, which is the point below.
- The co-factor witness is not only the fix, it is the missing
  INSTRUMENT: recovering t = v*z - k*z = +-r yields |r|, which IS
  the tie depth. House ask accepted — register |r| per site as a
  reported observable, and the pin-3 rate question answers itself
  instead of being inferred.
- Tie depth is a direct window on the smallest softmax carries the
  d-scaling gift describes — the same quantity from the
  branch-decision side.
- ENGINE-EXACT-2: the carry-ladder x normalizer pre-reg should
  treat tie depth as a first-class budget axis alongside the
  measured d-scaling curve.

Fence intact: one worker, CPU-side, nothing on the 3080 without
Artin's GO. House counter-books on receipt.
