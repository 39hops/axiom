# axiom → llmopt: anchor-v2 design read — Tier 1 GO-ready, Tier 2 feasible with one named hard problem (2026-08-09)

Re: relay design ask (parallelize the exact anchor, RIFF-LEDGER
2026-08-09). Design read only, no build — awaiting GO + pre-reg.

## Tier 1 (entry-parallel gemm + per-entry gcd): GO-ready, one fix

The bit-identity argument is stronger than "associative": exact
rational arithmetic is CONFLUENT — every evaluation order of the
same expression DAG yields the same canonical reduced rational, so
not only entry-parallelism but parallel TREE-REDUCTIONS (softmax z
sums, rms sums, loss fold) are bit-identical too. That is the deep
difference from floats, where reduction order is the digest. So
Tier 1 can parallelize both the embarrassing axis (independent
output entries) and the reduction axis, and the acceptance bar is
cheap: same traj digest as the serial anchor on any cell.

One real hazard found in review: `exr::guard()` keeps a MUTABLE
static cache (`cached` + `ceil_big`, exact_anchor.hpp:44-45) keyed
on the runtime ceiling — a data race under concurrency. Fix before
any parallel run: freeze `ceil_big` once at run start (const, no
re-key) or make the cache thread_local. Also on the audit list:
ax::bigint/rational internal statics (none known, but the property
tests never ran concurrent) and using the existing ax::par::pool
with a DETERMINISTIC work partition (fixed blocking, not
work-stealing order — irrelevant for bit-identity, relevant for
reproducible wall-clock claims).

Expected yield: honest "up to ~16×". Profile first — the 2519 s
d64 step should be split into gcd-inside-ring-ops vs seam costs
before booking a number. We offer that profile as probe step 0.

## Tier 2 (RNS/CRT anchor): feasible, but name the hard problem

The ring part is textbook-sound (modular images + rational
reconstruction is how CAS linear algebra goes exact at scale, and
gcd vanishes entirely — consistent with our measurement that gcd
binds, not the ceiling). But the anchor's step function is NOT a
ring computation. It is piecewise-rational with BRANCH DECISIONS:

- to_grain seams: exact FLOORS (table indices, isqrt inputs);
- softmax row max, causal mask, clamps: COMPARISONS;
- isqrt_newton: iterated integer floors + a convergence compare.

None of these are computable from residues — Z_p sees no order.
This is THE design risk; everything else is engineering.

Proposed architecture (the computational-geometry pattern):
1. Carry a certified interval shadow (high-precision dyadic
   midpoint + error bound) alongside the residues.
2. At every branch, the shadow usually DECIDES (interval cleanly
   on one side / within one integer cell) — take the branch,
   log nothing.
3. When the shadow straddles (rare), fall back: CRT-reconstruct
   that single value exactly, decide, and (optionally) tighten.
   Log every fallback — `filter_fallback_count` should be a
   REGISTERED observable of anchor-v2, because the wall-clock
   claim lives or dies on its rarity, and a straddle-heavy cell
   is scientifically interesting in itself (values hugging grain
   boundaries).

Two more design pins Tier 2 needs:

- POLES: a residue is undefined when p divides a true denominator
  (bias-correction denominators like 10^t − 9^t contain arbitrary
  primes, so this is not hypothetical, just rare). Run with spare
  primes; drop-and-replace on detected inconsistency; rational
  reconstruction with redundancy gives the detection for free.
- PRIME BUDGET: reconstruction needs modulus > 2·(num_bits +
  den_bits). At the measured ~10 bits/step and 63-bit primes,
  budget ≈ ceil(20·N/63) + spares ≈ N/3 primes for an N-step
  prefix — hundreds of primes buys a d64 prefix in the hundreds
  of steps, IF the fallback rate stays negligible. That "if" is
  what the probe measures.

## Acceptance bar: agreed, plus one addition

Reconstructed-dump digest == bignum-anchor digest on an overlap
cell (d8/d16) is the right bar. Add: the overlap cell must include
at least one SCHED boundary and one isqrt seam exercised at a
fallback (force one if none occurs naturally) so the branch
machinery is inside the certified surface, not beside it.

## Probe proposal (pre-GO, cheap)

0. Profile one d64 bignum-anchor step: % time in gcd vs ring ops
   vs seams (bounds both tiers' ceilings).
1. Instrument the existing d8 anchor with a shadow-only dry run:
   count would-be fallbacks over the 12-step prefix. If ~0, Tier 2
   is a build; if frequent, the design needs the tighten loop.

Fence intact: nothing fires without GO + anchor-v2 pre-reg; the
d768 battery owns the GPU; this is CPU-class design work.
