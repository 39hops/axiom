# FP32LIMB R1 — sloppiest-link table (PRE-REG FP32LIMB-METAL)

Per the relay 2026-08-10-10 contract: every link in the chain, the widest
value it can hold, and the proof it cannot round. Registered constants:
slice width `s = SLICE_W = 7`, block `b = BLOCK = 32`, cap
`MAX_SLICES = 8`; constraint `2s + log2(b) = 19 <= 24` (`static_assert`
in `include/ax/la/fp32limb.hpp`).

| Link | Widest value | Why it cannot round | Fence |
|---|---|---|---|
| align (`slice_row`) | fp32 scaled by 2^-e_align, held in double | element has 24 significant bits; power-of-two scale is exact in double (no underflow: double emin -1074 << -149 - 32) | non-finite input → throw |
| split (slice loop) | residual < 0.5, ≤ 24 significant bits | `R*2^s` is a power-of-two scale; `nearbyint` exact; `scaled - Q` extracts a bit-suffix of an exactly-held value | residual nonzero after MAX_SLICES → `throw "fp32limb: envelope"` (survives NDEBUG; verified in Release) |
| slice storage (float) | \|Q\| ≤ 2^(s-1) = 64 | integer < 2^24 is exact in fp32 | — (bounded by construction of round-to-nearest slicing) |
| product (fp32 mul) | \|Qa·Qb\| ≤ 2^(2s-2) = 2^12 | integer product < 2^24 exact in fp32 | covered by static_assert |
| local accumulate (fp32 add, ≤ b terms) | \|acc\| < 2^(2s-2+log2 b) = 2^17 | every partial sum is an integer < 2^24 | runtime guard: `|acc| >= 2^24 → throw "partial overflow"` after **every** add (release-safe; this is the v3-lesson fence — fp32 sums crossing 2^24 round before any two-sum can protect them) |
| simd reduce | (R1: same loop as above; R2 will split lanes) | same bound — lane-partials are subsets of the ≤ 32-term sum | same guard; R2 must re-verify under simdgroup reduction order |
| block carry / recombine | bigint at exponent −s(p+1)−s(q+1)+ea+eb | `acc()` aligns exponents by exact shifts and adds bigints — arbitrary precision, no rounding site exists | integer path only; never fp-vs-fp |
| expansion exit (rider) | Shewchuk expansion of exact floats | `two_sum` error-free under strict IEEE (build has no fast-math) | `exp_add_capped(cap=3)` throws on overflow past triple-double |

## Envelope inequality (the exactness condition)

An element is representable iff its lowest significant bit is ≥
2^(e_align − MAX_SLICES·s) — i.e. within 56 bits of the window max
exponent. This is a *lowest-bit* condition, not an exponent-spread
condition: a single-bit element at spread 40 is exact; a full-mantissa
element at spread 33 is not (fp32 mantissa is 24 bits: 24 + 32 = 56).
The fence is the residual check in `slice_row`, which rejects exactly
when representation fails — no false accepts possible by construction
(the residual IS the unrepresented remainder).

Input classes with unbounded downward tails (uniform, normal) are
therefore *defined* with a flush-to-zero below 2^-24 of class scale
(class names carry the `_f24` suffix in receipts; no silent caps). The
denormal-range class needs no flush: FTZ-free CPU denormals bottom out
at 2^-149 with truncated mantissas, keeping every lowest bit inside the
envelope automatically.

## FTZ / denormal note for the Metal port (R2)

R1 runs on CPU with no FTZ: denormal inputs and denormal slice scales
are exact (test `fp32limb_class.denormal_range_exact`). If the Metal
fp32 path flushes denormals, slices whose scale 2^(e_align − s·p) falls
in the denormal range would silently zero — the kernel must either
document a range restriction (inputs bounded away from 2^-126 + 56 bits)
or verify FTZ is off. This is the pre-reg risk item alongside fast-math
(two_sum requires strict IEEE; pin fast-math OFF, pin contraction).

## Deliberate non-ports

`dd_chain` (ozaki_rung2bc.py:89-102) NOT ported: its carrier term is
multiplied by a literal 0 (unmeasured placeholder). The chain rider in
`tools/fp32limb/r1_oracle.cpp` instead verifies each layer's GEMM
exactly and carries fp32-rounded (flushed) values between layers —
exact cross-layer carry is R3/RNS territory. The RNS promotion is NOT
triggered by this build (needs a live >6-layer chain); `ax::rns`
already exists if/when it fires.
