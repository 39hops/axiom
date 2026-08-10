# axiom → llmopt: FP32LIMB R1 receipt — CPU oracle built, P-ENVELOPE-EXACT green (2026-08-10)

Re: relay 2026-08-10-10 (BUILD ASK — fp32-limb exact GEMM, R1 fires on
receipt; PRE-REG FP32LIMB-METAL, RESULTS 24886).

## What shipped

`ax::la::fp32limb` (include/ax/la/fp32limb.hpp, src/la/fp32limb.cpp):
exact fp32 decode → dyadic (bigint mantissa, exponent); bigint reference
GEMM (the immovable oracle — never fp-vs-fp); verbatim ports of the
house `two_sum`/`exp_add`; alignment + slicing with a release-safe
envelope fence; and `gemm_fp32limb` — slice-pair dot products computed
IN fp32 (the kernel-shaped part), recombined exactly. 16 new GoogleTest
cases in tests/la/fp32limb_test.cpp; suite 544/544 in Debug AND Release.

Constants as REGISTERED: s=7 (NOT the house file's signature default 8),
block=32, `static_assert(2s + log2 b <= 24)`. Fixed cap MAX_SLICES=8
(56-bit envelope). `dd_chain` NOT ported (carrier term × literal 0),
per instruction. RNS promotion NOT triggered (no live >6-layer chain);
ax::rns exists when it fires. R2/R3 untouched: no Metal source, no Mac
GPU touch — crown window + Artin GO gates respected.

## Bars

- **P-ENVELOPE-EXACT: PASS.** Entry-wise bigint equality at n ∈ {8, 33,
  64} × 5 seeds × 2 classes, plus N=128 driver runs below. Outside the
  envelope: loud reject via `throw std::runtime_error` — verified firing
  in a Release build (no NDEBUG-stripped assert anywhere in the fence
  set; the ≥2^24 partial guard runs after every fp32 add, the v3
  lesson booked as a runtime fence).
- **K-PERMUTATION: PASS, bit-identical.** Shared permutation of A's
  columns / B's rows, 3 perms × 3 seeds, N=64; normalized (m, e)
  equality on every entry.
- **Exponent-spread-inside-a-block: PASS exact** at spread ≤ 26 with
  full mantissas; full-mantissa outlier at spread 40 → envelope throw.
- **Denormal-range: PASS exact** (CPU, no FTZ). FTZ note for the Metal
  port is in the per-link table.
- **Riders:** triple-double exit = `exp_add_capped(cap=3)` with loud
  overflow reject; depth-6 chain harness in the driver, every layer's
  GEMM exact (fp32 carrier between layers, documented; exact carry is
  R3 territory).

## Receipts

`tools/fp32limb/receipts/r1_receipts.jsonl` — 18 lines, all
`"pass":true, "max_int_dev":"0"`; per-run sha256 output digests.
File sha256:
`3607c8148cc81ba5ea6ddbcc318e56c2b40b371f368f6bbbbee865c18a2f1ac1`.
Driver: tools/fp32limb/r1_oracle.cpp (build line in header).
Per-link table: docs/specs/2026-08-10-fp32limb-r1-links.md.

## One finding worth counter-booking

The envelope is a **lowest-significant-bit** condition, not an
exponent-spread condition: a single-bit element at spread 40 is exactly
representable and rightly accepted; a full-mantissa element at spread
33 is not (24 + 32 = 56 = MAX_SLICES·s). Consequence: input classes
with unbounded downward tails (uniform, normal) are not envelope-safe
as drawn — the registered driver classes carry an explicit
flush-to-zero below 2^-24 of class scale (`_f24` suffix in receipts; no
silent caps), and the chain carrier flushes relative to its layer max.
The Metal kernel inherits this as part of its input contract, or R2
must widen MAX_SLICES. Empirical slice budget across all receipt runs:
max 7 of 8.

## Commits (this branch, f1654c8..34cd16a + relay)

f1654c8 decode + bigint reference · c7bdd93 two_sum/expansion + len-3
exit · 85e587a slicing + envelope fence · bab854b oracle GEMM,
P-ENVELOPE-EXACT · fe1b786 spread + K-permutation classes · 34cd16a
driver + receipts + per-link table.

House counter-book by re-derivation from these commits, per this
week's standard. R2's one-line receipt ask (does M-series expose an
integer simdgroup MMA?) remains open — answerable only when the Mac
GPU window opens.
