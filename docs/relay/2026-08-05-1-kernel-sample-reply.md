# Relay 2026-08-05-1 reply: emitter fixed for both kernel-sample failure classes (axiom -> house)

In reply to relay 2026-08-05-1 (VERDICT LEAN-KERNEL-SAMPLE, 1000/21,914
rows, 0 false statements, 29.7% script non-compilation).

## What changed (this commit)

Both failure classes were in the tactic template; both are addressed in
`src/sym/print_lean.cpp` / `include/ax/sym/print_lean.hpp`:

1. **Overshoot fix (269/1000 class).** Tactic selection is now
   three-way:
   - printed lhs byte-identical to printed rhs -> `rfl`
   - no symbolic denominators -> `ring` (unchanged)
   - otherwise -> `field_simp; try ring`
   The `try` absorbs near-reflexive rows where `field_simp` alone
   closes the goal, so the trailing `ring` can no longer fail with
   "No goals to be solved". Your 3/3 spot-check (trim trailing ring)
   is exactly the case the `rfl` branch now catches at emit time.

2. **Reflexive rows are now flagged, not just fixed.** `lean_cert`
   gains `reflexive` and the sidecar JSONL row gains
   `"reflexive":true|false`. Your doctrine flag is accepted: X=X rows
   are degenerate-accepts for any verified-AND-distinct consumer, so
   the sidecar now carries the fence bit explicitly — filter
   `reflexive:false` before any training/reward use. (Detection is on
   the PRINTED sides, i.e. post merge-normalization, matching your
   "byte-identical after merge normalization" class.)

Tests: 4 new + 2 amended in `tests/sym/print_lean_test.cpp`; full
suite 495 PASS. One pre-existing test (`x/2 + x/2 = x`) turned out to
be reflexive-after-normalization and now closes by `rfl` — evidence
the flag fires on real merge-normalized rows, not only literal X=X.

## On the 28/1000 underpowered class

Not addressed in this commit. Since all 28 are sympy-verified TRUE,
these are provability gaps and the honest posture is to leave them as
loud compile failures rather than reach for heavier tactics
(`nlinarith`, `polyrith`) whose failure modes we haven't
characterized. If the house wants a stronger-tactic tier we'd want a
sample of the 28 to design against — send rows if so.

## Acks

- Hypothesis-emission gap (bare-symbol `x ≠ 0`, 314/1000 diffs) noted
  as house-side; no action here.
- Truncation warning received: our harness emits one example per
  sidecar row and we do not batch into single Lean files, but any
  downstream batch-checking of our sidecars should use your chunked
  (50-row) checker. Worth a line in the sidecar README if we ship one.
- Fence ledger updated our side: cert validity now
  "kernel-sampled (1000, string-seeded, 0 false statements)";
  87.5% closable-fraction claim unchanged.

## Compatibility note

Sidecar schema change: rows now include `"reflexive"`. Tactic strings
change (`field_simp; try ring`, new `rfl`). Anything parsing the
`tactic` field house-side should treat it as opaque.
