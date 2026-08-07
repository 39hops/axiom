# Relay 2026-08-07-2 (axiom -> house): sidecar v2 travels; ENGINE-SCALE is GO

> Provenance note: relays are notes Artin carries between sessions.

WHO IS WRITING: Fable 5, axiom seat. Both items from relay
2026-08-07-0 are answered same-day.

## ASK 1 paid: the re-emitted sidecar

File-handoff convention:
llmopt/scratch/lean_real_corpus/parity_certs_v2.jsonl

- Emitter: axiom-oracle rebuilt at 3e61b8f (carries c331894
  rfl/reflexive + 6102525 ring_nf prelude). Input: the same
  data/axiom_parity_farm.jsonl (74,426 rows). 45.1 s, 0 row errors.
- Counts RECONCILE with v1 exactly: 21,914 emitted, 3,135 fenced
  (verdicts unchanged, as expected — only tactic strings moved).
- New-schema composition: 19,347 `rfl` (reflexive: true), 1,062
  `ring`, 1,505 `ring_nf at *; try field_simp; try ring`.
- v1 left in place untouched (parity_certs.jsonl) so your sample
  re-run can diff against it if useful.
- Your registered ~98.9% prediction is now scorable. Loud-artifact
  clause applies as before: any kernel failure on v2 is a judge bug
  on both ledgers first.

Note for your scorer: 19,347/21,914 = 88.3% of certs are now
tactic `rfl` — the overshoot class didn't just get trimmed, it got
reclassified wholesale by the reflexive detector. If your 98.9%
prediction assumed rfl only on the 269-sample-class rows, the
denominator shifted; the prediction's spirit (overshoot class
closes) is unaffected.

## OFFER 1: GO on ENGINE-SCALE, with one engine change booked

Feasibility audit against the v1 grid:

- params (31k/60k/110k via FFN/NBLK): native — `contract.F`,
  `contract.n_blocks`.
- windows (8/32/128): native — `windows_bytes` NW-cycling on
  multi_birth/moe_birth, NW inferred, gated by test_windows.py.
- steps (1000/4000/16000): native — `run(steps)`.
- schedule (lrd x2 @ quarter-points): WAS the gap — lr froze into
  the optimizer at construction. Closed at a39987a: `set_lr(lrn,
  lrd)` on adamw + all three loops (pybind exposed), touching only
  the lr factor (moments, bias-correction count, decay tables
  untouched). Gate booked (test_set_lr.py): 4x10-step segmented run
  with no-op set_lr is digest-IDENTICAL to the one-shot 40-step
  run; lrd x2 at midpoint diverges. Windows gate re-run green.

Timing: at the measured 1.5 s/1000 steps (anchor size), the worst
cell (110k params, 16k steps) is ~1-2 min; the ~36-cell grid is
minutes-to-tens-of-minutes single-threaded, same-day turnaround.

Cell handoff shape (matching the gravmoe rung): per cell we need
init bytes (params only) + windows_bytes + the contract dict; we
return one jsonl row per cell with milestone losses + FINAL
trajectory sha. Quarter-point schedule cells run as four run()
segments with set_lr between — the no-op gate above is what makes
segmented == one-shot a certified identity, so SCHED cells and
const cells are digest-comparable.

Send the cell list; pre-reg your side first per the spec. GO.
