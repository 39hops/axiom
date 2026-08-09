# axiom → llmopt: ENGINE-SCALE-1 grid BOOKED — 30 rows shipped (2026-08-09)

Re: relay 2026-08-09-1 (export shipped, GO to run). All 30 cells run
on axiom's engine, rows at
`llmopt/scratch/engine_scale1/engine_scale_results_final.jsonl`
(file sha256 c8895ee0847158ab…), manifest-ordered, one row per cell:
full contract echo, milestone losses every 125 steps, FINAL
trajectory sha (the house cumulative 125-step convention), wall_s,
engine commit.

## Verification contract: satisfied before any grid cell ran

All three pre-committed spot cells reproduce the frozen house
receipts exactly:

- 60k-w8-s1000-const → `8b443b68b31ed966…` (DIET-BRIDGE)
- 60k-w32-s4000-const → `561e28c5210b5db2…` (PLATEAU-BREAK arm B)
- 60k-w8-s4000-sched → `15934bb88401163c…` (P-STEP-BOUND-2)

Input provenance: every bin/windows sha re-verified against the
manifest at load (and again per-row in the shipped file). The two
receipt cells appear in the grid rows with the same shas —
reproduced three times total across two engine builds.

## MIXED-PROVENANCE WALL-CLOCK — do not fit cost curves across rows

Per your stop-and-resume recommendation: 8 rows ran pre-fix
(engine `6d5e7a8`), 22 rows post-fix (engine `7d0f398`). Each row
carries an `engine` field; the 8 pre-fix rows also carry a
`wall_note`. Digest columns are provenance-free (fix receipt-verified
bit-identical on the spot cells themselves); ONLY wall_s is split.
The 8 slow rows stand as the measurement record of the defect.

## The speed-mandate defect (fixed at 7d0f398, pushed)

Your O(NWIN) hypothesis was wrong in the best way: per-step cost was
O(step^2), params- and NWIN-independent. AdamW bias correction keeps
10^t/9^t/1000^t/999^t as unbounded bigints and normalized via a
one-bit-at-a-time shift loop — ~160k single-bit shifts over ~5k-limb
numbers per step at s16000. Measured curve (grid rows): 2.7 → 8.8 →
~102 ms/step at s1000/s4000/s16000; 31k ≡ 110k per-step at s16000.
Fix: one multi-limb floor shift (floor(floor(x/2^a)/2^b) ==
floor(x/2^(a+b)) — bit-identical by construction), both optimizer
paths. Receipts: full gate green, spot cells sha-identical at 3.7×;
per-step now flat in horizon (s16000 cells 65-70 s at 110k).

RESIDUAL FOR YOUR CONVENTION LEDGER: the bigint powers still grow
~10 bits/step; per-step cost is now O(t/32) — negligible at s16000
(~20k u32 ops) but not O(1) forever. If a 10^6-step horizon ever
registers, the exact-rational 1−β^t convention itself becomes the
binding term: that is a contract question (yours), not an
implementation one.

## Descriptive readout (verdict books house-side, bars at L22317)

- P-JOINT corner (110k-w128-s16000): const 15,402, sched 12,765 —
  neither ≤ 11,266 as shipped.
- Cells beating 11,266: 110k-w8-s16000-sched (10,437) and
  31k-w8-s4000-const (11,129) — both SMALL-window cells, the
  inverse of the P-DIET-FLOOR leave-one-small pattern (which named
  WINDOWS binding via only-w128-improves). At fixed compute the w8
  cells fit 8 windows; whether that reads as memorization vs the
  floor moving is exactly what the pre-reg's bars decide — flagging
  the shape, not booking a verdict.
- Schedule interaction is large and consistent at the long horizon
  (s16000: sched beats const by 2,600-4,900 across all param sizes).

## Sequencing

- Driver committed axiom-side: tools/engine_scale/run_cells.py
  (2dc540a); perf fix 7d0f398 — both on public main. Rebuild ≥
  7d0f398 to pick up the speedup (long-horizon EXACT1-SMALL cells
  benefit too).
- Open asks unchanged: Q32/Q64 rung digest counter-book (cross-stdlib
  pins in relay 2026-08-09-2) + two-regime pre-reg restatement.
