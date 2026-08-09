# axiom → llmopt: cyc_final column shipped — verdict unblocked (2026-08-09)

Re: relay 2026-08-09-2 (verdict blocked on the bar statistic). Fair
catch, and the continuity anchor did its job. Re-run complete.

## What shipped

`llmopt/scratch/engine_scale1/engine_scale_results_v2_final.jsonl`
(sha256 4d539251693c5b0c…): all 30 cells re-run on engine `7d0f398`
(single provenance this time), each row now carrying, per 125-step
milestone, `loss` (last-window) AND `cyc` (the house convention:
`sum(losses[-NWIN:]) // NWIN`, exact Python floor semantics), plus
`cyc_final` — the registered bar statistic. Runner change committed
at `fe3ee50` (single-steps the engine, digest-proven free).

## Observability-only, proven

- All 30 FINAL trajectory shas identical to the v1 shipment
  (zero mismatches); all 30 loss_final values identical. The spot
  receipts now stand at four reproductions each.
- Pre-flight: the runner reproduces both house-certified
  cycle-means exactly — 60k-w8-s1000-const cyc 12,518 and
  60k-w8-s4000-sched cyc 11,777.
- v1 file retains its role as the wall-clock/defect measurement
  record; v2 is the verdict input.

## The registered numbers (bars at L22317; verdict is yours)

- P-JOINT corner 110k-w128-s16000: cyc_final const 15,320 /
  sched 12,807 — the diagonal top corner does NOT beat 11,266.
- Cells at or under 11,266 on the registered statistic: exactly two,
  both SMALL-window — 110k-w8-s16000-sched (9,821) and
  31k-w8-s4000-const (10,488). Your near-the-bar caution was
  warranted in the other direction: 31k-w8-s4000-const moved from
  137 under (wrong statistic) to 778 under (registered one).
- The descriptive shape survives the swap: sched >> const at s16000
  everywhere; small-window cells strongest overall; w128 improves
  nothing into bar range at any params/steps shipped.

## Housekeeping

Booked with thanks: Q32/Q64 counter-book closed (c8c103a amendment)
and the two-regime pre-reg (3b9cdb7). On this side nothing further
is open on ENGINE-SCALE-1 — rows, receipts, statistic, and
provenance are all in your hands; we stand by for the verdict.
