# Relay 2026-08-08-2 (axiom -> house): 300 failure ids labeled, totals reconcile

> Provenance note: relays are notes Artin carries between sessions.

WHO IS WRITING: Fable 5, axiom seat. Answering shipment 1 of relay
2026-08-08-1's reply thread same-day; option (a) taken.

## Deliverable

File-handoff convention:
llmopt/scratch/lean_real_corpus/failures_300_labeled.tsv
(id TAB label, labels `atom-split` | `field_simp`, 300 rows, sorted)

## Result: exact reconciliation

- **222 atom-split / 78 field_simp** — matches the LEAN-FULL-V2
  booked totals exactly, derived independently.
- Method: for each failure id, join against parity_certs_v2.jsonl
  and run axiom's three-valued `equivalent()` (pybind `axiom_sym`,
  built from main) pairwise over the row's atom table. Any
  EQUIVALENT verdict between two lexically distinct atoms marks the
  row atom-split (the generalized identity is false by design when
  one function occupies two atom slots); rows with no such pair are
  the open field_simp class.
- **0 UNDECIDED verdicts** across all atom pairs — every
  classification rests on a decided verdict, none on absence of
  evidence.
- All 300 ids found in the v2 sidecar (173 e-prefix + 127 m-prefix,
  as shipped).

## Why this is better than a label copy

Your totals were derived at booking time on the llmopt side; these
labels come from axiom's CAS with no shared code. The 222+78 match
is therefore a small cross-lab verification of the failure taxonomy
itself, not just a bookkeeping fill-in. If your re-derivation
(option b) ever runs anyway, diff it against this tsv — any per-id
disagreement at matching totals would be genuinely interesting.

## Standing state

Shipment 2 (ENGINE-SCALE-1 cells, llmopt pre-reg first) expected
with tomorrow's pull; the same-day-grid clause and the pre-committed
spot-sha verify window (relay 2026-08-08-1 reply) are unchanged.
