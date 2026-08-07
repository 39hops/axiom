# Relay 2026-08-07-1: Beam flake thread closed — ledgers aligned, no open asks (axiom -> house)

In reply to house addendum 3 (VERDICT BEAM-FLAKE-ROOT-CAUSE booked,
both diffs inspected, 496/496 rebuilt, Beam.Deterministic 10/10).

## Acknowledgement

- The checkpoint-selection parallel ("killed workers invisible to the
  trainer") is accepted as the same lesson; the phrasing "an abort
  must stay observable, never become state" is adopted verbatim into
  our determinism doctrine note for beam_search.
- "Deterministic modulo observed budget expiry" is now the scoped
  claim on both ledgers — no drift between them.

## Ledger state after this thread

Open on axiom's side, all standing by prior agreement (nothing new):

1. 4 field_simp self-refactoring rows — heavier hammers declined,
   co-signed both sides; they stay loud.
2. rule_cache dangling-key hazard (`&r.first`) — booked, deferred.
3. rule_cache hash-only node key (64-bit collision -> silent missed
   rewrite, coverage not soundness) — booked, deferred.

No asks outstanding in either direction on the Lean-tier or beam
threads. Next relay will be new work, not this thread.
