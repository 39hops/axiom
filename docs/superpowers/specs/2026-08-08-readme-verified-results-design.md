# README verified-results section — design (2026-08-08)

Source: relay 2026-08-08-0 (Artin → axiom), approved with three
corrections from the llmopt seat same day.

## Goal

Bring axiom's README to the public-facing standard llmopt's README
meets: lead with what a stranger can verify, one measured claim per
bullet, honest fences stated in place, internal dialect glossed on
first use.

## Scope

1. **New README section "Verified results (cross-lab)"**, inserted
   immediately after the intro paragraph (before Modules). Bullets,
   one booked number each, every number pointing at its counterparty
   record (llmopt's RESULTS ledger):
   - E3 relay: 50/50 token-identical replay of llmopt engine
     trajectories through axiom's C++ leg.
   - Certified row factory: 167/167 emitted rows pass llmopt's
     production oracle (`verify_wave`), schema-exact, 0 diffs.
   - Lean certificates: 21,614/21,914 kernel-checked (98.63%) over
     the 74,426-row parity corpus; all 300 failures classified
     (222 atom-split by design, 78 open `field_simp`). Lineage:
     LEAN-TIER-1, 443 certificates booked 2026-08-04.
   - AXNN v1.1: 20/20 cross-lab parity.
   - Emission audits: 5-for-5 clean full audits llmopt-side
     (latest class: 0 contaminated rows in 145,011).
   - One-line gloss: llmopt is the sibling lab; its ledger is the
     counterparty record, so the two labs' public claims cannot
     drift independently.

## Fences (explicit)

- NO [R]/[H] status letters on results bullets — those are llmopt
  figure-generator statuses and would mislabel re-checkable
  artifacts as frozen history. The counterparty pointer carries the
  status.
- NO ENGINE-SCALE numbers — the 3 spot shas are unverified
  llmopt-side; embargoed until that verify books.
- Everything else in the README untouched.

## Non-README items

- GitHub repo description: already live (set 2026-08-08 via the
  same relay); verified matching, no action.
- Social preview image: manual step for Artin (Settings → Social
  preview); the 1280×640 crop was prepared llmopt-side.
