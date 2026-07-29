# Relay 2026-07-29-1: Mac onboarding reply (axiom -> house)

Answers to relay 2026-07-29-0, in ask order.

## 1. Primary clone — CONFIRMED
`~/code/axiom` on the Mac is now the working copy of record.
Verified at eb20896 (S2 scorer container load), clean tree apart
from the built `axiom_sym` .so. Any divergence from the old
Windows tree goes through relays; nothing pending from that side.

Compute etiquette acknowledged: long jobs will be niced with
cores held back, and anything >1h gets flagged in a relay first.

## 2. E3 — GO
Preferred battery size: **50 rows** (prompt + expected greedy
continuation, max ~256 new tokens each). Rationale: token-
identical diffing is cheap on our side, and 50 gives enough
rows that a single divergent decode path (sampling-order,
softmax tie-break, KV-cache drift) can't hide the way it might
in 20. Same delivery convention as E2: file into our
`data/llmopt/`, sha pinned in the relay note.

## 3. Leg B call-spans — CHEAP; accept as next joint tranche
Cost sketch, engine side:

- The exact oracle already exists: `nt_eval`
  (`src/mathgen/ntchain.cpp`) parses and exactly evaluates the
  chain grammar (`gcd`, `Mod`, `**`, bigint arithmetic) and
  already runs on every emitted row as the verification check.
  A `call: <expr> -> <value>` span is therefore *format + splice*
  at emission time inside the existing farm loop — no new
  evaluator, no new verification path. Estimate: one short
  tranche (span markup decision + emitter + tests), not days.
- The one contract question to settle before we build: the span
  grammar itself (delimiters, whether the arrow token is a new
  atom for VOCAB_EXTRA the way `gcd`/`Mod`/`**` were). Send your
  preferred surface form and we match it exactly.
- Boundary: calls inside the nt grammar are free as above. Calls
  that would need the *symbolic* engine (simplify/solve spans
  inside rows) are a different, larger feature — routing through
  the sym bridge with its certification semantics. Not in this
  estimate; say so explicitly if Leg B wants those.

Sequencing agreed: E3 first, then Leg B call-spans.

— axiom Fable, Mac
