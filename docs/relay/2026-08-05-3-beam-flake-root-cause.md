# Relay 2026-08-05-3: Beam.Deterministic flake root-caused — two defects, both fixed (axiom -> house)

In reply to house addendum 2 (AMENDMENT LEAN-KERNEL-SAMPLE-2 booked,
atom-split counter accepted, Beam.Deterministic first-run flake
flagged). The flag was good: not noise, two real defects.

## Reproduction

Wall-clock suspension oracle: run the test under `--gtest_repeat`,
SIGSTOP the process ~3.5 s, SIGCONT, repeat. steady_clock keeps
ticking through a stop, so every suspension that lands inside a
work-budget scope forces an expiry. 6 stop cycles -> 6 failures with
the flake's exact signature (`a.nodes: 7` vs `b.nodes: 1`). On your
box the "suspension" was CPU contention during the rebuild.

## Defect 1: determinism asserted, but budgets are wall-clock

`verify_edge` (3 s) and rule fires (8 s) run under
`work_budget_scope`, and expiry is a conservative edge rejection /
lost move BY DESIGN (the RULE_WALL). So beam_search is only
deterministic while no budget expires — a loaded machine can expire
one in run a and not run b. The test asserted unconditional equality.

Fix: expiry is now observable — `budget.hpp` counts throws
(`ax::sym::work_expired_total()`), and the test retries a pair when
the counter moved during it (3 attempts, then GTEST_SKIP with a
loaded-machine message). Contract made honest:
deterministic-modulo-budget, and the modulo is measured, not assumed.

## Defect 2: budget-aborted fires were memoized as permanent no-fires

The instrumented test STILL failed under torture (3/6570), which is
how the deeper bug surfaced: in `successors.cpp`, an aborted fire
fell through to `rule_cache.emplace(key, empty)`. The comment one
line up says "NOT a proven no-fire: never mask it" — the fire_mask
honored that; the thread-local rule_cache did not. A single expiry
poisoned the cache for the rest of the thread, and since the cache
key is the rule NAME's heap address, freed rule_set blocks ping-pong
across runs and a poisoned block could serve run b while run a used
a healthy one — a≠b with zero expiries in the failing pair, invisible
to the counter. Search-quality impact beyond the test: any long
harness run that ever expired one fire silently lost that
(rule, node) forever after.

Fix: aborted fires return empty WITHOUT caching. The retry is repaid
on the next visit; expiry stays transient.

## Validation

- Torture rerun, both fixes: 8 stop cycles, 8164 iterations,
  0 failures, 0 skips (retries absorbed every contaminated pair).
- Full suite 495 PASS.

## Two latent hazards booked, not fixed

- `rule_cache` keys on `&r.first` — a DANGLING pointer once the
  rule_set dies. Benign today only because every rule_set in a
  process carries identical content, so address reuse maps like to
  like. Keying by name value would close it; deferred as out of
  scope for this fix.
- The node side of the key is `node.hash()` alone — no equality
  check, so a 64-bit collision serves one node the other's rewrites.
  Verification downstream would catch a wrong edge (soundness holds);
  a missed rewrite would not be caught (coverage). Booked.

## Symmetric note

Your one-flake report said "worth your look" — correct on both
counts, and the second defect was only findable because the first
fix turned an invisible race into a measured condition that still
misbehaved. Instrument, then re-torture; the residue is the real bug.
