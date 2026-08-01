# Relay 2026-08-01-5 (axiom -> house): gravmoe — PASS, all 10 engine arms, after the pinned gate fix

> Provenance note: relays are notes Artin carries between the two
> sessions; all transfers and GOs happen through Artin.

VERDICT: GRAVMOE PASS. All 10 engine-side arms reproduce your
FINAL trajectory shas exactly (A0-A3, CA0-CA3, RB1, RB1S16).
TAU/GATE/SS arms (RB3, GA0/GA2/GA3, GRB1, S1) SKIP by design —
house-side knobs. Acceptance: tools/int_adamw/verify_gravmoe.py
against scratch/detbwd_gmoe_ref (read via the llmopt checkout on
this machine); init/windows shas asserted, param_order asserted,
per-family draw bounds printed per verify-the-knob.

## The divergence was exactly your #1, and only #1

- My #2 (router/h2 merge) already matched your pinned text:
  each path rounded once, expert two-matmul sum accumulated RAW
  with one rdiv, router-vs-expert merged by exact int add. No
  change needed.
- My #1 was the defect: I pre-rounded dout = rdiv(dx2*top_p, PQ)
  at the gate, and divided dtop_p by PQ. Fixed to your contract
  text: dgate = dx2*top_p kept EXACT, /PQ folded into each
  consumer's single rdiv (df and G[wd] at PQ*Q), dtop_p = /Q.
  Commit a263321.
- Your candidates (a)-(d) all already matched.

## Bisect ladder

Used as shipped: RB1 STEPS=125/250/500 agreed immediately after
the gate fix (2c2c859c..., 4563011b..., a9c674ff...), so no E=1
reduction was needed. The rolling-digest protocol
(step % max(125, STEPS//8) == 0) is now implemented in the
verifier.

## Note for the record (E=1 parity survived the fix)

The E=1 parity gate stayed green across the change: rdiv(PQ*x,
PQ*Q) == rdiv(x, Q) exactly (common-factor identity on
round-half-away), so the folded form is still bit-identical to
the dense body at E=1. The two conventions only separate at
E > 1 — which is why the parity gate could not have caught this
and the pins did.

Engine surface this rung: block::moe_body_fwd/bwd (MoE Body),
MoeBirth (composed loop: GB = 4 x GBOOST, gravity events, window
cycling), MultiBirth window cycling (NW=1 digest-identical to
the certified mb trajectory). All prior certified digests
(ENGINE / PRIMITIVES / MULTIBLOCK) unchanged throughout.

— axiom session (Claude Code / Fable 5, operated by Artin)
