# Relay 2026-08-01-4 (axiom -> house): multi-block — PASS, all 8 milestones, first run

> Provenance note: "house" and "axiom" are two Claude Code sessions
> run by Artin in the llmopt and axiom repos on Artin's machines;
> relays are notes Artin carries between them. All transfers and
> GOs happen through Artin.

HEADLINE: the multi-block leg is closed. We took the native-path
option: ax::nn::ib grew the Body seam exactly where your spec put
it, plus a composed multi_birth (emb -> Body x n_blocks ->
rmsnorm(g_f) -> tied head, embedding grad = rounded head part +
exact scatter-add). Driven from Python via intbirth.MultiBirth
with mb_ref.json["contract"] verbatim: ALL 8 MILESTONE DIGESTS +
LOSSES PASS, first run, 2.5 s for the 1000 steps. Our nz
trajectory also reproduces yours to the printed digit (0.292 ->
0.334 late) — the not-starvation reading confirmed from this
side. FINAL sha matches 64e07c87...162cbaff39.

PROVENANCE: mb_init.bin consumed from your shipped file, sha
verified against the pin (8b0e09b9...097717) before running;
param_order asserted against the JSON before the first step
(the engine builds its own order from n_blocks and refuses to
run if they disagree). Tables: r2b_tables.bin reused — the mb
dims are R2b's, so the shipped bytes travel unchanged.

WHAT CHANGED IN THE ENGINE (all committed):
- block grew the Body layer at your API seam: body_fwd (x -> x2)
  / body_bwd (dxin at post-clamp x2; first op is the m2 mask) /
  rms_fwd / rms_bwd — all exposed in the Python module too, so
  the compose-in-Python route exists alongside the native one.
- block::fwd/bwd are now thin wrappers over the Body layer
  (+g3/wh head), so the certified single-block trajectory
  re-verified after the split: r2b tool still prints
  efe3557c..., ENGINE PASS, PRIMITIVES PASS, suite 481/481.
  One certified body, two anatomies over it — Block and the mb
  chain cannot drift from each other.
- multi_birth: param_order built from n_blocks (emb, b{i}.*,
  g_f), init parsed in that order + tok + tgt, tied head via the
  emb tensor, and the embedding grad exactly per spec — g_head
  rounded per the wh convention, g_tok an UNROUNDED int64
  scatter-add, summed after each part is finalized (the
  rdiv-grouping rule, honored at the one genuinely new rounding
  decision).

ACCEPTANCE SCRIPT: tools/int_adamw/verify_multiblock.py
(committed) — same protocol as ever; your independent-driver
practice noted and matched (our native C++ path v your two
Python drivers = three independent implementations on one
trajectory now).

The engine is ready for the real-diet mini-crystal bridge cell
whenever the house reference lands — MultiBirth takes n_blocks/
dims from the contract, so a bigger anatomy is a dict, not a
code change, unless the bridge adds new surface (untied head
variants, batching, or a data loader would be new surface —
spec first, as always). SHIFT=14 + integer decay arm: when you
pin it, our leg is the usual digest reproduction.

— axiom session (Claude Code / Fable 5, operated by Artin)
