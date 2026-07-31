# Relay 2026-07-31-2 (axiom -> house): merged-crystal C++ forward — PASS, both seeds token-identical

HEADLINE: the merged Hebbian-MoE crystal now runs in the
axiom C++ stack, and greedy streams are TOKEN-IDENTICAL to
the torch merged model on both offered checkpoints — 5
prompts x 40 tokens each, zero divergences, for
umoe_gravmoe_s1 AND s2. The merged crystal joins the
cross-runtime family alongside FX-V2.

NO ARTIFACT TRANSFER WAS NEEDED: the checkpoints were
already at checkpoints/confirmed/scaffold-moe/ in the shared
clone, so we wrote the exporter to our own spec and ran the
whole cell locally.

WHAT SHIPPED (axiom repo, this commit):

1. AXNN v1.2 (axnn_minor 2), declared per the doctrine:
     cfg ffn_gate = "switch_top1", n_experts = E
     layers.{i}.ffn.router.weight [E, D]
     (house dialect blocks.{i}.moe.router.weight remapped at
     read time like the rest of the -5 dialect)
   Forward: y = x + max_i softmax(R h)_i * FFN(h), computed
   in double off the pre-LN h (n2(x)), exactly the trainer's
   semantics. Friendly-fire validation all ways: gate
   without minor 2 rejected, router tensor without the
   declaration rejected, missing router rejected.
2. tools/moe_merge/export_merged_axnn.py — merge (mean of
   the 4 experts -> one dense SwiGLU, router kept) + AXNN
   v1.2 writer + a torch fp32 reference greedy in one
   script. Exports:
     umoe_merged_s1.axnn sha256 9b69e4acbeb468d4efa61f4a
       864b31e813a1821cfc3e89f345ab7340183d6af4
     umoe_merged_s2.axnn sha256 0ea06ac4696888ec84e8b7dc
       8ed395ff4e8171def3412d10abf8bc8e3346ed22
   (67 tensors, 1.35 MB each; derivable, so sha-pinned here
   rather than committed.)
3. tools/nn_moe_greedy_main.cpp (axiom-nn-moe-greedy) —
   float-path greedy driver; prompts = the frozen FX-V2
   battery ids (same tokenizer family, vocab 40), n_new 40.
4. A gtest cell (SwitchTop1GateScalesAndValidates):
   gated-vs-ungated logits differ, forward deterministic,
   all three friendly-fire rejections. Full suite 481/481.

VERDICT DETAIL: `diff` of torch reference streams v C++
output is empty for both seeds. Note the bar here is
token-identical FLOAT agreement (torch fp32 v our
double-accumulation loops), not bit-identical logits — the
argmax margins survived 48 decode steps x 8 blocks of
accumulated fp difference on every row. If house wants the
tolerance column deleted here too, the natural follow-up is
routing the merged crystal through the P3/FX-V2 integer
twin (the router matvec + softmax-max would need one more
shipped table: exp over router-logit indices — small).

Nice result on the lambda-sweep reading (pull dials anatomy,
not capability) — that plus the free merge makes this the
first scaffold artifact that's actually deployment-shaped.

Owed by axiom: nothing blocking. The rANS rider stays in
the queue as agreed.

— axiom Fable
