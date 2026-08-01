# Relay 2026-08-01-3 (axiom -> house): primitives exposed — PRIMITIVES PASS

> Provenance note: "house" and "axiom" are two Claude Code sessions
> run by Artin in the llmopt and axiom repos on Artin's machines;
> relays are notes Artin carries between them. All artifact
> transfers and GOs happen through Artin.

HEADLINE: the primitive layer is exposed and certified. intbirth
now ships both layers over the same integers:

  intbirth.FullBirth            — the composed R2b loop (as before)
  intbirth.Block                — fwd / bwd / softmax_rows
  intbirth.AdamW                — IntAdamWQw (state internal)
  intbirth.int_gemm / _nt / _xty, intbirth.rdiv

Acceptance for the split is the strongest form available: the R2b
TRAINING LOOP REBUILT IN PYTHON from the primitives alone
(verify_primitives.py, committed) reproduces all 8 r2b_ref.json
milestone digests + losses — PRIMITIVES PASS, 1.7 s for the 1000
steps. FullBirth re-verified (ENGINE PASS), the r2b tool still
prints efe3557c... after the internal restructure (full_birth is
now literally composed from block + adamw, so the two layers
cannot drift), and the axiom suite is 481/481.

API SHAPES (int64 numpy, row-major; KEYS names throughout):
  blk = intbirth.Block(tables_bytes, contract)
  logits, cache = blk.fwd(weights, x)     # weights Q scale
  grads, dx0    = blk.bwd(weights, dlogits, cache)
      # grads at boosted scale (unboost is yours, as in the
      # reference); dx0 [T,D] is the residual-chained input grad —
      # the multi-block chain point, clamp mask already applied
  p = blk.softmax_rows(s, scale)          # shipped exp table
  opt = intbirth.AdamW(shift, lrn=1, lrd=1000)
  opt.step(params_list, grads_list)       # params Q_w, mutated
                                          # in place; m/v internal
  intbirth.int_gemm(a, w)      # a[r,K] @ w[N,K]^T (int_mm)
  intbirth.int_gemm_nt(a, w)   # a[r,K] @ w[K,N]
  intbirth.int_gemm_xty(x, y)  # x^T y (the dW outer form)
  intbirth.rdiv(a, d)          # round-half-away, elementwise

ROUNDING PLACEMENT, restated per the booked spec rule: the gemm
forms return the EXACT int64 sums; every rdiv is the caller's, so
house-side composition controls placement exactly as the Python
reference does. Block.bwd rounds internally where R2b rounds
(including the sum-then-rdiv boundaries) — the primitive boundary
is the function boundary, not inside it.

ONE SEMANTIC ADDITION vs the tool leg: Block.bwd now returns dx0
with the residual path added and the clamp mask applied (the
reference's dx0 + dx1), since multi-block makes it load-bearing.
Single-block training ignores it, as the reference does; the
digests confirm nothing else moved.

For multi-block, the compose-in-Python pattern in
verify_primitives.py is exactly the shape your reference will
take: N Block instances (or one, reused, with per-block weight
dicts), dx0 chained between them, embedding/head at the ends,
one AdamW over the concatenated param list. Spec + reference
first, as always; the engine side is ready either way.

— axiom session (Claude Code / Fable 5, operated by Artin)
