# 2026-07-27 — AXNN FX-V1: exact inference profile (rung 2b)

Doctrine link: precision-doctrine AMENDMENT 2026-07-27 — exact mode is
the SOLE retest condition; the house runs one paired arm (exact-gate v
rounded-gate, same weights) at sub-sigma resolution. Instrument fence
(a) stands: ax-gate v torch-gate numbers never compare unpaired.

FX-V1 is a DECLARED model definition, not an approximation claim: an
integer-only forward whose every operation is exactly computable and
platform-free. Bit-identity across Mac / 3080 / axiom C++ follows from
integer arithmetic alone; exact accumulation is associative, so the
result is order-free (threading cannot change it).

## Number formats

| thing                      | format  | storage | saturation |
|----------------------------|---------|---------|------------|
| weights / biases / embeds  | Q7.16   | int32   | ±128       |
| activations / residual     | Q11.16  | int64   | ±2048      |
| centered values (norms)    | Q9.16   | int64   | ±512       |
| linear accumulators        | Q.32    | int64   | none (headroom-proved) |
| logits (readout)           | Q.32    | int64   | none       |

Headroom proof sketch: |act| ≤ 2^27, |w| ≤ 2^23 → product ≤ 2^50;
longest contraction d_ff = 1024 ≤ 2^10 → |acc| ≤ 2^60 < 2^63. Norm
squares: |c| ≤ 2^25, c² ≤ 2^50, D ≤ 2^11 → ≤ 2^61.

## Declared rounding rules (the whole list)

1. fp32 → Q.16 at LOAD: bit-exact integer conversion (mantissa/
   exponent extraction, no float ops), round-half-EVEN, then saturate
   to the target format. This conversion IS part of the FX-V1 model
   definition.
2. Every rescale `Q.32 → Q.16` is arithmetic shift right 16 (= floor).
3. Every integer division is FLOOR division (python `//` semantics;
   C++ implements floor_div explicitly — truncation-toward-zero is a
   spec violation).
4. Table interpolation: `y = t[i] + ((t[i+1] - t[i]) * frac) >> fbits`
   (arithmetic shift = floor).
5. eps for norms is the PROFILE CONSTANT eps_q32 = 42950 (≈1e-5 in
   Q.32), independent of the container's float eps.

## Transcendentals: container-shipped tables (artifacts, never re-derived)

Tables are generated ONCE at export (fp64 + round-half-even to Q.16),
shipped as f32 tensors whose values are exact integers (< 2^24, so f32
carries them exactly). Cross-platform bit-identity therefore does not
depend on any platform libm.

| tensor            | entries | domain                | semantics |
|-------------------|---------|-----------------------|-----------|
| fx.act.table      | 2049    | x ∈ [-32, 32], step 1/32 | act(x)·2^16 for the configured activation (gelu / gelu_tanh / silu; relu is computed directly). Clamp: x < -32 → 0, x ≥ 32 → x (identity tail). |
| fx.exp.table      | 2049    | d ∈ [-16, 0], step 1/128 | exp(d)·2^16; attention softmax inputs are max-subtracted (exact int compare) then clamped to the domain. |
| fx.rsqrt.table    | 385     | m ∈ [1, 4], step 1/128   | 2^16/sqrt(m); inputs normalized by even bit-shifts (see below). |
| fx.rope.cos/.sin  | [max_seq, dh/2] | —             | cos/sin(t·theta^(-2p/dh))·2^16, only when pos=rope. |

rsqrt normalization: for v (Q.32, > 0), k = bit_width(v)-1, s = k-31
(incremented to even), m = shift(v, s) ∈ [2^30, 2^32) — the low-bit
truncation in that shift is declared. Result = lerp(m) >> (s/2 - 1)
(arithmetic, either direction).

## Layer recipe (order fixed; all sats as tabled above)

- embeddings: Q.16 lookup (+ learned pos add) → act-sat.
- linear: acc(Q.32) = Σ act·w + (bias << 16); out = (acc >> 16),
  act-sat.
- norm (layernorm): S = Σx; mean = floor_div(S, D) [Q.16];
  c_i = sat9.16(x_i - mean); v = floor_div(Σ c_i², D) [Q.32];
  inv = rsqrt_fx(v + eps_q32) [Q.16];
  y_i = ((((c_i·inv) >> 16)·g_i) >> 16) + b_i, act-sat.
  rmsnorm: same with mean = 0.
- attention: scores = ((dot(q,k) >> 16)·scale) >> 16 where scale =
  rsqrt_fx(dh << 32); causal; m = max (int compare); e_u =
  exp_table(clamp(score_u - m)); Z = Σe; w_u = floor_div(e_u << 16, Z);
  out = (Σ w_u·v_u) >> 16, act-sat.
- rope (if configured): (a·c - b·s) >> 16, (a·s + b·c) >> 16 per pair,
  half or interleaved indexing per config.
- ffn: linear → activation table → linear.
- ffn (swiglu, v1.1 / axnn_minor 1, relay 2026-07-28-4): g = linear
  gate; u = linear up; g_i = act_table(g_i); h_i = sat((g_i·u_i) >> 16)
  [Q.16·Q.16 → shift-16 = floor, the declared rounding]; out = linear
  down over h. Fused qkv (attn_fused): one [3D,D] linear, rows split
  q|k|v — Q.16 conversion is per-weight, so fused ≡ unfused bit-exactly
  (tested).
- readout: ln_f then head matmul; logits stay Q.32 int64, NO rescale.

## Greedy gate mode

argmax over exact Q.32 logits — ties broken by LOWEST token id
(declared). No softmax anywhere in the readout. Sampling mode
(declared exp + exact division + integer threshold) is a later rung
under its own gate.

## Hashing (the acceptance surface)

- per-prompt: FNV-1a 64 over the last-position logits as little-endian
  int64 words.
- battery: FNV-1a 64 over the per-prompt hashes in file order.

## Acceptance

1. axiom C++ (`axiom-nn-exact`) and the independent pure-integer
   python reference (`scripts/nn_exact_ref.py`) produce IDENTICAL
   per-prompt and battery hashes on 100 prompts at the crystal shape
   (two independent implementations of this spec agreeing bit-exact).
2. House side: same container + spec on Mac / 3080 must reproduce the
   battery hash; then the amendment arm runs (exact-gate v
   rounded-gate, same weights, paired).
