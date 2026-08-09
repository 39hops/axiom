# ENGINE-EXACT-1: the precision ladder + exact-prefix anchor

Status: GO (Artin, 2026-08-08 night; llmopt accepted relay
2026-08-08-3's counter-design verbatim). Fence: does NOT entangle with
ENGINE-SCALE-1's 30 cells; that battery fires first. CPU-only,
Mac-co-location friendly.

## Question under test

Does the learned object change as ring-op rounding grain → 0?
Absorption predicts rung-to-rung trajectory divergence → 0; structure
predicts it persists. Both sides' pre-regs register before any cell
fires (llmopt restates disagreement-#3 in ladder-limit form).

The literal zero-rounding arm is dead by the Ω(2^steps) bit-growth law
(relay 2026-08-08-3, measured receipts); the law books as its own leg.

## Design: three rungs + one anchor

Rungs are operand scales. CORRECTION over the relay's shorthand: the
shipped intbirth grain is **Q = 512 = 2^9** (the "Q.16 operands" line
in the README is the `ax::nn::exact` inference subsystem, a different
engine). The ladder is therefore p ∈ {9 (shipped, bit-identical), 32,
64}, written Q9/Q32/Q64 — identical rounding *placement* everywhere
(the placement contract is untouched; only the grain shrinks). The anchor is a dyadic-exact run
(ring ops exact, zero value-rounding) feasible for a short prefix
(target 8–12 steps at d64-class), giving the ladder a ground-truth
convergence point.

## Convention pin (the contract addition)

One rule covers every non-ring site, and it is chosen to make rungs
*comparable* rather than individually sharper:

> **Frozen-grain convention.** Every transcendental or
> algebraic-irrational site (exp table, RoPE sin/cos tables, rmsnorm
> isqrt carry, attention-scale isqrt, AdamW sqrt) evaluates at the
> frozen SHIPPED grain at every rung — operand grain Q = 2^9 and each
> site's shipped carry scale (R16 = 2^16 rmsnorm, RS = 2^14 RoPE,
> pq attention-prob carry, eps32 at 2^32): the input is
> floor-truncated from rung scale to shipped scale, the existing
> shipped table/exact-isqrt convention is applied unchanged, and the
> result is left-shifted back to rung scale.

Rationale: if each rung regenerated finer tables, transcendental
resolution would move together with ring-rounding grain and the
measurement would confound the two. Freezing the grain means
divergence-vs-p isolates exactly the quantity under test — ring-op
rounding — and the Q9 rung is bit-identical to the shipped engine by
construction (truncate+shift are no-ops at p=9). Tables therefore stay
opaque shipped bytes at all rungs (doctrine unchanged); no table
regeneration exists in this design after all.

The dyadic anchor uses the same frozen-grain conventions, so
anchor-vs-rung divergence is also pure ring-rounding.

## Contract changes

`contract` gains `int precision = 9` (operand scale Q_p = 2^precision).
Derived constants scale by declared rules, all in the spec, none
recomputed from libm:

- `Q = 1 << precision`; `Q_w = Q << shift` (shift unchanged).
- Constants carrying operand scale (`act_clamp`, attn scale input)
  left-shift by `(precision - 9)`; squared-scale constants (`eps32`
  basis) shift by `2*(precision - 9)`; carry scales frozen by the
  convention pin (`pq`, `gboost`, R16, RS) do not move.
- lr stays `lrn/lrd` (scale-free rational).

Digest gates: at `precision = 9` every existing reference digest
(r2b_ref, multiblock, gravmoe, test_set_lr, test_windows) must pass
unchanged — that is the ladder's own no-op gate.

## Arithmetic widths

- Q9 (today): i64 operands, i64 accumulation. Unchanged.
- Q32: i64 operands (values fit), `__int128` accumulation and product
  sites; rdiv at 128 bits.
- Q64: 128-bit operands, 256-bit accumulation. Implemented over a
  minimal fixed-width `i256` helper (add/mul/rdiv/shift/compare) rather
  than ax::core bigint — the hot loop wants fixed width; bigint remains
  the AdamW bias-correction path as today.
- Engine core becomes a width-templated implementation
  (`template <class Op, class Acc, class Round>`, `Round` =
  round-half-away at grain vs exact for the anchor); the shipped i64
  symbols remain the Q9 instantiation, ABI-compatible.

## Anchor implementation

The same templated core instantiated with a slow, obviously-correct
scalar and the `Exact` rounding policy — same frozen-grain
conventions, same rounding placement *sites*, but the round-at-grain
step replaced by exact arithmetic. Budget guard: abort loudly if any
scalar exceeds a declared bit ceiling (default 2^22 bits).

BUILD CORRECTIONS (as-implemented, 2026-08-08 night):

- The scalar is a full **rational** (`ax::rational` on bigint), NOT
  dyadic as first drafted: AdamW divides by 10/1000/100000, which
  leaves the dyadic domain. (`include/ax/nn/exact_anchor.hpp`.)
- eps32 does NOT scale with the rung: the rms `/(Q*Q)` normalization
  pins the isqrt input at fixed 2^32 scale at every grain.
- Causal floor convention: `-2^min(40+gshift, Op_bits-2)` — the
  naive re-embed overflows i64 at Q32; the cap is
  semantics-preserving (anything below every real logit floors the
  softmax term to 0) and bit-identical at Q9 (digest-gated).
- Rounding-policy division is width-generic (template members): a
  policy typed on the operand silently pre-narrowed wide
  accumulators before dividing on the builtin rungs; i256 caught it
  at compile time.
- Q64 scope: the dense composed loop (`full_birth`) only; multi/moe
  are wired through Q32. No registered rung needs Q64 MoE.
- Anchor wall-clock reality: step 2 is already minutes-class at toy
  size (d=8) — the measured d64 horizon is booked in the build
  relay, not predicted here.

## Deliverables and books

1. Law leg: bit-growth receipts (toy-loop tables from relay
   2026-08-08-3) booked as their own entry, both ledgers.
2. Engine: `precision` contract field + Q32/Q64 rungs, digest-gated
   (no-op gate at Q9; new per-rung reference digests booked before
   any comparison run).
3. Anchor binary/driver emitting per-step weight digests + milestone
   values for the prefix.
4. Divergence readout: per-step weight-space divergence anchor-vs-rung
   and rung-vs-rung over the shared prefix; capability race llmopt-side
   as a Mac-window WORK ORDER after ENGINE-SCALE-1 clears.

## Non-goals

- No changes to ENGINE-SCALE-1 cells, contracts, or schedule.
- No GPU work (d768 battery owns the GPU).
- No new transcendental resolution at any rung (see convention pin).
