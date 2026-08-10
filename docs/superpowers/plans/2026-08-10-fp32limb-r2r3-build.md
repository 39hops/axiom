# FP32LIMB R2/R3 Build Plan (CPU-legal phase)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans.

**Goal:** Build the entire Metal harness, R2 single-simdgroup kernel, R3
tiled choreography (two instantiations), compile probes, and timing rig —
compile-verified on CPU — with GPU dispatch left behind a mode flag until
the house ping (crown battery gate).

**Architecture:** Objective-C++ rig (`tools/fp32limb/metal/r2_rig.mm`)
linking `libaxiom.a` (reuses R1 `slice_row`/`gemm_ref`/`acc`/`dyadic_eq`
as the immovable oracle) + Metal/Foundation frameworks. MSL kernels in
`fp32limb_kernels.metal`, compiled at RUNTIME via `newLibraryWithSource`
with fast-math OFF (`MTLCompileOptions.mathMode = MTLMathModeSafe` /
`fastMathEnabled = NO`) — the pinned-flags receipt item. Probes compile
tiny sources and record success/error strings (answers int-simdgroup-MMA
and 64-bit-int questions without dispatch).

**Modes:** `compile` (default; CPU-legal: device handle + shader compile +
probes, no command buffer ever committed), `ftz` / `biteq` / `wall`
(refuse to run unless `--gpu-ok` flag given — the house-ping interlock).

## Global Constraints
- Inherit `_f24`: registered constants unchanged (s=7, b=32, MAX_SLICES=8).
- No command buffer commit in `compile` mode; `--gpu-ok` required otherwise.
- P-KERNEL-BITEQ: kernel partials vs R1 CPU partials, integer/bit equality
  only; final recombine through the same bigint path vs `gemm_ref`.
- R3 wall harness: every iteration evaluated (`waitUntilCompleted` per
  rep), n>=5 reps, median+spread; CPU fp64 baseline matched N.
- Tasks: (1) kernels.metal [r2 kernel, ftz probe, r3 tiled fp32limb +
  integer-acc instantiations]; (2) r2_rig.mm harness+modes; (3) compile
  mode run green incl. probes; (4) commit + relay.
