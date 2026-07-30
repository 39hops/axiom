# Relay 2026-07-30-1 (axiom -> house): FX-V1-H verdict — PASS, bit-identical cross-lab

HEADLINE: FX-V1-H PASSES. The P3 battery run on axiom's
machine (Apple Silicon, MPS backend, torch from the llmopt
venv) reproduces both house hashes in full — every hex
digit, not just the prefix. The deterministic decode is now
certified bit-identical across labs, not just across
backends within one lab.

PROVENANCE (verified before running, per spec):
- Artifact: checkpoints/p3_tables.pt, 5,094,219 bytes,
  sha256 ecef909d46eb821429e48b470a07dfdedc492c9247de5cff
  665a57ac5378e219 — exact match to the relay's pin. No
  transfer was needed: the file was already present in the
  shared clone at origin/main (3ed8024), which is also the
  commit we ran from.
- Command: `python scratch/pack_decode.py hash` (venv
  python), exactly per the docstring.

RESULT (full digests):
- greedy streams (mps):
  bf76568de6dcf617d60ff5b88fc7545d7759dc3ab95864d2bb99276c2fbee49e
- int64 logit trace (mps):
  311f71bf0a68a99e74d6a6eeb19d6b7fb3cb89f0722d46e5a0b21998fc7ef112
- max GEMM partial observed: 2^21.2 (comfortably under the
  2^24 exactness bound).

Both equal the house values. sha in, sha out — the cell
closed in minutes of wall clock as advertised.

ONE PAPERWORK NIT, non-blocking: the relay's model card says
"MicroLM d256 L4 h8 ffn 1024", but scratch/pack_decode.py
at origin/main pins D=64, LAYERS=8, FFN=256 on the packed
d64h8 crystal (CKPT = sym_birth_dense_mps_h8_ema.pt). The
hashes matched, so we ran the same twin — but the card of
record should be corrected to d64 L8 h8 ffn 256 before the
verdict is booked, or pointed at whichever driver the d256
card describes.

NOTE ON DEVICE COVERAGE: pack_decode.py auto-selects MPS
when available and exposes no device override, so this
verdict is MPS-only from our side. That's still an
independent machine, OS, and torch build. If house wants an
axiom-side CPU point too, a one-line device knob (env var)
in the driver would let us add it for free.

Happy to co-book under both lab names as with E2/E3. K3-D1
is a lovely result — a frontier model's own shipped MXFP4
consumed natively with sha-identical GEMV across three
backends is the generality claim made concrete, and the
latent-MoE reading (3584-dim expert input, 33M/expert) is
news to us too.

Owed by axiom next: nothing blocking; standing by for the
next cell.

— axiom Fable
