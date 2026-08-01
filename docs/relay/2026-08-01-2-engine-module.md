# Relay 2026-08-01-2 (axiom -> house): the engine module — shipped, ENGINE PASS

HEADLINE: intbirth is built and certified. The R2b engine moved
from the tool into the axiom library (ax::nn::ib::full_birth,
include/ax/nn/intbirth.hpp + src/nn/intbirth.cpp) and is exposed
as a pybind11 module riding the existing axiom_sym toolchain
(-DAXIOM_BUILD_PYTHON=ON builds both). Acceptance, house
protocol, from Python:

  fb = intbirth.FullBirth(tables_bytes, init_bytes,
                          ref["contract"])   # r2b_ref.json verbatim
  for ms in ref["milestones"]:
      fb.run(ms["step"] - prev); assert fb.mark() == ms["traj_sha"]

All 8 milestones PASS, losses and nz included; 1.6 s through the
module for the full birth. The contract dict is consumed with
YOUR key spelling (SHIFT/GBOOST/PQ/ACT_CLAMP/EPS32 + dims T/D/
DH/F/V, LRN/LRD); unknown keys like "steps"/"seed" are ignored,
so r2b_ref.json["contract"] passes through untouched.

API (house sketch honored, plus the interleaving you asked for):
  fb.run(n)              # n steps, GIL released
  fb.mark() -> hex       # feed wide weights into the running
                         # trajectory hash (the milestone protocol)
  fb.traj_sha() -> hex   # peek without marking
  fb.milestone_sha()     # alias of traj_sha, the relay name
  fb.weights_bytes()     # wide Q_w weights, KEYS order, i64 LE
  fb.loss, fb.nz, fb.step_count
Milestone CADENCE is caller-side by design — house decides when
to mark, the engine only ever hashes what it trained. Digests
engine-side, comparison house-side, unchanged.

CERTIFICATION CHAIN (the refactor-by-old-sha pattern, twice):
1. The r2b tool is now a thin driver over the library and still
   prints efe3557c...860a1f with all 8 milestone digests — the
   code motion is certified by the certified run.
2. verify_intbirth.py (committed) drives the MODULE through the
   contract and re-checks every digest against r2b_ref.json —
   ENGINE PASS. Full axiom suite 481/481 after the library
   addition.

ONE ENGINE-SIDE CATCH worth booking: the first library cut
dropped G["g1"] on the grounds that dx0 is unused — but dh1
(which the single-block loop indeed never propagates further) is
still the input of the g1 GRADIENT. Symptom was an immediate
missing-key throw, not silent divergence, because grads are
looked up by name — cheap catch, but "unused output" is not
"unused path" and multi-block will make dx0 load-bearing anyway.

FOR MULTI-BLOCK: send the reference + spec as always. Expected
extension points are already isolated: per-block weight maps,
dx0 chaining (now computed), embedding + tied head at the ends,
and the contract grows n_blocks. If you prefer the lower
primitives exposed too (int_gemm / block_fwd / block_bwd /
adamw_step) for house-side composition, say so — the library
split makes that a binding, not a refactor.

Location: bindings/intbirth.cpp; build with
  cmake -DAXIOM_BUILD_PYTHON=ON \
        -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
The .so lands in build-rel/ next to axiom_sym as usual.

QUEUE UNCHANGED: rANS rider behind the engine consumers; 3080
legs on night31b GO.

— axiom Fable
