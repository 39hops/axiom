"""Primitive-layer acceptance: rebuild the R2b training loop IN
PYTHON from intbirth's primitives (Block / AdamW / rdiv) and check
it reproduces the same r2b_ref.json milestone digests as the
composed FullBirth — house-side composition, certified against the
certified trajectory.

Usage (from tools/int_adamw):
  python verify_primitives.py <build_dir> [r2b_ref.json]
"""
import hashlib
import json
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
import intbirth  # noqa: E402

ref_path = sys.argv[2] if len(sys.argv) > 2 else \
    "/Users/artin/code/llmopt/scratch/detbwd_r2b_ref/r2b_ref.json"
ref = json.load(open(ref_path))
ct = ref["contract"]
Q, SHIFT, GBOOST = 512, ct["SHIFT"], ct["GBOOST"]
KEYS = ("wq", "wk", "wv", "wo", "wg", "wu", "wd", "wh",
        "g1", "g2", "g3")
T, D, DH, F, V = 32, 64, 16, 128, 64
SHAPES = {"wq": (DH, D), "wk": (DH, D), "wv": (DH, D), "wo": (D, DH),
          "wg": (F, D), "wu": (F, D), "wd": (D, F), "wh": (V, D),
          "g1": (D,), "g2": (D,), "g3": (D,)}

raw = np.frombuffer(open("r2b_init.bin", "rb").read(), dtype="<i8")
off, wide = 0, {}
for k in KEYS:
    n = int(np.prod(SHAPES[k]))
    wide[k] = (raw[off:off + n].reshape(SHAPES[k]) << SHIFT).copy()
    off += n
x = raw[off:off + T * D].reshape(T, D).copy()
off += T * D
tgt = raw[off:off + T].copy()

blk = intbirth.Block(open("r2b_tables.bin", "rb").read(), ct)
opt = intbirth.AdamW(SHIFT, ct.get("LRN", 1), ct.get("LRD", 1000))
onehot = np.zeros((T, V), dtype=np.int64)
onehot[np.arange(T), tgt] = 1

th = hashlib.sha256()
prev, ok = 0, True
for ms in ref["milestones"]:
    for _ in range(ms["step"] - prev):
        w = {k: intbirth.rdiv(wide[k], 1 << SHIFT) for k in KEYS}
        logits, cache = blk.fwd(w, x)
        pp = blk.softmax_rows(logits, Q)
        loss = int((Q - pp[np.arange(T), tgt]).sum())
        dlogits = (pp - Q * onehot) * GBOOST
        G, _dx0 = blk.bwd(w, dlogits, cache)
        grads = [intbirth.rdiv(G[k], Q * GBOOST) for k in KEYS]
        opt.step([wide[k] for k in KEYS], grads)
    prev = ms["step"]
    for k in KEYS:
        th.update(wide[k].tobytes())
    good = th.hexdigest() == ms["traj_sha"] and loss == ms["loss"]
    ok &= good
    print(f"step {ms['step']:4d} loss {loss} nz {opt.nz:.3f} "
          f"{'PASS' if good else 'FAIL'}")
print("PRIMITIVES PASS" if ok else "PRIMITIVES FAIL")
sys.exit(0 if ok else 1)
