"""E=1 MoE body must match the dense body bit-exactly (fwd + all
shared grads + dx0), and its expert grads must equal the dense
wg/wu/wd grads. Runs on the r2b fixture weights. The E=1 gate is
exact because softmax over one expert gives p = PQ, so the fx3
gate y = rdiv(out*PQ, PQ) = out, and softmax_bwd's inner =
rdiv(PQ*dp, PQ) = dp makes the router grad exactly zero."""
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
import intbirth  # noqa: E402

C = {"V": 64, "T": 32, "D": 64, "DH": 16, "F": 128, "SHIFT": 12}
tables = open("r2b_tables.bin", "rb").read()
init = open("r2b_init.bin", "rb").read()

# unpack the 11 KEYS tensors at Q scale
shapes = [("wq", (16, 64)), ("wk", (16, 64)), ("wv", (16, 64)),
          ("wo", (64, 16)), ("wg", (128, 64)), ("wu", (128, 64)),
          ("wd", (64, 128)), ("wh", (64, 64)), ("g1", (64,)),
          ("g2", (64,)), ("g3", (64,))]
w, off = {}, 0
buf = np.frombuffer(init, dtype="<i8")
for k, s in shapes:
    n = int(np.prod(s))
    w[k] = buf[off:off + n].reshape(s).copy()
    off += n
x = buf[off:off + 32 * 64].reshape(32, 64).copy()

blk = intbirth.Block(tables, C)
Cm = dict(C)
Cm["E"] = 1
blkm = intbirth.Block(tables, Cm)

wm = {k: w[k] for k in ("wq", "wk", "wv", "wo", "g1", "g2")}
wm["wr"] = np.ones((1, 64), dtype=np.int64)
wm["e0.wg"], wm["e0.wu"], wm["e0.wd"] = w["wg"], w["wu"], w["wd"]
wd = {k: w[k] for k in ("wq", "wk", "wv", "wo", "g1", "g2",
                        "wg", "wu", "wd")}

x2d, cd = blk.body_fwd(wd, x)
x2m, cm = blkm.moe_body_fwd(wm, x)
assert (x2d == x2m).all(), "fwd parity"

rng = np.random.RandomState(7)
dxin = rng.randint(-512, 512, size=(32, 64)).astype(np.int64)
Gd, dx0d = blk.body_bwd(wd, dxin, cd)
Gm, dx0m = blkm.moe_body_bwd(wm, dxin, cm)
assert (dx0d == dx0m).all(), "dx0 parity"
for k in ("wq", "wk", "wv", "wo", "g1", "g2"):
    assert (Gd[k] == Gm[k]).all(), k
for a, b in (("wg", "e0.wg"), ("wu", "e0.wu"), ("wd", "e0.wd")):
    assert (Gd[a] == Gm[b]).all(), b
assert (Gm["wr"] == 0).all(), "E=1 router grad must be exactly zero"
print("MOE PARITY PASS")
