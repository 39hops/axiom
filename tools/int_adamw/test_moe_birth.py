"""MoeBirth internal gates (no house pins yet): param_order matches
the relay contract, 208,192 params at gravmoe defaults, determinism,
and the gravity rdiv placement (all-zero pulls must be a no-op on
the trajectory; real pulls must move it)."""
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
import intbirth  # noqa: E402

C = {"V": 40, "T": 32, "D": 64, "DH": 16, "F": 128, "n_blocks": 2,
     "SHIFT": 14, "E": 4, "K": 100, "LN": 0, "LD": 1}
tables = open("r2b_tables.bin", "rb").read()

rng = np.random.RandomState(17)
order = ["emb"]
sizes = {"emb": 40 * 64}
for b in range(2):
    for k, n in (("wq", 16 * 64), ("wk", 16 * 64), ("wv", 16 * 64),
                 ("wo", 64 * 16), ("g1", 64), ("g2", 64),
                 ("wr", 4 * 64)):
        order.append(f"b{b}.{k}")
        sizes[f"b{b}.{k}"] = n
    for e in range(4):
        for k, n in (("wg", 128 * 64), ("wu", 128 * 64),
                     ("wd", 64 * 128)):
            order.append(f"b{b}.e{e}.{k}")
            sizes[f"b{b}.e{e}.{k}"] = n
order.append("g_f")
sizes["g_f"] = 64
assert sum(sizes.values()) == 208192

parts = []
for name in order:
    if name.split(".")[-1].startswith("g"):  # norms draw at Q
        v = np.full(sizes[name], 512, dtype=np.int64)
    else:
        v = rng.randint(-64, 65, size=sizes[name]).astype(np.int64)
    parts.append(v.tobytes())
tok = rng.randint(0, 40, size=32).astype(np.int64)
tgt = rng.randint(0, 40, size=32).astype(np.int64)
win = tok.tobytes() + tgt.tobytes()
init = b"".join(parts)


def run(c, steps=120):
    m = intbirth.MoeBirth(tables, init, c, windows_bytes=win)
    assert list(m.param_order) == order
    m.run(steps)
    return m.mark()


a = run(C)
assert a == run(C), "determinism"
c2 = dict(C)
c2["LN"], c2["LD"] = 1, 10**9
assert run(c2) == a, "all-zero pulls must not move the trajectory"
c3 = dict(C)
c3["LN"], c3["LD"] = 1, 2
assert run(c3) != a, "real gravity must move the trajectory"
print("MOEBIRTH PASS")
