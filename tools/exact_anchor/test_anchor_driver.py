"""Driver smoke on a tiny fixture: rungs + anchor 1 step, dumps
exist, divergence readout runs and reports anchor-q9 first-step
state. Usage: python test_anchor_driver.py <build_dir>"""
import json
import os
import struct
import subprocess
import sys
import tempfile

BUILD = sys.argv[1] if len(sys.argv) > 1 else "../../build-rel"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, BUILD) if not os.path.isabs(
    BUILD) else BUILD)


def u16(v): return struct.pack("<H", v)
def u32(v): return struct.pack("<I", v)
def u64(v): return struct.pack("<q", v)


def tensor(name, vals):
    return (u16(len(name)) + name.encode() + bytes([1]) +
            struct.pack("<Q", len(vals)) +
            b"".join(u64(v) for v in vals))


T, D, DH, F, V = 4, 8, 4, 16, 8
ts, tse, RS = 100, 50, 1 << 14
tables = (b"AXP3" + u32(5) +
          tensor("silu.tab", [(i - ts) // 2 for i in range(2 * ts + 1)]) +
          tensor("dsilu.tab", [256] * (2 * ts + 1)) +
          tensor("exp.tab", [1 + i * 7 for i in range(tse + 1)]) +
          tensor("rope.cos", [RS] * (T * (DH // 2))) +
          tensor("rope.sin", [0] * (T * (DH // 2))))
import random
random.seed(613)
sizes = [DH * D] * 3 + [D * DH, F * D, F * D, D * F, V * D, D, D, D]
init = b"".join(u64(random.randint(-256, 256))
                for n in sizes for _ in range(n))
init += b"".join(u64(random.randint(-256, 256)) for _ in range(T * D))
init += b"".join(u64(t % V) for t in range(T))

with tempfile.TemporaryDirectory() as td:
    tf, inf = os.path.join(td, "t.bin"), os.path.join(td, "i.bin")
    open(tf, "wb").write(tables)
    open(inf, "wb").write(init)
    out = os.path.join(td, "out")
    contract_env = dict(T=T, D=D, DH=DH, F=F, V=V)
    # drive via the module directly with the tiny contract (the CLI
    # defaults to r2b dims; tiny dims need the dict override)
    sys.path.insert(0, BUILD)
    import intbirth
    os.makedirs(out)
    for prec in (9, 32, 64):
        fb = intbirth.FullBirth(tables, init,
                                {**contract_env, "PRECISION": prec})
        fb.run(1)
        open(os.path.join(out, f"q{prec}_step1.w9"), "wb").write(
            fb.weights_grain9_bytes())
    an = intbirth.ExactAnchor(tables, init, contract_env)
    an.run(1)
    open(os.path.join(out, "anchor_step1.w9"), "wb").write(
        an.weights_grain9_bytes())
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "divergence.py"), out],
        capture_output=True, text=True, check=True)
    lines = [json.loads(l) for l in
             open(os.path.join(out, "divergence.jsonl"))]
    assert lines[0]["step"] == 1
    assert "anchor-q9" in lines[0]["pairs"], lines[0]
    sizes_ok = all(os.path.getsize(os.path.join(out, f)) ==
                   os.path.getsize(os.path.join(out, "q9_step1.w9"))
                   for f in os.listdir(out) if f.endswith(".w9"))
    assert sizes_ok
print("ANCHOR DRIVER PASS")
