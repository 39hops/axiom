"""Gravmoe acceptance: assert init sha + param_order + per-family
draw bounds (verify-the-knob: PRINT the bound per weight family at
arm start; a reproduction that cannot show its bounds is not a
reproduction), then reproduce FINAL trajectory shas from pins.json.

Refuse-if-disagree: any mismatch in param_order or init/windows sha
aborts before running a single step.

NOTE: the pins.json / contract.json field names below are a best
guess at the house schema; when the artifacts land, adapt the FIELD
ACCESS ONLY (never the engine). If the shipped contract disagrees
with the engine's param_order or shapes, that is a
refuse-if-disagree relay item, not something to paper over here.

Usage (from tools/int_adamw):
  python verify_gravmoe.py <build_dir> <ref_dir> [arm ...]
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, sys.argv[1])
import intbirth  # noqa: E402

ref_dir = sys.argv[2]
pins = json.load(open(os.path.join(ref_dir, "pins.json")))
tables = open("r2b_tables.bin", "rb").read()

# engine-side arms only: no TAU / GATE / SS knobs in this leg
SKIP_KEYS = ("TAU", "GATE", "SS")
arms = sys.argv[3:] or [a for a, e in pins.items()
                        if not any(int(e.get(k, 0)) for k in SKIP_KEYS)]

ok = True
for arm in arms:
    env = pins[arm]
    if any(int(env.get(k, 0)) for k in SKIP_KEYS):
        print(f"{arm}: SKIP (house-side knob)")
        continue
    cell = env["cell"]  # which {init,windows,contract} triple
    contract = json.load(
        open(os.path.join(ref_dir, f"{cell}_contract.json")))
    init = open(os.path.join(ref_dir, f"{cell}_init.bin"),
                "rb").read()
    win = open(os.path.join(ref_dir, f"{cell}_windows.bin"),
               "rb").read()
    assert hashlib.sha256(init).hexdigest() == contract["init_sha"], \
        f"{arm}: init sha DISAGREE - refusing"
    assert hashlib.sha256(win).hexdigest() == \
        contract["windows_sha"], \
        f"{arm}: windows sha DISAGREE - refusing"
    print(f"{arm}: draw bounds per family:")
    for fam, bound in contract["draw_bounds"].items():
        print(f"  {fam}: +-{bound}")
    c = dict(contract["contract"])
    for k in ("E", "K", "LN", "LD", "SHIFT"):
        if k in env:
            c[k] = int(env[k])
    steps = int(env.get("STEPS", c.pop("STEPS", 2000)))
    c.pop("STEPS", None)
    m = intbirth.MoeBirth(tables, init, c, windows_bytes=win)
    assert list(m.param_order) == contract["param_order"], \
        f"{arm}: param_order DISAGREE - refusing"
    # digest rolls at step % max(125, STEPS//8) == 0 (relay -6)
    interval = max(125, steps // 8)
    done = 0
    sha = None
    while done < steps:
        n = min(interval, steps - done)
        m.run(n)
        done += n
        if done % interval == 0 or done == steps:
            sha = m.mark()
    good = sha == env["final_sha"]
    ok &= good
    print(f"{arm}: steps {steps} loss {m.loss} nz {m.nz:.3f} "
          f"{'PASS' if good else 'FAIL want ' + env['final_sha'][:16]}")
print("GRAVMOE PASS" if ok else "GRAVMOE FAIL")
sys.exit(0 if ok else 1)
