#!/usr/bin/env python3
"""ENGINE-EXACT-1 anchor + ladder prefix driver.

Runs the exact-prefix anchor and the Q9/Q32/Q64 rungs step-by-step
over the same inputs, dumping per-step de-grained (shipped-scale i64)
weight snapshots + losses to an output directory for divergence.py.

Usage:
  python run_anchor.py <build_dir> [--steps N] [--ceiling BITS]
                       [--tables F] [--init F] [--out DIR]
                       [--budget SECONDS]

Defaults use the r2b reference inputs (T=32 D=64 DH=16 F=128 V=64 —
the d64-class anchor dims) and stop the anchor early on either the
bit ceiling (loud) or the per-run wall-clock budget (reported, never
silent). Softmax rows do NOT sum exactly to their carry (per-element
rounding) — no reader of these dumps may assume it.
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("build_dir")
ap.add_argument("--steps", type=int, default=12)
ap.add_argument("--ceiling", type=int, default=1 << 22)
ap.add_argument("--tables", default=None)
ap.add_argument("--init", default=None)
ap.add_argument("--out", default="anchor_out")
ap.add_argument("--budget", type=float, default=3600.0)
args = ap.parse_args()

sys.path.insert(0, args.build_dir)
import intbirth  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
tables = open(args.tables or
              os.path.join(HERE, "..", "int_adamw", "r2b_tables.bin"),
              "rb").read()
init = open(args.init or
            os.path.join(HERE, "..", "int_adamw", "r2b_init.bin"),
            "rb").read()
os.makedirs(args.out, exist_ok=True)

CONTRACT = {}  # r2b defaults live in the engine contract


def run_arm(name, obj, steps, budget):
    rows = []
    t0 = time.time()
    for s in range(1, steps + 1):
        try:
            st = time.time()
            obj.run(1)
            dt = time.time() - st
        except RuntimeError as e:  # bit ceiling — loud, expected
            rows.append({"step": s, "aborted": str(e)})
            print(f"{name}: step {s} aborted: {e}", flush=True)
            break
        with open(os.path.join(args.out, f"{name}_step{s}.w9"),
                  "wb") as f:
            f.write(obj.weights_grain9_bytes())
        rows.append({"step": s, "loss": obj.loss,
                     "wall_s": round(dt, 3)})
        print(f"{name}: step {s} loss {obj.loss} ({dt:.2f}s)",
              flush=True)
        if time.time() - t0 > budget:
            rows.append({"step": s, "aborted": "wall-clock budget"})
            print(f"{name}: budget hit after step {s}", flush=True)
            break
    with open(os.path.join(args.out, f"{name}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


for prec in (9, 32, 64):
    fb = intbirth.FullBirth(tables, init, {**CONTRACT,
                                           "PRECISION": prec})
    run_arm(f"q{prec}", fb, args.steps, args.budget)

intbirth.ExactAnchor.set_bit_ceiling(args.ceiling)
an = intbirth.ExactAnchor(tables, init, CONTRACT)
run_arm("anchor", an, args.steps, args.budget)
print("done ->", args.out)
