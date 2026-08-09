#!/usr/bin/env python3
"""Cross-arm divergence readout over run_anchor.py dumps.

For each step present in BOTH arms of a pair, compares the de-grained
(shipped-scale i64) weight snapshots: mean/max abs delta and the
first step at which the arms diverge. Emits one JSONL row per step.

Usage: python divergence.py <dump_dir> [--pairs a-b c-d ...]
"""
import argparse
import glob
import json
import os
import struct

ap = argparse.ArgumentParser()
ap.add_argument("dump_dir")
ap.add_argument("--pairs", nargs="*", default=[
    "anchor-q9", "anchor-q32", "anchor-q64", "q9-q32", "q32-q64"])
args = ap.parse_args()


def load(arm, step):
    p = os.path.join(args.dump_dir, f"{arm}_step{step}.w9")
    if not os.path.exists(p):
        return None
    raw = open(p, "rb").read()
    return struct.unpack(f"<{len(raw) // 8}q", raw)


def steps_for(arm):
    return sorted(int(f.rsplit("step", 1)[1].split(".")[0])
                  for f in glob.glob(
                      os.path.join(args.dump_dir, f"{arm}_step*.w9")))


rows = []
firsts = {}
all_steps = sorted({s for p in args.pairs for a in p.split("-")
                    for s in steps_for(a)})
for step in all_steps:
    row = {"step": step, "pairs": {}}
    for pair in args.pairs:
        a, b = pair.split("-")
        wa, wb = load(a, step), load(b, step)
        if wa is None or wb is None:
            continue
        deltas = [abs(x - y) for x, y in zip(wa, wb)]
        mx = max(deltas)
        row["pairs"][pair] = {
            "mean": sum(deltas) / len(deltas), "max": mx}
        if mx > 0 and pair not in firsts:
            firsts[pair] = step
    rows.append(row)

out = os.path.join(args.dump_dir, "divergence.jsonl")
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.write(json.dumps({"first_divergence_step": firsts}) + "\n")
print(json.dumps({"first_divergence_step": firsts}, indent=2))
print("->", out)
