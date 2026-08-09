"""ENGINE-SCALE-1 grid runner (relay 2026-08-09-1, pre-reg at
llmopt 3b9cdb7 / RESULTS L22317).

Consumes the shipped per-cell artifacts (DIET-BRIDGE layout: init
params in param_items order, then window token ids int64 [NWIN,33])
+ manifest.jsonl, drives intbirth.MultiBirth through each cell's
contract, and emits one jsonl row per cell: milestone losses every
125 steps + FINAL trajectory sha (mark(), the house cumulative
milestone-sha convention, digest-certified by verify_multiblock).

Conventions relied on (each digest-gated in tools/int_adamw):
- windows_bytes = NW x (tok[T] ++ tgt[T]): each shipped 33-id
  window w becomes tok=w[:32], tgt=w[1:33] (true next-token CE).
- segmented run() + no-op set_lr is byte-identical to one-shot
  (test_set_lr.py), so pausing every 125 steps to read .loss does
  not perturb the digest.
- SCHED=1: lrd *= 2 at ABSOLUTE steps 250/500/750 (lrd start 1000,
  the certified s4000-sched convention) — NOT pro-rata with STEPS.

Usage (from tools/engine_scale):
  python run_cells.py <build_dir> [--cells cell1,cell2] [--out out.jsonl]
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import time

CELL_DIR = ("/Users/artin/code/llmopt/docs/superpowers/specs/"
            "engine_scale_cells")
TABLES = os.path.join(os.path.dirname(__file__),
                      "../int_adamw/r2b_tables.bin")
MILESTONE = 125
SCHED_STEPS = (250, 500, 750)


def load_cell(row):
    """Split a shipped bin into (params_bytes, windows_bytes) with
    the axiom engine's tok/tgt window layout; verify shas first."""
    T = row["contract"]["T"]
    nwin = row["contract"]["NWIN"]
    data = open(f"{CELL_DIR}/{row['bin']}", "rb").read()
    assert hashlib.sha256(data).hexdigest() == row["bin_sha"], \
        f"{row['cell']}: bin sha mismatch"
    tail = nwin * (T + 1) * 8
    params, wins = data[:-tail], data[-tail:]
    assert hashlib.sha256(wins).hexdigest() == row["win_sha"], \
        f"{row['cell']}: windows sha mismatch"
    assert len(params) == row["n_params"] * 8, \
        f"{row['cell']}: param count mismatch"
    wb = b""
    for i in range(nwin):
        w = struct.unpack_from(f"<{T + 1}q", wins, i * (T + 1) * 8)
        wb += struct.pack(f"<{T}q", *w[:T])       # tok
        wb += struct.pack(f"<{T}q", *w[1:T + 1])  # tgt
    return params, wb


def run_cell(intbirth, tables, row):
    c = row["contract"]
    C = {"V": c["V"], "T": c["T"], "D": c["DIM"], "DH": c["DHEAD"],
         "F": c["FFN"], "n_blocks": c["NBLK"], "SHIFT": c["SHIFT"]}
    params, wb = load_cell(row)
    mb = intbirth.MultiBirth(tables, params, C, windows_bytes=wb)
    steps, sched = c["STEPS"], c["SCHED"]
    assert steps % MILESTONE == 0, "STEPS not milestone-aligned"
    # house convention: lrd *= 2 BEFORE step s in (250,500,750)
    # executes, so the engine must pause after steps 249/499/749.
    stops = set(range(MILESTONE, steps + 1, MILESTONE))
    if sched:
        stops |= {s - 1 for s in SCHED_STEPS}
    lrd, prev, sha = 1000, 0, ""
    milestones = []
    t0 = time.time()
    for stop in sorted(stops):
        mb.run(stop - prev)
        prev = stop
        if sched and stop + 1 in SCHED_STEPS:
            lrd *= 2
            mb.set_lr(1, lrd)
        if stop % MILESTONE == 0:
            # mark() FEEDS current weights into the cumulative traj
            # hash (the house 125-step convention) — call exactly
            # once per milestone, never elsewhere.
            sha = mb.mark()
            milestones.append({"step": stop, "loss": mb.loss})
    return {
        "cell": row["cell"], "bin": row["bin"],
        "bin_sha": row["bin_sha"], "win_sha": row["win_sha"],
        "n_params": row["n_params"], "contract": c,
        "milestones": milestones,
        "loss_final": milestones[-1]["loss"],
        "final_traj_sha": sha,
        "wall_s": round(time.time() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir")
    ap.add_argument("--cells", default="")
    ap.add_argument("--out", default="engine_scale_results.jsonl")
    a = ap.parse_args()
    sys.path.insert(0, a.build_dir)
    import intbirth  # noqa: E402

    tables = open(TABLES, "rb").read()
    rows = [json.loads(l) for l in
            open(f"{CELL_DIR}/manifest.jsonl")]
    want = set(a.cells.split(",")) if a.cells else None
    with open(a.out, "w") as f:
        for row in rows:
            if want and row["cell"] not in want:
                continue
            r = run_cell(intbirth, tables, row)
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"{r['cell']}: loss {r['loss_final']} "
                  f"traj {r['final_traj_sha'][:16]} "
                  f"({r['wall_s']}s)", flush=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
