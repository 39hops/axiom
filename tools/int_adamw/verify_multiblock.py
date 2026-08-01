"""Multi-block acceptance: drive intbirth.MultiBirth through the mb
contract and compare every milestone digest + loss against
mb_ref.json (house reference, two independent house drivers agree).

Usage (from tools/int_adamw):
  python verify_multiblock.py <build_dir> [mb_ref.json]
"""
import json
import sys

sys.path.insert(0, sys.argv[1])
import intbirth  # noqa: E402

ref_path = sys.argv[2] if len(sys.argv) > 2 else \
    "/Users/artin/code/llmopt/scratch/detbwd_mb_ref/mb_ref.json"
ref = json.load(open(ref_path))
tables = open("r2b_tables.bin", "rb").read()  # same dims as R2b
init = open("mb_init.bin", "rb").read()

mb = intbirth.MultiBirth(tables, init, ref["contract"])
assert list(mb.param_order) == ref["param_order"], "param order"
prev, ok = 0, True
for ms in ref["milestones"]:
    mb.run(ms["step"] - prev)
    prev = ms["step"]
    sha = mb.mark()
    good = sha == ms["traj_sha"] and mb.loss == ms["loss"]
    ok &= good
    print(f"step {ms['step']:4d} loss {mb.loss} nz {mb.nz:.3f} "
          f"{'PASS' if good else 'FAIL (want ' + ms['traj_sha'][:16] + ')'}")
print("MULTIBLOCK PASS" if ok else "MULTIBLOCK FAIL")
sys.exit(0 if ok else 1)
