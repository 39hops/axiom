"""Engine-module acceptance: drive intbirth.FullBirth through the
R2b contract and compare every milestone digest + loss against
r2b_ref.json — the same protocol as every leg (digests engine-side,
comparison house-side; this script plays house).

Usage (from tools/int_adamw, with the built module on the path):
  python verify_intbirth.py <build_dir> [r2b_ref.json]
"""
import json
import sys

sys.path.insert(0, sys.argv[1])
import intbirth  # noqa: E402

ref_path = sys.argv[2] if len(sys.argv) > 2 else \
    "/Users/artin/code/llmopt/scratch/detbwd_r2b_ref/r2b_ref.json"
ref = json.load(open(ref_path))
tables = open("r2b_tables.bin", "rb").read()
init = open("r2b_init.bin", "rb").read()

fb = intbirth.FullBirth(tables, init, ref["contract"])
prev = 0
ok = True
for ms in ref["milestones"]:
    fb.run(ms["step"] - prev)
    prev = ms["step"]
    sha = fb.mark()
    good = sha == ms["traj_sha"] and fb.loss == ms["loss"]
    ok &= good
    print(f"step {ms['step']:4d} loss {fb.loss} nz {fb.nz:.3f} "
          f"{'PASS' if good else 'FAIL (want ' + ms['traj_sha'][:16] + ')'}")
wb = fb.weights_bytes()
print(f"weights_bytes: {len(wb)} bytes")
print("ENGINE PASS" if ok else "ENGINE FAIL")
sys.exit(0 if ok else 1)
