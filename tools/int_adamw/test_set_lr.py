"""set_lr gate (engine-scale spec, relay 2026-08-07): a segmented
run with no-op set_lr calls must reproduce the one-shot trajectory
digest exactly, and an lrd x2 schedule point must diverge."""
import sys

sys.path.insert(0, sys.argv[1])
import intbirth  # noqa: E402

C = {"V": 64, "T": 32, "D": 64, "DH": 16, "F": 128,
     "n_blocks": 2, "SHIFT": 14}
tables = open("r2b_tables.bin", "rb").read()
init = open("mb_init.bin", "rb").read()

a = intbirth.MultiBirth(tables, init, C)
a.run(40)

b = intbirth.MultiBirth(tables, init, C)
for _ in range(4):
    b.run(10)
    b.set_lr(1, 1000)          # same lr — must be a byte no-op
assert a.mark() == b.mark(), "no-op set_lr must be digest-identical"

c = intbirth.MultiBirth(tables, init, C)
c.run(20)
c.set_lr(1, 2000)              # the SCHED arm shape: lrd x2 mid-run
c.run(20)
assert c.mark() != a.mark(), "schedule change must diverge"
print("SET_LR PASS")
