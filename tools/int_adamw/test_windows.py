"""Window-cycling gate: NW=1 windows path must reproduce the
certified no-windows mb trajectory digest exactly, and NW=2 with
distinct windows must diverge from it."""
import struct
import sys

sys.path.insert(0, sys.argv[1])
import intbirth  # noqa: E402

C = {"V": 64, "T": 32, "D": 64, "DH": 16, "F": 128,
     "n_blocks": 2, "SHIFT": 14}
tables = open("r2b_tables.bin", "rb").read()
init = open("mb_init.bin", "rb").read()
T = C["T"]
tail = 2 * T * 8                      # tok + tgt
params, win = init[:-tail], init[-tail:]

a = intbirth.MultiBirth(tables, init, C)
b = intbirth.MultiBirth(tables, params, C, windows_bytes=win)
a.run(20)
b.run(20)
assert a.mark() == b.mark(), "NW=1 must equal no-windows path"

# two distinct windows -> different trajectory
tok2 = struct.pack("<%dq" % T, *([1] * T))
tgt2 = struct.pack("<%dq" % T, *([2] * T))
c = intbirth.MultiBirth(tables, params, C,
                        windows_bytes=win + tok2 + tgt2)
c.run(20)
assert c.mark() != a.mark(), "NW=2 must diverge"
print("WINDOWS PASS")
