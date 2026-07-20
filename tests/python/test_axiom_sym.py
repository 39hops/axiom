"""Smoke + thread-safety tests for the axiom_sym pybind11 bridge.

Run: python tests/python/test_axiom_sym.py <path-to-pyd-dir>
(no pytest dependency; STL-only discipline extends to test tooling).
"""
import concurrent.futures
import sys

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "build-rel")
import axiom_sym as ax  # noqa: E402

failures = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failures.append(name)


# parse / str round-trip stays sympy-compatible
e = ax.parse_sstr("5*((3 - 4*x)*sin(log(x*(2*x - 3))) + "
                  "(2*x - 3)*cos(log(x*(2*x - 3))))/(2*x - 3)")
check("monster parses and prints without ^", "^" not in str(e))

# diff
d = ax.diff(ax.parse_sstr("x**3"), "x")
check("diff x**3", ax.equivalent(d, ax.parse_sstr("3*x**2"), "x")
      == "EQUIVALENT")

# canonical
c = ax.canonical(ax.parse_sstr("(x**2 - 1)/(x - 1)"), "x")
check("canonical cancels", c.same(ax.parse_sstr("x + 1")))

# verdict trichotomy incl. the soundness sentinel
check("equivalent", ax.equivalent(ax.parse_sstr("2*x + 2"),
                                  ax.parse_sstr("2*(x + 1)"), "x")
      == "EQUIVALENT")
check("not_equivalent", ax.equivalent(ax.parse_sstr("x**2"),
                                      ax.parse_sstr("x**3"), "x")
      == "NOT_EQUIVALENT")
check("undecided never guessed",
      ax.equivalent(ax.parse_sstr("sin(x)**2 + cos(x)**2"),
                    ax.parse_sstr("1"), "x") == "UNDECIDED")

# equivalent_mod_const
check("mod_const", ax.equivalent_mod_const(ax.parse_sstr("x**2/2 + 7"),
                                           ax.parse_sstr("x"), "x")
      == "EQUIVALENT")

# ValueError on rejects
for bad in ("oo", "foo(x)", "1 ++"):
    try:
        ax.parse_sstr(bad)
        check(f"reject {bad!r}", False)
    except ValueError:
        check(f"reject {bad!r}", True)

# thread safety: hammer the hash-cons pool + oracle from 8 threads; results
# must be deterministic and identical across threads.
def work(k):
    f = ax.parse_sstr(f"5*x*sin(log(x*(2*x - 3)))/(2*x - 3) + {k}")
    g = ax.diff(f, "x")
    return ax.equivalent_mod_const(f, g, "x")

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(work, range(200)))
check("8-thread hammer, 200 tasks, all EQUIVALENT",
      all(r == "EQUIVALENT" for r in results))

# ---- solver bindings (hybrid slots contract)

r = ax.solve(ax.parse_sstr("Integral(3*sin(x) + x**2, x)"))
check("native solve", r["solved"])

calls = []
def _fake_heurisch(node_sstr):
    calls.append(node_sstr)
    return ["x/2 - sin(x)*cos(x)/2"] if "sin(x)**2" in node_sstr else []
r = ax.solve(ax.parse_sstr("Integral(sin(x)**2, x)"), budget=100,
             heurisch=_fake_heurisch)
check("heurisch slot invoked + solved", r["solved"] and len(calls) >= 1)

def _lying(node_sstr):
    return ["x**3"]
r = ax.solve(ax.parse_sstr("Integral(sin(x)**2, x)"), budget=100,
             heurisch=_lying)
ok = True
if r["solved"]:
    ok = ax.equivalent_mod_const(ax.parse_sstr(r["answer"]),
                                 ax.parse_sstr("sin(x)**2"), "x") == "EQUIVALENT"
check("lying slot never corrupts the answer", ok)

def _crashing(node_sstr):
    raise RuntimeError("boom")
r = ax.solve(ax.parse_sstr("Integral(x**2, x)"), heurisch=_crashing)
check("crashing slot survives conservatively", r["solved"])

# ---- chain emission over the bridge (Phase D)

r = ax.emit_chain(ax.parse_sstr("Integral(21*x**2 + 9, x)"), 1)
rows = r["rows"]
check("emit_chain solves and emits", r["solved"] and len(rows) >= 2)
check("emit_chain schema exact",
      all(set(row) == {"cur", "nxt", "level", "source", "hints", "think"}
          for row in rows))
check("emit_chain drops nothing on a clean chain",
      r["dropped_pairs"] == 0 and r["replay_ok"])
check("emit_chain source tagged",
      all(row["source"] in ("axiom-chain", "axiom-oneply") for row in rows))
# every emitted state must round-trip the parser (rows are training text;
# an unparseable row would poison a shard). Value-level checking already
# happened C++-side: verify_edge gates every pair before it is emitted.
ok = True
for row in rows:
    for side in ("cur", "nxt"):
        try:
            ax.parse_sstr(row[side])
        except ValueError:
            ok = False
check("emit_chain rows round-trip the parser", ok)
# the chain must actually be a chain: each nxt is the following cur
check("emit_chain rows are contiguous",
      all(rows[i]["nxt"] == rows[i + 1]["cur"] for i in range(len(rows) - 1)))

# slot-served hints reach the rows (hybrid emission path)
seen_slot = []
def _slot_heurisch(node_sstr):
    seen_slot.append(node_sstr)
    return ["x/2 - sin(x)*cos(x)/2"] if "sin(x)**2" in node_sstr else []
r = ax.emit_chain(ax.parse_sstr("Integral(sin(x)**2, x)"), 6, budget=100,
                  heurisch=_slot_heurisch)
check("emit_chain hybrid path emits",
      r["solved"] and len(r["rows"]) >= 1 and len(seen_slot) >= 1)
check("emit_chain hints carry the slot",
      any("i_heurisch" in row["hints"] for row in r["rows"]))

def _emit_crash(node_sstr):
    raise RuntimeError("boom")
r = ax.emit_chain(ax.parse_sstr("Integral(x**2, x)"), 1,
                  heurisch=_emit_crash)
check("emit_chain survives a crashing slot", r["solved"])

print(f"\n{len(failures)} failure(s)")

# ---- slot-fire telemetry (L9 rung 4: HEAVY/LIGHT diet signal)

calls2 = []
def _slot2(node_sstr):
    calls2.append(node_sstr)
    return ["x/2 - sin(x)*cos(x)/2"] if "sin(x)**2" in node_sstr else []
r = ax.solve(ax.parse_sstr("Integral(sin(x)**2, x)"), budget=100,
             heurisch=_slot2)
check("slot telemetry counts fires", r["solved"] and r["slot_fires"] >= 1)
# decisive must agree with the winning history exactly (the slot may
# legitimately lose the race to a native rule - that IS the HEAVY signal)
check("slot decisive matches history",
      r["slot_decisive"] == sum(1 for h in r["history"]
                                if h.startswith("i_heurisch")))
r = ax.solve(ax.parse_sstr("Integral(x**2, x)"), heurisch=_slot2)
check("native solve has zero decisive slot steps",
      r["solved"] and r["slot_decisive"] == 0)
r = ax.emit_chain(ax.parse_sstr("Integral(sin(x)**2, x)"), 6, budget=100,
                  heurisch=_slot2)
check("emit_chain telemetry present",
      r["solved"] and r["slot_fires"] >= 1 and r["slot_decisive"] >= 0)

print(f"\n{len(failures)} failure(s)")
sys.exit(1 if failures else 0)
