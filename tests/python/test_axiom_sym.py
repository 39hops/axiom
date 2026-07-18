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

print(f"\n{len(failures)} failure(s)")
sys.exit(1 if failures else 0)
