"""Fuzz-the-oracle CI node (relay 2026-07-27-0 ask 2), bridge edition.

Property-based fuzz of the PRODUCTION edge oracle over the axiom_sym
bridge (`verify_edge` = exactly what the search pays, pure-native
slots). The value cache is persistent — a verifier bug would fossilize
into cached labels — so this is a standing CI node, not a one-off.

Properties
  1. ACCEPT-soundness: every accepted edge is numerically true
     (mod an additive constant): checked with an INDEPENDENT
     pure-Python evaluator (adaptive Simpson for Integral atoms) that
     shares nothing with the C++ symbolic layer.
  2. REJECT-completeness on corruptions: an x-dependent perturbation
     that the numerics convict must be rejected.
  3. Table-family acceptance: linearity-of-table edges (known-true by
     construction) must be accepted — guards against the oracle
     rotting into reject-everything, which properties 1-2 alone would
     never catch.

Runs under pytest (llmopt CI: `pytest tests/python/test_oracle_fuzz.py`)
AND standalone (`python tests/python/test_oracle_fuzz.py <pyd-dir>`),
matching the no-hard-dependency discipline of test_axiom_sym.py.
If sympy is importable, an extra cross-check fuzzes sympy-integrated
antiderivatives through the same properties.
"""
import math
import random
import sys

if __name__ == "__main__":
    sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "build-rel")
else:
    sys.path.insert(0, "build-rel")

import axiom_sym as ax  # noqa: E402

BASE = 0.31  # fixed lower quadrature limit; mod-const absorbs it
POINTS = (0.7, 1.3, 2.1, 0.45, 3.3)


# ----------------------------------------------------------- numerics
def _env_eval(src, x):
    """Evaluate an sstr expression numerically, Integral-aware."""
    return _eval_node(src, x)


def _eval_node(src, x):
    # Integral(f, x) atoms are handled textually only when they are the
    # WHOLE expression; fuzzed edges built here keep carriers top-level
    # additive, so split on top-level '+'/'-' first.
    terms = _split_top(src)
    if len(terms) > 1:
        return sum(_eval_node(t, x) for t in terms)
    t = terms[0].strip()
    neg = False
    while t.startswith("-"):
        neg = not neg
        t = t[1:].strip()
    val = None
    if t.startswith("Integral(") and t.endswith(",x)"):
        integrand = t[len("Integral("):-len(",x)")]
        val = _simpson(lambda u: _eval_scalar(integrand, u), BASE, x)
    else:
        val = _eval_scalar(t, x)
    return -val if neg else val


def _split_top(src):
    """Split on top-level +/- (keeping signs on the pieces)."""
    src = src.replace(" ", "")
    parts, depth, cur = [], 0, ""
    for i, ch in enumerate(src):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if (depth == 0 and ch in "+-" and i > 0
                and src[i - 1] not in "*/(+-e,"):
            parts.append(cur)
            cur = ch
        else:
            cur += ch
    parts.append(cur)
    return [p for p in parts if p.strip("+- ")]


def _eval_scalar(src, x):
    ns = {
        "x": x, "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
        "pi": math.pi, "E": math.e, "__builtins__": {},
    }
    return eval(src, ns)  # noqa: S307 - fuzz-local sstr, no user input


def _simpson(f, a, b, n=256):
    if a == b:
        return 0.0
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += f(a + i * h) * (4 if i % 2 else 2)
    return s * h / 3


def numeric_verdict(parent, child, mod_const=True):
    """+1 equal (mod const), -1 different, 0 inconclusive."""
    diffs = []
    for p in POINTS:
        try:
            a = _env_eval(parent, p)
            b = _env_eval(child, p)
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        if abs(a) > 1e8 or abs(b) > 1e8:
            continue
        diffs.append(a - b)
    if len(diffs) < 2:
        return 0
    spread = max(diffs) - min(diffs)
    scale = max(1.0, max(abs(d) for d in diffs))
    if mod_const:
        if spread < 1e-3 * scale:
            return +1
        if spread > 1e-1 * scale:
            return -1
    else:
        amax = max(abs(d) for d in diffs)
        if amax < 1e-4:
            return +1
        if amax > 1e-1:
            return -1
    return 0


# ---------------------------------------------------------- generator
TABLE = [
    ("x", "x**2/2"),
    ("x**2", "x**3/3"),
    ("x**3", "x**4/4"),
    ("x**-1", "log(x)"),
    ("sin(x)", "-cos(x)"),
    ("cos(x)", "sin(x)"),
    ("exp(x)", "exp(x)"),
    ("1", "x"),
]


def gen_pair(rng):
    """(parent sstr, true child sstr) by linearity over the table."""
    n = rng.randint(1, 3)
    fs, gs = [], []
    for _ in range(n):
        f, g = rng.choice(TABLE)
        c = rng.choice([1, 2, 3, 5, 7, -1, -2, -4])
        fs.append(f"{c}*({f})")
        gs.append(f"{c}*({g})")
    const = rng.choice(["", " + 7", " - 3"])
    return f"Integral({' + '.join(fs)}, x)", " + ".join(gs) + const


CORRUPTIONS = ["({}) + x**2", "({}) + sin(x)", "3*({})/2 + x"]


# -------------------------------------------------------------- tests
def test_table_family_accepted():
    rng = random.Random(20260727)
    rejected = []
    for _ in range(120):
        parent, child = gen_pair(rng)
        p = ax.parse_sstr(parent)
        c = ax.parse_sstr(child)
        if not ax.verify_edge(p, c):
            rejected.append((parent, child))
    assert not rejected, f"true table edges rejected: {rejected[:5]}"


def test_accepted_edges_numerically_sound():
    rng = random.Random(31415926)
    violations, checked = [], 0
    for _ in range(120):
        parent, child = gen_pair(rng)
        if ax.verify_edge(ax.parse_sstr(parent), ax.parse_sstr(child)):
            v = numeric_verdict(parent, child)
            if v != 0:
                checked += 1
            if v < 0:
                violations.append((parent, child))
    assert not violations, f"accepted-but-false: {violations[:5]}"
    assert checked >= 60, "numeric coverage collapsed"


def test_corruptions_rejected():
    rng = random.Random(27182818)
    missed, checked = [], 0
    for _ in range(120):
        parent, child = gen_pair(rng)
        bad = rng.choice(CORRUPTIONS).format(child)
        if numeric_verdict(parent, bad) < 0:
            checked += 1
            if ax.verify_edge(ax.parse_sstr(parent), ax.parse_sstr(bad)):
                missed.append((parent, bad))
    assert not missed, f"corrupted edges accepted: {missed[:5]}"
    assert checked >= 60, "corruption coverage collapsed"


def test_sympy_cross_check():
    try:
        import sympy
    except ImportError:
        return  # optional on the axiom side; llmopt CI has sympy
    x = sympy.Symbol("x")
    rng = random.Random(16180339)
    accepted_false, checked = [], 0
    for _ in range(40):
        parent, _ = gen_pair(rng)
        f = sympy.sympify(parent[len("Integral("):-len(", x)")])
        g = sympy.integrate(f, x)
        child = sympy.sstr(g)
        try:
            c = ax.parse_sstr(child)
        except ValueError:
            continue  # forms outside the axiom grammar: fine
        if ax.verify_edge(ax.parse_sstr(parent), c):
            checked += 1
            if numeric_verdict(parent, child) < 0:
                accepted_false.append((parent, child))
    assert not accepted_false, f"sympy cross-check: {accepted_false[:5]}"
    assert checked >= 20, "sympy cross-check coverage collapsed"


if __name__ == "__main__":
    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                print(f"FAIL {name}: {exc}")
                failures.append(name)
    sys.exit(1 if failures else 0)
