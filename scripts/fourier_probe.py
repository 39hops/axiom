"""Fourier grammar probe emitter (relay 2026-07-27-0 ask 4).

A SMALL grammar probe, not a farm: 20 rewrite rows over the trig-
polynomial (partial Fourier sum) grammar, gated on parse + soundness
before any volume ask — the ZX playbook.

Grammar: trig polynomials sum_k [ a_k*sin(k*x) + b_k*cos(k*x) ] + c
with rational coefficients, integer harmonics k, plus pi/2-multiple
phase shifts inside trig arguments. Every cur/nxt parses under axiom's
sstr grammar (sympy-compatible).

Moves (kind):
  f_ptos   product -> sum   sin(a)cos(b) = (sin(a+b)+sin(a-b))/2, ...
  f_stop   sum -> product   cos(a)-cos(b) = -2 sin((a+b)/2) sin((a-b)/2)
  f_pow2   power reduction  sin(kx)^2 = 1/2 - cos(2kx)/2, ...
  f_double frequency scale  sin(2kx) = 2 sin(kx) cos(kx), ...
  f_shift  phase normalize  sin(kx + pi/2) = cos(kx), ...

Boundary anchoring (the ZX discipline, translated): each row rewrites
ONE interior site; the surrounding partial-sum context appears
byte-identical on both sides of the row. The context terms are the
anchors — a trainee that touches them fails string-exact replay.

Gate (this script IS the gate; it refuses to write unsound rows):
  1. parse: both sides through ax.parse_sstr (native grammar).
  2. soundness: |cur - nxt| < 1e-9 at 16 sample points (numeric, so
     trig identities outside the symbolic oracle's reach still gate).
  3. dedupe on (cur, nxt).

Run: python scripts/fourier_probe.py [pyd-dir] [out.jsonl]
"""
import json
import math
import sys

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "build-rel")
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/fourier/fourier_probe.jsonl"

import axiom_sym as ax  # noqa: E402

# (kind, site_cur, site_nxt, context) — context terms are the anchors,
# appended verbatim to both sides.
ROWS = [
    # ---- f_ptos: product-to-sum
    ("f_ptos", "2*sin(3*x)*cos(x)", "sin(4*x) + sin(2*x)",
     " + cos(2*x)/2"),
    ("f_ptos", "2*sin(x)*sin(2*x)", "cos(x) - cos(3*x)",
     " - 5"),
    ("f_ptos", "2*cos(2*x)*cos(3*x)", "cos(x) + cos(5*x)",
     " + 3*sin(x)"),
    ("f_ptos", "6*sin(2*x)*cos(2*x)", "3*sin(4*x)",
     " + cos(x) - 1/2"),
    # ---- f_stop: sum-to-product (harmonic parities keep k integral)
    ("f_stop", "cos(x) - cos(3*x)", "2*sin(2*x)*sin(x)",
     " + 7*cos(2*x)"),
    ("f_stop", "sin(3*x) + sin(x)", "2*sin(2*x)*cos(x)",
     ""),
    ("f_stop", "cos(2*x) + cos(4*x)", "2*cos(3*x)*cos(x)",
     " - sin(x)/3"),
    ("f_stop", "sin(5*x) - sin(x)", "2*sin(2*x)*cos(3*x)",
     " + 1"),
    # ---- f_pow2: power reduction
    ("f_pow2", "sin(x)**2", "1/2 - cos(2*x)/2",
     " + 2*sin(3*x)"),
    ("f_pow2", "cos(2*x)**2", "1/2 + cos(4*x)/2",
     " - cos(x)"),
    ("f_pow2", "4*sin(3*x)**2", "2 - 2*cos(6*x)",
     " + sin(x) + cos(x)"),
    ("f_pow2", "6*cos(x)**2", "3 + 3*cos(2*x)",
     " - 5*sin(2*x)/2"),
    # ---- f_double: frequency doubling both directions
    ("f_double", "sin(4*x)", "2*sin(2*x)*cos(2*x)",
     " + cos(3*x)"),
    ("f_double", "cos(2*x)", "1 - 2*sin(x)**2",
     " + sin(5*x)/4"),
    ("f_double", "3*cos(6*x)", "3 - 6*sin(3*x)**2",
     " - 2*cos(x)"),
    ("f_double", "2*sin(2*x)", "4*sin(x)*cos(x)",
     " + cos(4*x) + 1/3"),
    # ---- f_shift: pi/2-multiple phase normalization
    ("f_shift", "sin(x + pi/2)", "cos(x)",
     " + 2*sin(2*x)"),
    ("f_shift", "cos(3*x + pi)", "-cos(3*x)",
     " + sin(x)/2"),
    ("f_shift", "sin(2*x - pi/2)", "-cos(2*x)",
     " - 4*cos(x)"),
    ("f_shift", "cos(x - pi/2)", "sin(x)",
     " + sin(3*x) - 2"),
]

POINTS = [0.35 + 0.41 * i for i in range(16)]


def num_eval(src, x):
    ns = {
        "x": x, "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
        "pi": math.pi, "E": math.e, "__builtins__": {},
    }
    return eval(src, ns)  # noqa: S307 - authored rows, no user input


def main():
    rows, seen = [], set()
    failures = []
    for n, (kind, cur_site, nxt_site, ctx) in enumerate(ROWS):
        cur = cur_site + ctx
        nxt = nxt_site + ctx
        # gate 1: parse (both sides, native grammar)
        try:
            ax.parse_sstr(cur)
            ax.parse_sstr(nxt)
        except ValueError as exc:
            failures.append(f"parse {kind} #{n}: {exc}")
            continue
        # gate 2: numeric soundness at 16 points
        bad = [p for p in POINTS
               if abs(num_eval(cur, p) - num_eval(nxt, p)) > 1e-9]
        if bad:
            failures.append(f"soundness {kind} #{n}: off at {bad[:3]}")
            continue
        # gate 3: dedupe
        if (cur, nxt) in seen:
            failures.append(f"dupe {kind} #{n}")
            continue
        seen.add((cur, nxt))
        rows.append({
            "family": "fourier", "level": 1, "seed": 0, "n": n,
            "kind": kind, "cur": cur, "nxt": nxt,
            "source": "axiom-fourier-probe",
        })
    if failures:
        for f in failures:
            print("FAIL", f)
        sys.exit(1)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"PASS {len(rows)}/{len(ROWS)} rows -> {OUT}")


if __name__ == "__main__":
    main()
