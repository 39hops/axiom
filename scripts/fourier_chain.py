"""Fourier chain emitter — volume batch 1 (relay 2026-07-27-2: volume
APPROVED with the STAY-IN-Q fence; batch 1 proposed at ~10k rows, the
ZX batch-1 scale).

Grammar: the probe grammar (docs/specs/2026-07-27-fourier-probe.md).
STAY-IN-Q fence honored: every coefficient is a Fraction, amplitude-
phase recombination is NOT a move (readout form only, per house).

States are term lists; a move rewrites the terms at ONE site and the
remaining term strings appear byte-identical on both sides (boundary
anchoring). Rows carry site indices ZX-style.

Moves (organic mix — whatever is applicable at each state):
  f_ptos   c*T(ax)*T(bx)         -> sum form (product-to-sum)
  f_stop   c*T(ax) +/- c*T(bx)   -> product form (parity-matched)
  f_pow2   c*T(kx)**2            -> half-angle sum
  f_double c*T(2mx)              -> half-frequency product / square
  f_shift  c*T(kx + m*pi/2)      -> normalized phase

Gate (refuses to write; any soundness failure ABORTS the batch):
  parse both sides (native sstr) + |cur-nxt| < 1e-9 at 16 points +
  (cur, nxt) dedupe.

Run: python scripts/fourier_chain.py [pyd-dir] [out.jsonl] [target_rows]
"""
import json
import math
import random
import sys
from fractions import Fraction

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "build-rel")
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/fourier/fourier_batch1.jsonl"
TARGET = int(sys.argv[3]) if len(sys.argv) > 3 else 10000

import axiom_sym as ax  # noqa: E402

# ------------------------------------------------------------- terms
# term = ("const", c) | ("trig", c, fn, k, m) | ("sq", c, fn, k)
#      | ("prod", c, fn1, k1, fn2, k2)        (c: Fraction, k: int>0,
#        m: 0..3 quarter-turn shift; prod renders fn1(k1x)*fn2(k2x))

SHIFT = {0: "", 1: " + pi/2", 2: " + pi", 3: " + 3*pi/2"}


def arg(k, m=0):
    core = "x" if k == 1 else f"{k}*x"
    return core + SHIFT[m]


def unit(term):
    """Render the coefficient-1 body of a term ('' for const)."""
    kind = term[0]
    if kind == "const":
        return ""
    if kind == "trig":
        _, _, fn, k, m = term
        return f"{fn}({arg(k, m)})"
    if kind == "sq":
        _, _, fn, k = term
        return f"{fn}({arg(k)})**2"
    _, _, f1, k1, f2, k2 = term
    return f"{f1}({arg(k1)})*{f2}({arg(k2)})"


def render(term):
    c = term[1]
    body = unit(term)
    if body == "":
        s = str(abs(c.numerator))
        if c.denominator != 1:
            s += f"/{c.denominator}"
        return ("-" if c < 0 else "") + s
    s = "" if abs(c.numerator) == 1 else f"{abs(c.numerator)}*"
    s += body
    if c.denominator != 1:
        s += f"/{c.denominator}"
    return ("-" if c < 0 else "") + s


def join(terms):
    out = render(terms[0])
    for t in terms[1:]:
        r = render(t)
        out += (" - " + r[1:]) if r.startswith("-") else (" + " + r)
    return out


# ------------------------------------------------------------- moves
def moves_at(terms):
    """[(kind, site_indices, replacement_terms)] deterministic order."""
    out = []
    for i, t in enumerate(terms):
        kind = t[0]
        if kind == "trig":
            _, c, fn, k, m = t
            if m:  # f_shift: quarter-turn normalization
                tbl = {("sin", 1): ("cos", 1), ("sin", 2): ("sin", -1),
                       ("sin", 3): ("cos", -1), ("cos", 1): ("sin", -1),
                       ("cos", 2): ("cos", -1), ("cos", 3): ("sin", 1)}
                nf, sgn = tbl[(fn, m)]
                out.append(("f_shift", [i], [("trig", c * sgn, nf, k, 0)]))
            elif k % 2 == 0:  # f_double: halve the frequency
                h = k // 2
                if fn == "sin":
                    out.append(("f_double", [i],
                                [("prod", 2 * c, "sin", h, "cos", h)]))
                else:
                    out.append(("f_double", [i],
                                [("const", c), ("sq", -2 * c, "sin", h)]))
        elif kind == "sq":
            _, c, fn, k = t
            half = c / 2
            cosc = -half if fn == "sin" else half
            out.append(("f_pow2", [i],
                        [("const", half), ("trig", cosc, "cos", 2 * k, 0)]))
        elif kind == "prod":
            _, c, f1, k1, f2, k2 = t
            half = c / 2
            rep = []
            if f1 == "sin" and f2 == "cos":
                rep.append(("trig", half, "sin", k1 + k2, 0))
                if k1 != k2:
                    d, s = abs(k1 - k2), (1 if k1 > k2 else -1)
                    rep.append(("trig", half * s, "sin", d, 0))
            elif f1 == "sin" and f2 == "sin":
                if k1 != k2:
                    rep.append(("trig", half, "cos", abs(k1 - k2), 0))
                else:
                    rep.append(("const", half))
                rep.append(("trig", -half, "cos", k1 + k2, 0))
            else:  # cos*cos
                if k1 != k2:
                    rep.append(("trig", half, "cos", abs(k1 - k2), 0))
                else:
                    rep.append(("const", half))
                rep.append(("trig", half, "cos", k1 + k2, 0))
            out.append(("f_ptos", [i], rep))
    # f_stop: parity-matched pairs of plain trig terms with equal |coeff|
    for i, a in enumerate(terms):
        for j, b in enumerate(terms):
            if i >= j or a[0] != "trig" or b[0] != "trig":
                continue
            _, ca, fa, ka, ma = a
            _, cb, fb, kb, mb = b
            if ma or mb or ka == kb or (ka - kb) % 2:
                continue
            hi, lo = max(ka, kb), min(ka, kb)
            s, d = (hi + lo) // 2, (hi - lo) // 2
            if fa == "sin" and fb == "sin" and ca == cb:
                # sin A + sin B = 2 sin((A+B)/2) cos((A-B)/2)
                out.append(("f_stop", [i, j],
                            [("prod", 2 * ca, "sin", s, "cos", d)]))
            elif fa == "cos" and fb == "cos" and ca == cb:
                out.append(("f_stop", [i, j],
                            [("prod", 2 * ca, "cos", s, "cos", d)]))
            elif fa == "cos" and fb == "cos" and ca == -cb:
                # cos LO - cos HI = 2 sin(s) sin(d); site order carries c
                clo = ca if ka == lo else cb
                out.append(("f_stop", [i, j],
                            [("prod", 2 * clo, "sin", s, "sin", d)]))
    return out


def apply_move(terms, site, rep):
    keep = [t for idx, t in enumerate(terms) if idx not in site]
    at = min(site)
    return keep[:at] + rep + keep[at:]


# --------------------------------------------------------- generator
def gen_state(rng):
    n = rng.randint(2, 4)
    terms = []
    for _ in range(n):
        c = Fraction(rng.choice([1, 2, 3, 5, 7, -1, -2, -3, -4]),
                     rng.choice([1, 1, 1, 2]))
        shape = rng.randrange(6)
        if shape == 0:
            terms.append(("const", c))
        elif shape == 1:
            terms.append(("trig", c, rng.choice(["sin", "cos"]),
                          rng.randint(1, 4) * 2, 0))  # even: f_double fires
        elif shape == 2:
            terms.append(("trig", c, rng.choice(["sin", "cos"]),
                          rng.randint(1, 6), rng.randint(1, 3)))
        elif shape == 3:
            terms.append(("sq", c, rng.choice(["sin", "cos"]),
                          rng.randint(1, 4)))
        elif shape == 4:
            terms.append(("prod", c, rng.choice(["sin", "cos", "sin"]),
                          rng.randint(1, 5), "cos", rng.randint(1, 5)))
        else:
            fn = rng.choice(["sin", "cos"])
            k = rng.randint(1, 3)
            terms.append(("trig", c, fn, k, 0))
            terms.append(("trig", c if fn == "sin" else
                          rng.choice([c, -c]), fn, k + 2, 0))
    return terms


POINTS = [0.35 + 0.41 * i for i in range(16)]


def num_eval(src, x):
    ns = {"x": x, "sin": math.sin, "cos": math.cos, "pi": math.pi,
          "__builtins__": {}}
    return eval(src, ns)  # noqa: S307 - generated rows, no user input


def main():
    rng = random.Random(20260727)
    rows, seen, kinds = [], set(), {}
    seed = 0
    while len(rows) < TARGET:
        seed += 1
        terms = gen_state(rng)
        plies = rng.randint(2, 4)
        for n in range(plies):
            if len(rows) >= TARGET:
                break
            options = moves_at(terms)
            if not options:
                break
            kind, site, rep = rng.choice(options)
            nxt_terms = apply_move(terms, site, rep)
            cur, nxt = join(terms), join(nxt_terms)
            terms = nxt_terms
            if cur == nxt or (cur, nxt) in seen:
                continue
            # gate: parse + numeric soundness (abort on failure — a
            # soundness miss is a rewriter bug, not a skippable row)
            ax.parse_sstr(cur)
            ax.parse_sstr(nxt)
            for p in POINTS:
                if abs(num_eval(cur, p) - num_eval(nxt, p)) > 1e-9:
                    print(f"ABORT unsound {kind}: {cur} -> {nxt} at {p}")
                    sys.exit(1)
            seen.add((cur, nxt))
            kinds[kind] = kinds.get(kind, 0) + 1
            rows.append({
                "family": "fourier", "level": 1, "size": len(terms),
                "seed": seed, "n": n, "kind": kind,
                "site": " ".join(map(str, site)),
                "cur": cur, "nxt": nxt,
                "source": "axiom-fourier-chain",
            })
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"PASS {len(rows)} rows from {seed} seeds -> {OUT}")
    for k in sorted(kinds):
        print(f"  {k} {kinds[k]}")


if __name__ == "__main__":
    main()
