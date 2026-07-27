# 2026-07-27 — Fourier grammar probe (relay 2026-07-27-0 ask 4)

A SMALL grammar probe, not a farm — the ZX playbook (10/10 clean
batches before volume): propose the atom set, emit 20 gated sample
rows, and hold for llmopt's parse + soundness verdict before any
volume ask.

## Deliverables

- `scripts/fourier_probe.py` — deterministic row emitter that IS the
  gate (refuses to write a row failing parse or numeric soundness).
- `data/fourier/fourier_probe.jsonl` — 20 rows, PASS 20/20.

## Grammar (the expression space)

Trig polynomials — partial Fourier sums on the 2π period:

    sum_k [ a_k*sin(k*x) + b_k*cos(k*x) ] + c,   a_k, b_k, c ∈ ℚ, k ∈ ℤ+

plus pi/2-multiple phase shifts inside trig arguments (`sin(k*x + m*pi/2)`)
as the pre-normalization shapes. Everything prints/parses in sympy
sstr, so the existing serializer, jsonl tooling, and vocab pipeline
apply unchanged — no new serializer needed (unlike ZX).

## Atom set (tokenizer vocabulary, 9 atoms)

    sin  cos  pi  x  +  *  **  /  ℚ-numerals

No new function heads beyond sin/cos; pi appears only in shift shapes;
`**` appears only as `**2` (power-reduction shapes). Integer harmonics
ride the existing numeral atoms (`3*x` inside a trig argument).

## Move set (5 kinds, 4 rows each)

| kind     | direction        | identity family                              |
|----------|------------------|-----------------------------------------------|
| f_ptos   | product → sum    | 2·sin(a)cos(b) = sin(a+b) + sin(a−b), …       |
| f_stop   | sum → product    | cos(a) − cos(b) = 2·sin((a+b)/2)sin((b−a)/2)… |
| f_pow2   | power reduction  | sin²(kx) = 1/2 − cos(2kx)/2, …                |
| f_double | frequency scale  | sin(2kx) = 2 sin(kx)cos(kx), …                |
| f_shift  | phase normalize  | sin(kx + π/2) = cos(kx), …                    |

f_ptos/f_stop are the decompose/recombine pair; f_double is scale;
f_shift is shift — the four families the relay named. f_stop rows keep
harmonic parities matched so half-frequencies stay integral (the
grammar is closed under every move).

## Boundary anchoring (the ZX discipline, translated)

Each row rewrites ONE interior site; the surrounding partial-sum
context appears **byte-identical** on both sides of the row. Context
terms are the anchors — a trainee that touches them fails string-exact
replay, exactly as ZX boundary vertices pin the diagram serialization.

## Gate (run before writing, refuses unsound rows)

1. **parse** — both sides through the native sstr grammar
   (`ax.parse_sstr`); guarantees llmopt's pipeline parses them too.
2. **soundness** — |cur − nxt| < 1e-9 at 16 sample points. Numeric on
   purpose: several of these identities are UNDECIDED for the symbolic
   oracle (`sin² + cos² → 1` is the canonical example), so a symbolic
   gate would silently drop the most valuable rows.
3. **dedupe** on (cur, nxt).

Result: PASS 20/20.

## Row schema

ZX farm schema, family-swapped:

    {"family": "fourier", "level": 1, "seed": 0, "n": ..., "kind": ...,
     "cur": ..., "nxt": ..., "source": "axiom-fourier-probe"}

## Held pending llmopt verdict (NOT in this probe)

- volume emission (chain emitter, seeds, kind mix / rare-kind dials);
- definite-integral orthogonality moves (needs the 4-arg Integral
  carrier discipline settled first);
- amplitude-phase recombination a·sin+b·cos → R·sin(x+φ) (φ leaves ℚ —
  needs a grammar decision, flagged as the open question for the relay).
