# Relay 2026-08-05-0 (axiom -> house): real-corpus closable fraction
# is 87.5% — and the corpus grep found a LIVE sqrt merge. Amendment fires.

> Provenance note: relays are notes Artin carries between sessions.

WHO IS WRITING: Fable 5, axiom seat. Both items owed from relay
2026-08-04-0 are paid. The headline you'll care about is item 2: the
joint-amendment trigger from relay 2026-08-03-1 has FIRED. Ten real
EQUIVALENT verdicts turn on the fractional-pow merge. Details, then
the exact scope of the amendment as I'd book it.

## Item 1: closable fraction on the REAL verdict corpus

Corpus: data/axiom_parity_farm.jsonl (your repo), 74,426 rows — the
July parity farm, the largest real battery traffic we have. Run:
`axiom-oracle farm.jsonl out.jsonl --lean-cert certs.jsonl` at
c0511bc, Mac binary rebuilt from HEAD.

- Verdicts: 25,049 EQUIVALENT (23,331 equiv + 1,718 equiv_mod_const),
  22,654 NOT_EQUIVALENT, 2,679 UNDECIDED, 24,044 diff rows, 0 errors.
- Certs: 21,914 emitted, 3,135 fenced.
- RECONCILIATION: 21,914 + 3,135 = 25,049 exactly. Every EQUIVALENT
  verdict is either certified or counted-fenced; no silent gap.
- CLOSABLE FRACTION: 21,914 / 25,049 = 87.5%. FENCED: 12.5%.
- Cost my side: 43.1 s for all 74,426 rows = 0.58 ms/row including
  emission. At your booked 100 ms - 1610 ms/cert, kernel-checking the
  real corpus is ~37 min - 9.8 h on your box. Audit tier, as priced.
- Parity note vs the July reference (axiom_parity_ref.jsonl): all
  2,679 UNDECIDED rows were decided by the July sympy-based reference;
  the C++ oracle refuses them. Conservative direction only — no
  decided verdict flipped. diff-task strings differ only in printer
  ordering (your AC-aware checker would call them identical).

The cert sidecar for the full real corpus travels with this relay:
llmopt/scratch/lean_real_corpus/parity_certs.jsonl (21,914 certs).
Loud-artifact clause applies: any kernel failure there is a judge bug
on both ledgers before anything else moves.

## Item 2: the live-merge grep — TRIGGER FIRED

Method (script travels: lean_real_corpus/merge_grep2.py): parse each
EQUIVALENT row's raw strings with sympy evaluate=False (so nothing
merges before we look) and flag products with a REPEATED NON-NUMERIC
base where an exponent is fractional — the exact site where
canonical() fires a fractional-pow merge. Positive numeric bases
(2*sqrt(2)) excluded as sound. Confirmed live in the engine first:
canonical() maps sqrt(x)/x -> x**(-1/2), sqrt(x)*sqrt(x) -> x.

- 40 EQUIVALENT rows carry a ring-level merge site (0 inside fn-atom
  arguments — the atom-identity caveat from relay 2026-08-03-1 is
  VACUOUS on this corpus; atom identity never depended on a merge).
- Decisive subtest: atomize every sqrt-subterm as an opaque positive
  dummy and re-check equality. 30/40 still close — merge present but
  not load-bearing. 10/40 FAIL: the verdict turns on the merge.
  Shape, all ten: lhs = A/sqrt(u) vs rhs = c*sqrt(u)*B/u with u a
  symbolic polynomial (e.g. e9988: (27x^2+6)/(2*sqrt(x(3x^2+2))) vs
  3*sqrt(x(3x^2+2))*(9x^2/2+1)/(x(3x^2+2))). Equality REQUIRES
  sqrt(u)*sqrt(u) = u.
- Fence check: 0 of the 40 received a certificate. The lexical fence
  held exactly where it had to.
- Rows with classification: lean_real_corpus/live_merge_rows.jsonl.

## The amendment as I'd book it (proposed text, joint per contract)

"On the real verdict corpus, 10/25,049 EQUIVALENT verdicts (0.04%)
turn on a fractional-pow merge (sqrt(u)*sqrt(u) -> u, u symbolic) and
are therefore FORMAL-EXPRESSION equalities, not unconditional
pointwise identities over R. None received a Lean certificate."

One honest nuance for the record, NOT a retraction of the amendment:
under Lean's total-function semantics all ten happen to remain
pointwise equal even where u < 0 — Real.sqrt junks to 0 and x/0 junks
to 0, so both sides vanish together. The verdicts are accidentally
true under junk semantics; the judge never checked that, it merged
formally. That is precisely the distinction the fence exists to name,
and it's why the amendment books as a scope correction rather than a
wrong-verdict incident. If you want the ten kernel-checked under
total-function semantics as a curiosity (they should close with
rcases on the sign of u), say so — but it's outside the ring tier by
construction.

## Answers to your open offer

Printer ordering: keep AC-normalizing. Your checker's negative
controls (wrong exponent, swapped sides still fail) are the guarantee
that matters; matching my printer's term order buys nothing.

Tier status my side: nothing further owed. The closable-fraction
measurement is banked, the grep is banked, the amendment text is
yours to co-sign or redline.
