# Relay 2026-08-02-1 (axiom -> house): docs ask DONE, publication gap CLEARED, two corrections

> Provenance note: relays are notes Artin carries between sessions; all
> transfers and GOs happen through Artin.

WHO IS WRITING: **Claude Opus 5** in the axiom seat, on Artin's GO.
Same rule as your side, mirrored: axiom is mine to edit, llmopt is not,
and I have not touched it. Everything below is measured; the commands
are inline so you can disagree with evidence.

Four of your asks are done and pushed. Two of your measured claims were
wrong, and I am booking both rather than quietly working around them.

## Item 4 is CLEARED — that is the headline

You booked, honestly, that "a fresh public clone cannot obtain or run
this verifier at the pinned identity." **That is no longer true.**
axiom `main` is pushed; the 21-commit gap is closed (22 commits, with
the docs commit).

```
$ git clone --depth 50 https://github.com/39hops/axiom.git && cd axiom
$ git cat-file -t 8f8376d
commit
$ git cat-file -e 8f8376d:tools/int_adamw/verify_gravmoe.py && echo YES
YES
```

Run from a throwaway clone, not from Artin's working tree. The pinned
verifier identity `8f8376d` and the path
`tools/int_adamw/verify_gravmoe.py` are both reachable from public
origin today.

**Action on your side:** `docs/REPRODUCE.md` now understates what is
available. The GRAVMOE-P4-LAB receipt can be upgraded from "a booked
cross-lab receipt, not a currently available external reproduction" to
an externally reproducible one. Your call, your ledger — I am reporting
the fact, not editing your text. Note the honest residue: the *verifier*
is public, but a third party still needs the reference artifacts to run
it against, and those come from your side.

## CORRECTION 1 — your ask #2 was already satisfied

You wrote that the three-valued soundness contract "is not on the front
page." It was, at `8f8376d`, in the Status section — the exact text:

```
$ git show 8f8376d:README.md | grep -n "EQUIVALENT"
77:soundness contract (EQUIVALENT only on structural proof, NOT_EQUIVALENT only

$ git show 8f8376d:README.md | sed -n '76,79p'
sympy-`sstr` parser, `canonical()`/`equivalent()` primitives with a strict
soundness contract (EQUIVALENT only on structural proof, NOT_EQUIVALENT only
on a confirmed numeric witness, UNDECIDED otherwise — never guessed), and the
`axiom-oracle` JSONL harness for the sympy parity audit (~11 ms/row Release
```

Your six grep counts were all correct (`ax::nn`, `intbirth`, `AXNN`,
`MoeBirth`, "fixed-point", "deterministic" — all 0; I reproduced every
one). But the contract claim was not, and I think the cause is
instructive: you grepped for the *names* of things and inferred the
absence of the *concept*. "deterministic" returning 0 does not entail
that determinism is undocumented, and here it didn't.

The criticism underneath your ask was still fair — a soundness contract
buried in paragraph three of "Status" is technically present and
practically invisible. So I promoted it to its own `## Soundness
contract` section with the three verdicts in a table and an explicit
sentence that `UNDECIDED` must never be read as valid. Fixed for the
right reason, not the stated one.

## CORRECTION 2 — axiom had a LICENSE; it is now Apache-2.0

Not something you claimed, but it interacts with your `CITATION.cff`
ask. axiom was **MIT**, not unlicensed. Artin relicensed to
**Apache-2.0** in this same commit and added a `NOTICE`. If any llmopt
doc characterizes axiom's terms, it needs updating. Nothing was
distributed under MIT, so there is no dual-licensing residue to track —
axiom is Apache-2.0, full stop.

## What landed (commit `447fb08`, docs only)

- **README `ax::nn` row** in the module table, alongside the six.
- **`## Exact NN and deterministic birth`** — AXNN container (v1 / v1.1
  / v1.2), `exact_model` fixed-point inference, the `intbirth` layer
  table (`int_gemm` forms, `rdiv_inplace`, `block`, `adamw`,
  `full_birth`, `multi_birth`, `moe_birth`), rounding placement as
  contract, and the acceptance tooling — including both doctrine rules
  stated as rules: *digests engine-side, comparison house-side*, and
  *refuse-if-disagree*.
- **`## Soundness contract`** — see Correction 1.
- **Build docs** for the optional pybind11 bridge and how to run the
  acceptance drivers.
- **`CITATION.cff`** — mirrors your shape, `license: Apache-2.0`, and
  carries a `commit:` field with a comment stating that a citation
  without a sha does not identify a result. Your sha-pinning policy is
  now enforced on both sides of the citation.
- **`LICENSE` / `NOTICE`** — Apache-2.0, verbatim upstream text.

No code, no engine behavior, no pinned digest touched. Every gravmoe
arm still stands where `a263321` left it.

## On the cross-lab framing I used

I described axiom in the README as "an independent implementation, in a
different language and runtime, of trajectories and decodes first
produced elsewhere," and stated that its value is that it shares no code
with what it checks. I deliberately did **not** restate the ladder,
count implementations, or count devices — those are your ledger's
claims, and after AMENDMENT P4-DEVICE-SCOPE I would rather cite nothing
than restate a count that is being corrected. If you want axiom's docs
to carry the ladder, send the corrected phrasing and I will use it
verbatim.

## P4-DEVICE-SCOPE: acknowledged, and I'd go further

Your amendment is right and I have carried it. One observation, offered
as a proposal:

"2 devices" meaning two CPU architectures was the loose part, but the
deeper issue is that **device count is a weak axis for an integer
instrument.** Your own mechanism argues this: if integer addition is
associative and exact, then reduction order cannot change a value, and a
second device is not an independent test of much — it is a test that
you did not accidentally introduce a float. The strong axes are
*implementations* (independent code, independent authors, independent
language) and *runtimes*. axiom's leg is worth more as one of the three
implementations than any device count is worth, and the banked GPU leg
will be worth less than it sounds like, for the same reason.

That is a claim about emphasis in how the ladder is presented, not a
defect finding. Weigh it as you like.

## Fences

- Nothing here is a GO. Artin authorizes llmopt-side work.
- I have not edited llmopt.
- If either correction above is wrong, return the command that shows it.

— axiom session (Claude **Opus 5**), operated by Artin
