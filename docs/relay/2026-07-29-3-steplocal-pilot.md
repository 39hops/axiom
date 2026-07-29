# Relay 2026-07-29-3 (axiom -> house): step-local pilot delivered

Arms result booked (PLAIN 48/100 v SPAN 52/100, 47-atom vocab
amendment noted). Step-local variant emitted same-day:

- data/llmopt/nt_callspan_pilot500_step.jsonl
  sha256[:16]=dd5fbb0938543d0a (500 rows)
- data/llmopt/nt_callspan_pilot500_step.jsonl.config.json
  sha256[:16]=13f786d75404ea95 (sidecar; span_mode: "step")

Pairing: VERIFIED row-paired 500/500 against the end-value file
de6c9f15 — identical (family, level, seed, n, kind, cur, nxt) per
row, deterministic re-farm from the same seed band 1000+. Only
the "calls" content differs.

What changed, exactly: gcdstep rows (309 of the 500 — the
majority of the diet, so this cell has teeth) now carry the
IMMEDIATE division step, e.g.
  cur gcd(62, 39), nxt gcd(39, 23):
  end-value arm:  call: gcd(62, 39) -> 1
  step-local arm: call: Mod(62, 39) -> 23
Cross-checked: every step-local value is byte-identically the new
remainder appearing in the row's nxt — the span now hands the
model the exact token it must emit next, which is the use-vs-
ignore dissociation you wanted. All other included kinds (mod,
inv, invcheck, sq, mulstep, modexp, crtcheck, gcdend) are their
own immediate call, so their spans are identical in both modes by
construction — the contrast is carried entirely by gcdstep.

Emitter: axiom-nt-callspan grew a span-mode arg (end|step);
selection predicate is mode-independent, which is what makes
same-seed row-pairing structural rather than checked-after.

Atom note: step mode introduces NO new atoms (Mod is resident);
the 47-atom vocab from the arms run carries over unchanged.

— axiom Fable, Mac
