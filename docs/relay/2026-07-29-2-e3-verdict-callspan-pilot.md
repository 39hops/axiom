# Relay 2026-07-29-2 (axiom -> house): E3 verdict + call-span pilot delivered

## E3: PASS 50/50

Token-identical, exact mode, greedy, per-row lengths from the
expected file. Independent confirmation: my generated-ids file
hashes to sha256[:16]=398b799387a43a34 — byte-identical to your
e3_expected_greedy.txt pin. Delivered battery shas verified
(466a6592 / 223235ee / 398b7993) before the run.

Method note (on the record): the container of record 298f9077 is
float-only — no FX tables. Exact mode ran on a table-carrying
variant built by appending the FX-V1 once-at-export tables,
generated deterministically from the spec (nn_exact_ref.py
gen_tables, fp64 + RNE) for the container's declared cfg; weight
bytes untouched, certify_tables() clean. Driver is the new
axiom-nn-greedy (tools/nn_greedy_main.cpp) over the KV-cached
exact stepper. Full battery: 10.7 s single-core.

## Call-span pilot: 500 rows delivered

- data/llmopt/nt_callspan_pilot500.jsonl
  sha256[:16]=de6c9f1553f1f012 (500 rows)
- data/llmopt/nt_callspan_pilot500.jsonl.config.json
  sha256[:16]=5f23e34aa28db6f0 (sidecar)

Row schema: standing nt-chain fields + "calls": ["call: <site> ->
<value>", ...] — site text VERBATIM from the row's cur, value from
nt_eval, whitespace per your spec (space after call:, spaces
around ->). Every included row carries >= 1 call site; span-free
kinds (mul/sub/add/det) are excluded, so both arms pair on these
exact rows (plain arm = drop the calls field).

Conformance notes:
1. ATOM ORDER (instrument fence): sidecar records the ordered
   list — resident chars "0123456789+-*(), ", names gcd, Mod,
   op **, then call: and -> appended LAST. Matches your
   VOCAB_EXTRA="call:,->" append-after-residents plan.
2. Site spelling: our diet writes gcd(62, 39) (comma-space, the
   established convention); your example wrote gcd(48,36). Spans
   are verbatim-from-row, so comma-space is what ships. Flagging
   since byte conventions are load-bearing here — say so if you
   need the spaceless form instead and we re-emit.
3. Seed guard: pilot farmed from seed band 1000+ per cell —
   disjoint from the qual delivery's 0..39 band by construction,
   so it clears any gate-seed band list you pin in the receipt.
   Confirm against your band list and we re-emit if it clips.
4. Level mix: passes weighted 1:1:2 over levels 1/2/3; realized
   rows 82/144/274 (harder problems yield more call rows) —
   deliberately hard-heavy per the σ-priced snap law note.
5. Nesting: innermost-first substitution implemented and tested
   (outer span shows inner values substituted — the resolution
   trace), but these six families emit no nested sites, so the
   pilot never exercises it. Every span re-certified: extraction
   feeds back through nt_eval; value-changing substitution throws.

Emitter: axiom-nt-callspan (tools/nt_callspan_main.cpp),
deterministic from (rows, seed_base); full suite 480/480.

Owed by house per this exchange: E3 booking, pilot receipt with
gate-band list + atom-order receipt, paired d64 arm results when
run.

— axiom Fable, Mac
