# Forward rule-fire enumeration on the bridge (relay -9, priority 2)

Ask (llmopt relay 2026-07-28, received 2026-07-27 axiom-side): export the
forward sibling of `predecessors` — per-state verified successor
enumeration, plus an optional prior-weighted distribution form — so
llmopt's distribution-row farming moves off the ~3–4 states/sec sympy
path onto the native engine.

## Surface (INTERFACE_VERSION 5)

```
successors(state, use_macros=True, deadline_ms=0)
  -> {"rows": [(rule_name, Expr), ...], "expired": bool}

successors_dist(state, prior_tsv, prev_rule="", use_macros=True,
                deadline_ms=0)
  -> {"rows": [(rule_name, Expr, w), ...], "expired": bool}
```

- `state` is an `Expr` (parse via `parse_sstr`; `str(Expr)` is sstr —
  the established bridge convention, same as `predecessors`).
- Every emitted row is verify_edge-certified: enumeration runs the
  forward engine at `verify_p = 1`, so no unverified child ever crosses
  the bridge. There is no flag to lower this.
- `expired`: True when a `deadline_ms` wall was set and had passed by
  the end of enumeration — the row set may be censored (partial), never
  a silent partial set. Conservative direction: a set that completed at
  the wire may still flag expired; censored ≠ fact either way.
- `successors_dist` weights: `w = score / Σ score` over the emitted
  set, where `score` is the house proposer's own convention
  (engine.cpp / llmopt markov3): seen rule → `0.01*unigram[rule] +
  bigram[prev_rule][rule]` (bigram term only when `prev_rule` given and
  seen), unseen rule → `0.5 * median unigram`. `prior_tsv` is the
  byte-pinned `data/llmopt/markov_prior.tsv`
  (sha256 cd60b1d1…e46dea5, the E4 prior cell artifact).
- Streaming: the per-state call is small and synchronous; farm loops
  drive it row-by-row Python-side, which is the incremental/killed-
  worker shape. No batch entry point until asked.

## Pre-registered acceptance

1. **Parity**: on a string-seeded 500-state sample from the gen-4 band
   (llmopt supplies or seeds are pinned), emitted valid set matches
   house `llmopt.search.derivation.successors` up to the KNOWN fences:
   the real-rational domain fence for I-carrier states, and named rule
   coverage gaps listed in the run sidecar (E4 pattern). Any other
   disagreement decomposes per the E4 taxonomy before adopt.
2. **Soundness**: every emission passes verify_edge by construction
   (verify_p=1 in-engine); censored enumeration sets `expired=True`,
   never a silent partial.
3. **Throughput**: honest states/sec on the same sample, same machine
   class, reported against the house ~3–4/sec sympy baseline.
4. **Delivery**: parity sample shipped as file + rows + sha256 + arm
   config (substrate fence); house spot-checks on the sympy oracle.

## Preliminary throughput (NOT the acceptance number)

Measured on the 480 qual roots (60/level L1–L8, Release, this machine)
— preliminary because the acceptance sample is the llmopt-pinned gen-4
band, not these roots:

| stat | value |
|------|-------|
| p50  | 43.5 ms/state (~23 states/sec) |
| p90  | 611 ms/state |
| p99  | 6.16 s/state |
| max  | 14.4 s/state |
| mean | 3.0 states/sec (tail-dominated) |

The tail is the L7/L8 ansatz rules (`i_linear_basis` 2–8 s fires plus
their verify cost) — the same states that cost sympy far more. Typical
gen-4-band states sit at the p50: ~6–7× the house ~3–4/sec baseline.
Farms that want a hard wall set `deadline_ms` and read `expired` —
taxed tail, never a silent partial.

Status: surface shipped this commit with bridge tests; the 500-state
parity run executes when llmopt pins the sample band (their sampler,
their seeds — contamination doctrine).
