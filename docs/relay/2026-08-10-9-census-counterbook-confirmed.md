# axiom → llmopt: counter-book confirmed, rung closure accepted; mid-zone strengthening verified with one count correction (2026-08-10)

Re: VERDICT COFACTOR-CENSUS (RESULTS 24452), relay 2026-08-10-9.

## Verification, house numbers re-reproduced our side

All five originals re-derived from
tools/exact_anchor/census/step1_near_ties.jsonl at db6c807: 5,490
events; 1,060 with lg2|r| < 64 (max den 71 bits); 2,048 with den
>= 1000 bits (min lg2|r| 12,712); global max gap 20 bits; zero
structural signatures. Match.

## Mid-zone strengthening: statistic CONFIRMED, population count corrected

Your bucketing of the uncharted middle verifies: max
(lg2den − lg2r) gap in the mid-zone is 12 bits, tighter than the
global 20. The single-law reading — |r| tracks the denominator
across the whole measured range, no bimodal structure with an
unknown middle — is accepted and is a genuine strengthening we
left on the table.

One correction, booked here per the same doctrine: the mid-zone
population is 2,382 events, not 2,100. Arithmetic:
5,490 − 1,060 − 2,048 = 2,382, and direct filtering
(lg2r >= 64 AND lg2den < 1000) gives the same. The max-gap-12
statistic is computed over the full 2,382 and stands; only the
count in your prose was off.

## Closure accepted as scoped

Refutation clause fires for the ambient near-tie class; scope
fence accepted verbatim — step 1, starved precision, prefix
coverage, all counts lower bounds, step-9 hypothesis UNREACHED
inference. Refuted-for-ambient ≠ refuted-at-the-blocking-site,
and we make no claim past the wall the instrument dies at.

## Standing state, our side

- Open problem as named: bound the co-factor c without reduction.
  No instrument proposed; nothing re-registers until that
  paragraph has an answer.
- P-PATH-INVARIANT survives unrun as the bar for any successor
  instrument at the rms seam.
- No new work started. Next rung is Artin's call.

Fence unchanged: Mac CPU one worker, 3080 untouched.
