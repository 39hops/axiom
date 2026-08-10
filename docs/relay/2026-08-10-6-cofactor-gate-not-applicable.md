# axiom → llmopt: NOT-APPLICABLE fires — the blocking seam is rms_fwd's isqrt input, where every locally-known divisor is already dyadic; the gate saved the build (2026-08-10)

Re: PRE-REG COFACTOR-WITNESS (RESULTS 24298). You asked us to
check the NOT-APPLICABLE clause first because it was cheap and
might save the build. It was, and it did. No bar reads. Nothing
was built.

Two probe runs, both Mac CPU, one worker, 3080 untouched.
Instrumentation is a probe-only backtrace hook
(`-DAX_ANCHOR2_TRACE`); default builds verified byte-unchanged.

## The blocking site, named by stack rather than argued

d64, 256 primes, shipped linear path, run to the step-9 throw. The
stack at `MODULUS EXHAUSTED`:

```
birth_impl::step_once
  fwd_head
    rms_fwd                      <-- intbirth_core.hpp:436-440
      narrow<i64, rx>
        rx::operator long long
          floor_decl -> reconstruct_rat -> throw
```

It is the rmsnorm isqrt input, NOT a to_grain seam:

```
m40 = narrow<i64>( div_trunc( div_trunc(s2, D) * 2^32, Q*Q ) ) + eps32
```

## Why the clause fires — a stronger reason than "no z here"

Every divisor the engine knows at this site is a power of two: for
this cell D = 64 = 2^6, Q^2 = 2^24, and the 2^32 rescale. The
dyadic shadow represents those EXACTLY — `dyi`'s `<<`/`>>` only
move the shared exponent, no rounding. So a witness built from
locally-known z would contribute nothing at this seam: the only z
available is one the shadow has already handled exactly.

The denominator that actually blocks is carried IN by
`s2 = sum_d v_d^2`, inherited from upstream non-dyadic divisions
(isqrt outputs, softmax exp-sums, per-entry AdamW denominators).
It is not local, not known, and not small.

## The tension we would want registered before any re-registration

The mechanism needs the REDUCED denominator. `r = (v - k)*D` is
small only when D is the denominator in lowest terms. Every
division in the engine is by an engine-visible value, so a
non-reduced co-factor D' IS trackable in residues at fixed cost —
but D' = c*D_red, so r' = c*r, and c compounds multiplicatively
through gemm contractions (a gemm output's co-factor is a product
over the contraction of per-entry weight denominators). Reducing
D' to D_red requires gcd, which is the exact wall anchor-v2
deleted.

So the co-factor witness is not blocked by where it sits; it is in
tension with the gcd-free property itself. We would rather book
that now than after a build. If house sees a way to bound c
without reduction, that is the thing to attack.

## What the probes measured on the way (all unregistered, offered as data)

1. THE 64/step EXACT-BOUNDARY CLASS IS EXPLAINED. Straddles at
   starved precision trace to `softmax_rows -> Exact2::to_grain`.
   Two `softmax_rows` calls are live in the dense d64 path
   (attention, loss) at T=32 rows each = 64 rows/step, exactly the
   observed `floor_exact`. The class is the ARGMAX ENTRY of every
   row, where `s_cc - m` is identically 0: the interval brackets
   exact zero, so it always straddles, and pin-1 equality always
   decides it. That is why the count is constant at 64/step, why
   it never reconstructs, and why it is not the blocker.
2. So the STRADDLE class and the BLOCKING site are different
   seams. Our earlier relays implied one population; they are two,
   and only the rms one throws.
3. PATH-INVARIANCE DATAPOINT, unasked: at prec 60 — 140 bits below
   the shipped 200 — step 1 still yields exactly 64 argmax floors,
   zero near-ties, zero reconstructions, and the correct digest
   7c9b8f0bfb592185.... Offered toward your P-PATH-INVARIANT
   instinct, though it tests one step, not the bar.

An |r| harvest at starved precision across 8 steps is running to
see whether genuine near-ties (as opposed to argmax zeros) appear
early enough to reconstruct and measure. If they do, you get the
registered observable without a witness build; if they do not,
that itself bounds how rare the class is. Result follows.

## Bars

P-DIGEST-INVARIANT, P-WITNESS-DECIDES, P-HORIZON-2 and
P-PATH-INVARIANT all read NOT-APPLICABLE — no witness exists to
test. Your registered prediction on |r| is neither confirmed nor
refuted; it was never reached.

Fence unchanged: Mac CPU, one worker, 3080 untouched, Artin's GO
required for the next rung. Pin-3 contract observed throughout —
schedules quoted as whole expressions with the prime budget beside
them, and no prefix length quoted as a property of the anchor.
