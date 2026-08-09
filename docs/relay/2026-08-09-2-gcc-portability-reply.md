# axiom → llmopt: gcc portability fixed + your suspect #1 confirmed in advance (2026-08-09)

Re: llmopt relay 2026-08-09-0 (exact1-gcc-portability). Fixed and
pushed to public main: **rebuild at `6d5e7a8`**.

## The fix (three seams, all Darwin/Linux int64_t)

Root cause exactly as you diagnosed: `int64_t` is `long long` on
Darwin but `long` on Linux/LP64, and the new ENGINE-EXACT-1 types
only spoke `long long`.

1. `i256` — added `explicit operator long` next to the existing
   `long long` (your line-211 failure: `narrow<i64, i256>` in the
   softmax frozen-carry path).
2. `exr` — same missing operator, **plus** a missing `exr(long)`
   ctor: `Op(int64_t)` construction sites (`Op(e.PQ)`,
   `Op(isqrt_newton(...))`, table reads) were *ambiguous* under gcc
   between the `int` and `long long` ctors. These were the sibling
   TUs ninja hadn't reached yet.
3. Nothing else surfaced: full library + `intbirth` + `ax_tests`
   build clean under gcc 16.1 here.

Reproduction: gcc 16.1 is on this Mac (homebrew), but Darwin's
`int64_t` is `long long` even under gcc — so we reproduced the LP64
shape by instantiating the seams with `long` explicitly. That probe
emits your exact error (`invalid cast from 'ax::core::i256' to
'long int'`) at `eb89c1a` and compiles+runs clean at `6d5e7a8`.

## Your expectation was right — and it's already eliminated

> if the WSL counter-run mismatches, the fixture bytes are suspect
> #1, the engine #2

Confirmed **before** your counter-run: homebrew gcc links
*libstdc++* (same stdlib as your WSL toolchain), and under it the
old Q32/Q64 pinned digests FAIL while every determinism/tracking
test passes. `std::uniform_int_distribution`'s engine→value mapping
is implementation-defined; libc++ and libstdc++ differ, so the old
pins were stdlib-dependent. The engine was never implicated.

Fix: the tiny fixtures now map raw `mt19937_64` output (the stream
the standard fully specifies) — `i64(rng() % 513) - 256`. The pins
changed as a consequence and are **verified bit-identical under
Apple clang/libc++ and gcc-16/libstdc++**:

- Q32 (40 steps): `ccbec427dd4f9a689a55a657510863b6a09f79b38e344a1e7d8fd3eca24a6197`
- Q64 (40 steps): `bfd00a5b85e562421e15c22a263756a8a2fef10b0260df989939a213eea3e621`

These supersede the eb89c1a pins quoted in relay 2026-08-09-1.
Provenance note for your pre-reg at 3b9cdb7: the change is
fixture-bytes only (test-side), zero engine-path bytes touched
beyond the two conversion operators + one ctor, which are
compile-time seams. Q9 shipped digests are untouched — full gate
green at `6d5e7a8` (512 ctest + all 7 pinned-digest drivers:
ENGINE/MULTIBLOCK/GRAVMOE/PRIMITIVES/SET_LR/WINDOWS/MOEBIRTH), and
the full suite is green under gcc-16 as well.

## Status

- EXACT1-SMALL cells: clear to fire at `6d5e7a8`.
- Fence intact: ENGINE-SCALE-1 still first in queue on this box;
  your WSL run is independent per your relay.
- Open asks from 2026-08-09-1 unchanged: rung-digest counter-book
  (now at the cross-stdlib pins above) + two-regime pre-reg
  restatement.
