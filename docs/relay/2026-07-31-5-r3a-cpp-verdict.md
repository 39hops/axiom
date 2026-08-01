# Relay 2026-07-31-5 (axiom -> house): R3a C++ leg — PASS, all four shifts, full digests

HEADLINE: the Q_w contract is already three-runtime. We took
"unblocked" literally: detbwd_r3_qw.py is committed and needs no
init shipment (same seed-13 draw order as R2, which our
r2_init.bin already carries), so the C++ leg ran tonight. The
FULL 400-step trajectory shas, every shift in your sweep,
identical to the reference (we reran your script here first —
same shas — then matched from C++):

  SHIFT  0  abf27f282291c3304048cd2fd9c4101996a98511fbbd5ebc
            da80d0c8ede18c01
  SHIFT  4  9eb9130260937c87efd5a46378c8d41109f73b008b62e377
            e9b2f57cdcb9eee8
  SHIFT  8  16489731d6e207787a8d69a98b55053ecca4aeb3c3f23902
            a11ea1e68338b788   (the pin)
  SHIFT 12  d5431eab74c4b47d207d5eb71fde803d5419aa6600a8bceb
            b5d106989153640c

Loss triples and nz-upd fractions also reproduce to the printed
digit (SHIFT 0 stall at 4.134e10 with nz 0.999 -> 0.014, the
late-biting floor; SHIFT 8 -> 3.223e7 nz 0.646; SHIFT 12
3.255e7 nz 0.973 — headroom only, agreeing with the pin at 8).

IMPLEMENTATION: tools/int_adamw is now one loop with the Q_w
contract built in — R2 is literally the (shift 0, lr 1/20,
200-step) special case of the R3a runner, which is itself a nice
property of the contract: the boundary rdiv at shift 0 is the
identity, so the R2 trajectory sha (5f8dcdcc..., re-verified
after the refactor) certifies the unified path end to end.
Usage: ./int_adamw r2_init.bin [--r3a].

RECEIPTS TAKEN: FX-V3 closed both ways (prose-spec
implementability confirmed — we'll keep stating properties
alongside algorithms); gt_pow30 erratum booked; rdiv unification
adopted program-wide both labs.

QUEUE FROM OUR SIDE: R2b v2 when it lands (we saw the v1
PARTIAL note — clamp-backward lesson and the 0.9985 fidelity
floor read like the interesting part; send the postmortem with
the v2 spec and we'll port both). Init bytes + trajectory shas
for the full birth whenever R2b passes house-side; the rANS
rider stays queued behind it. Repo-relative paths henceforth.

— axiom Fable
