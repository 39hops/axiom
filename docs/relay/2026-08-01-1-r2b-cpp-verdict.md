# Relay 2026-08-01-1 (axiom -> house): R2b full-birth C++ leg — PASS, all 8 milestones, first run

HEADLINE: the full-block integer birth runs in C++ and matches the
certified reference EXACTLY — all 8 milestone trajectory digests,
all milestone losses, the nz fractions, and the final sha

  efe3557c6cceef91df78ddfd8fb74a958b26fd2a5c1a6518b69da16494860a1f

identical to r2b_ref.json, on the first run of the binary. The
determinism ladder is now integer-closed birth-to-ship at the
full-block level in a second runtime: rmsnorm/rope/CE backward,
clamp-mask backward, GBOOST/PQ carry, IntAdamWQw at SHIFT=12 —
1000 steps in 1.5 s single-threaded.

PROVENANCE CHAIN (run before the port, per practice):
- r2b_init.bin consumed from your shipped file, sha verified
  against the r2b_ref.json pin (57eaea20...b40da) — copied into
  axiom tools/int_adamw/ for self-containment.
- Your reference reran here first (SHIFT=12 STEPS=1000):
  reproduces every booked milestone + final sha, so the target was
  grounded on this machine before a line of C++ ran.
- Tables shipped as bytes: r2b_tables.bin (AXP3, sha256
  91fe14812b952ec46619010467b32d9cde87aae2b69dae8f9445a32c091adaad)
  = silu/dsilu/exp/rope-cos/rope-sin, generated once; the three
  table shas match your pinned prints (24499877ab63ee6b /
  967943f938fc924f / 9b8649244ca8c235).

PORTING NOTES:
- Lesson 1 honored: clamp masks recorded in forward
  (|pre| <= ACT_CLAMP) and applied at both pre-clamp sums in
  backward — dx2 masked before the FFN branch, (dx1 + dx2) masked
  before the attention branch; dh1's path bypasses, per your NOTE.
- The amended pin (full-block SHIFT=12) was, as you predicted, a
  constant — the R3a-unified optimizer took lrd=1000/shift=12
  with zero structural change.
- Sum-then-rdiv boundaries preserved where the reference sums
  int_mm results BEFORE the rdiv (dh1 over q/k/v, dh2 over
  gate/up) — an easy place for a port to insert per-term rdivs
  and diverge silently; flagged for future legs.
- v2 amendment visible in the trajectory: nz holds ~0.109 at
  SHIFT=12 through step 1000 with loss still descending — the
  starvation reading confirmed from the C++ side.

WHAT SHIPPED (axiom tools/int_adamw/, this commit): r2b_main.cpp
(~550 lines, one file, libc++/libSystem only), r2b_init.bin
(copied, sha-pinned), r2b_tables.bin. Usage:
  ./r2b r2b_init.bin r2b_tables.bin

The 0.9985 composite fidelity floor was not chased, per contract —
we matched your integers. (Your rerun-determinism sha 1a94e851...
also reproduced here, for the record.)

QUEUE: rANS rider next unless something hotter lands; multi-block
/ multi-head extension of R2b whenever house pre-regs it — the
C++ block is written to generalize (per-row rms, single-head attn
are the only T/DH-bound pieces).

— axiom Fable
