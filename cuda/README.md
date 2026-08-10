# RNSCHAIN CUDA leg (C2 + C3)

House-executed on the RTX 3080. No axiom process runs on the 3080; the 3080
runs this code. Blind-authored (no nvcc on the authoring machine).

## Build and run

```sh
nvcc -O3 -arch=sm_86 rns_chain.cu -o rns_chain
./rns_chain          # N=1024 default
./rns_chain 4096     # production-shape N for the honest-loss probe
```

Exit code 0 = C2 bit-equality (slice kernel vs naive mulmod kernel) passed
on every channel; 2 = biteq failure (details on stdout). Receipt is JSONL
on stdout — return it raw with logs and shas.

## What it measures

- **C2 biteq**: `rns_slice_gemm` (int8-slice, int64 accumulation,
  digit-weight fold mod p) against `rns_naive_gemm` (__int128 mulmod),
  depth 1, all 8 channels, entrywise.
- **C3 wall**: depth ladder L in {2,4,6,8,12} — RNS chain (permute +
  slice-GEMM per channel + fused CRT exit) vs native fp64 chain
  (permute + fp64 GEMM). P-BREAKEVEN bar: measured crossover <= depth 8.
  Honest-loss clause: if launch overhead kills the crossover at
  production N, that refutes the law's production shape and books as a
  finding.

## Caveats (state in the receipt)

- Deep chains at large N exceed the 8x61-bit RNS capacity (~488 bits);
  residues then represent the value mod M. The wall is a timing
  instrument — exactness is C1's bar (CPU oracle, N=32, capacity-proved).
- The fused exit emits nonnegative mixed-radix values as double-double
  hi/lo; centered lift is host-side.
- Choreography is our own. `scratch/ozaki_fused.py` is absent from the
  axiom repo; nothing was ported.
- No cross-device wall comparisons: these numbers never sit next to Metal
  numbers.
