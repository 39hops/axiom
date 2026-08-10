# RNSCHAIN C2/C3 receipt (3080 leg, 2026-08-10)

Host: WSL2 on the 3080 box; compile: nvcc.exe 13.3 (CUDA 13.3) + MSVC
14.51 host, -O3 -arch=sm_86. Fence delta: relay said house executes the
3080 leg; operator redirected - this seat IS the 3080 host, compiled and
ran locally on GO.

## C2 - biteq: FIRED. slice kernel == naive mulmod kernel, all 8
channels, entrywise, N=1024.

## C3 - wall: P-BREAKEVEN REFUTED on this instrument. RNS chain is
~74x fp64 at every depth; ratio is depth-flat, so no crossover exists
at any depth. Books as the honest-loss finding: the 13-vs-43 ms/layer
law does not transfer to this kernel shape. Instrument caveats: no
tensor cores, mulmod is a portable shift-add ladder (MSVC host, no
__int128), 8 u64 channels vs one fp64 GEMM. A Montgomery/tensor-core
reimplementation is the counter-book, not a rerun.

```jsonl
{"receipt":"rnschain-c2c3","n":1024,"channels":8,"choreography":"own; scratch/ozaki_fused.py absent from axiom repo, nothing ported","input_contract":"_f24: integers in [-2^23, 2^23-1]","seed":"0x20260810524e53"}
{"check":"slice_vs_naive_biteq","pass":true}
{"wall":{"depth":2,"rns_ms":633.211,"fp64_ms":8.820,"ratio":71.792}}
{"wall":{"depth":4,"rns_ms":1318.813,"fp64_ms":17.872,"ratio":73.792}}
{"wall":{"depth":6,"rns_ms":2024.745,"fp64_ms":26.789,"ratio":75.582}}
{"wall":{"depth":8,"rns_ms":2729.792,"fp64_ms":37.064,"ratio":73.651}}
{"wall":{"depth":12,"rns_ms":4132.376,"fp64_ms":53.778,"ratio":76.841}}
{"receipt":"rnschain-c2c3","biteq":true}
```

sha:
```
45c3fc5fd886b4bb96de11dc6ca779fa2bd44f939fb87c3185cafb683841c97a  cuda/rns_chain.cu
```
