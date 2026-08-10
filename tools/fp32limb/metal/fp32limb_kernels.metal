// FP32LIMB R2/R3 kernels (PRE-REG FP32LIMB-METAL; relays 2026-08-10-10/-13).
//
// Input contract: _f24 inherited from R1 (relay 2026-08-10-13 axiom-side) —
// registered constants unchanged: SLICE_W=7, BLOCK=32, MAX_SLICES=8,
// 2s + log2(b) = 19 <= 24. Slicing/alignment happens CPU-side (R1
// slice_row); these kernels are exactly the fp32 multiply-accumulate link,
// so P-KERNEL-BITEQ failures can only implicate GPU arithmetic.
//
// MUST be compiled with fast-math OFF (MTLCompileOptions, see r2_rig.mm):
// two_sum-adjacent exactness dies silently under contraction/reassociation.
// No fp simdgroup_matrix fragments anywhere: fragment-internal precision is
// vendor-defined and unprovable from outside (banked rule).
#include <metal_stdlib>
using namespace metal;

constant constexpr int SLICE_W = 7;
constant constexpr int BLOCK = 32;
constant constexpr int MAX_SLICES = 8;

// ---------------------------------------------------------------------------
// R2: single-simdgroup rung. One threadgroup = one (output entry, K-block);
// 32 lanes span the K-block. Slices arrive zero-padded to MAX_SLICES so
// every threadgroup runs the identical shape (zeros are exact).
// Each product |Qa*Qb| <= 2^(2*SLICE_W-2) = 2^12 and each 32-term sum
// < 2^17 < 2^24: every fp32 op is exact, so simd_sum's vendor-defined
// order cannot change the value (exact adds are associative).
// Guard link: lane-local |partial| >= 2^24 flags loudly into out-of-band
// buffer (device-side equivalent of the R1 throw fence).
kernel void r2_slicepair_dots(
    device const float* asl [[buffer(0)]],   // [ntg][MAX_SLICES][BLOCK]
    device const float* bsl [[buffer(1)]],   // [ntg][MAX_SLICES][BLOCK]
    device float* partials [[buffer(2)]],    // [ntg][MAX_SLICES][MAX_SLICES]
    device atomic_uint* overflow [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  device const float* a = asl + (ulong)tg * MAX_SLICES * BLOCK;
  device const float* b = bsl + (ulong)tg * MAX_SLICES * BLOCK;
  device float* out = partials + (ulong)tg * MAX_SLICES * MAX_SLICES;
  for (int p = 0; p < MAX_SLICES; ++p) {
    const float av = a[p * BLOCK + lane];
    for (int q = 0; q < MAX_SLICES; ++q) {
      const float prod = av * b[q * BLOCK + lane];
      const float s = simd_sum(prod);
      if (lane == 0) {
        if (fabs(s) >= 16777216.0f)
          atomic_fetch_add_explicit(overflow, 1u, memory_order_relaxed);
        out[p * MAX_SLICES + q] = s;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// FTZ / denormal probe (R2 receipt item 3): measured, not assumed.
// in[0] = a denormal (e.g. 2^-140 as fp32). If arithmetic flushes to zero,
// the scheme needs an explicit range restriction (pre-reg risk item).
kernel void ftz_probe(device const float* in [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      uint t [[thread_position_in_grid]]) {
  if (t != 0) return;
  out[0] = in[0] * 0.5f;        // denormal * 0.5: preserved -> 2^-141
  out[1] = in[0] + in[0];       // denormal + denormal -> 2^-139
  out[2] = in[0] * 1.0f;        // identity multiply: FTZ kills even this
  out[3] = in[1] - in[2];       // subtract producing a denormal result
}

// ---------------------------------------------------------------------------
// R3: ONE tiling choreography, TWO instantiations (pre-reg 24886).
// Choreography: grid tiles the output; each threadgroup owns a TM x TN
// output tile and marches the K-blocks, staging the A-tile slices in
// threadgroup memory (B streams from device memory). One thread = one
// output entry of the tile; the inner slice-pair loops are identical to
// R2 so the exactness argument transfers unchanged.
constant constexpr int TM = 8;
constant constexpr int TN = 8;

// Instantiation 1: fp32-limb. Emits per-entry per-block partial sums
// (recombination to the exact result stays host-side through the same
// bigint path as R1 — the shared-page unified-memory exit: no staging
// copy on M-series).
kernel void r3_fp32limb_tiled(
    device const float* asl [[buffer(0)]],  // [rows][nblk][MAX_SLICES][BLOCK]
    device const float* bsl [[buffer(1)]],  // [cols][nblk][MAX_SLICES][BLOCK]
    device float* partials [[buffer(2)]],   // [rows][cols][nblk][MS][MS]
    device atomic_uint* overflow [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& nblk [[buffer(6)]],
    uint2 tgpos [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]) {
  const uint i = tgpos.y * TM + tid.y;
  const uint j = tgpos.x * TN + tid.x;
  if (i >= rows || j >= cols) return;
  threadgroup float atile[TM][MAX_SLICES][BLOCK];
  const ulong ms2 = (ulong)MAX_SLICES * MAX_SLICES;
  for (uint blk = 0; blk < nblk; ++blk) {
    // stage this K-block's A-tile slices cooperatively
    const uint nthreads = TM * TN;
    const uint flat = tid.y * TN + tid.x;
    const uint span = (uint)(TM * MAX_SLICES * BLOCK);
    for (uint idx = flat; idx < span; idx += nthreads) {
      const uint r = idx / (MAX_SLICES * BLOCK);
      const uint rem = idx % (MAX_SLICES * BLOCK);
      const uint gi = tgpos.y * TM + r;
      atile[r][rem / BLOCK][rem % BLOCK] =
          (gi < rows)
              ? asl[(((ulong)gi * nblk + blk) * MAX_SLICES * BLOCK) + rem]
              : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float* b =
        bsl + ((ulong)j * nblk + blk) * MAX_SLICES * BLOCK;
    device float* out =
        partials + (((ulong)i * cols + j) * nblk + blk) * ms2;
    for (int p = 0; p < MAX_SLICES; ++p)
      for (int q = 0; q < MAX_SLICES; ++q) {
        // contraction note: acc + a*b -> fma(a,b,acc) cannot change the
        // value here — every product is an exact integer < 2^14 and every
        // partial sum < 2^24, so mul, add, and fused mul-add all round to
        // the same (exact) result. fast-math stays pinned OFF regardless.
        float acc = 0.0f;
        for (int k = 0; k < BLOCK; ++k)
          acc += atile[tid.y][p][k] * b[q * BLOCK + k];
        if (fabs(acc) >= 16777216.0f)
          atomic_fetch_add_explicit(overflow, 1u, memory_order_relaxed);
        out[p * MAX_SLICES + q] = acc;
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Instantiation 2: integer accumulator on decoded 24-bit mantissas
// (the exact_gemm half; the 2^47 product bound survives any tiling, so
// tiling is pure speed). Accumulator width probed at compile time: the
// rig first probes 64-bit `long` support and falls back to recording
// NOT-AVAILABLE (see r2_rig.mm probes). Products of 24-bit mantissas
// are < 2^48; BLOCK of them < 2^53 — needs the wide accumulator.
kernel void r3_intacc_tiled(
    device const int* am [[buffer(0)]],   // [rows][K] aligned mantissas
    device const int* bm [[buffer(1)]],   // [cols][K]
    device long* acc_out [[buffer(2)]],   // [rows][cols][nblk]
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& nblk [[buffer(6)]],
    uint2 tgpos [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]) {
  const uint i = tgpos.y * TM + tid.y;
  const uint j = tgpos.x * TN + tid.x;
  if (i >= rows || j >= cols) return;
  const uint K = nblk * BLOCK;
  for (uint blk = 0; blk < nblk; ++blk) {
    long acc = 0;
    for (uint k = blk * BLOCK; k < (blk + 1) * BLOCK; ++k)
      acc += (long)am[(ulong)i * K + k] * (long)bm[(ulong)j * K + k];
    acc_out[((ulong)i * cols + j) * nblk + blk] = acc;
  }
}
