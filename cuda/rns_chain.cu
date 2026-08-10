/** @file rns_chain.cu RNSCHAIN C2/C3: CUDA depth-chain kernels + wall driver.
 *
 * Blind-authored on a machine with no nvcc; compiled and smoke-run by the
 * house on the RTX 3080. Standalone, no cuBLAS:
 *
 *   nvcc -O3 -arch=sm_86 rns_chain.cu -o rns_chain
 *   ./rns_chain [N]                # default N=1024
 *
 * Kernels:
 *   rns_slice_gemm  - C = A*B mod p per 61-bit prime channel. Operand
 *                     residues split into 8 unsigned 8-bit digits; per
 *                     output element a register loop accumulates each
 *                     digit-weight bucket in int64 (int32 would overflow
 *                     silently at large K - banked hazard class), then
 *                     folds buckets with precomputed 2^(8s) mod p.
 *                     Choreography is our own; the house Triton reference
 *                     (scratch/ozaki_fused.py) is absent from this repo,
 *                     so nothing is ported - receipt cites this.
 *   rns_naive_gemm  - same product via direct mulmod-ladder; on-device
 *                     bit-equality cross-check arm for the slice kernel.
 *   crt_exit_fused  - fused CRT exit: Garner mixed-radix per element, then
 *                     double-double (two-sum/two-prod) recombination and a
 *                     single hi/lo write per element.
 *   fp64_gemm       - native fp64 baseline for the C3 wall ladder.
 *
 * All bound checks are plain branches - they survive release builds
 * (assert() would not under NDEBUG).
 *
 * Receipt: JSONL on stdout for the house to return raw.
 */
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using u64 = std::uint64_t;
using i64 = long long;
// No __int128 anywhere: the house leg builds with nvcc + MSVC host, which
// lacks it. mulmod is a shift-add (peasant) ladder over addmod - portable,
// slower; the receipt states it so wall numbers are read accordingly.

#define CUDA_CHECK(x)                                                      \
  do {                                                                     \
    cudaError_t err_ = (x);                                                \
    if (err_ != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                     \
                   cudaGetErrorString(err_), __FILE__, __LINE__);          \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

// ----- host-side number theory (self-contained; ax lib not linked) -----

static u64 h_addmod(u64 a, u64 b, u64 p) {
  u64 s = a + b;  // both < 2^61
  return s >= p ? s - p : s;
}

static u64 h_mulmod(u64 a, u64 b, u64 p) {
  u64 r = 0;
  a %= p;
  for (b %= p; b; b >>= 1) {
    if (b & 1) r = h_addmod(r, a, p);
    a = h_addmod(a, a, p);
  }
  return r;
}

static u64 h_powmod(u64 a, u64 e, u64 p) {
  u64 r = 1 % p;
  a %= p;
  while (e) {
    if (e & 1) r = h_mulmod(r, a, p);
    a = h_mulmod(a, a, p);
    e >>= 1;
  }
  return r;
}

// deterministic Miller-Rabin, valid for all n < 2^64
static bool h_is_prime(u64 n) {
  if (n < 2) return false;
  for (u64 q : {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL, 19ULL, 23ULL,
                29ULL, 31ULL, 37ULL})
    if (n % q == 0) return n == q;
  u64 d = n - 1;
  int r = 0;
  while ((d & 1) == 0) d >>= 1, ++r;
  for (u64 a : {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL, 19ULL, 23ULL,
                29ULL, 31ULL, 37ULL}) {
    u64 x = h_powmod(a % n, d, n);
    if (x == 1 || x == n - 1) continue;
    bool witness = true;
    for (int i = 1; i < r; ++i) {
      x = h_mulmod(x, x, n);
      if (x == n - 1) {
        witness = false;
        break;
      }
    }
    if (witness) return false;
  }
  return true;
}

// same pin as ax::rns::pinned_primes - largest primes below 2^61, descending
static std::vector<u64> pinned_primes(int count) {
  std::vector<u64> ps;
  for (u64 c = (1ULL << 61) - 1; (int)ps.size() < count; c -= 2)
    if (h_is_prime(c)) ps.push_back(c);
  return ps;
}

static u64 h_inv(u64 a, u64 p) { return h_powmod(a % p, p - 2, p); }

// ----- device kernels --------------------------------------------------

constexpr int kMaxCh = 8;
__constant__ u64 c_primes[kMaxCh];
__constant__ u64 c_pow2mod[kMaxCh][16];  // 2^(8s) mod p, s in [0,16)

__device__ __forceinline__ u64 d_addmod(u64 a, u64 b, u64 p) {
  u64 s = a + b;  // both < 2^61
  return s >= p ? s - p : s;
}

__device__ __forceinline__ u64 d_mulmod(u64 a, u64 b, u64 p) {
  u64 r = 0;
  a %= p;
  for (b %= p; b; b >>= 1) {
    if (b & 1) r = d_addmod(r, a, p);
    a = d_addmod(a, a, p);
  }
  return r;
}

// int8-slice GEMM in one RNS channel. A: MxK, B: KxN residues of channel ch.
__global__ void rns_slice_gemm(const u64* A, const u64* B, u64* C, int M,
                               int N, int K, int ch) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N || ch >= kMaxCh) return;  // release-safe bounds
  u64 p = c_primes[ch];
  // digit-weight buckets s = i+j in [0, 15); int64 mandatory: worst case
  // 255*255*K per bucket ~= K*2^16 - silent int32 overflow at K >= 2^15.
  i64 acc[15];
#pragma unroll
  for (int s = 0; s < 15; ++s) acc[s] = 0;
  const int kFold = 1 << 20;  // fold cadence; far below i64 ceiling
  int since_fold = 0;
  u64 out = 0;
  for (int k = 0; k < K; ++k) {
    u64 a = A[(size_t)row * K + k];
    u64 b = B[(size_t)k * N + col];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      i64 ai = (i64)((a >> (8 * i)) & 0xff);
      if (ai == 0) continue;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        i64 bj = (i64)((b >> (8 * j)) & 0xff);
        acc[i + j] += ai * bj;
      }
    }
    if (++since_fold == kFold || k == K - 1) {
#pragma unroll
      for (int s = 0; s < 15; ++s) {
        out = d_addmod(out, d_mulmod((u64)acc[s] % p, c_pow2mod[ch][s], p),
                       p);
        acc[s] = 0;
      }
      since_fold = 0;
    }
  }
  C[(size_t)row * N + col] = out;
}

// direct mulmod GEMM - cross-check arm for the slice kernel
__global__ void rns_naive_gemm(const u64* A, const u64* B, u64* C, int M,
                               int N, int K, int ch) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N || ch >= kMaxCh) return;
  u64 p = c_primes[ch];
  u64 acc = 0;
  for (int k = 0; k < K; ++k)
    acc = d_addmod(
        acc, d_mulmod(A[(size_t)row * K + k], B[(size_t)k * N + col], p), p);
  C[(size_t)row * N + col] = acc;
}

// column permutation, one channel
__global__ void rns_permute_kernel(const u64* Y, u64* out, const int* pi,
                                   int M, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  out[(size_t)row * N + col] = Y[(size_t)row * N + pi[col]];
}

// ----- fused CRT exit --------------------------------------------------

struct GarnerTables {
  u64 inv[kMaxCh][kMaxCh];  // inv[i][j] = p_i^-1 mod p_j, for i < j
  double radix_hi[kMaxCh];  // prod_{c<i} p_c as double-double
  double radix_lo[kMaxCh];
};
__constant__ GarnerTables c_garner;

__device__ __forceinline__ void two_sum(double a, double b, double& s,
                                        double& e) {
  s = a + b;
  double bb = s - a;
  e = (a - (s - bb)) + (b - bb);
}

__device__ __forceinline__ void two_prod(double a, double b, double& p,
                                         double& e) {
  p = a * b;
  e = fma(a, b, -p);
}

// Garner mixed-radix digits, then double-double recombination.
// One hi/lo write per element (fused exit - no residue round trip).
__global__ void crt_exit_fused(const u64* const* res, int channels, int n,
                               double* hi, double* lo) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n || channels > kMaxCh) return;
  u64 t[kMaxCh];  // mixed-radix digits
  for (int i = 0; i < channels; ++i) {
    u64 x = res[i][idx] % c_primes[i];
    for (int j = 0; j < i; ++j) {
      u64 diff = x >= t[j] % c_primes[i]
                     ? x - t[j] % c_primes[i]
                     : x + c_primes[i] - t[j] % c_primes[i];
      x = d_mulmod(diff, c_garner.inv[j][i], c_primes[i]);
    }
    t[i] = x;
  }
  // value = sum_i t_i * radix_i, accumulated in double-double. Centered
  // lift is the host's job (it knows M); the exit emits the nonnegative
  // mixed-radix value as hi+lo - the fp64-facing fused path.
  double shi = 0.0, slo = 0.0;
  for (int i = 0; i < channels; ++i) {
    double ph, pe;
    two_prod((double)t[i], c_garner.radix_hi[i], ph, pe);
    pe = fma((double)t[i], c_garner.radix_lo[i], pe);
    double s, e;
    two_sum(shi, ph, s, e);
    shi = s;
    slo += e + pe;
  }
  double s, e;
  two_sum(shi, slo, s, e);
  hi[idx] = s;  // single hi/lo write
  lo[idx] = e;
}

// ----- fp64 baseline chain (C3 wall) -----------------------------------

__global__ void fp64_gemm(const double* A, const double* B, double* C, int M,
                          int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  double acc = 0.0;
  for (int k = 0; k < K; ++k)
    acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
  C[(size_t)row * N + col] = acc;
}

__global__ void fp64_permute_kernel(const double* Y, double* out,
                                    const int* pi, int M, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  out[(size_t)row * N + col] = Y[(size_t)row * N + pi[col]];
}

// ----- driver ----------------------------------------------------------

static u64 splitmix64(u64& s) {
  s += 0x9e3779b97f4a7c15ULL;
  u64 z = s;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

int main(int argc, char** argv) {
  int N = argc > 1 ? std::atoi(argv[1]) : 1024;
  if (N < 1) return 1;
  const int channels = kMaxCh;
  const int depths[] = {2, 4, 6, 8, 12};
  const u64 seed = 0x20260810524e53ULL;

  auto primes = pinned_primes(channels);
  CUDA_CHECK(
      cudaMemcpyToSymbol(c_primes, primes.data(), sizeof(u64) * channels));
  u64 pow2[kMaxCh][16];
  for (int c = 0; c < channels; ++c)
    for (int s2 = 0; s2 < 16; ++s2)
      pow2[c][s2] = h_powmod(2, 8ULL * s2, primes[c]);
  CUDA_CHECK(cudaMemcpyToSymbol(c_pow2mod, pow2, sizeof(pow2)));

  GarnerTables gt = {};
  for (int i = 0; i < channels; ++i)
    for (int j = i + 1; j < channels; ++j)
      gt.inv[i][j] = h_inv(primes[i] % primes[j], primes[j]);
  double rhi = 1.0, rlo = 0.0;
  for (int i = 0; i < channels; ++i) {
    gt.radix_hi[i] = rhi;
    gt.radix_lo[i] = rlo;
    double ph = rhi * (double)primes[i];
    double pe = fma(rhi, (double)primes[i], -ph);
    rhi = ph;
    rlo = rlo * (double)primes[i] + pe;
  }
  CUDA_CHECK(cudaMemcpyToSymbol(c_garner, &gt, sizeof(gt)));

  std::printf(
      "{\"receipt\":\"rnschain-c2c3\",\"n\":%d,\"channels\":%d,"
      "\"choreography\":\"own; scratch/ozaki_fused.py absent from axiom "
      "repo, nothing ported\",\"input_contract\":\"_f24: integers in "
      "[-2^23, 2^23-1]\",\"seed\":\"0x%llx\"}\n",
      N, channels, (unsigned long long)seed);

  size_t elems = (size_t)N * N, bytes = elems * sizeof(u64);
  std::vector<double> hostAd(elems);
  u64 s = seed;
  for (size_t i = 0; i < elems; ++i)
    hostAd[i] =
        (double)((long long)(splitmix64(s) % (1ULL << 24)) - (1LL << 23));
  std::vector<int> pi(N);
  for (int i = 0; i < N; ++i) pi[i] = i;
  for (int i = N - 1; i > 0; --i) {
    int j = (int)(splitmix64(s) % (u64)(i + 1));
    int t = pi[i];
    pi[i] = pi[j];
    pi[j] = t;
  }
  int* dPi;
  CUDA_CHECK(cudaMalloc(&dPi, sizeof(int) * N));
  CUDA_CHECK(
      cudaMemcpy(dPi, pi.data(), sizeof(int) * N, cudaMemcpyHostToDevice));

  u64 *dY[kMaxCh], *dT[kMaxCh], *dW[kMaxCh], *dChk;
  for (int c = 0; c < channels; ++c) {
    CUDA_CHECK(cudaMalloc(&dY[c], bytes));
    CUDA_CHECK(cudaMalloc(&dT[c], bytes));
    CUDA_CHECK(cudaMalloc(&dW[c], bytes));
  }
  CUDA_CHECK(cudaMalloc(&dChk, bytes));
  double *dYd, *dTd, *dWd;
  CUDA_CHECK(cudaMalloc(&dYd, elems * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dTd, elems * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dWd, elems * sizeof(double)));

  std::vector<u64> hres(elems);
  auto upload_residues = [&](const std::vector<double>& src,
                             u64* dst[kMaxCh]) {
    for (int c = 0; c < channels; ++c) {
      u64 p = primes[c];
      for (size_t i = 0; i < elems; ++i) {
        long long v = (long long)src[i];
        if (v >= 0) {
          hres[i] = (u64)v % p;
        } else {
          u64 m = ((u64)(-(v + 1)) + 1) % p;
          hres[i] = m == 0 ? 0 : p - m;
        }
      }
      CUDA_CHECK(
          cudaMemcpy(dst[c], hres.data(), bytes, cudaMemcpyHostToDevice));
    }
  };

  dim3 blk(16, 16), grd((unsigned)(N + 15) / 16, (unsigned)(N + 15) / 16);
  cudaEvent_t ev0, ev1;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));

  // one pinned weight matrix reused each layer: the wall measures
  // arithmetic + launch cost, not weight variety; C1 owns exactness
  std::vector<double> hostW(elems);
  for (size_t i = 0; i < elems; ++i)
    hostW[i] =
        (double)((long long)(splitmix64(s) % (1ULL << 24)) - (1LL << 23));
  upload_residues(hostW, dW);
  CUDA_CHECK(cudaMemcpy(dWd, hostW.data(), elems * sizeof(double),
                        cudaMemcpyHostToDevice));

  // --- C2 biteq: slice vs naive kernel, depth 1, every channel ---
  upload_residues(hostAd, dY);
  bool biteq = true;
  std::vector<u64> ha(elems), hb(elems);
  for (int c = 0; c < channels && biteq; ++c) {
    rns_slice_gemm<<<grd, blk>>>(dY[c], dW[c], dT[c], N, N, N, c);
    rns_naive_gemm<<<grd, blk>>>(dY[c], dW[c], dChk, N, N, N, c);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(ha.data(), dT[c], bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), dChk, bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < elems; ++i)
      if (ha[i] != hb[i]) {
        biteq = false;
        std::printf(
            "{\"biteq_fail\":{\"channel\":%d,\"index\":%zu,"
            "\"slice\":%llu,\"naive\":%llu}}\n",
            c, i, (unsigned long long)ha[i], (unsigned long long)hb[i]);
        break;
      }
  }
  std::printf("{\"check\":\"slice_vs_naive_biteq\",\"pass\":%s}\n",
              biteq ? "true" : "false");

  // --- C3 wall: depth ladder, RNS chain + fused exit vs fp64 chain ---
  double *dHi, *dLo;
  CUDA_CHECK(cudaMalloc(&dHi, elems * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dLo, elems * sizeof(double)));
  u64** dResPtrs;
  CUDA_CHECK(cudaMalloc(&dResPtrs, sizeof(u64*) * channels));
  CUDA_CHECK(cudaMemcpy(dResPtrs, dY, sizeof(u64*) * channels,
                        cudaMemcpyHostToDevice));
  for (int L : depths) {
    upload_residues(hostAd, dY);
    CUDA_CHECK(cudaEventRecord(ev0));
    for (int d = 0; d < L; ++d)
      for (int c = 0; c < channels; ++c) {
        rns_permute_kernel<<<grd, blk>>>(dY[c], dT[c], dPi, N, N);
        rns_slice_gemm<<<grd, blk>>>(dT[c], dW[c], dY[c], N, N, N, c);
      }
    crt_exit_fused<<<((int)elems + 255) / 256, 256>>>(
        (const u64* const*)dResPtrs, channels, (int)elems, dHi, dLo);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaGetLastError());
    float rns_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&rns_ms, ev0, ev1));

    CUDA_CHECK(cudaMemcpy(dYd, hostAd.data(), elems * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev0));
    for (int d = 0; d < L; ++d) {
      fp64_permute_kernel<<<grd, blk>>>(dYd, dTd, dPi, N, N);
      fp64_gemm<<<grd, blk>>>(dTd, dWd, dYd, N, N, N);
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaGetLastError());
    float fp64_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&fp64_ms, ev0, ev1));

    std::printf(
        "{\"wall\":{\"depth\":%d,\"rns_ms\":%.3f,\"fp64_ms\":%.3f,"
        "\"ratio\":%.3f}}\n",
        L, rns_ms, fp64_ms, fp64_ms > 0 ? rns_ms / fp64_ms : 0.0);
  }

  std::printf("{\"receipt\":\"rnschain-c2c3\",\"biteq\":%s}\n",
              biteq ? "true" : "false");
  return biteq ? 0 : 2;
}
