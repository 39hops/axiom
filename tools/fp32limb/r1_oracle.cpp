/** @file r1_oracle.cpp
 *  FP32LIMB R1 receipt driver (PRE-REG FP32LIMB-METAL, relay 2026-08-10-10).
 *
 *  Runs every registered input class at N=128 plus a depth-6 chain rider,
 *  each verified entry-wise against the ax::bigint reference, and emits
 *  one JSONL receipt line per run.
 *
 *  Build (not in CMake, per tools/ convention):
 *    c++ -std=c++2b -O2 -Iinclude tools/fp32limb/r1_oracle.cpp \
 *        build-rel/libaxiom.a -o /tmp/r1_oracle
 *  Run:
 *    /tmp/r1_oracle > tools/fp32limb/receipts/r1_receipts.jsonl
 */
#include <ax/la/fp32limb.hpp>
#include <ax/st/rng.hpp>

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace ax::la::fp32limb;
using ax::bigint;

namespace {

std::string sha256_hex(const std::string& s) {
  unsigned char h[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(s.data(), static_cast<CC_LONG>(s.size()), h);
  std::string out;
  char buf[3];
  for (unsigned char b : h) {
    std::snprintf(buf, sizeof buf, "%02x", b);
    out += buf;
  }
  return out;
}

dyadic normalized(dyadic d) {
  if (d.m == bigint(0)) return {bigint(0), 0};
  while (d.m % bigint(2) == bigint(0)) {
    d.m = d.m >> 1u;
    d.e += 1;
  }
  return d;
}

std::string serialize(const std::vector<dyadic>& c) {
  std::string s;
  for (const auto& d : c) {
    const dyadic n = normalized(d);
    s += n.m.to_string();
    s += 'e';
    s += std::to_string(n.e);
    s += ';';
  }
  return s;
}

struct run_stats {
  bool pass = true;
  std::size_t slices_max = 0;
  std::string digest;
};

std::size_t max_slices(const matf& m, bool rows) {
  std::size_t mx = 0;
  std::vector<float> seg(BLOCK);
  const int outer = rows ? m.rows : m.cols;
  const int K = rows ? m.cols : m.rows;
  for (int i = 0; i < outer; ++i)
    for (int b0 = 0; b0 < K; b0 += BLOCK) {
      const int n = std::min(BLOCK, K - b0);
      for (int k = 0; k < n; ++k)
        seg[static_cast<std::size_t>(k)] =
            rows ? m.at(i, b0 + k) : m.at(b0 + k, i);
      mx = std::max(mx, slice_row(seg.data(), n).sl.size());
    }
  return mx;
}

run_stats verify(const matf& a, const matf& b) {
  run_stats st;
  const auto ours = gemm_fp32limb(a, b);
  const auto ref = gemm_ref(a, b);
  for (std::size_t t = 0; t < ref.size(); ++t)
    if (!dyadic_eq(ours[t], ref[t])) st.pass = false;
  st.slices_max = std::max(max_slices(a, true), max_slices(b, false));
  st.digest = sha256_hex(serialize(ours));
  return st;
}

void emit(const char* cls, int n, unsigned long long seed,
          const run_stats& st) {
  std::printf(
      "{\"class\":\"%s\",\"n\":%d,\"seed\":%llu,\"pass\":%s,"
      "\"max_int_dev\":\"0\",\"slices_max\":%zu,\"digest\":\"%s\"}\n",
      cls, n, seed, st.pass ? "true" : "false", st.slices_max, st.digest.c_str());
  if (!st.pass) std::exit(1);
}

// Unbounded-tail draws (uniform/normal) are flushed to zero below
// 2^-24 * scale: a full-mantissa element that far under the window max
// would breach the 56-bit envelope. The flush is part of the class
// definition and is named in the receipt class string (no silent caps).
matf random_matf(int r, int c, ax::st::rng& g, double scale, bool normal) {
  matf m(r, c);
  const double flush = scale * std::ldexp(1.0, -24);
  for (auto& v : m.v) {
    const double d = (normal ? g.normal() : g.uniform(-1.0, 1.0)) * scale;
    v = std::fabs(d) < flush ? 0.0f : static_cast<float>(d);
  }
  return m;
}

}  // namespace

int main() {
  const int N = 128;
  for (unsigned long long seed : {1ULL, 2ULL, 3ULL}) {
    {
      ax::st::rng g(seed);
      emit("uniform_f24", N, seed,
           verify(random_matf(N, N, g, 1.0, false),
                  random_matf(N, N, g, 1.0, false)));
    }
    {
      ax::st::rng g(seed + 100);
      emit("normal005_f24", N, seed,
           verify(random_matf(N, N, g, 0.05, true),
                  random_matf(N, N, g, 0.05, true)));
    }
    {
      // exponent spread inside blocks, within envelope: full mantissas
      // in [1,2) so the spread axis (the exponent) is the only variable
      ax::st::rng g(seed + 200);
      matf a(N, N), b(N, N);
      for (auto* m : {&a, &b})
        for (auto& v : m->v)
          v = static_cast<float>((g.below(2) ? 1.0 : -1.0) *
                                 g.uniform(1.0, 2.0) *
                                 std::ldexp(1.0, -int(g.below(26))));
      emit("exp_spread26", N, seed, verify(a, b));
    }
    {
      ax::st::rng g(seed + 300);
      matf a(N, N), b(N, N);
      for (auto* m : {&a, &b})
        for (auto& v : m->v)
          v = std::ldexp(static_cast<float>(g.uniform(-1.0, 1.0)), -120);
      emit("denormal_range", N, seed, verify(a, b));
    }
  }
  // depth-6 chain rider: each layer's GEMM verified exact; the carrier
  // between layers is fp32 rounding (exact cross-layer carry is R3/RNS
  // territory; dd_chain deliberately NOT ported — placeholder carrier).
  {
    const int n = 64, L = 6;
    ax::st::rng g(20260810);
    matf x = random_matf(n, n, g, 1.0 / 8.0, true);
    for (int l = 0; l < L; ++l) {
      const matf m = random_matf(n, n, g, 1.0 / 8.0, true);
      const run_stats st = verify(m, x);
      emit(("chain_layer" + std::to_string(l)).c_str(), n, 20260810, st);
      const auto exact = gemm_ref(m, x);
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
          const dyadic& d = exact[static_cast<std::size_t>(i) * n + j];
          // fp32 rounding via decimal-free path: m * 2^e in long double
          double v = 0.0;
          {
            // reconstruct exact value approximately for the carrier only
            const dyadic nd = normalized(d);
            const std::string ms = nd.m.to_string();
            v = std::strtod(ms.c_str(), nullptr) * std::ldexp(1.0, nd.e);
          }
          x.at(i, j) = static_cast<float>(v);
        }
      // carrier flush: cancellation can leave full-mantissa entries far
      // below the layer max; flush below max*2^-24 (part of the carrier
      // definition, same clause as the input classes)
      float mx = 0.0f;
      for (float v : x.v) mx = std::max(mx, std::fabs(v));
      for (float& v : x.v)
        if (std::fabs(v) < mx * std::ldexp(1.0f, -24)) v = 0.0f;
    }
  }
  return 0;
}
