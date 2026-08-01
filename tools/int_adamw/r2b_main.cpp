// R2b C++ leg: the FULL transformer block trained end-to-end in
// int64 (reference: llmopt scratch/detbwd_r2b.py; contract:
// scratch/detbwd_r2b_ref/r2b_ref.json — SHIFT=12, GBOOST=256,
// PQ=8192, ACT_CLAMP=16384, lr 1/1000, 1000 steps). New machinery
// over the R2/R3a leg: rmsnorm backward (isqrt-based), rope backward
// (transpose rotation), CE gradient (p - Q*onehot via the r1b
// softmax), clamp masks applied in backward (house lesson 1), GBOOST
// at the loss / unboost at the optimizer, attention probs at PQ.
// Init + tables consumed as shipped bytes; PASS = the 8 milestone
// trajectory digests + losses of the certified reference run.
//
// Build: c++ -O2 -std=c++17 r2b_main.cpp -o r2b
// Run:   ./r2b r2b_init.bin r2b_tables.bin
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using i64 = int64_t;
using u64 = uint64_t;
using u32 = uint32_t;
using u8 = uint8_t;

static const i64 Q = 512, TSs = 4096, TSE = 4096;   // silu / exp ranges
static const i64 RS = 1 << 14, R16 = 1 << 16, EPS32 = 42950;
static const i64 PQ = 8192, GBOOST = 256, ACT_CLAMP = 16384;
static const i64 SCALE = 2048;  // round(Q * sqrt(DH)), DH = 16
static const int T = 32, D = 64, DH = 16, F = 128, V = 64;
static const int SHIFT = 12, STEPS = 1000;
static const i64 LRN = 1, LRD = 1000, AEPS = 4;
static const i64 B1N = 9, B1D = 10, B2N = 999, B2D = 1000;
static const i64 WDN = 1, WDD = 100000;

// ---------- sha256 (as the sibling tools) ----------
struct Sha256 {
  u32 h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
              0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  u8 buf[64];
  u64 len = 0;
  size_t fill = 0;
  static u32 rot(u32 x, int n) { return (x >> n) | (x << (32 - n)); }
  void block(const u8* p) {
    static const u32 K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b,
        0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
        0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7,
        0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152,
        0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
        0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
        0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
        0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    u32 w[64];
    for (int i = 0; i < 16; i++)
      w[i] = (u32(p[4 * i]) << 24) | (u32(p[4 * i + 1]) << 16) |
             (u32(p[4 * i + 2]) << 8) | u32(p[4 * i + 3]);
    for (int i = 16; i < 64; i++) {
      u32 s0 = rot(w[i - 15], 7) ^ rot(w[i - 15], 18) ^ (w[i - 15] >> 3);
      u32 s1 = rot(w[i - 2], 17) ^ rot(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    u32 a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5],
        g = h[6], hh = h[7];
    for (int i = 0; i < 64; i++) {
      u32 S1 = rot(e, 6) ^ rot(e, 11) ^ rot(e, 25);
      u32 ch = (e & f) ^ (~e & g);
      u32 t1 = hh + S1 + ch + K[i] + w[i];
      u32 S0 = rot(a, 2) ^ rot(a, 13) ^ rot(a, 22);
      u32 mj = (a & b) ^ (a & c) ^ (b & c);
      u32 t2 = S0 + mj;
      hh = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  void update(const void* data, size_t n) {
    const u8* p = (const u8*)data;
    len += n;
    while (n) {
      size_t take = 64 - fill < n ? 64 - fill : n;
      memcpy(buf + fill, p, take);
      fill += take; p += take; n -= take;
      if (fill == 64) { block(buf); fill = 0; }
    }
  }
  std::string hex() {
    u64 bits = len * 8;
    u8 pad = 0x80;
    update(&pad, 1);
    u8 z = 0;
    while (fill != 56) update(&z, 1);
    u8 lb[8];
    for (int i = 0; i < 8; i++) lb[i] = u8(bits >> (56 - 8 * i));
    update(lb, 8);
    char out[65];
    for (int i = 0; i < 8; i++) snprintf(out + 8 * i, 9, "%08x", h[i]);
    return std::string(out, 64);
  }
};

// ---------- big-uint for the exact bias correction (as R2 leg) ----------
struct Big {
  std::vector<u32> d{1};
  void trim() { while (d.size() > 1 && d.back() == 0) d.pop_back(); }
  void mul_small(u64 k) {
    u64 carry = 0;
    for (auto& x : d) {
      u64 p = u64(x) * k + carry;
      x = u32(p);
      carry = p >> 32;
    }
    while (carry) { d.push_back(u32(carry)); carry >>= 32; }
  }
  int bits() const {
    u32 top = d.back();
    int b = 0;
    while (top) { b++; top >>= 1; }
    return int(d.size() - 1) * 32 + b;
  }
  void shr1() {
    for (size_t i = 0; i < d.size(); i++) {
      u32 lo = (i + 1 < d.size()) ? (d[i + 1] & 1) : 0;
      d[i] = (d[i] >> 1) | (lo << 31);
    }
    trim();
  }
  bool gt_pow30() const {
    const int b = bits();
    if (b != 31) return b > 31;
    return !(d.size() == 1 && d[0] == 0x40000000u);
  }
  i64 to_i64() const {
    u64 v = 0;
    for (size_t i = d.size(); i-- > 0;) v = (v << 32) | d[i];
    return i64(v);
  }
  static Big sub(const Big& a, const Big& b) {
    Big r;
    r.d.assign(a.d.size(), 0);
    i64 borrow = 0;
    for (size_t i = 0; i < a.d.size(); i++) {
      i64 x = i64(a.d[i]) - (i < b.d.size() ? i64(b.d[i]) : 0) - borrow;
      borrow = x < 0;
      r.d[i] = u32(x + (borrow << 32));
    }
    r.trim();
    return r;
  }
};

// ---------- primitives ----------
static inline i64 rdiv(i64 x, i64 d) {
  i64 ax = x < 0 ? -x : x;
  i64 r = (ax + d / 2) / d;
  return x < 0 ? -r : r;
}
static i64 isqrt_newton(i64 x) {
  if (x <= 0) return 0;
  i64 r = x;
  for (int i = 0; i < 40; i++) {
    r = (r + x / r) / 2;
    if (r < 1) r = 1;
  }
  if (r * r > x) r -= 1;
  if ((r + 1) * (r + 1) <= x) r += 1;
  return r;
}
static inline i64 clampi(i64 x, i64 lo, i64 hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

using Mat = std::vector<i64>;

// a[rows,K] @ w[N,K]^T -> [rows,N]  (torch int_mm(a, w))
static Mat mmT(const Mat& a, int rows, int K, const Mat& w, int N) {
  Mat y((size_t)rows * N);
  for (int t = 0; t < rows; t++)
    for (int n = 0; n < N; n++) {
      i64 acc = 0;
      const i64* ar = &a[(size_t)t * K];
      const i64* wr = &w[(size_t)n * K];
      for (int k = 0; k < K; k++) acc += ar[k] * wr[k];
      y[(size_t)t * N + n] = acc;
    }
  return y;
}
// a[rows,K] @ w[K,N] -> [rows,N]  (torch int_mm(a, w.T))
static Mat mm(const Mat& a, int rows, int K, const Mat& w, int N) {
  Mat y((size_t)rows * N);
  for (int t = 0; t < rows; t++)
    for (int n = 0; n < N; n++) {
      i64 acc = 0;
      for (int k = 0; k < K; k++)
        acc += a[(size_t)t * K + k] * w[(size_t)k * N + n];
      y[(size_t)t * N + n] = acc;
    }
  return y;
}
// x[rows,K]^T @ y[rows,N] -> [K,N]  (torch int_mm(x.T, y.T))
static Mat xty(const Mat& x, int rows, int K, const Mat& y, int N) {
  Mat o((size_t)K * N, 0);
  for (int t = 0; t < rows; t++)
    for (int k = 0; k < K; k++) {
      const i64 xv = x[(size_t)t * K + k];
      if (!xv) continue;
      for (int n = 0; n < N; n++)
        o[(size_t)k * N + n] += xv * y[(size_t)t * N + n];
    }
  return o;
}
static void rdiv_all(Mat& m, i64 d) {
  for (auto& v : m) v = rdiv(v, d);
}

// ---------- tables ----------
struct Tab {
  Mat silu, dsilu, exp, cos, sin;  // shipped bytes
};
static inline i64 lut_silu(const Tab& tb, i64 x) {
  if (x > TSs) return x;
  if (x < -TSs) return 0;
  return tb.silu[x + TSs];
}
static inline i64 lut_dsilu(const Tab& tb, i64 x) {
  if (x > TSs) return Q;
  if (x < -TSs) return 0;
  return tb.dsilu[x + TSs];
}
static inline i64 lut_exp(const Tab& tb, i64 d) {
  return d < -TSE ? 0 : tb.exp[d + TSE];
}

// rope on [T, DH], cos/sin [T, DH/2]
static Mat rope_fwd(const Mat& x, const Tab& tb) {
  const int half = DH / 2;
  Mat y((size_t)T * DH);
  for (int t = 0; t < T; t++)
    for (int i = 0; i < half; i++) {
      const i64 c = tb.cos[(size_t)t * half + i];
      const i64 s = tb.sin[(size_t)t * half + i];
      const i64 v1 = x[(size_t)t * DH + i];
      const i64 v2 = x[(size_t)t * DH + half + i];
      y[(size_t)t * DH + i] = rdiv(v1 * c - v2 * s, RS);
      y[(size_t)t * DH + half + i] = rdiv(v1 * s + v2 * c, RS);
    }
  return y;
}
static Mat rope_bwd(const Mat& dx, const Tab& tb) {
  const int half = DH / 2;
  Mat y((size_t)T * DH);
  for (int t = 0; t < T; t++)
    for (int i = 0; i < half; i++) {
      const i64 c = tb.cos[(size_t)t * half + i];
      const i64 s = tb.sin[(size_t)t * half + i];
      const i64 d1 = dx[(size_t)t * DH + i];
      const i64 d2 = dx[(size_t)t * DH + half + i];
      y[(size_t)t * DH + i] = rdiv(d1 * c + d2 * s, RS);
      y[(size_t)t * DH + half + i] = rdiv(-d1 * s + d2 * c, RS);
    }
  return y;
}

// rmsnorm forward on [T, D]; returns y and per-row isq
static Mat rms_fwd(const Mat& x, const Mat& g, Mat& isq) {
  Mat y((size_t)T * D);
  isq.assign(T, 0);
  for (int t = 0; t < T; t++) {
    i64 s2 = 0;
    for (int d = 0; d < D; d++) {
      const i64 v = x[(size_t)t * D + d];
      s2 += v * v;
    }
    const i64 m40 = (s2 / D) * (i64(1) << 32) / (Q * Q) + EPS32;
    isq[t] = isqrt_newton(m40);
    for (int d = 0; d < D; d++)
      y[(size_t)t * D + d] =
          rdiv(rdiv(x[(size_t)t * D + d] * g[d], Q) * R16, isq[t]);
  }
  return y;
}
static Mat rms_bwd(const Mat& dy, const Mat& x, const Mat& g,
                   const Mat& isq, Mat& dg) {
  Mat dx((size_t)T * D);
  dg.assign(D, 0);
  for (int t = 0; t < T; t++) {
    i64 inner = 0;
    Mat tv(D);
    for (int d = 0; d < D; d++) {
      tv[d] = rdiv(g[d] * dy[(size_t)t * D + d], Q);
      inner += rdiv(tv[d] * x[(size_t)t * D + d], Q);
    }
    for (int d = 0; d < D; d++) {
      const i64 xv = x[(size_t)t * D + d];
      const i64 term1 = rdiv(tv[d] * R16, isq[t]);
      i64 c = rdiv(xv * inner, i64(D) * Q);
      for (int r = 0; r < 3; r++) c = rdiv(c * R16, isq[t]);
      dx[(size_t)t * D + d] = term1 - c;
      dg[d] += rdiv(rdiv(dy[(size_t)t * D + d] * xv, Q) * R16, isq[t]);
    }
  }
  return dx;
}

// row softmax on [rows, C] at `scale` units
static Mat softmax_rows(const Mat& s, int rows, int C, const Tab& tb,
                        i64 scale) {
  Mat p((size_t)rows * C);
  for (int t = 0; t < rows; t++) {
    i64 m = INT64_MIN;
    for (int c = 0; c < C; c++) m = std::max(m, s[(size_t)t * C + c]);
    i64 z = 0;
    Mat e(C);
    for (int c = 0; c < C; c++) {
      i64 d = s[(size_t)t * C + c] - m;
      if (d < -TSE - 1) d = -TSE - 1;
      e[c] = lut_exp(tb, d);
      z += e[c];
    }
    for (int c = 0; c < C; c++) p[(size_t)t * C + c] = rdiv(e[c] * scale, z);
  }
  return p;
}
static Mat softmax_bwd(const Mat& p, const Mat& dp, int rows, int C,
                       i64 scale) {
  Mat ds((size_t)rows * C);
  for (int t = 0; t < rows; t++) {
    i64 inner = 0;
    for (int c = 0; c < C; c++)
      inner += rdiv(p[(size_t)t * C + c] * dp[(size_t)t * C + c], scale);
    for (int c = 0; c < C; c++)
      ds[(size_t)t * C + c] = rdiv(
          p[(size_t)t * C + c] * (dp[(size_t)t * C + c] - inner), scale);
  }
  return ds;
}

// ---------- the block ----------
static const char* KEYS[11] = {"wq", "wk", "wv", "wo", "wg", "wu",
                               "wd", "wh", "g1", "g2", "g3"};
struct Shapes {
  int r, c;  // c == 0 -> vector
};
static const std::map<std::string, Shapes> SH = {
    {"wq", {DH, D}}, {"wk", {DH, D}}, {"wv", {DH, D}}, {"wo", {D, DH}},
    {"wg", {F, D}},  {"wu", {F, D}},  {"wd", {D, F}},  {"wh", {V, D}},
    {"g1", {D, 0}},  {"g2", {D, 0}},  {"g3", {D, 0}}};

struct Cache {
  Mat x, h1, i1, q0, k0, v, qr, kr, p, a, m1, x1, h2, i2, gp, u, sg, f,
      m2, x2, h3, i3;
};

static Mat block_fwd(const std::map<std::string, Mat>& w, const Mat& x,
                     const Tab& tb, Cache& c) {
  c.x = x;
  c.h1 = rms_fwd(x, w.at("g1"), c.i1);
  c.q0 = mmT(c.h1, T, D, w.at("wq"), DH);
  c.k0 = mmT(c.h1, T, D, w.at("wk"), DH);
  c.v = mmT(c.h1, T, D, w.at("wv"), DH);
  rdiv_all(c.q0, Q);
  rdiv_all(c.k0, Q);
  rdiv_all(c.v, Q);
  c.qr = rope_fwd(c.q0, tb);
  c.kr = rope_fwd(c.k0, tb);
  Mat s = mmT(c.qr, T, DH, c.kr, T);  // [T,T]
  rdiv_all(s, SCALE);
  for (int t = 0; t < T; t++)
    for (int u2 = t + 1; u2 < T; u2++)
      s[(size_t)t * T + u2] = -(i64(1) << 40);  // causal
  c.p = softmax_rows(s, T, T, tb, PQ);
  c.a = mm(c.p, T, T, c.v, DH);  // p @ v
  rdiv_all(c.a, PQ);
  Mat pre1 = mmT(c.a, T, DH, w.at("wo"), D);
  rdiv_all(pre1, Q);
  c.m1.assign((size_t)T * D, 0);
  c.x1.assign((size_t)T * D, 0);
  for (size_t i = 0; i < pre1.size(); i++) {
    pre1[i] += x[i];
    c.m1[i] = (pre1[i] <= ACT_CLAMP && pre1[i] >= -ACT_CLAMP);
    c.x1[i] = clampi(pre1[i], -ACT_CLAMP, ACT_CLAMP);
  }
  c.h2 = rms_fwd(c.x1, w.at("g2"), c.i2);
  c.gp = mmT(c.h2, T, D, w.at("wg"), F);
  c.u = mmT(c.h2, T, D, w.at("wu"), F);
  rdiv_all(c.gp, Q);
  rdiv_all(c.u, Q);
  c.sg.resize(c.gp.size());
  c.f.resize(c.gp.size());
  for (size_t i = 0; i < c.gp.size(); i++) {
    c.sg[i] = lut_silu(tb, c.gp[i]);
    c.f[i] = rdiv(c.sg[i] * c.u[i], Q);
  }
  Mat pre2 = mmT(c.f, T, F, w.at("wd"), D);
  rdiv_all(pre2, Q);
  c.m2.assign((size_t)T * D, 0);
  c.x2.assign((size_t)T * D, 0);
  for (size_t i = 0; i < pre2.size(); i++) {
    pre2[i] += c.x1[i];
    c.m2[i] = (pre2[i] <= ACT_CLAMP && pre2[i] >= -ACT_CLAMP);
    c.x2[i] = clampi(pre2[i], -ACT_CLAMP, ACT_CLAMP);
  }
  c.h3 = rms_fwd(c.x2, w.at("g3"), c.i3);
  Mat logits = mmT(c.h3, T, D, w.at("wh"), V);
  rdiv_all(logits, Q);
  return logits;
}

static std::map<std::string, Mat> block_bwd(
    const std::map<std::string, Mat>& w, const Mat& dlogits,
    const Cache& c, const Tab& tb) {
  std::map<std::string, Mat> G;
  G["wh"] = xty(dlogits, T, V, c.h3, D);
  rdiv_all(G["wh"], Q);
  Mat dh3 = mm(dlogits, T, V, w.at("wh"), D);
  rdiv_all(dh3, Q);
  Mat dx2 = rms_bwd(dh3, c.x2, w.at("g3"), c.i3, G["g3"]);
  for (size_t i = 0; i < dx2.size(); i++) dx2[i] *= c.m2[i];
  // FFN branch
  Mat df = mm(dx2, T, D, w.at("wd"), F);
  rdiv_all(df, Q);
  G["wd"] = xty(dx2, T, D, c.f, F);
  rdiv_all(G["wd"], Q);
  Mat du(df.size()), dgp(df.size());
  for (size_t i = 0; i < df.size(); i++) {
    du[i] = rdiv(c.sg[i] * df[i], Q);
    dgp[i] = rdiv(rdiv(c.u[i] * df[i], Q) * lut_dsilu(tb, c.gp[i]), Q);
  }
  Mat dh2 = mm(du, T, F, w.at("wu"), D);
  {
    Mat t2 = mm(dgp, T, F, w.at("wg"), D);
    for (size_t i = 0; i < dh2.size(); i++) dh2[i] += t2[i];
  }
  rdiv_all(dh2, Q);
  G["wu"] = xty(du, T, F, c.h2, D);
  rdiv_all(G["wu"], Q);
  G["wg"] = xty(dgp, T, F, c.h2, D);
  rdiv_all(G["wg"], Q);
  Mat dx1 = rms_bwd(dh2, c.x1, w.at("g2"), c.i2, G["g2"]);
  for (size_t i = 0; i < dx1.size(); i++)
    dx1[i] = (dx1[i] + dx2[i]) * c.m1[i];  // residual + clamp bwd
  // attention branch
  Mat da = mm(dx1, T, D, w.at("wo"), DH);
  rdiv_all(da, Q);
  G["wo"] = xty(dx1, T, D, c.a, DH);
  rdiv_all(G["wo"], Q);
  Mat dp = mmT(da, T, DH, c.v, T);
  rdiv_all(dp, Q);
  Mat dv = xty(c.p, T, T, da, DH);
  rdiv_all(dv, PQ);
  Mat ds = softmax_bwd(c.p, dp, T, T, PQ);
  Mat dqr = mm(ds, T, T, c.kr, DH);
  rdiv_all(dqr, SCALE);
  Mat dkr = xty(ds, T, T, c.qr, DH);
  rdiv_all(dkr, SCALE);
  Mat dq = rope_bwd(dqr, tb);
  Mat dk = rope_bwd(dkr, tb);
  G["wq"] = xty(dq, T, DH, c.h1, D);
  rdiv_all(G["wq"], Q);
  G["wk"] = xty(dk, T, DH, c.h1, D);
  rdiv_all(G["wk"], Q);
  G["wv"] = xty(dv, T, DH, c.h1, D);
  rdiv_all(G["wv"], Q);
  Mat dh1 = mm(dq, T, DH, w.at("wq"), D);
  {
    Mat t2 = mm(dk, T, DH, w.at("wk"), D);
    Mat t3 = mm(dv, T, DH, w.at("wv"), D);
    for (size_t i = 0; i < dh1.size(); i++) dh1[i] += t2[i] + t3[i];
  }
  rdiv_all(dh1, Q);
  Mat g1g;
  Mat dx0 = rms_bwd(dh1, c.x, w.at("g1"), c.i1, g1g);
  G["g1"] = g1g;
  return G;  // dx0 unused by training
}

// ---------- IntAdamWQw ----------
struct IntAdamW {
  std::vector<Mat*> p;
  std::vector<Mat> m, v;
  int t = 0;
  double nz_last = 0;
  Big p10, p9, p1000, p999;
  explicit IntAdamW(std::vector<Mat*> params) : p(std::move(params)) {
    for (auto* w : p) {
      m.emplace_back(w->size(), 0);
      v.emplace_back(w->size(), 0);
    }
  }
  void step(const std::vector<Mat>& grads) {
    t += 1;
    p10.mul_small(10); p9.mul_small(9);
    p1000.mul_small(1000); p999.mul_small(999);
    Big n1 = p10, d1 = Big::sub(p10, p9);
    Big n2 = p1000, d2 = Big::sub(p1000, p999);
    while (n1.gt_pow30()) { n1.shr1(); d1.shr1(); }
    while (n2.gt_pow30()) { n2.shr1(); d2.shr1(); }
    i64 bc1n = n1.to_i64(), bc1d = std::max<i64>(d1.to_i64(), 1);
    i64 bc2n = n2.to_i64(), bc2d = std::max<i64>(d2.to_i64(), 1);
    i64 nz = 0, tot = 0;
    for (size_t j = 0; j < p.size(); j++) {
      Mat& w = *p[j];
      for (size_t i = 0; i < w.size(); i++) {
        const i64 g = grads[j][i];
        m[j][i] = rdiv(B1N * m[j][i] + (B1D - B1N) * g, B1D);
        v[j][i] = rdiv(B2N * v[j][i] + (B2D - B2N) * rdiv(g * g, Q),
                       B2D);
        const i64 mh = rdiv(m[j][i] * bc1n, bc1d);
        const i64 vh = rdiv(v[j][i] * bc2n, bc2d);
        const i64 den = isqrt_newton(vh * Q) + AEPS;
        const i64 upd = rdiv(LRN * mh * (Q << SHIFT), LRD * den);
        nz += upd != 0;
        tot += 1;
        w[i] -= upd;
        w[i] -= rdiv(w[i] * WDN, WDD);
      }
    }
    nz_last = double(nz) / double(tot);
  }
};

// ---------- loaders ----------
static std::map<std::string, Mat> load_axp3(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  char magic[4];
  f.read(magic, 4);
  if (memcmp(magic, "AXP3", 4)) { fprintf(stderr, "bad magic\n"); exit(1); }
  u32 n;
  f.read((char*)&n, 4);
  std::map<std::string, Mat> t;
  for (u32 i = 0; i < n; i++) {
    uint16_t nl; f.read((char*)&nl, 2);
    std::string name(nl, 0); f.read(&name[0], nl);
    u8 nd; f.read((char*)&nd, 1);
    u64 numel = 1;
    for (int k = 0; k < nd; k++) {
      u64 dd; f.read((char*)&dd, 8);
      numel *= dd;
    }
    Mat m(numel);
    f.read((char*)m.data(), 8 * numel);
    t[name] = std::move(m);
  }
  return t;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s r2b_init.bin r2b_tables.bin\n", argv[0]);
    return 1;
  }
  // init: 11 weight tensors in KEYS order, then x [T,D], tgt [T]
  std::ifstream f(argv[1], std::ios::binary);
  if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
  std::map<std::string, Mat> w;
  for (const char* k : KEYS) {
    const auto sh = SH.at(k);
    Mat m((size_t)sh.r * (sh.c ? sh.c : 1));
    f.read((char*)m.data(), 8 * m.size());
    w[k] = std::move(m);
  }
  Mat x((size_t)T * D), tgt(T);
  f.read((char*)x.data(), 8 * x.size());
  f.read((char*)tgt.data(), 8 * tgt.size());
  if (!f) { fprintf(stderr, "truncated init\n"); return 1; }

  auto tt = load_axp3(argv[2]);
  Tab tb{tt.at("silu.tab"), tt.at("dsilu.tab"), tt.at("exp.tab"),
         tt.at("rope.cos"), tt.at("rope.sin")};

  // lift to Q_w; optimizer holds the wide weights
  std::map<std::string, Mat> wide = w;
  for (const char* k : KEYS)
    for (auto& v : wide[k]) v <<= SHIFT;
  std::vector<Mat*> params;
  for (const char* k : KEYS) params.push_back(&wide[k]);
  IntAdamW opt(params);

  Sha256 th;
  i64 loss0 = 0, loss_mid = 0, loss_last = 0;
  for (int step = 1; step <= STEPS; step++) {
    for (const char* k : KEYS) {
      w[k] = wide[k];
      for (auto& v : w[k]) v = rdiv(v, i64(1) << SHIFT);
    }
    Cache c;
    Mat logits = block_fwd(w, x, tb, c);
    Mat pp = softmax_rows(logits, T, V, tb, Q);
    i64 loss = 0;
    for (int t = 0; t < T; t++)
      loss += Q - pp[(size_t)t * V + tgt[t]];
    if (step == 1) loss0 = loss;
    if (step == STEPS / 2 + 1) loss_mid = loss;
    loss_last = loss;
    Mat dlogits((size_t)T * V);
    for (int t = 0; t < T; t++)
      for (int vv = 0; vv < V; vv++)
        dlogits[(size_t)t * V + vv] =
            (pp[(size_t)t * V + vv] - Q * (tgt[t] == vv)) * GBOOST;
    auto G = block_bwd(w, dlogits, c, tb);
    std::vector<Mat> grads;
    for (const char* k : KEYS) {
      Mat g = G.at(k);
      for (auto& v : g) v = rdiv(v, Q * GBOOST);  // unboost
      grads.push_back(std::move(g));
    }
    opt.step(grads);
    if (step % (STEPS / 8) == 0) {
      for (const char* k : KEYS)
        th.update(wide[k].data(), wide[k].size() * 8);
      Sha256 peek = th;
      printf("[r2b-cpp] step %d loss %lld nz %.3f traj-sha %s\n", step,
             (long long)loss, opt.nz_last, peek.hex().c_str());
    }
  }
  printf("[r2b-cpp] loss %lld -> %lld -> %lld  falling: %s\n",
         (long long)loss0, (long long)loss_mid, (long long)loss_last,
         loss_last < loss_mid && loss_mid < loss0 ? "true" : "false");
  printf("[r2b-cpp] FINAL trajectory sha %s\n", th.hex().c_str());
  return 0;
}
