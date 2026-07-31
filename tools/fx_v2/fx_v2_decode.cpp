// FX-V2: C++ implementation of the P3 deterministic fixed-point twin
// (reference: llmopt/decoding/deterministic.py, promoted P3 verdict
// 2026-07-30). Pure int64 arithmetic end to end — no fp carrier, no
// libm, no torch. Consumes the AXP3 flat export of the sha-pinned
// p3_tables.pt and the frozen battery prompt ids; emits the same two
// digests as the Python driver:
//   streams  = sha256(Python repr of the list of 40-token greedy streams)
//   trace    = sha256(concatenated int64-LE logits of every prompt step)
//
// Build: c++ -O2 -std=c++17 fx_v2_decode.cpp -o fx_v2_decode
// Run:   ./fx_v2_decode p3_tables.bin battery_ids.json
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

static const i64 A = 1024, ACT_CLAMP = 32 * 1024;
static const i64 ROPE_S = 1 << 14, EXP_K = 1024, SQ = 46341;
static const int D = 64, LAYERS = 8, HEADS = 8, HD = 8, HALF = 4;

// ---------- sha256 (FIPS 180-4, minimal) ----------
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

// ---------- AXP3 loader ----------
struct Tensor {
  std::vector<u64> dims;
  std::vector<i64> data;
};
static std::map<std::string, Tensor> load_axp3(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  char magic[4];
  f.read(magic, 4);
  if (memcmp(magic, "AXP3", 4)) { fprintf(stderr, "bad magic\n"); exit(1); }
  u32 n;
  f.read((char*)&n, 4);
  std::map<std::string, Tensor> t;
  for (u32 i = 0; i < n; i++) {
    uint16_t nl; f.read((char*)&nl, 2);
    std::string name(nl, 0); f.read(&name[0], nl);
    u8 nd; f.read((char*)&nd, 1);
    Tensor tt; u64 numel = 1;
    for (int k = 0; k < nd; k++) {
      u64 d; f.read((char*)&d, 8);
      tt.dims.push_back(d); numel *= d;
    }
    tt.data.resize(numel);
    f.read((char*)tt.data.data(), 8 * numel);
    t[name] = std::move(tt);
  }
  return t;
}

// ---------- exact integer ops (twins of the Python reference) ----------
// round-half-away division, d > 0; numerators stay nonnegative so
// C++ truncation equals Python floor here.
static inline i64 rdiv(i64 x, i64 d) {
  return x >= 0 ? (2 * x + d) / (2 * d) : -((-2 * x + d) / (2 * d));
}
static inline i64 clampi(i64 x, i64 lo, i64 hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}
static i64 isqrt_newton(i64 n) {
  i64 x = i64(1) << 32;
  for (int i = 0; i < 30; i++) {
    i64 xa = x < 1 ? 1 : x;
    x = (x + rdiv(n, xa)) >> 1;
  }
  return x < 1 ? 1 : x;
}

struct DetLM {
  std::map<std::string, Tensor> t;
  i64 max_partial = 0;
  int vocab;
  // KV cache: [layer][head] -> vector of HD-wide rows
  std::vector<std::vector<std::vector<std::vector<i64>>>> kc, vc;

  explicit DetLM(std::map<std::string, Tensor> tables) : t(std::move(tables)) {
    vocab = (int)t.at("head.weight.codes").dims[0];
    reset();
  }
  void reset() {
    kc.assign(LAYERS, {}); vc.assign(LAYERS, {});
    for (int l = 0; l < LAYERS; l++) { kc[l].resize(HEADS); vc[l].resize(HEADS); }
  }
  // y[out] = rdiv(sum_i a[i]*W[o][i], q) — pure int64, no partial bound
  // needed, but track the reference's diagnostic bound anyway.
  std::vector<i64> gemm(const std::vector<i64>& a, const std::string& key) {
    const Tensor& W = t.at(key + ".weight.codes");
    i64 q = t.at(key + ".weight.q").data[0];
    int out = (int)W.dims[0], in = (int)W.dims[1];
    i64 wmax = 0, himax = 63;
    for (i64 w : W.data) { i64 aw = w < 0 ? -w : w; if (aw > wmax) wmax = aw; }
    for (i64 v : a) { i64 hi = (v >> 6); if (hi < 0) hi = -hi; if (hi > himax) himax = hi; }
    i64 bound = himax * wmax * in;
    if (bound > max_partial) max_partial = bound;
    std::vector<i64> y(out);
    for (int o = 0; o < out; o++) {
      const i64* w = &W.data[(size_t)o * in];
      i64 acc = 0;
      for (int i = 0; i < in; i++) acc += a[i] * w[i];
      y[o] = rdiv(acc, q);
    }
    return y;
  }
  std::vector<i64> rmsnorm(const std::vector<i64>& a, const std::string& gkey) {
    const Tensor& g = t.at(gkey + ".g.int");
    i64 s2 = 0;
    for (i64 v : a) s2 += v * v;
    i64 m = (s2 / D) * (i64(1) << 32) / (A * A) + 4295;
    i64 r = isqrt_newton(m);
    std::vector<i64> y(a.size());
    for (size_t i = 0; i < a.size(); i++) {
      i64 v = rdiv(a[i] * g.data[i], A);
      v = rdiv(v * (i64(1) << 16), r < 1 ? 1 : r);
      y[i] = clampi(v, -ACT_CLAMP, ACT_CLAMP);
    }
    return y;
  }
  void rope(i64* v, int pos) {  // one head vector, in place
    const Tensor& C = t.at("rope.cos");
    const Tensor& S = t.at("rope.sin");
    const i64* c = &C.data[(size_t)pos * HALF];
    const i64* s = &S.data[(size_t)pos * HALF];
    i64 r[HD];
    for (int i = 0; i < HALF; i++) {
      i64 v1 = v[i], v2 = v[i + HALF];
      r[i] = rdiv(v1 * c[i] - v2 * s[i], ROPE_S);
      r[i + HALF] = rdiv(v1 * s[i] + v2 * c[i], ROPE_S);
    }
    memcpy(v, r, sizeof r);
  }
  // logits for one token; updates KV cache
  std::vector<i64> step(int tok, int pos) {
    const Tensor& E = t.at("emb.weight.codes");
    i64 qe = t.at("emb.weight.q").data[0];
    std::vector<i64> x(D);
    for (int i = 0; i < D; i++) x[i] = rdiv(E.data[(size_t)tok * D + i] * A, qe);
    const Tensor& ET = t.at("exp.tab");
    const Tensor& SI = t.at("silu.tab");
    for (int l = 0; l < LAYERS; l++) {
      std::string p = "blocks." + std::to_string(l);
      std::vector<i64> h = rmsnorm(x, p + ".n1");
      std::vector<i64> qkv = gemm(h, p + ".qkv");
      std::vector<i64> attn_out(D);
      for (int hd = 0; hd < HEADS; hd++) {
        i64 qv[HD], kv[HD];
        std::vector<i64> vv(HD);
        for (int i = 0; i < HD; i++) {
          qv[i] = qkv[hd * HD + i];
          kv[i] = qkv[D + hd * HD + i];
          vv[i] = qkv[2 * D + hd * HD + i];
        }
        rope(qv, pos);
        rope(kv, pos);
        kc[l][hd].emplace_back(std::vector<i64>(kv, kv + HD));
        vc[l][hd].push_back(vv);
        int T = (int)kc[l][hd].size();
        std::vector<i64> idx(T);
        i64 mx = INT64_MIN;
        for (int tt = 0; tt < T; tt++) {
          i64 s = 0;
          for (int i = 0; i < HD; i++) s += qv[i] * kc[l][hd][tt][i];
          idx[tt] = rdiv(s * EXP_K * (i64(1) << 14), A * A * SQ);
          if (idx[tt] > mx) mx = idx[tt];
        }
        i64 num[HD] = {0}, den = 0;
        for (int tt = 0; tt < T; tt++) {
          i64 ix = clampi(idx[tt] - mx, -(i64(1) << 15), 0);
          i64 w = ET.data[ix + (i64(1) << 15)];
          den += w;
          for (int i = 0; i < HD; i++) num[i] += w * vc[l][hd][tt][i];
        }
        i64 dn = den < 1 ? 1 : den;
        for (int i = 0; i < HD; i++) attn_out[hd * HD + i] = rdiv(num[i], dn);
      }
      std::vector<i64> o = gemm(attn_out, p + ".o");
      for (int i = 0; i < D; i++) x[i] = clampi(x[i] + o[i], -ACT_CLAMP, ACT_CLAMP);
      h = rmsnorm(x, p + ".n2");
      std::vector<i64> g = gemm(h, p + ".gate");
      std::vector<i64> u = gemm(h, p + ".up");
      std::vector<i64> ff(g.size());
      for (size_t i = 0; i < g.size(); i++) {
        i64 gi = clampi(g[i], -(i64(1) << 15), i64(1) << 15);
        i64 sg = SI.data[gi + (i64(1) << 15)];
        i64 v = rdiv(sg * u[i], A);
        ff[i] = clampi(v, -(i64(1) << 15), i64(1) << 15);
      }
      std::vector<i64> dwn = gemm(ff, p + ".down");
      for (int i = 0; i < D; i++) x[i] = clampi(x[i] + dwn[i], -ACT_CLAMP, ACT_CLAMP);
    }
    x = rmsnorm(x, "norm");
    return gemm(x, "head");
  }
  std::vector<int> greedy(const std::vector<int>& ids, int n_new) {
    reset();
    std::vector<i64> logits;
    int pos = 0;
    for (int tk : ids) logits = step(tk, pos++);
    std::vector<int> out;
    for (int j = 0; j < n_new; j++) {
      int best = 0;
      for (int i = 1; i < vocab; i++) if (logits[i] > logits[best]) best = i;
      out.push_back(best);
      logits = step(best, pos++);
    }
    return out;
  }
};

// minimal JSON reader for {"battery": [[int,...],...], "trace": [...], "n_new": N}
struct Js {
  const char* p;
  void ws() { while (*p == ' ' || *p == '\n' || *p == '\t' || *p == ',' || *p == ':') p++; }
  bool seek(const char* key) {
    const char* q = strstr(p, key);
    if (!q) return false;
    p = q + strlen(key);
    return true;
  }
  std::vector<int> arr() {  // parse [int, int, ...] at p
    std::vector<int> v;
    ws(); p++;  // '['
    while (true) {
      ws();
      if (*p == ']') { p++; return v; }
      v.push_back((int)strtol(p, (char**)&p, 10));
    }
  }
};

int main(int argc, char** argv) {
  if (argc != 3) { fprintf(stderr, "usage: %s tables.bin battery.json\n", argv[0]); return 1; }
  DetLM m(load_axp3(argv[1]));
  std::ifstream jf(argv[2]);
  std::string js((std::istreambuf_iterator<char>(jf)), std::istreambuf_iterator<char>());
  Js j{js.c_str()};
  if (!j.seek("\"battery\"")) return 1;
  j.ws(); j.p++;  // outer '['
  std::vector<std::vector<int>> battery;
  while (true) {
    j.ws();
    if (*j.p == ']') { j.p++; break; }
    battery.push_back(j.arr());
  }
  if (!j.seek("\"trace\"")) return 1;
  std::vector<int> trace = j.arr();
  if (!j.seek("\"n_new\"")) return 1;
  j.ws();
  int n_new = (int)strtol(j.p, nullptr, 10);

  // streams hash: sha256 of the Python repr of the list of streams
  std::string rep = "[";
  for (size_t s = 0; s < battery.size(); s++) {
    std::vector<int> toks = m.greedy(battery[s], n_new);
    rep += "[";
    for (size_t i = 0; i < toks.size(); i++) {
      rep += std::to_string(toks[i]);
      if (i + 1 < toks.size()) rep += ", ";
    }
    rep += "]";
    if (s + 1 < battery.size()) rep += ", ";
  }
  rep += "]";
  Sha256 hs;
  hs.update(rep.data(), rep.size());
  printf("FX-V2 streams (c++): %s\n", hs.hex().c_str());

  // logit-trace hash: int64-LE logits of every prompt step, first prompt
  m.reset();
  Sha256 ht;
  for (size_t pos = 0; pos < trace.size(); pos++) {
    std::vector<i64> lg = m.step(trace[pos], (int)pos);
    ht.update(lg.data(), lg.size() * 8);
  }
  double lb = 0;
  for (i64 b = m.max_partial; b > 1; b >>= 1) lb += 1;
  printf("FX-V2 logit-trace (c++): %s | max GEMM partial ~2^%.0f\n",
         ht.hex().c_str(), lb);
  return 0;
}
