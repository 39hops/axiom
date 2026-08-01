/** @file intbirth.cpp Integer-birth engine (see intbirth.hpp).
    Primitive layer (int_gemm / block / adamw) + the composed
    full_birth; certified by the r2b_ref.json milestone digests. */
#include <ax/nn/intbirth.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace ax::nn::ib {

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using u8 = std::uint8_t;

constexpr i64 Q = 512;
constexpr i64 RS = 1 << 14, R16 = 1 << 16;
constexpr i64 B1N = 9, B1D = 10, B2N = 999, B2D = 1000;
constexpr i64 AEPS = 4, WDN = 1, WDD = 100000;

inline i64 rdiv(i64 x, i64 d) {
  const i64 ax = x < 0 ? -x : x;
  const i64 r = (ax + d / 2) / d;
  return x < 0 ? -r : r;
}
i64 isqrt_newton(i64 x) {
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
inline i64 clampi(i64 x, i64 lo, i64 hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}
/** round(sqrt(n)) exactly (ties impossible for integer n). */
i64 isqrt_round(i64 n) {
  const i64 r = isqrt_newton(n);
  return (n - r * r > (r + 1) * (r + 1) - n) ? r + 1 : r;
}

// ---- big-uint limbs (little-endian u32) for the bias correction
using BigV = std::vector<u32>;
void big_trim(BigV& d) {
  while (d.size() > 1 && d.back() == 0) d.pop_back();
}
void big_mul(BigV& d, u64 k) {
  u64 carry = 0;
  for (auto& x : d) {
    const u64 p = u64(x) * k + carry;
    x = u32(p);
    carry = p >> 32;
  }
  while (carry) { d.push_back(u32(carry)); carry >>= 32; }
}
int big_bits(const BigV& d) {
  u32 top = d.back();
  int b = 0;
  while (top) { b++; top >>= 1; }
  return int(d.size() - 1) * 32 + b;
}
void big_shr1(BigV& d) {
  for (std::size_t i = 0; i < d.size(); i++) {
    const u32 lo = (i + 1 < d.size()) ? (d[i + 1] & 1) : 0;
    d[i] = (d[i] >> 1) | (lo << 31);
  }
  big_trim(d);
}
bool big_gt_pow30(const BigV& d) {  // strictly greater than 2^30
  const int b = big_bits(d);
  if (b != 31) return b > 31;
  return !(d.size() == 1 && d[0] == 0x40000000u);
}
i64 big_i64(const BigV& d) {
  u64 v = 0;
  for (std::size_t i = d.size(); i-- > 0;) v = (v << 32) | d[i];
  return i64(v);
}
BigV big_sub(const BigV& a, const BigV& b) {  // a >= b
  BigV r(a.size(), 0);
  i64 borrow = 0;
  for (std::size_t i = 0; i < a.size(); i++) {
    const i64 x = i64(a[i]) - (i < b.size() ? i64(b[i]) : 0) - borrow;
    borrow = x < 0;
    r[i] = u32(x + (borrow << 32));
  }
  big_trim(r);
  return r;
}

std::map<std::string, Mat> parse_axp3(const std::string& b) {
  const auto need = [&](std::size_t off, std::size_t n) {
    if (off + n > b.size())
      throw std::runtime_error("intbirth: truncated AXP3");
  };
  need(0, 8);
  if (std::memcmp(b.data(), "AXP3", 4) != 0)
    throw std::runtime_error("intbirth: bad AXP3 magic");
  u32 count;
  std::memcpy(&count, b.data() + 4, 4);
  std::size_t off = 8;
  std::map<std::string, Mat> t;
  for (u32 i = 0; i < count; i++) {
    need(off, 2);
    std::uint16_t nl;
    std::memcpy(&nl, b.data() + off, 2);
    off += 2;
    need(off, nl + std::size_t(1));
    std::string name(b.data() + off, nl);
    off += nl;
    const u8 nd = u8(b[off++]);
    u64 numel = 1;
    need(off, std::size_t(nd) * 8);
    for (int k = 0; k < nd; k++) {
      u64 dd;
      std::memcpy(&dd, b.data() + off, 8);
      off += 8;
      numel *= dd;
    }
    need(off, numel * 8);
    Mat m(numel);
    std::memcpy(m.data(), b.data() + off, numel * 8);
    off += numel * 8;
    t[name] = std::move(m);
  }
  return t;
}

struct Shape {
  int r, c;
};
std::map<std::string, Shape> shapes(const contract& c) {
  return {{"wq", {c.DH, c.D}}, {"wk", {c.DH, c.D}},
          {"wv", {c.DH, c.D}}, {"wo", {c.D, c.DH}},
          {"wg", {c.F, c.D}},  {"wu", {c.F, c.D}},
          {"wd", {c.D, c.F}},  {"wh", {c.V, c.D}},
          {"g1", {c.D, 1}},    {"g2", {c.D, 1}},
          {"g3", {c.D, 1}}};
}

}  // namespace

// ---------------------------------------------------------- sha256
namespace detail {

void sha256::block(const u8* p) {
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
  const auto rot = [](u32 x, int n) {
    return (x >> n) | (x << (32 - n));
  };
  u32 w[64];
  for (int i = 0; i < 16; i++)
    w[i] = (u32(p[4 * i]) << 24) | (u32(p[4 * i + 1]) << 16) |
           (u32(p[4 * i + 2]) << 8) | u32(p[4 * i + 3]);
  for (int i = 16; i < 64; i++) {
    const u32 s0 =
        rot(w[i - 15], 7) ^ rot(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const u32 s1 =
        rot(w[i - 2], 17) ^ rot(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  u32 a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5],
      g = h[6], hh = h[7];
  for (int i = 0; i < 64; i++) {
    const u32 S1 = rot(e, 6) ^ rot(e, 11) ^ rot(e, 25);
    const u32 ch = (e & f) ^ (~e & g);
    const u32 t1 = hh + S1 + ch + K[i] + w[i];
    const u32 S0 = rot(a, 2) ^ rot(a, 13) ^ rot(a, 22);
    const u32 mj = (a & b) ^ (a & c) ^ (b & c);
    const u32 t2 = S0 + mj;
    hh = g; g = f; f = e; e = d + t1;
    d = c; c = b; b = a; a = t1 + t2;
  }
  h[0] += a; h[1] += b; h[2] += c; h[3] += d;
  h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

void sha256::update(const void* data, std::size_t n) {
  const u8* p = static_cast<const u8*>(data);
  len += n;
  while (n) {
    const std::size_t take = 64 - fill < n ? 64 - fill : n;
    std::memcpy(buf + fill, p, take);
    fill += take;
    p += take;
    n -= take;
    if (fill == 64) {
      block(buf);
      fill = 0;
    }
  }
}

std::string sha256::hex() {
  const u64 bits = len * 8;
  const u8 pad = 0x80;
  update(&pad, 1);
  const u8 z = 0;
  while (fill != 56) update(&z, 1);
  u8 lb[8];
  for (int i = 0; i < 8; i++) lb[i] = u8(bits >> (56 - 8 * i));
  update(lb, 8);
  char out[65];
  for (int i = 0; i < 8; i++)
    std::snprintf(out + 8 * i, 9, "%08x", h[i]);
  return std::string(out, 64);
}

}  // namespace detail

// ---------------------------------------------------- int_gemm forms

Mat int_gemm(const Mat& a, int rows, int K, const Mat& w, int N) {
  Mat y(std::size_t(rows) * N);
  for (int t = 0; t < rows; t++)
    for (int n = 0; n < N; n++) {
      i64 acc = 0;
      const i64* ar = &a[std::size_t(t) * K];
      const i64* wr = &w[std::size_t(n) * K];
      for (int k = 0; k < K; k++) acc += ar[k] * wr[k];
      y[std::size_t(t) * N + n] = acc;
    }
  return y;
}
Mat int_gemm_nt(const Mat& a, int rows, int K, const Mat& w, int N) {
  Mat y(std::size_t(rows) * N);
  for (int t = 0; t < rows; t++)
    for (int n = 0; n < N; n++) {
      i64 acc = 0;
      for (int k = 0; k < K; k++)
        acc += a[std::size_t(t) * K + k] * w[std::size_t(k) * N + n];
      y[std::size_t(t) * N + n] = acc;
    }
  return y;
}
Mat int_gemm_xty(const Mat& x, int rows, int K, const Mat& y, int N) {
  Mat o(std::size_t(K) * N, 0);
  for (int t = 0; t < rows; t++)
    for (int k = 0; k < K; k++) {
      const i64 xv = x[std::size_t(t) * K + k];
      if (!xv) continue;
      for (int n = 0; n < N; n++)
        o[std::size_t(k) * N + n] += xv * y[std::size_t(t) * N + n];
    }
  return o;
}
void rdiv_inplace(Mat& m, i64 d) {
  for (auto& v : m) v = rdiv(v, d);
}

// ------------------------------------------------------------ block

const char* const block::KEYS[11] = {"wq", "wk", "wv", "wo", "wg",
                                     "wu", "wd", "wh", "g1", "g2",
                                     "g3"};
const char* const block::BODY_KEYS[9] = {"wq", "wk", "wv", "wo",
                                         "wg", "wu", "wd", "g1",
                                         "g2"};

block::block(const std::string& tables_bytes, const contract& c)
    : c_(c) {
  if (c_.T <= 0 || c_.D <= 0 || c_.DH <= 0 || c_.F <= 0 ||
      c_.V <= 0 || c_.DH % 2 || c_.shift < 0 || c_.gboost < 1 ||
      c_.pq < 1 || c_.act_clamp < 1 || c_.lrd < 1)
    throw std::runtime_error("intbirth: bad contract");
  scale_ = isqrt_round(Q * Q * i64(c_.DH));
  tab_ = parse_axp3(tables_bytes);
  for (const char* k : {"silu.tab", "dsilu.tab", "exp.tab",
                        "rope.cos", "rope.sin"})
    if (!tab_.count(k))
      throw std::runtime_error(std::string("intbirth: missing ") + k);
  ts_ = (i64(tab_.at("silu.tab").size()) - 1) / 2;
  tse_ = i64(tab_.at("exp.tab").size()) - 1;
  if (i64(tab_.at("dsilu.tab").size()) != 2 * ts_ + 1 ||
      i64(tab_.at("rope.cos").size()) != i64(c_.T) * (c_.DH / 2) ||
      i64(tab_.at("rope.sin").size()) != i64(c_.T) * (c_.DH / 2))
    throw std::runtime_error("intbirth: table size mismatch");
}

namespace {
void check_keys(const contract& c, const std::map<std::string, Mat>& w,
                const char* const* keys, int n) {
  const auto sh = shapes(c);
  for (int i = 0; i < n; i++) {
    const auto it = w.find(keys[i]);
    if (it == w.end())
      throw std::runtime_error(std::string("intbirth: missing ") +
                               keys[i]);
    const auto s = sh.at(keys[i]);
    if (i64(it->second.size()) != i64(s.r) * s.c)
      throw std::runtime_error(std::string("intbirth: bad shape ") +
                               keys[i]);
  }
}
}  // namespace

void block::check_weights(const std::map<std::string, Mat>& w) const {
  check_keys(c_, w, KEYS, 11);
}

Mat block::softmax_rows(const Mat& s, int rows, int C,
                        i64 scale) const {
  const Mat& ex = tab_.at("exp.tab");
  Mat p(std::size_t(rows) * C), e(C);
  for (int t = 0; t < rows; t++) {
    i64 m = s[std::size_t(t) * C];
    for (int cc = 1; cc < C; cc++)
      m = std::max(m, s[std::size_t(t) * C + cc]);
    i64 z = 0;
    for (int cc = 0; cc < C; cc++) {
      i64 d = s[std::size_t(t) * C + cc] - m;
      if (d < -tse_ - 1) d = -tse_ - 1;
      e[cc] = d < -tse_ ? 0 : ex[d + tse_];
      z += e[cc];
    }
    for (int cc = 0; cc < C; cc++)
      p[std::size_t(t) * C + cc] = rdiv(e[cc] * scale, z);
  }
  return p;
}

Mat block::rms_fwd(const Mat& xx, const Mat& g, Mat& isq) const {
  const int T = c_.T, D = c_.D;
  if (i64(xx.size()) != i64(T) * D || i64(g.size()) != D)
    throw std::runtime_error("intbirth: rms_fwd shape");
  Mat y(std::size_t(T) * D);
  isq.assign(T, 0);
  for (int t = 0; t < T; t++) {
    i64 s2 = 0;
    for (int d = 0; d < D; d++) {
      const i64 v = xx[std::size_t(t) * D + d];
      s2 += v * v;
    }
    const i64 m40 = (s2 / D) * (i64(1) << 32) / (Q * Q) + c_.eps32;
    isq[t] = isqrt_newton(m40);
    for (int d = 0; d < D; d++)
      y[std::size_t(t) * D + d] =
          rdiv(rdiv(xx[std::size_t(t) * D + d] * g[d], Q) * R16,
               isq[t]);
  }
  return y;
}

Mat block::rms_bwd(const Mat& dy, const Mat& xx, const Mat& g,
                   const Mat& isq, Mat& dg) const {
  const int T = c_.T, D = c_.D;
  Mat dx(std::size_t(T) * D);
  dg.assign(D, 0);
  Mat tv(D);
  for (int t = 0; t < T; t++) {
    i64 inner = 0;
    for (int d = 0; d < D; d++) {
      tv[d] = rdiv(g[d] * dy[std::size_t(t) * D + d], Q);
      inner += rdiv(tv[d] * xx[std::size_t(t) * D + d], Q);
    }
    for (int d = 0; d < D; d++) {
      const i64 xv = xx[std::size_t(t) * D + d];
      const i64 term1 = rdiv(tv[d] * R16, isq[t]);
      i64 cc = rdiv(xv * inner, i64(D) * Q);
      for (int r = 0; r < 3; r++) cc = rdiv(cc * R16, isq[t]);
      dx[std::size_t(t) * D + d] = term1 - cc;
      dg[d] += rdiv(rdiv(dy[std::size_t(t) * D + d] * xv, Q) * R16,
                    isq[t]);
    }
  }
  return dx;
}

Mat block::attn_fwd(const std::map<std::string, Mat>& w, const Mat& x,
                    block_cache& c) const {
  const int T = c_.T, D = c_.D, DH = c_.DH;
  const i64 PQ = c_.pq, CL = c_.act_clamp;
  if (i64(x.size()) != i64(T) * D)
    throw std::runtime_error("intbirth: bad x shape");
  const Mat& tcos = tab_.at("rope.cos");
  const Mat& tsin = tab_.at("rope.sin");
  const int half = DH / 2;
  const auto rope_f = [&](const Mat& v) {
    Mat y(std::size_t(T) * DH);
    for (int t = 0; t < T; t++)
      for (int i = 0; i < half; i++) {
        const i64 co = tcos[std::size_t(t) * half + i];
        const i64 si = tsin[std::size_t(t) * half + i];
        const i64 a = v[std::size_t(t) * DH + i];
        const i64 b = v[std::size_t(t) * DH + half + i];
        y[std::size_t(t) * DH + i] = rdiv(a * co - b * si, RS);
        y[std::size_t(t) * DH + half + i] = rdiv(a * si + b * co, RS);
      }
    return y;
  };

  c.x = x;
  c.h1 = rms_fwd(x, w.at("g1"), c.i1);
  c.q0 = int_gemm(c.h1, T, D, w.at("wq"), DH);
  c.k0 = int_gemm(c.h1, T, D, w.at("wk"), DH);
  c.v0 = int_gemm(c.h1, T, D, w.at("wv"), DH);
  rdiv_inplace(c.q0, Q);
  rdiv_inplace(c.k0, Q);
  rdiv_inplace(c.v0, Q);
  c.qr = rope_f(c.q0);
  c.kr = rope_f(c.k0);
  Mat s = int_gemm(c.qr, T, DH, c.kr, T);
  rdiv_inplace(s, scale_);
  for (int t = 0; t < T; t++)
    for (int u = t + 1; u < T; u++)
      s[std::size_t(t) * T + u] = -(i64(1) << 40);  // causal
  c.p = softmax_rows(s, T, T, PQ);
  c.a = int_gemm_nt(c.p, T, T, c.v0, DH);
  rdiv_inplace(c.a, PQ);
  Mat pre1 = int_gemm(c.a, T, DH, w.at("wo"), D);
  rdiv_inplace(pre1, Q);
  c.m1.assign(pre1.size(), 0);
  c.x1.assign(pre1.size(), 0);
  for (std::size_t i = 0; i < pre1.size(); i++) {
    pre1[i] += x[i];
    c.m1[i] = (pre1[i] <= CL && pre1[i] >= -CL);
    c.x1[i] = clampi(pre1[i], -CL, CL);
  }
  return c.x1;
}

Mat block::ffn_fwd(const std::map<std::string, Mat>& w, const Mat& x1,
                   block_cache& c) const {
  const int T = c_.T, D = c_.D, F = c_.F;
  const i64 CL = c_.act_clamp;
  const Mat& sil = tab_.at("silu.tab");
  (void)x1;  // residual base read from c.x1 (== x1)
  c.h2 = rms_fwd(c.x1, w.at("g2"), c.i2);
  c.gp = int_gemm(c.h2, T, D, w.at("wg"), F);
  c.u = int_gemm(c.h2, T, D, w.at("wu"), F);
  rdiv_inplace(c.gp, Q);
  rdiv_inplace(c.u, Q);
  c.sg.resize(c.gp.size());
  c.f.resize(c.gp.size());
  for (std::size_t i = 0; i < c.gp.size(); i++) {
    const i64 z = c.gp[i];
    c.sg[i] = z > ts_ ? z : z < -ts_ ? 0 : sil[z + ts_];
    c.f[i] = rdiv(c.sg[i] * c.u[i], Q);
  }
  Mat pre2 = int_gemm(c.f, T, F, w.at("wd"), D);
  rdiv_inplace(pre2, Q);
  c.m2.assign(pre2.size(), 0);
  c.x2.assign(pre2.size(), 0);
  for (std::size_t i = 0; i < pre2.size(); i++) {
    pre2[i] += c.x1[i];
    c.m2[i] = (pre2[i] <= CL && pre2[i] >= -CL);
    c.x2[i] = clampi(pre2[i], -CL, CL);
  }
  return c.x2;
}

Mat block::body_fwd(const std::map<std::string, Mat>& w, const Mat& x,
                    block_cache& c) const {
  check_keys(c_, w, BODY_KEYS, 9);
  return ffn_fwd(w, attn_fwd(w, x, c), c);
}

// ------------------------------------------------------- MoE body

namespace {
std::vector<std::string> moe_body_keys(int E) {
  std::vector<std::string> k = {"wq", "wk", "wv", "wo",
                                "g1", "g2", "wr"};
  for (int e = 0; e < E; e++)
    for (const char* s : {".wg", ".wu", ".wd"})
      k.push_back("e" + std::to_string(e) + s);
  return k;
}
}  // namespace

Mat block::moe_body_fwd(const std::map<std::string, Mat>& w,
                        const Mat& x, block_cache& c) const {
  const int T = c_.T, D = c_.D, F = c_.F, E = c_.n_experts;
  const i64 PQ = c_.pq, CL = c_.act_clamp;
  if (E < 1) throw std::runtime_error("intbirth: E < 1");
  for (const auto& k : moe_body_keys(E))
    if (!w.count(k))
      throw std::runtime_error("intbirth: missing " + k);
  if (i64(w.at("wr").size()) != i64(E) * D)
    throw std::runtime_error("intbirth: bad wr shape");
  const Mat& sil = tab_.at("silu.tab");

  attn_fwd(w, x, c);  // fills c.x .. c.x1
  c.h2 = rms_fwd(c.x1, w.at("g2"), c.i2);

  // router: r = rdiv(h2 @ wr^T, Q); p_r = softmax(PQ); top-1,
  // lowest expert index wins ties (strict > scanning upward)
  c.r = int_gemm(c.h2, T, D, w.at("wr"), E);
  rdiv_inplace(c.r, Q);
  c.pr = softmax_rows(c.r, T, E, PQ);
  c.top.assign(T, 0);
  c.top_p.assign(T, 0);
  for (int t = 0; t < T; t++) {
    int best = 0;
    for (int e = 1; e < E; e++)
      if (c.pr[std::size_t(t) * E + e] >
          c.pr[std::size_t(t) * E + best])
        best = e;
    c.top[t] = best;
    c.top_p[t] = c.pr[std::size_t(t) * E + best];
  }

  // selected expert's FFN per row (per-row independent, so
  // computing row t with expert top[t]'s weights is exact)
  c.egp.assign(std::size_t(T) * F, 0);
  c.eu.assign(std::size_t(T) * F, 0);
  c.esg.assign(std::size_t(T) * F, 0);
  c.ef.assign(std::size_t(T) * F, 0);
  c.eout.assign(std::size_t(T) * D, 0);
  for (int t = 0; t < T; t++) {
    const std::string p = "e" + std::to_string(c.top[t]);
    const Mat& wg = w.at(p + ".wg");
    const Mat& wu = w.at(p + ".wu");
    const Mat& wd = w.at(p + ".wd");
    for (int f = 0; f < F; f++) {
      i64 ag = 0, au = 0;
      for (int d = 0; d < D; d++) {
        const i64 h = c.h2[std::size_t(t) * D + d];
        ag += h * wg[std::size_t(f) * D + d];
        au += h * wu[std::size_t(f) * D + d];
      }
      const i64 z = rdiv(ag, Q);
      c.egp[std::size_t(t) * F + f] = z;
      c.eu[std::size_t(t) * F + f] = rdiv(au, Q);
      c.esg[std::size_t(t) * F + f] =
          z > ts_ ? z : z < -ts_ ? 0 : sil[z + ts_];
      c.ef[std::size_t(t) * F + f] =
          rdiv(c.esg[std::size_t(t) * F + f] *
                   c.eu[std::size_t(t) * F + f],
               Q);
    }
    for (int d = 0; d < D; d++) {
      i64 acc = 0;
      for (int f = 0; f < F; f++)
        acc += c.ef[std::size_t(t) * F + f] *
               wd[std::size_t(d) * F + f];
      c.eout[std::size_t(t) * D + d] = rdiv(acc, Q);
    }
  }

  // gate + residual + clamp (fx3 multiplicative-gate convention)
  Mat pre2(std::size_t(T) * D);
  for (int t = 0; t < T; t++)
    for (int d = 0; d < D; d++)
      pre2[std::size_t(t) * D + d] =
          rdiv(c.eout[std::size_t(t) * D + d] * c.top_p[t], PQ) +
          c.x1[std::size_t(t) * D + d];
  c.m2.assign(pre2.size(), 0);
  c.x2.assign(pre2.size(), 0);
  for (std::size_t i = 0; i < pre2.size(); i++) {
    c.m2[i] = (pre2[i] <= CL && pre2[i] >= -CL);
    c.x2[i] = clampi(pre2[i], -CL, CL);
  }
  return c.x2;
}

std::map<std::string, Mat> block::moe_body_bwd(
    const std::map<std::string, Mat>& w, const Mat& dxin,
    const block_cache& c, Mat* dx0_out) const {
  const int T = c_.T, D = c_.D, F = c_.F, E = c_.n_experts;
  const i64 PQ = c_.pq;
  if (E < 1) throw std::runtime_error("intbirth: E < 1");
  if (i64(dxin.size()) != i64(T) * D)
    throw std::runtime_error("intbirth: bad dxin shape");
  const Mat& dsl = tab_.at("dsilu.tab");
  std::map<std::string, Mat> G;

  Mat dx2 = dxin;
  for (std::size_t i = 0; i < dx2.size(); i++) dx2[i] *= c.m2[i];

  // gate chain: d(out) = rdiv(dx2 * top_p, PQ);
  // d(top_p)[t] = rdiv(sum_d out[t,d]*dx2[t,d], PQ)
  Mat dout(std::size_t(T) * D);
  Mat dtp(T);
  for (int t = 0; t < T; t++) {
    i64 acc = 0;
    for (int d = 0; d < D; d++) {
      const std::size_t i = std::size_t(t) * D + d;
      dout[i] = rdiv(dx2[i] * c.top_p[t], PQ);
      acc += c.eout[i] * dx2[i];
    }
    dtp[t] = rdiv(acc, PQ);
  }

  // expert FFN backward per row; dW accumulated RAW per expert,
  // one rdiv(Q) per expert after the token loop (the rdiv-grouping
  // rule: same placement as the dense xty-then-round)
  for (int e = 0; e < E; e++) {
    const std::string p = "e" + std::to_string(e);
    G[p + ".wg"].assign(std::size_t(F) * D, 0);
    G[p + ".wu"].assign(std::size_t(F) * D, 0);
    G[p + ".wd"].assign(std::size_t(D) * F, 0);
  }
  Mat dh2(std::size_t(T) * D, 0);
  Mat df(F), du(F), dgp(F);
  for (int t = 0; t < T; t++) {
    const std::string p = "e" + std::to_string(c.top[t]);
    const Mat& wg = w.at(p + ".wg");
    const Mat& wu = w.at(p + ".wu");
    const Mat& wd = w.at(p + ".wd");
    Mat& Gwg = G.at(p + ".wg");
    Mat& Gwu = G.at(p + ".wu");
    Mat& Gwd = G.at(p + ".wd");
    for (int f = 0; f < F; f++) {
      i64 acc = 0;
      for (int d = 0; d < D; d++)
        acc += dout[std::size_t(t) * D + d] *
               wd[std::size_t(d) * F + f];
      df[f] = rdiv(acc, Q);
      const i64 z = c.egp[std::size_t(t) * F + f];
      const i64 dsv = z > ts_ ? Q : z < -ts_ ? 0 : dsl[z + ts_];
      du[f] = rdiv(c.esg[std::size_t(t) * F + f] * df[f], Q);
      dgp[f] = rdiv(
          rdiv(c.eu[std::size_t(t) * F + f] * df[f], Q) * dsv, Q);
    }
    for (int f = 0; f < F; f++)
      for (int d = 0; d < D; d++) {
        Gwg[std::size_t(f) * D + d] +=
            dgp[f] * c.h2[std::size_t(t) * D + d];
        Gwu[std::size_t(f) * D + d] +=
            du[f] * c.h2[std::size_t(t) * D + d];
      }
    for (int d = 0; d < D; d++)
      for (int f = 0; f < F; f++)
        Gwd[std::size_t(d) * F + f] +=
            dout[std::size_t(t) * D + d] *
            c.ef[std::size_t(t) * F + f];
    // dh2 from expert weights (two-term sum, one rdiv after loop)
    for (int d = 0; d < D; d++) {
      i64 acc = 0;
      for (int f = 0; f < F; f++)
        acc += du[f] * wu[std::size_t(f) * D + d] +
               dgp[f] * wg[std::size_t(f) * D + d];
      dh2[std::size_t(t) * D + d] = acc;
    }
  }
  for (int e = 0; e < E; e++) {
    const std::string p = "e" + std::to_string(e);
    rdiv_inplace(G.at(p + ".wg"), Q);
    rdiv_inplace(G.at(p + ".wu"), Q);
    rdiv_inplace(G.at(p + ".wd"), Q);
  }
  rdiv_inplace(dh2, Q);

  // router: scatter d(top_p) into dp_r, softmax_bwd at PQ, then
  // wr + h2 paths (each group finalized once, then summed)
  Mat dpr(std::size_t(T) * E, 0);
  for (int t = 0; t < T; t++)
    dpr[std::size_t(t) * E + c.top[t]] = dtp[t];
  Mat dr(std::size_t(T) * E);
  for (int t = 0; t < T; t++) {
    i64 inner = 0;
    for (int e = 0; e < E; e++)
      inner += rdiv(c.pr[std::size_t(t) * E + e] *
                        dpr[std::size_t(t) * E + e],
                    PQ);
    for (int e = 0; e < E; e++)
      dr[std::size_t(t) * E + e] =
          rdiv(c.pr[std::size_t(t) * E + e] *
                   (dpr[std::size_t(t) * E + e] - inner),
               PQ);
  }
  G["wr"] = int_gemm_xty(dr, T, E, c.h2, D);
  rdiv_inplace(G["wr"], Q);
  {
    Mat dh2r = int_gemm_nt(dr, T, E, w.at("wr"), D);
    rdiv_inplace(dh2r, Q);
    for (std::size_t i = 0; i < dh2.size(); i++) dh2[i] += dh2r[i];
  }

  Mat dx1 = rms_bwd(dh2, c.x1, w.at("g2"), c.i2, G["g2"]);
  for (std::size_t i = 0; i < dx1.size(); i++)
    dx1[i] = (dx1[i] + dx2[i]) * c.m1[i];
  Mat dx0 = attn_bwd(w, dx1, c, G);
  if (dx0_out) {
    for (std::size_t i = 0; i < dx0.size(); i++) dx0[i] += dx1[i];
    *dx0_out = std::move(dx0);
  }
  return G;
}

Mat block::fwd(const std::map<std::string, Mat>& w, const Mat& x,
               block_cache& c) const {
  check_weights(w);
  const Mat x2 = body_fwd(w, x, c);
  c.h3 = rms_fwd(x2, w.at("g3"), c.i3);
  Mat logits = int_gemm(c.h3, c_.T, c_.D, w.at("wh"), c_.V);
  rdiv_inplace(logits, Q);
  return logits;
}

Mat block::ffn_bwd(const std::map<std::string, Mat>& w,
                   const Mat& dx2_masked, const block_cache& c,
                   std::map<std::string, Mat>& G) const {
  const int T = c_.T, D = c_.D, F = c_.F;
  const Mat& dsl = tab_.at("dsilu.tab");
  const Mat& dx2 = dx2_masked;
  Mat df = int_gemm_nt(dx2, T, D, w.at("wd"), F);
  rdiv_inplace(df, Q);
  G["wd"] = int_gemm_xty(dx2, T, D, c.f, F);
  rdiv_inplace(G["wd"], Q);
  Mat du(df.size()), dgp(df.size());
  for (std::size_t i = 0; i < df.size(); i++) {
    const i64 z = c.gp[i];
    const i64 dsv = z > ts_ ? Q : z < -ts_ ? 0 : dsl[z + ts_];
    du[i] = rdiv(c.sg[i] * df[i], Q);
    dgp[i] = rdiv(rdiv(c.u[i] * df[i], Q) * dsv, Q);
  }
  Mat dh2 = int_gemm_nt(du, T, F, w.at("wu"), D);
  {
    const Mat t2 = int_gemm_nt(dgp, T, F, w.at("wg"), D);
    for (std::size_t i = 0; i < dh2.size(); i++) dh2[i] += t2[i];
  }
  rdiv_inplace(dh2, Q);  // one rdiv after the two-term sum
  G["wu"] = int_gemm_xty(du, T, F, c.h2, D);
  rdiv_inplace(G["wu"], Q);
  G["wg"] = int_gemm_xty(dgp, T, F, c.h2, D);
  rdiv_inplace(G["wg"], Q);
  return rms_bwd(dh2, c.x1, w.at("g2"), c.i2, G["g2"]);
}

Mat block::attn_bwd(const std::map<std::string, Mat>& w,
                    const Mat& dx1_masked, const block_cache& c,
                    std::map<std::string, Mat>& G) const {
  const int T = c_.T, D = c_.D, DH = c_.DH;
  const i64 PQ = c_.pq;
  const Mat& dx1 = dx1_masked;
  const Mat& tcos = tab_.at("rope.cos");
  const Mat& tsin = tab_.at("rope.sin");
  const int half = DH / 2;
  const auto rope_b = [&](const Mat& dx) {
    Mat y(std::size_t(T) * DH);
    for (int t = 0; t < T; t++)
      for (int i = 0; i < half; i++) {
        const i64 co = tcos[std::size_t(t) * half + i];
        const i64 si = tsin[std::size_t(t) * half + i];
        const i64 a = dx[std::size_t(t) * DH + i];
        const i64 b = dx[std::size_t(t) * DH + half + i];
        y[std::size_t(t) * DH + i] = rdiv(a * co + b * si, RS);
        y[std::size_t(t) * DH + half + i] =
            rdiv(-a * si + b * co, RS);
      }
    return y;
  };
  Mat da = int_gemm_nt(dx1, T, D, w.at("wo"), DH);
  rdiv_inplace(da, Q);
  G["wo"] = int_gemm_xty(dx1, T, D, c.a, DH);
  rdiv_inplace(G["wo"], Q);
  Mat dp = int_gemm(da, T, DH, c.v0, T);
  rdiv_inplace(dp, Q);
  Mat dv = int_gemm_xty(c.p, T, T, da, DH);
  rdiv_inplace(dv, PQ);
  Mat ds(std::size_t(T) * T);
  for (int t = 0; t < T; t++) {
    i64 inner = 0;
    for (int cc = 0; cc < T; cc++)
      inner += rdiv(
          c.p[std::size_t(t) * T + cc] * dp[std::size_t(t) * T + cc],
          PQ);
    for (int cc = 0; cc < T; cc++)
      ds[std::size_t(t) * T + cc] =
          rdiv(c.p[std::size_t(t) * T + cc] *
                   (dp[std::size_t(t) * T + cc] - inner),
               PQ);
  }
  Mat dqr = int_gemm_nt(ds, T, T, c.kr, DH);
  rdiv_inplace(dqr, scale_);
  Mat dkr = int_gemm_xty(ds, T, T, c.qr, DH);
  rdiv_inplace(dkr, scale_);
  const Mat dq = rope_b(dqr), dk = rope_b(dkr);
  G["wq"] = int_gemm_xty(dq, T, DH, c.h1, D);
  rdiv_inplace(G["wq"], Q);
  G["wk"] = int_gemm_xty(dk, T, DH, c.h1, D);
  rdiv_inplace(G["wk"], Q);
  G["wv"] = int_gemm_xty(dv, T, DH, c.h1, D);
  rdiv_inplace(G["wv"], Q);
  Mat dh1 = int_gemm_nt(dq, T, DH, w.at("wq"), D);
  {
    const Mat t2 = int_gemm_nt(dk, T, DH, w.at("wk"), D);
    const Mat t3 = int_gemm_nt(dv, T, DH, w.at("wv"), D);
    for (std::size_t i = 0; i < dh1.size(); i++)
      dh1[i] += t2[i] + t3[i];
  }
  rdiv_inplace(dh1, Q);  // one rdiv after the three-term sum
  return rms_bwd(dh1, c.x, w.at("g1"), c.i1, G["g1"]);
}

std::map<std::string, Mat> block::body_bwd(
    const std::map<std::string, Mat>& w, const Mat& dxin,
    const block_cache& c, Mat* dx0_out) const {
  check_keys(c_, w, BODY_KEYS, 9);
  if (i64(dxin.size()) != i64(c_.T) * c_.D)
    throw std::runtime_error("intbirth: bad dxin shape");
  std::map<std::string, Mat> G;
  Mat dx2 = dxin;
  for (std::size_t i = 0; i < dx2.size(); i++) dx2[i] *= c.m2[i];
  Mat dx1 = ffn_bwd(w, dx2, c, G);
  for (std::size_t i = 0; i < dx1.size(); i++)
    dx1[i] = (dx1[i] + dx2[i]) * c.m1[i];
  Mat dx0 = attn_bwd(w, dx1, c, G);
  if (dx0_out) {
    // residual to input: the multi-block chain point (dx1 already
    // carries the clamp mask)
    for (std::size_t i = 0; i < dx0.size(); i++) dx0[i] += dx1[i];
    *dx0_out = std::move(dx0);
  }
  return G;
}

std::map<std::string, Mat> block::bwd(
    const std::map<std::string, Mat>& w, const Mat& dlogits,
    const block_cache& c, Mat* dx0_out) const {
  check_weights(w);
  const int T = c_.T, D = c_.D, V = c_.V;
  if (i64(dlogits.size()) != i64(T) * V)
    throw std::runtime_error("intbirth: bad dlogits shape");
  std::map<std::string, Mat> G;
  G["wh"] = int_gemm_xty(dlogits, T, V, c.h3, D);
  rdiv_inplace(G["wh"], Q);
  Mat dh3 = int_gemm_nt(dlogits, T, V, w.at("wh"), D);
  rdiv_inplace(dh3, Q);
  const Mat dx2in = rms_bwd(dh3, c.x2, w.at("g3"), c.i3, G["g3"]);
  auto Gb = body_bwd(w, dx2in, c, dx0_out);
  for (auto& [k, g] : Gb) G[k] = std::move(g);
  return G;
}

// ------------------------------------------------------------ adamw

adamw::adamw(int shift, i64 lrn, i64 lrd)
    : shift_(shift), lrn_(lrn), lrd_(lrd) {
  if (shift < 0 || lrn < 1 || lrd < 1)
    throw std::runtime_error("intbirth: bad adamw params");
  p10_ = p9_ = p1000_ = p999_ = BigV{1};
}

void adamw::step(const std::vector<Mat*>& params,
                 const std::vector<const Mat*>& grads) {
  if (params.size() != grads.size())
    throw std::runtime_error("intbirth: params/grads length mismatch");
  if (m_.empty()) {
    for (const Mat* p : params) {
      m_.emplace_back(p->size(), 0);
      v_.emplace_back(p->size(), 0);
    }
  }
  if (m_.size() != params.size())
    throw std::runtime_error("intbirth: param count changed");
  t_ += 1;
  big_mul(p10_, 10);
  big_mul(p9_, 9);
  big_mul(p1000_, 1000);
  big_mul(p999_, 999);
  BigV n1 = p10_, d1 = big_sub(p10_, p9_);
  BigV n2 = p1000_, d2 = big_sub(p1000_, p999_);
  while (big_gt_pow30(n1)) { big_shr1(n1); big_shr1(d1); }
  while (big_gt_pow30(n2)) { big_shr1(n2); big_shr1(d2); }
  const i64 bc1n = big_i64(n1), bc1d = std::max<i64>(big_i64(d1), 1);
  const i64 bc2n = big_i64(n2), bc2d = std::max<i64>(big_i64(d2), 1);
  i64 nz = 0, tot = 0;
  for (std::size_t j = 0; j < params.size(); j++) {
    Mat& w = *params[j];
    const Mat& g = *grads[j];
    if (w.size() != g.size() || w.size() != m_[j].size())
      throw std::runtime_error("intbirth: grad shape mismatch");
    for (std::size_t i = 0; i < w.size(); i++) {
      m_[j][i] = rdiv(B1N * m_[j][i] + (B1D - B1N) * g[i], B1D);
      v_[j][i] =
          rdiv(B2N * v_[j][i] + (B2D - B2N) * rdiv(g[i] * g[i], Q),
               B2D);
      const i64 mh = rdiv(m_[j][i] * bc1n, bc1d);
      const i64 vh = rdiv(v_[j][i] * bc2n, bc2d);
      const i64 den = isqrt_newton(vh * Q) + AEPS;
      const i64 upd = rdiv(lrn_ * mh * (Q << shift_), lrd_ * den);
      nz += upd != 0;
      tot += 1;
      w[i] -= upd;
      w[i] -= rdiv(w[i] * WDN, WDD);
    }
  }
  nz_ = double(nz) / double(tot);
}

// ------------------------------------------------------- full_birth

full_birth::full_birth(const std::string& tables_bytes,
                       const std::string& init_bytes,
                       const contract& c)
    : blk_(tables_bytes, c), opt_(c.shift, c.lrn, c.lrd) {
  const auto sh = shapes(c);
  std::size_t off = 0;
  const auto take = [&](std::size_t n) {
    if (off + n * 8 > init_bytes.size())
      throw std::runtime_error("intbirth: truncated init");
    Mat m(n);
    std::memcpy(m.data(), init_bytes.data() + off, n * 8);
    off += n * 8;
    return m;
  };
  for (const char* k : block::KEYS) {
    const auto s = sh.at(k);
    w_[k] = take(std::size_t(s.r) * s.c);
  }
  x_ = take(std::size_t(c.T) * c.D);
  tgt_ = take(std::size_t(c.T));
  if (off != init_bytes.size())
    throw std::runtime_error("intbirth: trailing init bytes");
  for (const i64 t : tgt_)
    if (t < 0 || t >= c.V)
      throw std::runtime_error("intbirth: target out of vocab");
  for (const char* k : block::KEYS)
    for (auto& v : w_[k]) v <<= c.shift;  // lift to Q_w
}

void full_birth::run(int steps) {
  for (int i = 0; i < steps; i++) step_once();
}

std::string full_birth::mark() {
  for (const char* k : block::KEYS)
    th_.update(w_.at(k).data(), w_.at(k).size() * 8);
  return traj_sha();
}

std::string full_birth::traj_sha() const {
  detail::sha256 peek = th_;
  return peek.hex();
}

std::string full_birth::weights_bytes() const {
  std::string out;
  for (const char* k : block::KEYS) {
    const auto& w = w_.at(k);
    out.append(reinterpret_cast<const char*>(w.data()), w.size() * 8);
  }
  return out;
}

void full_birth::step_once() {
  const contract& c = blk_.cfg();
  // Q-scale view of the wide weights (the matmul boundary)
  std::map<std::string, Mat> w;
  for (const char* k : block::KEYS) {
    w[k] = w_.at(k);
    for (auto& v : w[k]) v = rdiv(v, i64(1) << c.shift);
  }
  block_cache bc;
  const Mat logits = blk_.fwd(w, x_, bc);
  const Mat pp = blk_.softmax_rows(logits, c.T, c.V, Q);
  i64 loss = 0;
  for (int t = 0; t < c.T; t++)
    loss += Q - pp[std::size_t(t) * c.V + tgt_[t]];
  loss_ = loss;
  Mat dlogits(std::size_t(c.T) * c.V);
  for (int t = 0; t < c.T; t++)
    for (int vv = 0; vv < c.V; vv++)
      dlogits[std::size_t(t) * c.V + vv] =
          (pp[std::size_t(t) * c.V + vv] - Q * (tgt_[t] == vv)) *
          c.gboost;
  auto G = blk_.bwd(w, dlogits, bc);
  std::vector<Mat*> params;
  std::vector<Mat> unboosted;
  unboosted.reserve(11);
  for (const char* k : block::KEYS) {
    Mat g = std::move(G.at(k));
    for (auto& v : g) v = rdiv(v, Q * c.gboost);  // unboost
    unboosted.push_back(std::move(g));
    params.push_back(&w_.at(k));
  }
  std::vector<const Mat*> gp;
  for (const auto& g : unboosted) gp.push_back(&g);
  opt_.step(params, gp);
  step_ += 1;
}

// ------------------------------------------------------ multi_birth

multi_birth::multi_birth(const std::string& tables_bytes,
                         const std::string& init_bytes,
                         const contract& c,
                         const std::string& windows_bytes)
    : blk_(tables_bytes, c), opt_(c.shift, c.lrn, c.lrd) {
  if (c.n_blocks < 1)
    throw std::runtime_error("intbirth: n_blocks < 1");
  const auto sh = shapes(c);
  order_.push_back("emb");
  for (int b = 0; b < c.n_blocks; b++)
    for (const char* k : block::BODY_KEYS)
      order_.push_back("b" + std::to_string(b) + "." + k);
  order_.push_back("g_f");
  const auto numel = [&](const std::string& name) -> std::size_t {
    if (name == "emb") return std::size_t(c.V) * c.D;
    if (name == "g_f") return std::size_t(c.D);
    const auto s = sh.at(name.substr(name.find('.') + 1));
    return std::size_t(s.r) * s.c;
  };
  std::size_t off = 0;
  const auto take = [&](std::size_t n) {
    if (off + n * 8 > init_bytes.size())
      throw std::runtime_error("intbirth: truncated init");
    Mat m(n);
    std::memcpy(m.data(), init_bytes.data() + off, n * 8);
    off += n * 8;
    return m;
  };
  for (const auto& name : order_) w_[name] = take(numel(name));
  if (windows_bytes.empty()) {
    wtok_.push_back(take(std::size_t(c.T)));
    wtgt_.push_back(take(std::size_t(c.T)));
  } else {
    const std::size_t rec = std::size_t(c.T) * 2 * 8;
    if (windows_bytes.size() % rec || windows_bytes.empty())
      throw std::runtime_error("intbirth: bad windows length");
    const std::size_t nw = windows_bytes.size() / rec;
    for (std::size_t i = 0; i < nw; i++) {
      Mat tk(c.T), tg(c.T);
      std::memcpy(tk.data(), windows_bytes.data() + i * rec,
                  std::size_t(c.T) * 8);
      std::memcpy(tg.data(),
                  windows_bytes.data() + i * rec + std::size_t(c.T) * 8,
                  std::size_t(c.T) * 8);
      wtok_.push_back(std::move(tk));
      wtgt_.push_back(std::move(tg));
    }
  }
  if (off != init_bytes.size())
    throw std::runtime_error("intbirth: trailing init bytes");
  for (const Mat& tk : wtok_)
    for (const i64 t : tk)
      if (t < 0 || t >= c.V)
        throw std::runtime_error("intbirth: token out of vocab");
  for (const Mat& tg : wtgt_)
    for (const i64 t : tg)
      if (t < 0 || t >= c.V)
        throw std::runtime_error("intbirth: target out of vocab");
  for (const auto& name : order_)
    for (auto& v : w_[name]) v <<= c.shift;  // lift to Q_w
}

void multi_birth::run(int steps) {
  for (int i = 0; i < steps; i++) step_once();
}

std::string multi_birth::mark() {
  for (const auto& name : order_)
    th_.update(w_.at(name).data(), w_.at(name).size() * 8);
  return traj_sha();
}

std::string multi_birth::traj_sha() const {
  detail::sha256 peek = th_;
  return peek.hex();
}

std::string multi_birth::weights_bytes() const {
  std::string out;
  for (const auto& name : order_) {
    const auto& w = w_.at(name);
    out.append(reinterpret_cast<const char*>(w.data()), w.size() * 8);
  }
  return out;
}

void multi_birth::step_once() {
  const contract& c = blk_.cfg();
  const int T = c.T, D = c.D, V = c.V, NB = c.n_blocks;
  const Mat& tok_ = wtok_[std::size_t(step_) % wtok_.size()];
  const Mat& tgt_ = wtgt_[std::size_t(step_) % wtgt_.size()];
  // Q-scale view of the wide params (the matmul boundary)
  std::map<std::string, Mat> nar;
  for (const auto& name : order_) {
    nar[name] = w_.at(name);
    for (auto& v : nar[name]) v = rdiv(v, i64(1) << c.shift);
  }
  const Mat& emb = nar.at("emb");
  // per-block weight views (BODY_KEYS names)
  std::vector<std::map<std::string, Mat>> bw(NB);
  for (int b = 0; b < NB; b++)
    for (const char* k : block::BODY_KEYS)
      bw[b][k] = nar.at("b" + std::to_string(b) + "." + k);

  // ---- forward: emb lookup -> bodies -> final norm -> tied head
  Mat x(std::size_t(T) * D);
  for (int t = 0; t < T; t++)
    for (int d = 0; d < D; d++)
      x[std::size_t(t) * D + d] = emb[std::size_t(tok_[t]) * D + d];
  std::vector<block_cache> bc(NB);
  for (int b = 0; b < NB; b++) x = blk_.body_fwd(bw[b], x, bc[b]);
  Mat i_f;
  const Mat xf = x;
  const Mat hf = blk_.rms_fwd(xf, nar.at("g_f"), i_f);
  Mat logits = int_gemm(hf, T, D, emb, V);  // tied head
  rdiv_inplace(logits, Q);

  // ---- loss + CE gradient (boosted)
  const Mat pp = blk_.softmax_rows(logits, T, V, Q);
  i64 loss = 0;
  for (int t = 0; t < T; t++)
    loss += Q - pp[std::size_t(t) * V + tgt_[t]];
  loss_ = loss;
  Mat dlogits(std::size_t(T) * V);
  for (int t = 0; t < T; t++)
    for (int vv = 0; vv < V; vv++)
      dlogits[std::size_t(t) * V + vv] =
          (pp[std::size_t(t) * V + vv] - Q * (tgt_[t] == vv)) *
          c.gboost;

  // ---- backward
  std::map<std::string, Mat> G;
  // tied head: wh-convention grad wrt emb + dh into the norm
  Mat g_head = int_gemm_xty(dlogits, T, V, hf, D);
  rdiv_inplace(g_head, Q);
  Mat dhf = int_gemm_nt(dlogits, T, V, emb, D);
  rdiv_inplace(dhf, Q);
  Mat dx = blk_.rms_bwd(dhf, xf, nar.at("g_f"), i_f, G["g_f"]);
  for (int b = NB - 1; b >= 0; b--) {
    Mat dx0;
    auto Gb = blk_.body_bwd(bw[b], dx, bc[b], &dx0);
    for (const char* k : block::BODY_KEYS)
      G["b" + std::to_string(b) + "." + k] = std::move(Gb.at(k));
    dx = std::move(dx0);
  }
  // embedding: EXACT scatter-add of dx rows by token, summed with
  // the already-rounded head part (each part finalized BEFORE the
  // sum — the rdiv-grouping rule)
  Mat g_emb = std::move(g_head);
  for (int t = 0; t < T; t++)
    for (int d = 0; d < D; d++)
      g_emb[std::size_t(tok_[t]) * D + d] += dx[std::size_t(t) * D + d];
  G["emb"] = std::move(g_emb);

  // ---- optimizer over param_order
  std::vector<Mat*> params;
  std::vector<Mat> unboosted;
  unboosted.reserve(order_.size());
  for (const auto& name : order_) {
    Mat g = std::move(G.at(name));
    for (auto& v : g) v = rdiv(v, Q * c.gboost);  // unboost
    unboosted.push_back(std::move(g));
    params.push_back(&w_.at(name));
  }
  std::vector<const Mat*> gp;
  for (const auto& g : unboosted) gp.push_back(&g);
  opt_.step(params, gp);
  step_ += 1;
}

// -------------------------------------------------------- moe_birth

moe_birth::moe_birth(const std::string& tables_bytes,
                     const std::string& init_bytes, const contract& c,
                     const std::string& windows_bytes)
    : blk_(tables_bytes, c), opt_(c.shift, c.lrn, c.lrd) {
  if (c.n_blocks < 1)
    throw std::runtime_error("intbirth: n_blocks < 1");
  if (c.n_experts < 1)
    throw std::runtime_error("intbirth: n_experts < 1");
  const auto sh = shapes(c);
  order_.push_back("emb");
  for (int b = 0; b < c.n_blocks; b++) {
    const std::string bp = "b" + std::to_string(b) + ".";
    for (const char* k : {"wq", "wk", "wv", "wo", "g1", "g2", "wr"})
      order_.push_back(bp + k);
    for (int e = 0; e < c.n_experts; e++)
      for (const char* k : {"wg", "wu", "wd"})
        order_.push_back(bp + "e" + std::to_string(e) + "." + k);
  }
  order_.push_back("g_f");
  const auto numel = [&](const std::string& name) -> std::size_t {
    if (name == "emb") return std::size_t(c.V) * c.D;
    if (name == "g_f") return std::size_t(c.D);
    const std::string leaf = name.substr(name.rfind('.') + 1);
    if (leaf == "wr") return std::size_t(c.n_experts) * c.D;
    const auto s = sh.at(leaf);
    return std::size_t(s.r) * s.c;
  };
  std::size_t off = 0;
  const auto take = [&](std::size_t n) {
    if (off + n * 8 > init_bytes.size())
      throw std::runtime_error("intbirth: truncated init");
    Mat m(n);
    std::memcpy(m.data(), init_bytes.data() + off, n * 8);
    off += n * 8;
    return m;
  };
  for (const auto& name : order_) w_[name] = take(numel(name));
  if (windows_bytes.empty()) {
    wtok_.push_back(take(std::size_t(c.T)));
    wtgt_.push_back(take(std::size_t(c.T)));
  } else {
    const std::size_t rec = std::size_t(c.T) * 2 * 8;
    if (windows_bytes.size() % rec)
      throw std::runtime_error("intbirth: bad windows length");
    const std::size_t nw = windows_bytes.size() / rec;
    for (std::size_t i = 0; i < nw; i++) {
      Mat tk(c.T), tg(c.T);
      std::memcpy(tk.data(), windows_bytes.data() + i * rec,
                  std::size_t(c.T) * 8);
      std::memcpy(tg.data(),
                  windows_bytes.data() + i * rec + std::size_t(c.T) * 8,
                  std::size_t(c.T) * 8);
      wtok_.push_back(std::move(tk));
      wtgt_.push_back(std::move(tg));
    }
  }
  if (off != init_bytes.size())
    throw std::runtime_error("intbirth: trailing init bytes");
  for (const Mat& tk : wtok_)
    for (const i64 t : tk)
      if (t < 0 || t >= c.V)
        throw std::runtime_error("intbirth: token out of vocab");
  for (const Mat& tg : wtgt_)
    for (const i64 t : tg)
      if (t < 0 || t >= c.V)
        throw std::runtime_error("intbirth: target out of vocab");
  for (const auto& name : order_)
    for (auto& v : w_[name]) v <<= c.shift;  // lift to Q_w
}

void moe_birth::run(int steps) {
  for (int i = 0; i < steps; i++) step_once();
}

std::string moe_birth::mark() {
  for (const auto& name : order_)
    th_.update(w_.at(name).data(), w_.at(name).size() * 8);
  return traj_sha();
}

std::string moe_birth::traj_sha() const {
  detail::sha256 peek = th_;
  return peek.hex();
}

std::string moe_birth::weights_bytes() const {
  std::string out;
  for (const auto& name : order_) {
    const auto& w = w_.at(name);
    out.append(reinterpret_cast<const char*>(w.data()), w.size() * 8);
  }
  return out;
}

void moe_birth::step_once() {
  const contract& c = blk_.cfg();
  const int T = c.T, D = c.D, V = c.V, NB = c.n_blocks;
  const i64 gb = c.gboost * 4;  // GB = 4 x GBOOST (gravmoe spec)
  const Mat& tok = wtok_[std::size_t(step_) % wtok_.size()];
  const Mat& tgt = wtgt_[std::size_t(step_) % wtgt_.size()];
  // Q-scale view of the wide params (the matmul boundary)
  std::map<std::string, Mat> nar;
  for (const auto& name : order_) {
    nar[name] = w_.at(name);
    for (auto& v : nar[name]) v = rdiv(v, i64(1) << c.shift);
  }
  const Mat& emb = nar.at("emb");
  // per-body weight views (MoE names, body-local)
  std::vector<std::map<std::string, Mat>> bw(NB);
  for (int b = 0; b < NB; b++) {
    const std::string bp = "b" + std::to_string(b) + ".";
    for (const char* k : {"wq", "wk", "wv", "wo", "g1", "g2", "wr"})
      bw[b][k] = nar.at(bp + k);
    for (int e = 0; e < c.n_experts; e++)
      for (const char* k : {"wg", "wu", "wd"}) {
        const std::string ek = "e" + std::to_string(e) + "." + k;
        bw[b][ek] = nar.at(bp + ek);
      }
  }

  // ---- forward: emb lookup -> MoE bodies -> final norm -> tied head
  Mat x(std::size_t(T) * D);
  for (int t = 0; t < T; t++)
    for (int d = 0; d < D; d++)
      x[std::size_t(t) * D + d] = emb[std::size_t(tok[t]) * D + d];
  std::vector<block_cache> bc(NB);
  for (int b = 0; b < NB; b++) x = blk_.moe_body_fwd(bw[b], x, bc[b]);
  Mat i_f;
  const Mat xf = x;
  const Mat hf = blk_.rms_fwd(xf, nar.at("g_f"), i_f);
  Mat logits = int_gemm(hf, T, D, emb, V);  // tied head
  rdiv_inplace(logits, Q);

  // ---- loss + CE gradient (boosted at GB)
  const Mat pp = blk_.softmax_rows(logits, T, V, Q);
  i64 loss = 0;
  for (int t = 0; t < T; t++)
    loss += Q - pp[std::size_t(t) * V + tgt[t]];
  loss_ = loss;
  Mat dlogits(std::size_t(T) * V);
  for (int t = 0; t < T; t++)
    for (int vv = 0; vv < V; vv++)
      dlogits[std::size_t(t) * V + vv] =
          (pp[std::size_t(t) * V + vv] - Q * (tgt[t] == vv)) * gb;

  // ---- backward
  std::map<std::string, Mat> G;
  Mat g_head = int_gemm_xty(dlogits, T, V, hf, D);
  rdiv_inplace(g_head, Q);
  Mat dhf = int_gemm_nt(dlogits, T, V, emb, D);
  rdiv_inplace(dhf, Q);
  Mat dx = blk_.rms_bwd(dhf, xf, nar.at("g_f"), i_f, G["g_f"]);
  for (int b = NB - 1; b >= 0; b--) {
    Mat dx0;
    auto Gb = blk_.moe_body_bwd(bw[b], dx, bc[b], &dx0);
    const std::string bp = "b" + std::to_string(b) + ".";
    for (auto& [k, g] : Gb) G[bp + k] = std::move(g);
    dx = std::move(dx0);
  }
  // embedding: rounded head part + exact scatter-add (the
  // rdiv-grouping rule, unchanged from mb)
  Mat g_emb = std::move(g_head);
  for (int t = 0; t < T; t++)
    for (int d = 0; d < D; d++)
      g_emb[std::size_t(tok[t]) * D + d] += dx[std::size_t(t) * D + d];
  G["emb"] = std::move(g_emb);

  // ---- optimizer over param_order (unboost at GB)
  std::vector<Mat*> params;
  std::vector<Mat> unboosted;
  unboosted.reserve(order_.size());
  for (const auto& name : order_) {
    Mat g = std::move(G.at(name));
    for (auto& v : g) v = rdiv(v, Q * gb);
    unboosted.push_back(std::move(g));
    params.push_back(&w_.at(name));
  }
  std::vector<const Mat*> gp;
  for (const auto& g : unboosted) gp.push_back(&g);
  opt_.step(params, gp);
  step_ += 1;

  // ---- gravity event (wide Q_w space, after the optimizer step)
  if (c.grav_k > 0 && step_ % c.grav_k == 0 && c.grav_ln != 0) {
    static const char* const KINDS[3] = {"wg", "wu", "wd"};
    for (int b = 0; b < NB; b++)
      for (const char* kind : KINDS) {
        std::vector<Mat*> ws;
        for (int e = 0; e < c.n_experts; e++)
          ws.push_back(&w_.at("b" + std::to_string(b) + ".e" +
                              std::to_string(e) + "." + kind));
        const std::size_t n = ws[0]->size();
        for (std::size_t i = 0; i < n; i++) {
          i64 s = 0;
          for (Mat* wp : ws) s += (*wp)[i];
          const i64 mean = rdiv(s, c.n_experts);  // finalized ONCE
          for (Mat* wp : ws)
            (*wp)[i] +=
                rdiv((mean - (*wp)[i]) * c.grav_ln, c.grav_ld);
        }
      }
  }
}

}  // namespace ax::nn::ib
