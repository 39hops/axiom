/** @file intbirth.cpp Integer-birth engine (see intbirth.hpp).
    Line-for-line port of the certified R2b leg
    (tools/int_adamw/r2b_main.cpp) into contract-parameterized form;
    certified by reproducing the r2b_ref.json milestone digests. */
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
using Mat = std::vector<i64>;

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
/** round(sqrt(n)) exactly: floor-sqrt then round to the nearer
    square (ties cannot occur — n is never exactly (r+1/2)^2). */
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
    const i64 x =
        i64(a[i]) - (i < b.size() ? i64(b[i]) : 0) - borrow;
    borrow = x < 0;
    r[i] = u32(x + (borrow << 32));
  }
  big_trim(r);
  return r;
}

// ---- matmul forms (rounding placement per the booked spec rule:
// callers round ONCE, after any multi-term sum)
Mat mmT(const Mat& a, int rows, int K, const Mat& w, int N) {
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
Mat mm(const Mat& a, int rows, int K, const Mat& w, int N) {
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
Mat xty(const Mat& x, int rows, int K, const Mat& y, int N) {
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
void rdiv_all(Mat& m, i64 d) {
  for (auto& v : m) v = rdiv(v, d);
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

// ------------------------------------------------------- full_birth

const char* const full_birth::KEYS[11] = {"wq", "wk", "wv", "wo",
                                          "wg", "wu", "wd", "wh",
                                          "g1", "g2", "g3"};

full_birth::full_birth(const std::string& tables_bytes,
                       const std::string& init_bytes,
                       const contract& c)
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

  const std::map<std::string, Shape> sh = {
      {"wq", {c_.DH, c_.D}}, {"wk", {c_.DH, c_.D}},
      {"wv", {c_.DH, c_.D}}, {"wo", {c_.D, c_.DH}},
      {"wg", {c_.F, c_.D}},  {"wu", {c_.F, c_.D}},
      {"wd", {c_.D, c_.F}},  {"wh", {c_.V, c_.D}},
      {"g1", {c_.D, 1}},     {"g2", {c_.D, 1}},
      {"g3", {c_.D, 1}}};
  std::size_t off = 0;
  const auto take = [&](std::size_t n) {
    if (off + n * 8 > init_bytes.size())
      throw std::runtime_error("intbirth: truncated init");
    Mat m(n);
    std::memcpy(m.data(), init_bytes.data() + off, n * 8);
    off += n * 8;
    return m;
  };
  for (const char* k : KEYS) {
    const auto s = sh.at(k);
    w_[k] = take(std::size_t(s.r) * s.c);
  }
  x_ = take(std::size_t(c_.T) * c_.D);
  tgt_ = take(std::size_t(c_.T));
  if (off != init_bytes.size())
    throw std::runtime_error("intbirth: trailing init bytes");
  for (const i64 t : tgt_)
    if (t < 0 || t >= c_.V)
      throw std::runtime_error("intbirth: target out of vocab");

  for (const char* k : KEYS)
    for (auto& v : w_[k]) v <<= c_.shift;  // lift to Q_w
  for (const char* k : KEYS) {
    m_.emplace_back(w_[k].size(), 0);
    v_.emplace_back(w_[k].size(), 0);
  }
  p10_ = p9_ = p1000_ = p999_ = BigV{1};
}

void full_birth::run(int steps) {
  for (int i = 0; i < steps; i++) step_once();
}

std::string full_birth::mark() {
  for (const char* k : KEYS)
    th_.update(w_.at(k).data(), w_.at(k).size() * 8);
  return traj_sha();
}

std::string full_birth::traj_sha() const {
  detail::sha256 peek = th_;
  return peek.hex();
}

std::string full_birth::weights_bytes() const {
  std::string out;
  for (const char* k : KEYS) {
    const auto& w = w_.at(k);
    out.append(reinterpret_cast<const char*>(w.data()),
               w.size() * 8);
  }
  return out;
}

void full_birth::step_once() {
  const int T = c_.T, D = c_.D, DH = c_.DH, F = c_.F, V = c_.V;
  const i64 PQ = c_.pq, CL = c_.act_clamp;
  const Mat& sil = tab_.at("silu.tab");
  const Mat& dsl = tab_.at("dsilu.tab");
  const Mat& ex = tab_.at("exp.tab");
  const Mat& tcos = tab_.at("rope.cos");
  const Mat& tsin = tab_.at("rope.sin");
  const int half = DH / 2;
  const auto lut_silu = [&](i64 z) {
    return z > ts_ ? z : z < -ts_ ? 0 : sil[z + ts_];
  };
  const auto lut_dsilu = [&](i64 z) {
    return z > ts_ ? Q : z < -ts_ ? 0 : dsl[z + ts_];
  };
  const auto lut_exp = [&](i64 d) {
    return d < -tse_ ? 0 : ex[d + tse_];
  };
  const auto rope = [&](const Mat& v, bool bwd) {
    Mat y(std::size_t(T) * DH);
    for (int t = 0; t < T; t++)
      for (int i = 0; i < half; i++) {
        const i64 co = tcos[std::size_t(t) * half + i];
        const i64 si = tsin[std::size_t(t) * half + i];
        const i64 a = v[std::size_t(t) * DH + i];
        const i64 b = v[std::size_t(t) * DH + half + i];
        if (!bwd) {
          y[std::size_t(t) * DH + i] = rdiv(a * co - b * si, RS);
          y[std::size_t(t) * DH + half + i] =
              rdiv(a * si + b * co, RS);
        } else {
          y[std::size_t(t) * DH + i] = rdiv(a * co + b * si, RS);
          y[std::size_t(t) * DH + half + i] =
              rdiv(-a * si + b * co, RS);
        }
      }
    return y;
  };
  const auto rms_fwd = [&](const Mat& x, const Mat& g, Mat& isq) {
    Mat y(std::size_t(T) * D);
    isq.assign(T, 0);
    for (int t = 0; t < T; t++) {
      i64 s2 = 0;
      for (int d = 0; d < D; d++) {
        const i64 v = x[std::size_t(t) * D + d];
        s2 += v * v;
      }
      const i64 m40 =
          (s2 / D) * (i64(1) << 32) / (Q * Q) + c_.eps32;
      isq[t] = isqrt_newton(m40);
      for (int d = 0; d < D; d++)
        y[std::size_t(t) * D + d] =
            rdiv(rdiv(x[std::size_t(t) * D + d] * g[d], Q) * R16,
                 isq[t]);
    }
    return y;
  };
  const auto rms_bwd = [&](const Mat& dy, const Mat& x, const Mat& g,
                           const Mat& isq, Mat& dg) {
    Mat dx(std::size_t(T) * D);
    dg.assign(D, 0);
    Mat tv(D);
    for (int t = 0; t < T; t++) {
      i64 inner = 0;
      for (int d = 0; d < D; d++) {
        tv[d] = rdiv(g[d] * dy[std::size_t(t) * D + d], Q);
        inner += rdiv(tv[d] * x[std::size_t(t) * D + d], Q);
      }
      for (int d = 0; d < D; d++) {
        const i64 xv = x[std::size_t(t) * D + d];
        const i64 term1 = rdiv(tv[d] * R16, isq[t]);
        i64 cc = rdiv(xv * inner, i64(D) * Q);
        for (int r = 0; r < 3; r++) cc = rdiv(cc * R16, isq[t]);
        dx[std::size_t(t) * D + d] = term1 - cc;
        dg[d] += rdiv(
            rdiv(dy[std::size_t(t) * D + d] * xv, Q) * R16, isq[t]);
      }
    }
    return dx;
  };
  const auto softmax_rows = [&](const Mat& s, int rows, int C,
                                i64 sc) {
    Mat p(std::size_t(rows) * C);
    Mat e(C);
    for (int t = 0; t < rows; t++) {
      i64 m = s[std::size_t(t) * C];
      for (int cc = 1; cc < C; cc++)
        m = std::max(m, s[std::size_t(t) * C + cc]);
      i64 z = 0;
      for (int cc = 0; cc < C; cc++) {
        i64 d = s[std::size_t(t) * C + cc] - m;
        if (d < -tse_ - 1) d = -tse_ - 1;
        e[cc] = lut_exp(d);
        z += e[cc];
      }
      for (int cc = 0; cc < C; cc++)
        p[std::size_t(t) * C + cc] = rdiv(e[cc] * sc, z);
    }
    return p;
  };

  // Q-scale view of the wide weights (the matmul boundary)
  std::map<std::string, Mat> w;
  for (const char* k : KEYS) {
    w[k] = w_.at(k);
    for (auto& v : w[k]) v = rdiv(v, i64(1) << c_.shift);
  }

  // ---- forward
  Mat i1, i2, i3;
  const Mat h1 = rms_fwd(x_, w.at("g1"), i1);
  Mat q0 = mmT(h1, T, D, w.at("wq"), DH);
  Mat k0 = mmT(h1, T, D, w.at("wk"), DH);
  Mat v0 = mmT(h1, T, D, w.at("wv"), DH);
  rdiv_all(q0, Q);
  rdiv_all(k0, Q);
  rdiv_all(v0, Q);
  const Mat qr = rope(q0, false), kr = rope(k0, false);
  Mat s = mmT(qr, T, DH, kr, T);
  rdiv_all(s, scale_);
  for (int t = 0; t < T; t++)
    for (int u = t + 1; u < T; u++)
      s[std::size_t(t) * T + u] = -(i64(1) << 40);  // causal
  const Mat p = softmax_rows(s, T, T, PQ);
  Mat a = mm(p, T, T, v0, DH);
  rdiv_all(a, PQ);
  Mat pre1 = mmT(a, T, DH, w.at("wo"), D);
  rdiv_all(pre1, Q);
  Mat m1(pre1.size()), x1(pre1.size());
  for (std::size_t i = 0; i < pre1.size(); i++) {
    pre1[i] += x_[i];
    m1[i] = (pre1[i] <= CL && pre1[i] >= -CL);
    x1[i] = clampi(pre1[i], -CL, CL);
  }
  const Mat h2 = rms_fwd(x1, w.at("g2"), i2);
  Mat gp = mmT(h2, T, D, w.at("wg"), F);
  Mat u = mmT(h2, T, D, w.at("wu"), F);
  rdiv_all(gp, Q);
  rdiv_all(u, Q);
  Mat sg(gp.size()), ff(gp.size());
  for (std::size_t i = 0; i < gp.size(); i++) {
    sg[i] = lut_silu(gp[i]);
    ff[i] = rdiv(sg[i] * u[i], Q);
  }
  Mat pre2 = mmT(ff, T, F, w.at("wd"), D);
  rdiv_all(pre2, Q);
  Mat m2(pre2.size()), x2(pre2.size());
  for (std::size_t i = 0; i < pre2.size(); i++) {
    pre2[i] += x1[i];
    m2[i] = (pre2[i] <= CL && pre2[i] >= -CL);
    x2[i] = clampi(pre2[i], -CL, CL);
  }
  const Mat h3 = rms_fwd(x2, w.at("g3"), i3);
  Mat logits = mmT(h3, T, D, w.at("wh"), V);
  rdiv_all(logits, Q);

  // ---- loss + CE gradient (boosted)
  const Mat pp = softmax_rows(logits, T, V, Q);
  i64 loss = 0;
  for (int t = 0; t < T; t++)
    loss += Q - pp[std::size_t(t) * V + tgt_[t]];
  loss_ = loss;
  Mat dlogits(std::size_t(T) * V);
  for (int t = 0; t < T; t++)
    for (int vv = 0; vv < V; vv++)
      dlogits[std::size_t(t) * V + vv] =
          (pp[std::size_t(t) * V + vv] - Q * (tgt_[t] == vv)) *
          c_.gboost;

  // ---- backward (clamp masks per the R2b contract)
  std::map<std::string, Mat> G;
  G["wh"] = xty(dlogits, T, V, h3, D);
  rdiv_all(G["wh"], Q);
  Mat dh3 = mm(dlogits, T, V, w.at("wh"), D);
  rdiv_all(dh3, Q);
  Mat dx2 = rms_bwd(dh3, x2, w.at("g3"), i3, G["g3"]);
  for (std::size_t i = 0; i < dx2.size(); i++) dx2[i] *= m2[i];
  Mat df = mm(dx2, T, D, w.at("wd"), F);
  rdiv_all(df, Q);
  G["wd"] = xty(dx2, T, D, ff, F);
  rdiv_all(G["wd"], Q);
  Mat du(df.size()), dgp(df.size());
  for (std::size_t i = 0; i < df.size(); i++) {
    du[i] = rdiv(sg[i] * df[i], Q);
    dgp[i] = rdiv(rdiv(u[i] * df[i], Q) * lut_dsilu(gp[i]), Q);
  }
  Mat dh2 = mm(du, T, F, w.at("wu"), D);
  {
    const Mat t2 = mm(dgp, T, F, w.at("wg"), D);
    for (std::size_t i = 0; i < dh2.size(); i++) dh2[i] += t2[i];
  }
  rdiv_all(dh2, Q);  // one rdiv after the two-term sum
  G["wu"] = xty(du, T, F, h2, D);
  rdiv_all(G["wu"], Q);
  G["wg"] = xty(dgp, T, F, h2, D);
  rdiv_all(G["wg"], Q);
  Mat dx1 = rms_bwd(dh2, x1, w.at("g2"), i2, G["g2"]);
  for (std::size_t i = 0; i < dx1.size(); i++)
    dx1[i] = (dx1[i] + dx2[i]) * m1[i];
  Mat da = mm(dx1, T, D, w.at("wo"), DH);
  rdiv_all(da, Q);
  G["wo"] = xty(dx1, T, D, a, DH);
  rdiv_all(G["wo"], Q);
  Mat dp = mmT(da, T, DH, v0, T);
  rdiv_all(dp, Q);
  Mat dv = xty(p, T, T, da, DH);
  rdiv_all(dv, PQ);
  Mat ds(std::size_t(T) * T);
  for (int t = 0; t < T; t++) {
    i64 inner = 0;
    for (int cc = 0; cc < T; cc++)
      inner += rdiv(p[std::size_t(t) * T + cc] *
                        dp[std::size_t(t) * T + cc],
                    PQ);
    for (int cc = 0; cc < T; cc++)
      ds[std::size_t(t) * T + cc] =
          rdiv(p[std::size_t(t) * T + cc] *
                   (dp[std::size_t(t) * T + cc] - inner),
               PQ);
  }
  Mat dqr = mm(ds, T, T, kr, DH);
  rdiv_all(dqr, scale_);
  Mat dkr = xty(ds, T, T, qr, DH);
  rdiv_all(dkr, scale_);
  const Mat dq = rope(dqr, true), dk = rope(dkr, true);
  G["wq"] = xty(dq, T, DH, h1, D);
  rdiv_all(G["wq"], Q);
  G["wk"] = xty(dk, T, DH, h1, D);
  rdiv_all(G["wk"], Q);
  G["wv"] = xty(dv, T, DH, h1, D);
  rdiv_all(G["wv"], Q);
  Mat dh1 = mm(dq, T, DH, w.at("wq"), D);
  {
    const Mat t2 = mm(dk, T, DH, w.at("wk"), D);
    const Mat t3 = mm(dv, T, DH, w.at("wv"), D);
    for (std::size_t i = 0; i < dh1.size(); i++)
      dh1[i] += t2[i] + t3[i];
  }
  rdiv_all(dh1, Q);  // one rdiv after the three-term sum
  rms_bwd(dh1, x_, w.at("g1"), i1, G["g1"]);  // dx0 itself unused

  // ---- optimizer (IntAdamWQw, exact big-int bias correction)
  t_ += 1;
  big_mul(p10_, 10);
  big_mul(p9_, 9);
  big_mul(p1000_, 1000);
  big_mul(p999_, 999);
  BigV n1 = p10_, d1 = big_sub(p10_, p9_);
  BigV n2 = p1000_, d2 = big_sub(p1000_, p999_);
  while (big_gt_pow30(n1)) { big_shr1(n1); big_shr1(d1); }
  while (big_gt_pow30(n2)) { big_shr1(n2); big_shr1(d2); }
  const i64 bc1n = big_i64(n1),
            bc1d = std::max<i64>(big_i64(d1), 1);
  const i64 bc2n = big_i64(n2),
            bc2d = std::max<i64>(big_i64(d2), 1);
  i64 nz = 0, tot = 0;
  for (std::size_t j = 0; j < 11; j++) {
    Mat& wd = w_.at(KEYS[j]);
    const Mat& gk = G.at(KEYS[j]);
    for (std::size_t i = 0; i < wd.size(); i++) {
      const i64 g = rdiv(gk[i], Q * c_.gboost);  // unboost
      m_[j][i] = rdiv(B1N * m_[j][i] + (B1D - B1N) * g, B1D);
      v_[j][i] =
          rdiv(B2N * v_[j][i] + (B2D - B2N) * rdiv(g * g, Q), B2D);
      const i64 mh = rdiv(m_[j][i] * bc1n, bc1d);
      const i64 vh = rdiv(v_[j][i] * bc2n, bc2d);
      const i64 den = isqrt_newton(vh * Q) + AEPS;
      const i64 upd =
          rdiv(c_.lrn * mh * (Q << c_.shift), c_.lrd * den);
      nz += upd != 0;
      tot += 1;
      wd[i] -= upd;
      wd[i] -= rdiv(wd[i] * WDN, WDD);
    }
  }
  nz_ = double(nz) / double(tot);
  step_ += 1;
}

}  // namespace ax::nn::ib
