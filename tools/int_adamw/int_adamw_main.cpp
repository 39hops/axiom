// R2/R3a C++ leg: IntAdamW + the integer FFN fwd/bwd mini-birth
// (references: llmopt scratch/detbwd_r2_adamw.py, detbwd_r3_qw.py,
// detbwd_r1.py). All state int64; bias-correction rationals are
// exact big-int per step, both sides right-shifted together while
// the numerator exceeds 2^30 (strict >, the corrected house spec).
// R3a wide-accumulator contract: weights carried at Q_w = Q << S,
// rdiv(w, 1 << S) at the matmul boundary, update applied at Q_w
// via rdiv(LRN*mh*(Q<<S), LRD*den). R2 is the S=0 / lr=1/20 /
// 200-step special case of the same loop. Init + tables consumed
// as shipped bytes (r2_init.bin, AXP3) — never re-drawn.
//
// Build: c++ -O2 -std=c++17 int_adamw_main.cpp -o int_adamw
// Run:   ./int_adamw r2_init.bin          (R2, house sha 5f8dcdcc...)
//        ./int_adamw r2_init.bin --r3a    (R3a shift sweep 0/4/8/12)
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

static const i64 Q = 512, TS = 4096, EPS = 4;
static const i64 B1N = 9, B1D = 10, B2N = 999, B2D = 1000;
static const i64 WDN = 1, WDD = 100000;
static const int D = 64, F = 256, T = 32;

// ---------- sha256 (as FX-V2/V3) ----------
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

// ---------- minimal big-uint (for the exact bias-correction) ----------
struct Big {  // little-endian u32 limbs, always trimmed
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
  bool gt_pow30() const {  // strictly greater than 2^30 (house cap)
    const int b = bits();
    if (b != 31) return b > 31;
    return !(d.size() == 1 && d[0] == 0x40000000u);
  }
  i64 to_i64() const {  // caller guarantees <= 63 bits
    u64 v = 0;
    for (size_t i = d.size(); i-- > 0;) v = (v << 32) | d[i];
    return i64(v);
  }
  static Big sub(const Big& a, const Big& b) {  // a >= b
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

// ---------- integer kernels (detbwd_r1 twins) ----------
static inline i64 rdiv(i64 x, i64 d) {
  i64 ax = x < 0 ? -x : x;
  i64 r = (ax + d / 2) / d;
  return x < 0 ? -r : r;
}
static i64 isqrt_newton(i64 x) {
  if (x <= 0) return 0;
  i64 r = x < 1 ? 1 : x;
  for (int i = 0; i < 40; i++) {
    r = (r + x / r) / 2;
    if (r < 1) r = 1;
  }
  if (r * r > x) r -= 1;
  if ((r + 1) * (r + 1) <= x) r += 1;
  return r;
}

using Mat = std::vector<i64>;  // row-major

// [T,K] x W[N,K] -> [T,N] exact int64 sum-reduce
static Mat int_mm(const Mat& a, int rows, int K, const Mat& w, int N) {
  Mat y((size_t)rows * N);
  for (int t = 0; t < rows; t++)
    for (int n = 0; n < N; n++) {
      const i64* ar = &a[(size_t)t * K];
      const i64* wr = &w[(size_t)n * K];
      i64 acc = 0;
      for (int k = 0; k < K; k++) acc += ar[k] * wr[k];
      y[(size_t)t * N + n] = acc;
    }
  return y;
}
// [T,K] x W^T where W[K,N] row-major -> [T,N]
static Mat int_mm_tr(const Mat& a, int rows, int K, const Mat& w, int N) {
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
// A^T[K,rows] x B^T[N,rows] -> [K? ] — dW = X^T Y with X[rows,K],
// Y[rows,N]: out[k][n] = sum_t X[t][k]*Y[t][n]  (shape [K,N])
static Mat outer_acc(const Mat& x, int rows, int K, const Mat& y, int N) {
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

struct Tables {
  Mat silu, dsilu;  // domain [-TS, TS] in x*Q units
};
static inline i64 lut_silu(const Tables& tb, i64 x) {
  if (x > TS) return x;      // silu(x) -> x above TS
  if (x < -TS) return 0;
  return tb.silu[x + TS];
}
static inline i64 lut_dsilu(const Tables& tb, i64 x) {
  if (x > TS) return Q;      // dsilu -> 1 (Q) above TS
  if (x < -TS) return 0;
  return tb.dsilu[x + TS];
}

struct Cache {
  Mat g, u, s, p;
};
static Mat ffn_fwd(const Mat& xq, const Mat& wg, const Mat& wu,
                   const Mat& wd, const Tables& tb, Cache& c) {
  c.g = int_mm(xq, T, D, wg, F);
  c.u = int_mm(xq, T, D, wu, F);
  for (auto& v : c.g) v = rdiv(v, Q);
  for (auto& v : c.u) v = rdiv(v, Q);
  c.s.resize(c.g.size());
  c.p.resize(c.g.size());
  for (size_t i = 0; i < c.g.size(); i++) {
    c.s[i] = lut_silu(tb, c.g[i]);
    c.p[i] = rdiv(c.s[i] * c.u[i], Q);
  }
  Mat y = int_mm_tr(c.p, T, F, wd, D);  // p [T,F] x wd^T (wd [F,D])
  for (auto& v : y) v = rdiv(v, Q);
  return y;
}
static void ffn_bwd(const Mat& dy, const Mat& xq, const Mat& wg,
                    const Mat& wu, const Mat& wd, const Cache& c,
                    const Tables& tb, Mat& dwg, Mat& dwu, Mat& dwd) {
  Mat dp = int_mm(dy, T, D, wd, F);  // dy [T,D] x wd [F,D] -> [T,F]
  for (auto& v : dp) v = rdiv(v, Q);
  dwd = outer_acc(c.p, T, F, dy, D);  // [F,D]
  Mat du(dp.size()), dg(dp.size());
  for (size_t i = 0; i < dp.size(); i++) {
    du[i] = rdiv(dp[i] * c.s[i], Q);
    i64 ds = rdiv(dp[i] * c.u[i], Q);
    dg[i] = rdiv(ds * lut_dsilu(tb, c.g[i]), Q);
  }
  dwg = outer_acc(dg, T, F, xq, D);  // dg^T x xq -> [F,D]
  dwu = outer_acc(du, T, F, xq, D);
}

// ---------- IntAdamW ----------
struct IntAdamW {
  std::vector<Mat*> p;
  std::vector<Mat> m, v;
  int t = 0;
  int shift;      // weights at Q_w = Q << shift (R3a; R2 = 0)
  i64 lrn, lrd;
  i64 nz = 0, tot = 0;  // last step's nonzero-update stats
  Big p10, p9, p1000, p999;  // running B1D^t etc.
  IntAdamW(std::vector<Mat*> params, int shift_, i64 lrn_, i64 lrd_)
      : p(std::move(params)), shift(shift_), lrn(lrn_), lrd(lrd_) {
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
    i64 bc1n = n1.to_i64(), bc1d = d1.to_i64();
    i64 bc2n = n2.to_i64(), bc2d = d2.to_i64();
    if (bc1d < 1) bc1d = 1;
    if (bc2d < 1) bc2d = 1;
    nz = tot = 0;
    for (size_t j = 0; j < p.size(); j++) {
      Mat& w = *p[j];
      for (size_t i = 0; i < w.size(); i++) {
        const i64 g = grads[j][i];
        m[j][i] = rdiv(B1N * m[j][i] + (B1D - B1N) * g, B1D);
        v[j][i] = rdiv(B2N * v[j][i] + (B2D - B2N) * rdiv(g * g, Q),
                       B2D);
        const i64 mh = rdiv(m[j][i] * bc1n, bc1d);
        const i64 vh = rdiv(v[j][i] * bc2n, bc2d);
        const i64 den = isqrt_newton(vh * Q) + EPS;
        const i64 upd = rdiv(lrn * mh * (Q << shift), lrd * den);
        nz += upd != 0;
        tot += 1;
        w[i] -= upd;
        w[i] -= rdiv(w[i] * WDN, WDD);
      }
    }
  }
};

// ---------- AXP3 loader (FX-V2 format) ----------
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

static void run(const std::map<std::string, Mat>& tt, int shift,
                i64 lrn, i64 lrd, int steps, int hash_every,
                bool r2_prints) {
  Tables tb{tt.at("silu.tab"), tt.at("dsilu.tab")};
  const Mat& xq = tt.at("xq");
  Cache c0;
  Mat tgt = ffn_fwd(xq, tt.at("tw0"), tt.at("tw1"), tt.at("tw2"),
                    tb, c0);
  Mat st[3] = {tt.at("s0"), tt.at("s1"), tt.at("s2")};
  for (auto& w : st)
    for (auto& v : w) v <<= shift;  // carry weights at Q_w
  IntAdamW opt({&st[0], &st[1], &st[2]}, shift, lrn, lrd);
  Sha256 h;
  double nz_first = -1, nz_last = 0;
  i64 loss0 = 0, loss_mid = 0, loss_last = 0;
  for (int step = 1; step <= steps; step++) {
    Mat wq[3];
    for (int j = 0; j < 3; j++) {
      wq[j] = st[j];
      for (auto& v : wq[j]) v = rdiv(v, i64(1) << shift);
    }
    Cache c;
    Mat y = ffn_fwd(xq, wq[0], wq[1], wq[2], tb, c);
    Mat dy(y.size());
    i64 loss = 0;
    for (size_t i = 0; i < y.size(); i++) {
      dy[i] = y[i] - tgt[i];
      loss += dy[i] * dy[i];
    }
    if (step == 1) loss0 = loss;
    if (step == steps / 2 + 1) loss_mid = loss;  // losses[len//2]
    loss_last = loss;
    std::vector<Mat> grads(3);
    ffn_bwd(dy, xq, wq[0], wq[1], wq[2], c, tb,
            grads[0], grads[1], grads[2]);
    for (auto& g : grads)
      for (auto& v : g) v = rdiv(v, Q);  // Q^2 -> Q (loss boundary)
    opt.step(grads);
    if (nz_first < 0) nz_first = double(opt.nz) / double(opt.tot);
    nz_last = double(opt.nz) / double(opt.tot);
    if (step % hash_every == 0) {
      for (const auto& w : st) h.update(w.data(), w.size() * 8);
      if (r2_prints) {
        Sha256 peek = h;  // running digest, as the reference prints
        printf("[r2-cpp] step %d loss %lld traj-sha %.16s\n", step,
               (long long)loss, peek.hex().c_str());
      }
    }
  }
  if (r2_prints) {
    printf("[r2-cpp] loss %lld -> %lld -> %lld monotone-ish: %s\n",
           (long long)loss0, (long long)loss_mid, (long long)loss_last,
           loss_last < loss_mid && loss_mid < loss0 ? "true" : "false");
    printf("[r2-cpp] FINAL trajectory sha %s\n", h.hex().c_str());
  } else {
    printf("[r3a-cpp] SHIFT=%2d loss %.3e -> %.3e -> %.3e  "
           "nz-upd first %.3f last %.3f  sha %s\n", shift,
           double(loss0), double(loss_mid), double(loss_last),
           nz_first, nz_last, h.hex().c_str());
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s r2_init.bin [--r3a]\n", argv[0]);
    return 1;
  }
  const auto tt = load_axp3(argv[1]);
  if (argc > 2 && std::string(argv[2]) == "--r3a") {
    printf("[r3a-cpp] lr 1/1000, 400 steps\n");
    for (int s : {0, 4, 8, 12}) run(tt, s, 1, 1000, 400, 100, false);
  } else {
    run(tt, 0, 1, 20, 200, 50, true);
  }
  return 0;
}
