/** @file intbirth.cpp Integer-birth engine (see intbirth.hpp).
    Primitive layer (int_gemm / block / adamw) + the composed
    full_birth; certified by the r2b_ref.json milestone digests. */
#include <ax/nn/intbirth.hpp>
#include <ax/nn/intbirth_core.hpp>

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
  return core::gemm<i64, i64>(a, rows, K, w, N);
}
Mat int_gemm_nt(const Mat& a, int rows, int K, const Mat& w, int N) {
  return core::gemm_nt<i64, i64>(a, rows, K, w, N);
}
Mat int_gemm_xty(const Mat& x, int rows, int K, const Mat& y, int N) {
  return core::gemm_xty<i64, i64>(x, rows, K, y, N);
}
void rdiv_inplace(Mat& m, i64 d) {
  core::rdiv_inplace<i64, core::RoundHalfAway>(m, d);
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
  // ENGINE-EXACT-1 ladder: wired rungs only; an unwired precision
  // aborts here, before any step runs (refuse-if-disagree). Every
  // loop builds a block, so this one check gates the loops too.
  if (c_.precision != 9 && c_.precision != 32)
    throw std::runtime_error("intbirth: bad precision");
  // eps32 >= 1 makes rmsnorm isq >= 1 structural: rms_bwd divides
  // by isq, and m40 >= eps32 in rms_fwd is the only floor.
  if (c_.eps32 < 1)
    throw std::runtime_error("intbirth: bad eps32");
  // Frozen-grain attn scale: floor-isqrt does not commute with
  // *4^k, so the shipped-grain isqrt is computed first and then
  // re-embedded (declared convention; no-op at precision 9).
  scale_ = isqrt_round(Q * Q * i64(c_.DH)) << (c_.precision - 9);
  tab_ = core::parse_axp3(tables_bytes);
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
  const auto sh = core::shapes(c.DH, c.D, c.F, c.V);
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
  return core::softmax_rows<i64, i64, core::RoundHalfAway>(s, rows, C, scale,
                                 tab_.at("exp.tab"), tse_,
                                 c_.precision - 9);
}

Mat block::rms_fwd(const Mat& xx, const Mat& g, Mat& isq) const {
  const int T = c_.T, D = c_.D;
  if (i64(xx.size()) != i64(T) * D || i64(g.size()) != D)
    throw std::runtime_error("intbirth: rms_fwd shape");
  if (c_.precision != 9)
    return core::rms_fwd<i64, __int128, core::RoundHalfAway>(
  xx, g, isq, T, D, contract_Q(c_), c_.eps32);
  return core::rms_fwd<i64, i64, core::RoundHalfAway>(
      xx, g, isq, T, D, contract_Q(c_), c_.eps32);
}

Mat block::rms_bwd(const Mat& dy, const Mat& xx, const Mat& g,
                   const Mat& isq, Mat& dg) const {
  if (c_.precision != 9)
    return core::rms_bwd<i64, __int128, core::RoundHalfAway>(
  dy, xx, g, isq, dg, c_.T, c_.D, contract_Q(c_));
  return core::rms_bwd<i64, i64, core::RoundHalfAway>(
      dy, xx, g, isq, dg, c_.T, c_.D, contract_Q(c_));
}

core::env<i64> block::make_env() const {
  core::env<i64> e;
  e.T = c_.T; e.D = c_.D; e.DH = c_.DH; e.F = c_.F; e.V = c_.V;
  e.E = c_.n_experts;
  e.PQ = c_.pq;                                 // frozen carry
  e.CL = c_.act_clamp << (c_.precision - 9);    // operand scale
  e.Q = contract_Q(c_);
  e.scale = scale_;
  e.eps32 = c_.eps32;
  e.ts = ts_; e.tse = tse_;
  e.tcos = &tab_.at("rope.cos"); e.tsin = &tab_.at("rope.sin");
  e.sil = &tab_.at("silu.tab"); e.dsl = &tab_.at("dsilu.tab");
  e.ex = &tab_.at("exp.tab");
  e.gshift = c_.precision - 9;
  return e;
}

Mat block::attn_fwd(const std::map<std::string, Mat>& w, const Mat& x,
                    block_cache& c) const {
  if (i64(x.size()) != i64(c_.T) * c_.D)
    throw std::runtime_error("intbirth: bad x shape");
  if (c_.precision != 9)
    return core::attn_fwd<i64, __int128, core::RoundHalfAway>(
  w, x, c, make_env());
  return core::attn_fwd<i64, i64, core::RoundHalfAway>(
      w, x, c, make_env());
}

Mat block::ffn_fwd(const std::map<std::string, Mat>& w, const Mat& x1,
                   block_cache& c) const {
  (void)x1;  // residual base read from c.x1 (== x1)
  if (c_.precision != 9)
    return core::ffn_fwd<i64, __int128, core::RoundHalfAway>(
  w, c, make_env());
  return core::ffn_fwd<i64, i64, core::RoundHalfAway>(
      w, c, make_env());
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
  const int E = c_.n_experts;
  if (E < 1) throw std::runtime_error("intbirth: E < 1");
  for (const auto& k : moe_body_keys(E))
    if (!w.count(k))
      throw std::runtime_error("intbirth: missing " + k);
  if (i64(w.at("wr").size()) != i64(E) * c_.D)
    throw std::runtime_error("intbirth: bad wr shape");
  if (c_.precision != 9)
    return core::moe_body_fwd<i64, __int128, core::RoundHalfAway>(
  w, x, c, make_env());
  return core::moe_body_fwd<i64, i64, core::RoundHalfAway>(
      w, x, c, make_env());
}

std::map<std::string, Mat> block::moe_body_bwd(
    const std::map<std::string, Mat>& w, const Mat& dxin,
    const block_cache& c, Mat* dx0_out) const {
  const int E = c_.n_experts;
  if (E < 1) throw std::runtime_error("intbirth: E < 1");
  if (i64(dxin.size()) != i64(c_.T) * c_.D)
    throw std::runtime_error("intbirth: bad dxin shape");
  if (c_.precision != 9)
    return core::moe_body_bwd<i64, __int128, core::RoundHalfAway>(
  w, dxin, c, dx0_out, make_env());
  return core::moe_body_bwd<i64, i64, core::RoundHalfAway>(
      w, dxin, c, dx0_out, make_env());
}

Mat block::fwd(const std::map<std::string, Mat>& w, const Mat& x,
               block_cache& c) const {
  check_weights(w);
  const Mat x2 = body_fwd(w, x, c);
  if (c_.precision != 9)
    return core::fwd_head<i64, __int128, core::RoundHalfAway>(
  w, x2, c, make_env());
  return core::fwd_head<i64, i64, core::RoundHalfAway>(
      w, x2, c, make_env());
}

Mat block::ffn_bwd(const std::map<std::string, Mat>& w,
                   const Mat& dx2_masked, const block_cache& c,
                   std::map<std::string, Mat>& G) const {
  if (c_.precision != 9)
    return core::ffn_bwd<i64, __int128, core::RoundHalfAway>(
  w, dx2_masked, c, G, make_env());
  return core::ffn_bwd<i64, i64, core::RoundHalfAway>(
      w, dx2_masked, c, G, make_env());
}

Mat block::attn_bwd(const std::map<std::string, Mat>& w,
                    const Mat& dx1_masked, const block_cache& c,
                    std::map<std::string, Mat>& G) const {
  if (c_.precision != 9)
    return core::attn_bwd<i64, __int128, core::RoundHalfAway>(
  w, dx1_masked, c, G, make_env());
  return core::attn_bwd<i64, i64, core::RoundHalfAway>(
      w, dx1_masked, c, G, make_env());
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
  const Mat dx2in =
      c_.precision != 9
          ? core::bwd_head<i64, __int128, core::RoundHalfAway>(
                w, dlogits, c, G, make_env())
          : core::bwd_head<i64, i64, core::RoundHalfAway>(
                w, dlogits, c, G, make_env());
  auto Gb = body_bwd(w, dx2in, c, dx0_out);
  for (auto& [k, g] : Gb) G[k] = std::move(g);
  return G;
}

// ------------------------------------------------------------ adamw

adamw::adamw(int shift, i64 lrn, i64 lrd, int precision)
    : shift_(shift), precision_(precision), lrn_(lrn), lrd_(lrd) {
  if (shift < 0 || lrn < 1 || lrd < 1 ||
      (precision != 9 && precision != 32))
    throw std::runtime_error("intbirth: bad adamw params");
  p10_ = p9_ = p1000_ = p999_ = core::BigV{1};
}

void adamw::set_lr(i64 lrn, i64 lrd) {
  if (lrn < 1 || lrd < 1)
    throw std::runtime_error("intbirth: bad set_lr params");
  lrn_ = lrn;
  lrd_ = lrd;
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
  core::big_mul(p10_, 10);
  core::big_mul(p9_, 9);
  core::big_mul(p1000_, 1000);
  core::big_mul(p999_, 999);
  core::BigV n1 = p10_, d1 = core::big_sub(p10_, p9_);
  core::BigV n2 = p1000_, d2 = core::big_sub(p1000_, p999_);
  core::big_norm30(n1, d1);
  core::big_norm30(n2, d2);
  const i64 bc1n = core::big_i64(n1), bc1d = std::max<i64>(core::big_i64(d1), 1);
  const i64 bc2n = core::big_i64(n2), bc2d = std::max<i64>(core::big_i64(d2), 1);
  i64 nz = 0, tot = 0;
  for (std::size_t j = 0; j < params.size(); j++) {
    Mat& w = *params[j];
    const Mat& g = *grads[j];
    if (w.size() != g.size() || w.size() != m_[j].size())
      throw std::runtime_error("intbirth: grad shape mismatch");
    if (precision_ != 9)
      core::adamw_update<i64, __int128, core::RoundHalfAway>(
          w, g, m_[j], v_[j], bc1n, bc1d, bc2n, bc2d,
          i64{1} << precision_, shift_, lrn_, lrd_, precision_ - 9,
          nz, tot);
    else
      core::adamw_update<i64, i64, core::RoundHalfAway>(
          w, g, m_[j], v_[j], bc1n, bc1d, bc2n, bc2d,
          i64{1} << precision_, shift_, lrn_, lrd_, precision_ - 9,
          nz, tot);
  }
  nz_ = double(nz) / double(tot);
}

// ------------------------------------------------------- full_birth

namespace {
core::birth_cfg_t birth_cfg(const contract& c) {
  if (c.precision != 9 && c.precision != 32 && c.precision != 64)
    throw std::runtime_error("intbirth: bad precision");
  return {c.T,      c.D,  c.DH,        c.F,     c.V,
          c.shift,  c.precision,       c.gboost, c.pq,
          c.act_clamp, c.eps32, c.lrn, c.lrd};
}
}  // namespace

full_birth::full_birth(const std::string& tables_bytes,
                       const std::string& init_bytes,
                       const contract& c)
    : impl_([&]() -> decltype(impl_) {
        const auto bc = birth_cfg(c);
        using RH64 = core::RoundHalfAway;
        using RH128 = core::RoundHalfAway;
        if (c.precision == 9)
          return core::birth_impl<i64, i64, RH64>(tables_bytes,
                                                  init_bytes, bc);
        if (c.precision == 32)
          return core::birth_impl<i64, __int128, RH64>(
              tables_bytes, init_bytes, bc);
        return core::birth_impl<__int128, ax::core::i256, RH128>(
            tables_bytes, init_bytes, bc);
      }()) {}

void full_birth::run(int steps) {
  std::visit([&](auto& b) { b.run(steps); }, impl_);
}
void full_birth::set_lr(i64 lrn, i64 lrd) {
  std::visit([&](auto& b) { b.set_lr(lrn, lrd); }, impl_);
}
int full_birth::step_count() const {
  return std::visit([](const auto& b) { return b.step_count(); },
                    impl_);
}
i64 full_birth::last_loss() const {
  return std::visit([](const auto& b) { return b.last_loss(); },
                    impl_);
}
double full_birth::nz_last() const {
  return std::visit([](const auto& b) { return b.nz_last(); },
                    impl_);
}
std::string full_birth::mark() {
  return std::visit([](auto& b) { return b.mark(); }, impl_);
}
std::string full_birth::traj_sha() const {
  return std::visit([](const auto& b) { return b.traj_sha(); },
                    impl_);
}
std::string full_birth::weights_bytes() const {
  return std::visit(
      [](const auto& b) { return b.weights_bytes(); }, impl_);
}
std::string full_birth::weights_grain9_bytes() const {
  return std::visit(
      [](const auto& b) {
        const auto g9 = b.weights_grain9();
        return std::string(reinterpret_cast<const char*>(g9.data()),
                           g9.size() * 8);
      },
      impl_);
}

// ------------------------------------------------------ multi_birth

multi_birth::multi_birth(const std::string& tables_bytes,
                         const std::string& init_bytes,
                         const contract& c,
                         const std::string& windows_bytes)
    : blk_(tables_bytes, c), opt_(c.shift, c.lrn, c.lrd, c.precision) {
  if (c.n_blocks < 1)
    throw std::runtime_error("intbirth: n_blocks < 1");
  const auto sh = core::shapes(c.DH, c.D, c.F, c.V);
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
    for (auto& v : w_[name])
      v <<= c.shift + (c.precision - 9);  // lift to Q_w at rung grain
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
  const i64 qc = contract_Q(c);
  const Mat pp = blk_.softmax_rows(logits, T, V, qc);
  i64 loss = 0;
  for (int t = 0; t < T; t++)
    loss += qc - pp[std::size_t(t) * V + tgt_[t]];
  loss_ = loss;
  Mat dlogits(std::size_t(T) * V);
  for (int t = 0; t < T; t++)
    for (int vv = 0; vv < V; vv++)
      dlogits[std::size_t(t) * V + vv] =
          (pp[std::size_t(t) * V + vv] - qc * (tgt_[t] == vv)) *
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
    for (auto& v : g) v = rdiv(v, contract_Q(c) * c.gboost);  // unboost
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
    : blk_(tables_bytes, c), opt_(c.shift, c.lrn, c.lrd, c.precision) {
  if (c.n_blocks < 1)
    throw std::runtime_error("intbirth: n_blocks < 1");
  if (c.n_experts < 1)
    throw std::runtime_error("intbirth: n_experts < 1");
  const auto sh = core::shapes(c.DH, c.D, c.F, c.V);
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
    for (auto& v : w_[name])
      v <<= c.shift + (c.precision - 9);  // lift to Q_w at rung grain
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
  const i64 qc = contract_Q(c);
  const Mat pp = blk_.softmax_rows(logits, T, V, qc);
  i64 loss = 0;
  for (int t = 0; t < T; t++)
    loss += qc - pp[std::size_t(t) * V + tgt[t]];
  loss_ = loss;
  Mat dlogits(std::size_t(T) * V);
  for (int t = 0; t < T; t++)
    for (int vv = 0; vv < V; vv++)
      dlogits[std::size_t(t) * V + vv] =
          (pp[std::size_t(t) * V + vv] - qc * (tgt[t] == vv)) * gb;

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
