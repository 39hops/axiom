#pragma once
/** @file intbirth_core.hpp ENGINE-EXACT-1: the intbirth arithmetic
    core, templated on <Op, Acc, Round> (operand scalar, accumulator,
    rounding policy). The shipped engine is the <i64, i64,
    RoundHalfAway<i64>> instantiation — the free functions in
    intbirth.cpp delegate here and are digest-gated bit-identical.
    Bodies are verbatim moves from intbirth.cpp; the digest drivers
    in tools/int_adamw/ are the arbiter of every move.

    Rounding placement is part of the engine contract and lives at
    the CALL sites, unchanged; this header only generalizes widths
    and the rounding grain. Spec:
    docs/specs/2026-08-08-engine-exact-ladder.md */
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ax::nn::ib::core {

/** Forward activations/masks the backward needs (templated on the
    operand scalar; the shipped block_cache is cache_t<i64>). */
template <class Op>
struct cache_t {
  using V = std::vector<Op>;
  // i1/i2/i3 are rmsnorm isq values: fixed 2^32 scale at every rung
  // (see rms_fwd), so they stay i64 for any Op.
  V x, h1, q0, k0, v0, qr, kr, p, a, m1, x1, h2, gp, u, sg,
      f, m2, x2, h3;
  std::vector<std::int64_t> i1, i2, i3;
  // gravmoe MoE fields: router logits/probs, top-1 choice + prob,
  // and the SELECTED expert's FFN intermediates laid out [T,*]
  V r, pr, top_p, egp, eu, esg, ef, eout;
  std::vector<int> top;
};

/** Everything a core block function needs beyond its operands:
    dims, frozen carries (i64 by the convention pin), rung-scaled
    constants (Op), and the shipped tables (i64 at shipped scale).
    Built once per call by the delegating shipped class. */
template <class Op>
struct env {
  int T, D, DH, F, V, E;
  std::int64_t PQ;    // attention-prob carry (frozen)
  Op CL;              // act_clamp at rung scale
  Op Q;               // operand grain 2^precision
  Op scale;           // attn scale: isqrt_round(Q9^2*DH) << gshift
  std::int64_t eps32;  // rmsnorm eps at fixed 2^32 scale (no shift)
  std::int64_t ts, tse;
  const std::vector<std::int64_t>*tcos, *tsin, *sil, *dsl, *ex;
  int gshift;
};

struct Shape {
  int r, c;
};

// ---- big-uint limbs (little-endian u32) for the bias correction
using BigV = std::vector<std::uint32_t>;
inline void big_trim(BigV& d) {
  while (d.size() > 1 && d.back() == 0) d.pop_back();
}
inline void big_mul(BigV& d, std::uint64_t k) {
  std::uint64_t carry = 0;
  for (auto& x : d) {
    const std::uint64_t p = std::uint64_t(x) * k + carry;
    x = std::uint32_t(p);
    carry = p >> 32;
  }
  while (carry) { d.push_back(std::uint32_t(carry)); carry >>= 32; }
}
inline int big_bits(const BigV& d) {
  std::uint32_t top = d.back();
  int b = 0;
  while (top) { b++; top >>= 1; }
  return int(d.size() - 1) * 32 + b;
}
inline void big_shr1(BigV& d) {
  for (std::size_t i = 0; i < d.size(); i++) {
    const std::uint32_t lo = (i + 1 < d.size()) ? (d[i + 1] & 1) : 0;
    d[i] = (d[i] >> 1) | (lo << 31);
  }
  big_trim(d);
}
inline void big_shr(BigV& d, int k) {  // floor shift right by k bits
  if (k <= 0) return;
  const int limb = k / 32, bit = k % 32;
  const std::size_t n = d.size();
  for (std::size_t i = 0; i < n; i++) {
    const std::size_t s = i + std::size_t(limb);
    std::uint64_t v = (s < n) ? (d[s] >> bit) : 0;
    if (bit && s + 1 < n) v |= std::uint64_t(d[s + 1]) << (32 - bit);
    d[i] = std::uint32_t(v);
  }
  big_trim(d);
}
inline bool big_gt_pow30(const BigV& d) {  // strictly greater than 2^30
  const int b = big_bits(d);
  if (b != 31) return b > 31;
  return !(d.size() == 1 && d[0] == 0x40000000u);
}
inline std::int64_t big_i64(const BigV& d) {
  std::uint64_t v = 0;
  for (std::size_t i = d.size(); i-- > 0;) v = (v << 32) | d[i];
  return std::int64_t(v);
}
/** bias-correction normalization: floor-shift n (and d in lockstep)
    right until n <= 2^30. One multi-limb shift then at most a step
    or two of the strict-compare loop — bit-identical to the former
    one-bit-at-a-time loop (floor shifts compose:
    floor(floor(x/2^a)/2^b) == floor(x/2^(a+b))), which cost
    O(bits^2) PER STEP once beta^t grew (~10 bits/step for the v
    moment): the ENGINE-SCALE-1 grid measured 2.7 -> 8.8 -> ~102
    ms/step at s1000/s4000/s16000 from exactly this loop. */
inline void big_norm30(BigV& n, BigV& d) {
  const int k = big_bits(n) - 31;
  if (k > 0) { big_shr(n, k); big_shr(d, k); }
  while (big_gt_pow30(n)) { big_shr1(n); big_shr1(d); }
}
inline BigV big_sub(const BigV& a, const BigV& b) {  // a >= b
  BigV r(a.size(), 0);
  std::int64_t borrow = 0;
  for (std::size_t i = 0; i < a.size(); i++) {
    const std::int64_t x = std::int64_t(a[i]) - (i < b.size() ? std::int64_t(b[i]) : 0) - borrow;
    borrow = x < 0;
    r[i] = std::uint32_t(x + (borrow << 32));
  }
  big_trim(r);
  return r;
}

inline std::map<std::string, std::vector<std::int64_t>> parse_axp3(const std::string& b) {
  const auto need = [&](std::size_t off, std::size_t n) {
    if (off + n > b.size())
      throw std::runtime_error("intbirth: truncated AXP3");
  };
  need(0, 8);
  if (std::memcmp(b.data(), "AXP3", 4) != 0)
    throw std::runtime_error("intbirth: bad AXP3 magic");
  std::uint32_t count;
  std::memcpy(&count, b.data() + 4, 4);
  std::size_t off = 8;
  std::map<std::string, std::vector<std::int64_t>> t;
  for (std::uint32_t i = 0; i < count; i++) {
    need(off, 2);
    std::uint16_t nl;
    std::memcpy(&nl, b.data() + off, 2);
    off += 2;
    need(off, nl + std::size_t(1));
    std::string name(b.data() + off, nl);
    off += nl;
    const std::uint8_t nd = std::uint8_t(b[off++]);
    std::uint64_t numel = 1;
    need(off, std::size_t(nd) * 8);
    for (int k = 0; k < nd; k++) {
      std::uint64_t dd;
      std::memcpy(&dd, b.data() + off, 8);
      off += 8;
      numel *= dd;
    }
    need(off, numel * 8);
    std::vector<std::int64_t> m(numel);
    std::memcpy(m.data(), b.data() + off, numel * 8);
    off += numel * 8;
    t[name] = std::move(m);
  }
  return t;
}

/** KEYS-order tensor shapes from the four dims (contract-free so
    the core header stays independent of intbirth.hpp). */
inline std::map<std::string, Shape> shapes(int DH, int D, int F,
                                           int V) {
  return {{"wq", {DH, D}}, {"wk", {DH, D}}, {"wv", {DH, D}},
          {"wo", {D, DH}}, {"wg", {F, D}},  {"wu", {F, D}},
          {"wd", {D, F}},  {"wh", {V, D}},  {"g1", {D, 1}},
          {"g2", {D, 1}},  {"g3", {D, 1}}};
}


template <class Op>
inline Op clampi(Op x, Op lo, Op hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

/** Round-half-away division at grain d (the program-wide rdiv),
    plus the frozen-grain seam helpers (spec §Convention pin):
    transcendental sites truncate rung scale -> shipped scale,
    apply the shipped convention, and shift back. Both are no-ops
    at gshift == 0 (the Q9 rung). */
struct RoundHalfAway {
  /** verbatim shipped rdiv semantics, at the WIDTH OF ITS ARGUMENT
      — division of a wide accumulator must never pre-narrow (the
      Q64 build caught exactly that; i256 has no implicit narrowing,
      the builtins silently did). */
  template <class T>
  static T div(T x, T d) {
    const T ax = x < T(0) ? -x : x;
    const T r = (ax + d / T(2)) / d;
    return x < T(0) ? -r : r;
  }
  /** Rung scale -> shipped scale: FLOOR (arithmetic shift toward
      -inf), not round-half-away and not toward-zero truncation —
      the frozen-grain seam is declared as exactly this shift. */
  template <class T>
  static T to_grain(T x, int gshift) {
    return gshift ? (x >> gshift) : x;
  }
  template <class T>
  static T from_grain(T x, int gshift) {  // exact re-embed
    return gshift ? (x << gshift) : x;
  }
  /** the engine's plain truncating division sites (isqrt-input
      prep in rmsnorm); the exact policy divides exactly instead —
      the subsequent declared floor is the only quantization. */
  template <class T>
  static T div_trunc(T a, T b) {
    return a / b;
  }
};

/** Acc -> Op, loud on overflow (never wrap; refuse-if-disagree). */
template <class Op, class Acc>
inline Op narrow(Acc v) {
  if constexpr (sizeof(Acc) > sizeof(Op)) {
    if (v > Acc(std::numeric_limits<Op>::max()) ||
        v < Acc(std::numeric_limits<Op>::min()))
      throw std::runtime_error("intbirth: accumulator narrow overflow");
  }
  return Op(v);
}

/** a[rows,K] @ w[N,K]^T -> [rows,N], Acc sum-reduce (caller rounds).
    Narrowing back to Op throws rather than wraps when Acc is wider. */
template <class Op, class Acc>
std::vector<Op> gemm(const std::vector<Op>& a, int rows, int K,
                     const std::vector<Op>& w, int N) {
  std::vector<Op> y(std::size_t(rows) * N);
  for (int t = 0; t < rows; t++)
    for (int n = 0; n < N; n++) {
      Acc acc = 0;
      const Op* ar = &a[std::size_t(t) * K];
      const Op* wr = &w[std::size_t(n) * K];
      for (int k = 0; k < K; k++) acc += Acc(ar[k]) * Acc(wr[k]);
      y[std::size_t(t) * N + n] = narrow<Op, Acc>(acc);
    }
  return y;
}

/** a[rows,K] @ w[K,N] -> [rows,N]. */
template <class Op, class Acc>
std::vector<Op> gemm_nt(const std::vector<Op>& a, int rows, int K,
                        const std::vector<Op>& w, int N) {
  std::vector<Op> y(std::size_t(rows) * N);
  for (int t = 0; t < rows; t++)
    for (int n = 0; n < N; n++) {
      Acc acc = 0;
      for (int k = 0; k < K; k++)
        acc += Acc(a[std::size_t(t) * K + k]) *
               Acc(w[std::size_t(k) * N + n]);
      y[std::size_t(t) * N + n] = narrow<Op, Acc>(acc);
    }
  return y;
}

/** x[rows,K]^T @ y[rows,N] -> [K,N] (the dW outer form). */
template <class Op, class Acc>
std::vector<Op> gemm_xty(const std::vector<Op>& x, int rows, int K,
                         const std::vector<Op>& y, int N) {
  std::vector<Acc> o(std::size_t(K) * N, 0);
  for (int t = 0; t < rows; t++)
    for (int k = 0; k < K; k++) {
      const Op xv = x[std::size_t(t) * K + k];
      if (xv == Op(0)) continue;
      for (int n = 0; n < N; n++)
        o[std::size_t(k) * N + n] +=
            Acc(xv) * Acc(y[std::size_t(t) * N + n]);
    }
  std::vector<Op> r(o.size());
  for (std::size_t i = 0; i < o.size(); i++) r[i] = narrow<Op, Acc>(o[i]);
  return r;
}

template <class Op, class Round>
void rdiv_inplace(std::vector<Op>& m, Op d) {
  for (auto& v : m) v = Round::div(v, d);
}

// ---- Acc-form gemms: the raw wide sums, for callers that must sum
// several gemm results (or fold a divisor) before the ONE rounding
// the placement contract allows. finalize_rdiv is that rounding.

template <class Op, class Acc>
std::vector<Acc> gemm_acc(const std::vector<Op>& a, int rows, int K,
                          const std::vector<Op>& w, int N) {
  std::vector<Acc> y(std::size_t(rows) * N);
  for (int t = 0; t < rows; t++)
    for (int n = 0; n < N; n++) {
      Acc acc = 0;
      const Op* ar = &a[std::size_t(t) * K];
      const Op* wr = &w[std::size_t(n) * K];
      for (int k = 0; k < K; k++) acc += Acc(ar[k]) * Acc(wr[k]);
      y[std::size_t(t) * N + n] = acc;
    }
  return y;
}

template <class Op, class Acc>
std::vector<Acc> gemm_nt_acc(const std::vector<Op>& a, int rows,
                             int K, const std::vector<Op>& w, int N) {
  std::vector<Acc> y(std::size_t(rows) * N);
  for (int t = 0; t < rows; t++)
    for (int n = 0; n < N; n++) {
      Acc acc = 0;
      for (int k = 0; k < K; k++)
        acc += Acc(a[std::size_t(t) * K + k]) *
               Acc(w[std::size_t(k) * N + n]);
      y[std::size_t(t) * N + n] = acc;
    }
  return y;
}

template <class Op, class Acc>
std::vector<Acc> gemm_xty_acc(const std::vector<Op>& x, int rows,
                              int K, const std::vector<Op>& y, int N) {
  std::vector<Acc> o(std::size_t(K) * N, 0);
  for (int t = 0; t < rows; t++)
    for (int k = 0; k < K; k++) {
      const Op xv = x[std::size_t(t) * K + k];
      if (xv == Op(0)) continue;
      for (int n = 0; n < N; n++)
        o[std::size_t(k) * N + n] +=
            Acc(xv) * Acc(y[std::size_t(t) * N + n]);
    }
  return o;
}

/** The single post-sum rounding: Round::div each wide value by d,
    narrow to Op. Placement identical to gemm-then-rdiv_inplace. */
template <class Op, class Acc, class Round>
std::vector<Op> finalize_rdiv(const std::vector<Acc>& a, Acc d) {
  std::vector<Op> r(a.size());
  for (std::size_t i = 0; i < a.size(); i++)
    r[i] = narrow<Op, Acc>(Round::div(a[i], d));
  return r;
}

/** rdiv of a two-Op product, accumulated wide: Round::div(a*b, d)
    with the product carried in Acc and the quotient narrowed. The
    ubiquitous engine idiom `rdiv(x * y, Q)` generalized. */
template <class Op, class Acc, class Round>
inline Op mul_rdiv(Op a, Op b, Op d) {
  return narrow<Op, Acc>(Round::div(Acc(a) * Acc(b), Acc(d)));
}

/** Exact integer sqrt pair (moved verbatim from intbirth.cpp; the
    isqrt convention is frozen-grain, i64 by construction — its
    inputs are normalized to fixed 2^32 scale at every rung). */
inline std::int64_t isqrt_newton(std::int64_t x) {
  if (x <= 0) return 0;
  std::int64_t r = x;
  for (int i = 0; i < 40; i++) {
    r = (r + x / r) / 2;
    if (r < 1) r = 1;
  }
  if (r * r > x) r -= 1;
  if ((r + 1) * (r + 1) <= x) r += 1;
  return r;
}
/** round(sqrt(n)) exactly (ties impossible for integer n). */
inline std::int64_t isqrt_round(std::int64_t n) {
  const std::int64_t r = isqrt_newton(n);
  return (n - r * r > (r + 1) * (r + 1) - n) ? r + 1 : r;
}

/** Integer row softmax at `scale` units via the shipped exp table
    (the r1b construction). Frozen-grain seam: the table INDEX is
    the max-shifted logit FLOORED to the shipped grain (to_grain);
    e and z live at shipped/frozen table scale (i64) at every rung —
    the freeze cannot be violated invisibly. `scale` is the caller's
    carry (frozen PQ class, or the operand grain for the loss path);
    the e*scale product runs wide in Acc. Row sums of the output are
    NOT exactly `scale` (per-element rounding, no residual
    assignment) — nothing downstream may assume it. */
template <class Op, class Acc, class Round>
std::vector<Op> softmax_rows(const std::vector<Op>& s, int rows,
                             int C, Op scale,
                             const std::vector<std::int64_t>& ex,
                             std::int64_t tse, int gshift) {
  using i64 = std::int64_t;
  std::vector<Op> p(std::size_t(rows) * C);
  std::vector<i64> e(C);
  for (int t = 0; t < rows; t++) {
    Op m = s[std::size_t(t) * C];
    for (int cc = 1; cc < C; cc++)
      m = std::max(m, s[std::size_t(t) * C + cc]);
    i64 z = 0;
    for (int cc = 0; cc < C; cc++) {
      i64 d = i64(Round::to_grain(  // floor to shipped
          s[std::size_t(t) * C + cc] - m, gshift));
      if (d < -tse - 1) d = -tse - 1;
      e[cc] = d < -tse ? 0 : ex[d + tse];
      z += e[cc];
    }
    for (int cc = 0; cc < C; cc++)
      p[std::size_t(t) * C + cc] = narrow<Op, Acc>(
          Round::div(Acc(e[cc]) * Acc(scale), Acc(z)));
  }
  return p;
}

/** rmsnorm forward over [T, D]: isq stays i64 at fixed 2^32 scale
    at every rung (the /(Q*Q) normalization freezes it); R16 is a
    frozen carry. */
template <class Op, class Acc, class Round>
std::vector<Op> rms_fwd(const std::vector<Op>& xx,
                        const std::vector<Op>& g,
                        std::vector<std::int64_t>& isq, int T, int D,
                        Op Q, std::int64_t eps32) {
  using i64 = std::int64_t;
  constexpr i64 R16 = i64{1} << 16;
  std::vector<Op> y(std::size_t(T) * D);
  isq.assign(T, 0);
  for (int t = 0; t < T; t++) {
    Acc s2 = 0;
    for (int d = 0; d < D; d++) {
      const Op v = xx[std::size_t(t) * D + d];
      s2 += Acc(v) * Acc(v);
    }
    const i64 m40 =
        narrow<i64, Acc>(Round::div_trunc(
            Round::div_trunc(s2, Acc(D)) * (Acc(1) << 32),
            Acc(Q) * Acc(Q))) +
        eps32;
    isq[t] = isqrt_newton(m40);
    for (int d = 0; d < D; d++)
      y[std::size_t(t) * D + d] = narrow<Op, Acc>(Round::div(
          Acc(mul_rdiv<Op, Acc, Round>(xx[std::size_t(t) * D + d],
                                       g[d], Q)) *
              R16,
          Acc(isq[t])));
  }
  return y;
}

/** rmsnorm backward over [T, D] (verbatim move; same scale notes
    as rms_fwd). */
template <class Op, class Acc, class Round>
std::vector<Op> rms_bwd(const std::vector<Op>& dy,
                        const std::vector<Op>& xx,
                        const std::vector<Op>& g,
                        const std::vector<std::int64_t>& isq,
                        std::vector<Op>& dg, int T, int D, Op Q) {
  using i64 = std::int64_t;
  constexpr i64 R16 = i64{1} << 16;
  std::vector<Op> dx(std::size_t(T) * D);
  dg.assign(D, 0);
  std::vector<Op> tv(D);
  for (int t = 0; t < T; t++) {
    Acc inner = 0;
    for (int d = 0; d < D; d++) {
      tv[d] = mul_rdiv<Op, Acc, Round>(g[d],
                                       dy[std::size_t(t) * D + d], Q);
      inner += Acc(mul_rdiv<Op, Acc, Round>(
          tv[d], xx[std::size_t(t) * D + d], Q));
    }
    for (int d = 0; d < D; d++) {
      const Op xv = xx[std::size_t(t) * D + d];
      const Op term1 = narrow<Op, Acc>(
          Round::div(Acc(tv[d]) * R16, Acc(isq[t])));
      Op cc = narrow<Op, Acc>(
          Round::div(Acc(xv) * inner, Acc(D) * Acc(Q)));
      for (int r = 0; r < 3; r++)
        cc = narrow<Op, Acc>(
            Round::div(Acc(cc) * R16, Acc(isq[t])));
      dx[std::size_t(t) * D + d] = term1 - cc;
      dg[d] += narrow<Op, Acc>(Round::div(
          Acc(mul_rdiv<Op, Acc, Round>(dy[std::size_t(t) * D + d],
                                       xv, Q)) *
              R16,
          Acc(isq[t])));
    }
  }
  return dx;
}

// ---- attention (verbatim moves from intbirth.cpp; the RoPE and
// exp tables stay i64 at shipped scale — Round policy handles the
// grain seams; RS is a frozen carry).

/** RoPE rotate fwd/bwd over [T, DH]; sign=+1 fwd, -1 bwd. */
template <class Op, class Acc, class Round>
std::vector<Op> rope_apply(const std::vector<Op>& v, int T, int DH,
                           const std::vector<std::int64_t>& tcos,
                           const std::vector<std::int64_t>& tsin,
                           int sign) {
  constexpr std::int64_t RS = std::int64_t{1} << 14;
  const int half = DH / 2;
  std::vector<Op> y(std::size_t(T) * DH);
  for (int t = 0; t < T; t++)
    for (int i = 0; i < half; i++) {
      const Acc co = Acc(tcos[std::size_t(t) * half + i]);
      const Acc si = Acc(tsin[std::size_t(t) * half + i]) * sign;
      const Acc a = Acc(v[std::size_t(t) * DH + i]);
      const Acc b = Acc(v[std::size_t(t) * DH + half + i]);
      y[std::size_t(t) * DH + i] =
          narrow<Op, Acc>(Round::div(a * co - b * si, Acc(RS)));
      y[std::size_t(t) * DH + half + i] =
          narrow<Op, Acc>(Round::div(a * si + b * co, Acc(RS)));
    }
  return y;
}

template <class Op, class Acc, class Round>
std::vector<Op> attn_fwd(const std::map<std::string, std::vector<Op>>& w,
                         const std::vector<Op>& x, cache_t<Op>& c,
                         const env<Op>& e) {
  const int T = e.T, D = e.D, DH = e.DH;
  c.x = x;
  c.h1 = rms_fwd<Op, Acc, Round>(x, w.at("g1"), c.i1, T, D, e.Q,
                                 e.eps32);
  c.q0 = finalize_rdiv<Op, Acc, Round>(
      gemm_acc<Op, Acc>(c.h1, T, D, w.at("wq"), DH), Acc(e.Q));
  c.k0 = finalize_rdiv<Op, Acc, Round>(
      gemm_acc<Op, Acc>(c.h1, T, D, w.at("wk"), DH), Acc(e.Q));
  c.v0 = finalize_rdiv<Op, Acc, Round>(
      gemm_acc<Op, Acc>(c.h1, T, D, w.at("wv"), DH), Acc(e.Q));
  c.qr = rope_apply<Op, Acc, Round>(c.q0, T, DH, *e.tcos, *e.tsin, 1);
  c.kr = rope_apply<Op, Acc, Round>(c.k0, T, DH, *e.tcos, *e.tsin, 1);
  std::vector<Op> s = finalize_rdiv<Op, Acc, Round>(
      gemm_acc<Op, Acc>(c.qr, T, DH, c.kr, T), Acc(e.scale));
  // Causal floor: -(2^40) re-embedded to rung scale, CAPPED at the
  // operand width (40 + 23 would overflow i64 at Q32). The cap is
  // semantics-preserving: any value below every real logit floors
  // the softmax index to e = 0 identically. Declared convention.
  const int fbits =
      std::min(40 + e.gshift, int(sizeof(Op)) * 8 - 2);
  const Op floor_v = -(Op(1) << fbits);
  for (int t = 0; t < T; t++)
    for (int u = t + 1; u < T; u++)
      s[std::size_t(t) * T + u] = floor_v;  // causal
  c.p = softmax_rows<Op, Acc, Round>(s, T, T, Op(e.PQ), *e.ex,
                                     e.tse, e.gshift);
  c.a = finalize_rdiv<Op, Acc, Round>(
      gemm_nt_acc<Op, Acc>(c.p, T, T, c.v0, DH), Acc(e.PQ));
  std::vector<Op> pre1 = finalize_rdiv<Op, Acc, Round>(
      gemm_acc<Op, Acc>(c.a, T, DH, w.at("wo"), D), Acc(e.Q));
  c.m1.assign(pre1.size(), 0);
  c.x1.assign(pre1.size(), 0);
  for (std::size_t i = 0; i < pre1.size(); i++) {
    pre1[i] += x[i];
    c.m1[i] = (pre1[i] <= e.CL && pre1[i] >= -e.CL);
    c.x1[i] = clampi<Op>(pre1[i], -e.CL, e.CL);
  }
  return c.x1;
}

template <class Op, class Acc, class Round>
std::vector<Op> attn_bwd(const std::map<std::string, std::vector<Op>>& w,
                         const std::vector<Op>& dx1_masked,
                         const cache_t<Op>& c,
                         std::map<std::string, std::vector<Op>>& G,
                         const env<Op>& e) {
  using Vec = std::vector<Op>;
  const int T = e.T, D = e.D, DH = e.DH;
  const Op PQ = Op(e.PQ);
  const Vec& dx1 = dx1_masked;
  Vec da = finalize_rdiv<Op, Acc, Round>(
      gemm_nt_acc<Op, Acc>(dx1, T, D, w.at("wo"), DH), Acc(e.Q));
  G["wo"] = finalize_rdiv<Op, Acc, Round>(
      gemm_xty_acc<Op, Acc>(dx1, T, D, c.a, DH), Acc(e.Q));
  Vec dp = finalize_rdiv<Op, Acc, Round>(
      gemm_acc<Op, Acc>(da, T, DH, c.v0, T), Acc(e.Q));
  Vec dv = finalize_rdiv<Op, Acc, Round>(
      gemm_xty_acc<Op, Acc>(c.p, T, T, da, DH), Acc(PQ));
  Vec ds(std::size_t(T) * T);
  for (int t = 0; t < T; t++) {
    Acc inner = 0;
    for (int cc = 0; cc < T; cc++)
      inner += Acc(mul_rdiv<Op, Acc, Round>(
          c.p[std::size_t(t) * T + cc], dp[std::size_t(t) * T + cc],
          PQ));
    for (int cc = 0; cc < T; cc++)
      ds[std::size_t(t) * T + cc] = narrow<Op, Acc>(Round::div(
          Acc(c.p[std::size_t(t) * T + cc]) *
              (Acc(dp[std::size_t(t) * T + cc]) - inner),
          Acc(PQ)));
  }
  Vec dqr = finalize_rdiv<Op, Acc, Round>(
      gemm_nt_acc<Op, Acc>(ds, T, T, c.kr, DH), Acc(e.scale));
  Vec dkr = finalize_rdiv<Op, Acc, Round>(
      gemm_xty_acc<Op, Acc>(ds, T, T, c.qr, DH), Acc(e.scale));
  const Vec dq =
      rope_apply<Op, Acc, Round>(dqr, T, DH, *e.tcos, *e.tsin, -1);
  const Vec dk =
      rope_apply<Op, Acc, Round>(dkr, T, DH, *e.tcos, *e.tsin, -1);
  G["wq"] = finalize_rdiv<Op, Acc, Round>(
      gemm_xty_acc<Op, Acc>(dq, T, DH, c.h1, D), Acc(e.Q));
  G["wk"] = finalize_rdiv<Op, Acc, Round>(
      gemm_xty_acc<Op, Acc>(dk, T, DH, c.h1, D), Acc(e.Q));
  G["wv"] = finalize_rdiv<Op, Acc, Round>(
      gemm_xty_acc<Op, Acc>(dv, T, DH, c.h1, D), Acc(e.Q));
  std::vector<Acc> dh1a = gemm_nt_acc<Op, Acc>(dq, T, DH, w.at("wq"), D);
  {
    const std::vector<Acc> t2 =
        gemm_nt_acc<Op, Acc>(dk, T, DH, w.at("wk"), D);
    const std::vector<Acc> t3 =
        gemm_nt_acc<Op, Acc>(dv, T, DH, w.at("wv"), D);
    for (std::size_t i = 0; i < dh1a.size(); i++)
      dh1a[i] += t2[i] + t3[i];
  }
  // one rdiv after the 3-term sum (placement contract)
  const Vec dh1 = finalize_rdiv<Op, Acc, Round>(dh1a, Acc(e.Q));
  return rms_bwd<Op, Acc, Round>(dh1, c.x, w.at("g1"), c.i1, G["g1"],
                                 T, D, e.Q);
}

// ---- FFN (dense SwiGLU) ----
// Frozen-grain silu seam: the table index is the gate value FLOORED
// to shipped grain; table values re-embed to rung scale. Above/below
// the table: identity arm stays at rung scale, derivative arm is Q_p.

template <class Op, class Acc, class Round>
std::vector<Op> ffn_fwd(const std::map<std::string, std::vector<Op>>& w,
                        cache_t<Op>& c, const env<Op>& e) {
  const int T = e.T, D = e.D, F = e.F;
  const std::vector<std::int64_t>& sil = *e.sil;
  c.h2 = rms_fwd<Op, Acc, Round>(c.x1, w.at("g2"), c.i2, T, D, e.Q,
                                 e.eps32);
  c.gp = finalize_rdiv<Op, Acc, Round>(
      gemm_acc<Op, Acc>(c.h2, T, D, w.at("wg"), F), Acc(e.Q));
  c.u = finalize_rdiv<Op, Acc, Round>(
      gemm_acc<Op, Acc>(c.h2, T, D, w.at("wu"), F), Acc(e.Q));
  c.sg.resize(c.gp.size());
  c.f.resize(c.gp.size());
  for (std::size_t i = 0; i < c.gp.size(); i++) {
    const Op z = c.gp[i];
    const std::int64_t zg =
        std::int64_t(Round::to_grain(z, e.gshift));
    c.sg[i] = zg > e.ts   ? z
              : zg < -e.ts ? Op(0)
                           : Round::from_grain(Op(sil[zg + e.ts]),
                                               e.gshift);
    c.f[i] = mul_rdiv<Op, Acc, Round>(c.sg[i], c.u[i], e.Q);
  }
  std::vector<Acc> pre2 = gemm_acc<Op, Acc>(c.f, T, F, w.at("wd"), D);
  c.m2.assign(pre2.size(), 0);
  c.x2.assign(pre2.size(), 0);
  std::vector<Op> out(pre2.size());
  for (std::size_t i = 0; i < pre2.size(); i++) {
    const Op v =
        narrow<Op, Acc>(Round::div(pre2[i], Acc(e.Q))) + c.x1[i];
    c.m2[i] = (v <= e.CL && v >= -e.CL);
    c.x2[i] = clampi<Op>(v, -e.CL, e.CL);
  }
  return c.x2;
}

template <class Op, class Acc, class Round>
std::vector<Op> ffn_bwd(const std::map<std::string, std::vector<Op>>& w,
                        const std::vector<Op>& dx2_masked,
                        const cache_t<Op>& c,
                        std::map<std::string, std::vector<Op>>& G,
                        const env<Op>& e) {
  const int T = e.T, D = e.D, F = e.F;
  const std::vector<std::int64_t>& dsl = *e.dsl;
  const std::vector<Op>& dx2 = dx2_masked;
  std::vector<Op> df = finalize_rdiv<Op, Acc, Round>(
      gemm_nt_acc<Op, Acc>(dx2, T, D, w.at("wd"), F), Acc(e.Q));
  G["wd"] = finalize_rdiv<Op, Acc, Round>(
      gemm_xty_acc<Op, Acc>(dx2, T, D, c.f, F), Acc(e.Q));
  std::vector<Op> du(df.size()), dgp(df.size());
  for (std::size_t i = 0; i < df.size(); i++) {
    const std::int64_t zg =
        std::int64_t(Round::to_grain(c.gp[i], e.gshift));
    const Op dsv = zg > e.ts   ? e.Q
                   : zg < -e.ts ? Op(0)
                                : Round::from_grain(Op(dsl[zg + e.ts]),
                                                    e.gshift);
    du[i] = mul_rdiv<Op, Acc, Round>(c.sg[i], df[i], e.Q);
    dgp[i] = narrow<Op, Acc>(Round::div(
        Acc(mul_rdiv<Op, Acc, Round>(c.u[i], df[i], e.Q)) * Acc(dsv),
        Acc(e.Q)));
  }
  std::vector<Acc> dh2a = gemm_nt_acc<Op, Acc>(du, T, F, w.at("wu"), D);
  {
    const std::vector<Acc> t2 =
        gemm_nt_acc<Op, Acc>(dgp, T, F, w.at("wg"), D);
    for (std::size_t i = 0; i < dh2a.size(); i++) dh2a[i] += t2[i];
  }
  // one rdiv after the two-term sum (placement contract)
  const std::vector<Op> dh2 =
      finalize_rdiv<Op, Acc, Round>(dh2a, Acc(e.Q));
  G["wu"] = finalize_rdiv<Op, Acc, Round>(
      gemm_xty_acc<Op, Acc>(du, T, F, c.h2, D), Acc(e.Q));
  G["wg"] = finalize_rdiv<Op, Acc, Round>(
      gemm_xty_acc<Op, Acc>(dgp, T, F, c.h2, D), Acc(e.Q));
  return rms_bwd<Op, Acc, Round>(dh2, c.x1, w.at("g2"), c.i2,
                                 G["g2"], T, D, e.Q);
}

// ---- MoE body (gravmoe): attn half shared, FFN becomes E experts,
// top-1 router with fx3 multiplicative gate. Verbatim move; same
// silu/softmax frozen-grain seams as the dense path.

template <class Op, class Acc, class Round>
std::vector<Op> moe_body_fwd(
    const std::map<std::string, std::vector<Op>>& w,
    const std::vector<Op>& x, cache_t<Op>& c, const env<Op>& e) {
  using Vec = std::vector<Op>;
  const int T = e.T, D = e.D, F = e.F, E = e.E;
  const std::vector<std::int64_t>& sil = *e.sil;

  attn_fwd<Op, Acc, Round>(w, x, c, e);  // fills c.x .. c.x1
  c.h2 = rms_fwd<Op, Acc, Round>(c.x1, w.at("g2"), c.i2, T, D, e.Q,
                                 e.eps32);

  // router: r = rdiv(h2 @ wr^T, Q); p_r = softmax(PQ); top-1,
  // lowest expert index wins ties (strict > scanning upward)
  c.r = finalize_rdiv<Op, Acc, Round>(
      gemm_acc<Op, Acc>(c.h2, T, D, w.at("wr"), E), Acc(e.Q));
  c.pr = softmax_rows<Op, Acc, Round>(c.r, T, E, Op(e.PQ), *e.ex,
                                      e.tse, e.gshift);
  c.top.assign(T, 0);
  c.top_p.assign(T, 0);
  for (int t = 0; t < T; t++) {
    int best = 0;
    for (int ee = 1; ee < E; ee++)
      if (c.pr[std::size_t(t) * E + ee] >
          c.pr[std::size_t(t) * E + best])
        best = ee;
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
    const Vec& wg = w.at(p + ".wg");
    const Vec& wu = w.at(p + ".wu");
    const Vec& wd = w.at(p + ".wd");
    for (int f = 0; f < F; f++) {
      Acc ag = 0, au = 0;
      for (int d = 0; d < D; d++) {
        const Acc h = Acc(c.h2[std::size_t(t) * D + d]);
        ag += h * Acc(wg[std::size_t(f) * D + d]);
        au += h * Acc(wu[std::size_t(f) * D + d]);
      }
      const Op z = narrow<Op, Acc>(Round::div(ag, Acc(e.Q)));
      c.egp[std::size_t(t) * F + f] = z;
      c.eu[std::size_t(t) * F + f] =
          narrow<Op, Acc>(Round::div(au, Acc(e.Q)));
      const std::int64_t zg =
          std::int64_t(Round::to_grain(z, e.gshift));
      c.esg[std::size_t(t) * F + f] =
          zg > e.ts   ? z
          : zg < -e.ts ? Op(0)
                       : Round::from_grain(Op(sil[zg + e.ts]),
                                           e.gshift);
      c.ef[std::size_t(t) * F + f] = mul_rdiv<Op, Acc, Round>(
          c.esg[std::size_t(t) * F + f],
          c.eu[std::size_t(t) * F + f], e.Q);
    }
    for (int d = 0; d < D; d++) {
      Acc acc = 0;
      for (int f = 0; f < F; f++)
        acc += Acc(c.ef[std::size_t(t) * F + f]) *
               Acc(wd[std::size_t(d) * F + f]);
      c.eout[std::size_t(t) * D + d] =
          narrow<Op, Acc>(Round::div(acc, Acc(e.Q)));
    }
  }

  // gate + residual + clamp (fx3 multiplicative-gate convention)
  Vec pre2(std::size_t(T) * D);
  for (int t = 0; t < T; t++)
    for (int d = 0; d < D; d++)
      pre2[std::size_t(t) * D + d] =
          mul_rdiv<Op, Acc, Round>(c.eout[std::size_t(t) * D + d],
                                   c.top_p[t], Op(e.PQ)) +
          c.x1[std::size_t(t) * D + d];
  c.m2.assign(pre2.size(), 0);
  c.x2.assign(pre2.size(), 0);
  for (std::size_t i = 0; i < pre2.size(); i++) {
    c.m2[i] = (pre2[i] <= e.CL && pre2[i] >= -e.CL);
    c.x2[i] = clampi<Op>(pre2[i], -e.CL, e.CL);
  }
  return c.x2;
}

template <class Op, class Acc, class Round>
std::map<std::string, std::vector<Op>> moe_body_bwd(
    const std::map<std::string, std::vector<Op>>& w,
    const std::vector<Op>& dxin, const cache_t<Op>& c,
    std::vector<Op>* dx0_out, const env<Op>& e) {
  using Vec = std::vector<Op>;
  using VAcc = std::vector<Acc>;
  const int T = e.T, D = e.D, F = e.F, E = e.E;
  const Op PQ = Op(e.PQ);
  const std::vector<std::int64_t>& dsl = *e.dsl;
  std::map<std::string, Vec> G;

  Vec dx2 = dxin;
  for (std::size_t i = 0; i < dx2.size(); i++) dx2[i] *= c.m2[i];

  // gate chain (relay 2026-08-01-6 pinned text): dgate = dx2*top_p
  // kept EXACT (PQ-scaled, NO rounding at the gate); every consumer
  // folds the /PQ into its own single rdiv (PQ*Q). d(top_p)[t] =
  // rdiv(sum_d out[t,d]*dx2[t,d], Q) — /Q, the attention convention.
  VAcc dgate(std::size_t(T) * D);
  Vec dtp(T);
  for (int t = 0; t < T; t++) {
    Acc acc = 0;
    for (int d = 0; d < D; d++) {
      const std::size_t i = std::size_t(t) * D + d;
      dgate[i] = Acc(dx2[i]) * Acc(c.top_p[t]);
      acc += Acc(c.eout[i]) * Acc(dx2[i]);
    }
    dtp[t] = narrow<Op, Acc>(Round::div(acc, Acc(e.Q)));
  }

  // expert FFN backward per row; dW accumulated RAW per expert,
  // one rdiv per expert after the token loop (the rdiv-grouping
  // rule: same placement as the dense xty-then-round)
  std::map<std::string, VAcc> GA;
  for (int ee = 0; ee < E; ee++) {
    const std::string p = "e" + std::to_string(ee);
    GA[p + ".wg"].assign(std::size_t(F) * D, 0);
    GA[p + ".wu"].assign(std::size_t(F) * D, 0);
    GA[p + ".wd"].assign(std::size_t(D) * F, 0);
  }
  VAcc dh2a(std::size_t(T) * D, 0);
  Vec df(F), du(F), dgp(F);
  for (int t = 0; t < T; t++) {
    const std::string p = "e" + std::to_string(c.top[t]);
    const Vec& wg = w.at(p + ".wg");
    const Vec& wu = w.at(p + ".wu");
    const Vec& wd = w.at(p + ".wd");
    VAcc& Gwg = GA.at(p + ".wg");
    VAcc& Gwu = GA.at(p + ".wu");
    VAcc& Gwd = GA.at(p + ".wd");
    for (int f = 0; f < F; f++) {
      Acc acc = 0;
      for (int d = 0; d < D; d++)
        acc += dgate[std::size_t(t) * D + d] *
               Acc(wd[std::size_t(d) * F + f]);
      df[f] = narrow<Op, Acc>(
          Round::div(acc, Acc(PQ) * Acc(e.Q)));  // fold the gate /PQ
      const std::int64_t zg = std::int64_t(
          Round::to_grain(c.egp[std::size_t(t) * F + f], e.gshift));
      const Op dsv = zg > e.ts   ? e.Q
                     : zg < -e.ts ? Op(0)
                                  : Round::from_grain(
                                        Op(dsl[zg + e.ts]), e.gshift);
      du[f] = mul_rdiv<Op, Acc, Round>(
          c.esg[std::size_t(t) * F + f], df[f], e.Q);
      dgp[f] = narrow<Op, Acc>(Round::div(
          Acc(mul_rdiv<Op, Acc, Round>(c.eu[std::size_t(t) * F + f],
                                       df[f], e.Q)) *
              Acc(dsv),
          Acc(e.Q)));
    }
    for (int f = 0; f < F; f++)
      for (int d = 0; d < D; d++) {
        Gwg[std::size_t(f) * D + d] +=
            Acc(dgp[f]) * Acc(c.h2[std::size_t(t) * D + d]);
        Gwu[std::size_t(f) * D + d] +=
            Acc(du[f]) * Acc(c.h2[std::size_t(t) * D + d]);
      }
    for (int d = 0; d < D; d++)
      for (int f = 0; f < F; f++)
        Gwd[std::size_t(d) * F + f] +=
            dgate[std::size_t(t) * D + d] *
            Acc(c.ef[std::size_t(t) * F + f]);
    // dh2 from expert weights (two-term sum, one rdiv after loop)
    for (int d = 0; d < D; d++) {
      Acc acc = 0;
      for (int f = 0; f < F; f++)
        acc += Acc(du[f]) * Acc(wu[std::size_t(f) * D + d]) +
               Acc(dgp[f]) * Acc(wg[std::size_t(f) * D + d]);
      dh2a[std::size_t(t) * D + d] = acc;
    }
  }
  for (int ee = 0; ee < E; ee++) {
    const std::string p = "e" + std::to_string(ee);
    G[p + ".wg"] =
        finalize_rdiv<Op, Acc, Round>(GA.at(p + ".wg"), Acc(e.Q));
    G[p + ".wu"] =
        finalize_rdiv<Op, Acc, Round>(GA.at(p + ".wu"), Acc(e.Q));
    G[p + ".wd"] = finalize_rdiv<Op, Acc, Round>(
        GA.at(p + ".wd"), Acc(PQ) * Acc(e.Q));  // dgate carries PQ
  }
  Vec dh2 = finalize_rdiv<Op, Acc, Round>(dh2a, Acc(e.Q));

  // router: scatter d(top_p) into dp_r, softmax_bwd at PQ, then
  // wr + h2 paths (each group finalized once, then summed)
  Vec dpr(std::size_t(T) * E, 0);
  for (int t = 0; t < T; t++)
    dpr[std::size_t(t) * E + c.top[t]] = dtp[t];
  Vec dr(std::size_t(T) * E);
  for (int t = 0; t < T; t++) {
    Acc inner = 0;
    for (int ee = 0; ee < E; ee++)
      inner += Acc(mul_rdiv<Op, Acc, Round>(
          c.pr[std::size_t(t) * E + ee],
          dpr[std::size_t(t) * E + ee], PQ));
    for (int ee = 0; ee < E; ee++)
      dr[std::size_t(t) * E + ee] = narrow<Op, Acc>(Round::div(
          Acc(c.pr[std::size_t(t) * E + ee]) *
              (Acc(dpr[std::size_t(t) * E + ee]) - inner),
          Acc(PQ)));
  }
  G["wr"] = finalize_rdiv<Op, Acc, Round>(
      gemm_xty_acc<Op, Acc>(dr, T, E, c.h2, D), Acc(e.Q));
  {
    const Vec dh2r = finalize_rdiv<Op, Acc, Round>(
        gemm_nt_acc<Op, Acc>(dr, T, E, w.at("wr"), D), Acc(e.Q));
    for (std::size_t i = 0; i < dh2.size(); i++) dh2[i] += dh2r[i];
  }

  Vec dx1 = rms_bwd<Op, Acc, Round>(dh2, c.x1, w.at("g2"), c.i2,
                                    G["g2"], T, D, e.Q);
  for (std::size_t i = 0; i < dx1.size(); i++)
    dx1[i] = (dx1[i] + dx2[i]) * c.m1[i];
  Vec dx0 = attn_bwd<Op, Acc, Round>(w, dx1, c, G, e);
  if (dx0_out) {
    for (std::size_t i = 0; i < dx0.size(); i++) dx0[i] += dx1[i];
    *dx0_out = std::move(dx0);
  }
  return G;
}

// ---- head (dense fwd/bwd around the body) ----

template <class Op, class Acc, class Round>
std::vector<Op> fwd_head(const std::map<std::string, std::vector<Op>>& w,
                         const std::vector<Op>& x2, cache_t<Op>& c,
                         const env<Op>& e) {
  c.h3 = rms_fwd<Op, Acc, Round>(x2, w.at("g3"), c.i3, e.T, e.D, e.Q,
                                 e.eps32);
  return finalize_rdiv<Op, Acc, Round>(
      gemm_acc<Op, Acc>(c.h3, e.T, e.D, w.at("wh"), e.V), Acc(e.Q));
}

template <class Op, class Acc, class Round>
std::vector<Op> bwd_head(const std::map<std::string, std::vector<Op>>& w,
                         const std::vector<Op>& dlogits,
                         const cache_t<Op>& c,
                         std::map<std::string, std::vector<Op>>& G,
                         const env<Op>& e) {
  G["wh"] = finalize_rdiv<Op, Acc, Round>(
      gemm_xty_acc<Op, Acc>(dlogits, e.T, e.V, c.h3, e.D), Acc(e.Q));
  const std::vector<Op> dh3 = finalize_rdiv<Op, Acc, Round>(
      gemm_nt_acc<Op, Acc>(dlogits, e.T, e.V, w.at("wh"), e.D),
      Acc(e.Q));
  return rms_bwd<Op, Acc, Round>(dh3, c.x2, w.at("g3"), c.i3,
                                 G["g3"], e.T, e.D, e.Q);
}

// ---- adamw elementwise update (one parameter tensor) ----
// Verbatim move of the shipped inner loop. Frozen-grain sqrt seam:
// den = from_grain(isqrt(to_grain(vh) * Q9)) — declared, no-op at
// gshift 0. Bias-correction factors bc* are scale-free i64 (the
// BigV machinery stays with the adamw class).
template <class Op, class Acc, class Round>
void adamw_update(std::vector<Op>& w, const std::vector<Op>& g,
                  std::vector<Op>& m, std::vector<Op>& v,
                  std::int64_t bc1n, std::int64_t bc1d,
                  std::int64_t bc2n, std::int64_t bc2d, Op Q,
                  int shift, std::int64_t lrn, std::int64_t lrd,
                  int gshift, std::int64_t& nz, std::int64_t& tot) {
  using i64 = std::int64_t;
  constexpr i64 B1N = 9, B1D = 10, B2N = 999, B2D = 1000;
  constexpr i64 AEPS = 4, WDN = 1, WDD = 100000;
  constexpr i64 Q9 = 512;
  for (std::size_t i = 0; i < w.size(); i++) {
    m[i] = narrow<Op, Acc>(Round::div(
        Acc(B1N) * Acc(m[i]) + Acc(B1D - B1N) * Acc(g[i]),
        Acc(B1D)));
    v[i] = narrow<Op, Acc>(Round::div(
        Acc(B2N) * Acc(v[i]) +
            Acc(B2D - B2N) *
                Acc(mul_rdiv<Op, Acc, Round>(g[i], g[i], Q)),
        Acc(B2D)));
    const Op mh = narrow<Op, Acc>(
        Round::div(Acc(m[i]) * Acc(bc1n), Acc(bc1d)));
    const Op vh = narrow<Op, Acc>(
        Round::div(Acc(v[i]) * Acc(bc2n), Acc(bc2d)));
    const Op den =
        Round::from_grain(
            Op(isqrt_newton(i64(Round::to_grain(vh, gshift)) * Q9)),
            gshift) +
        AEPS;
    const Op upd = narrow<Op, Acc>(
        Round::div(Acc(lrn) * Acc(mh) * (Acc(Q) << shift),
                   Acc(lrd) * Acc(den)));
    nz += upd != 0;
    tot += 1;
    w[i] -= upd;
    w[i] -= mul_rdiv<Op, Acc, Round>(w[i], Op(WDN), Op(WDD));
  }
}

// ---- the composed dense training loop, width-generic ----
// (ENGINE-EXACT-1: full_birth is the <i64,i64> instantiation, gated
// bit-identical by the r2b_ref digests; Q32 = <i64,__int128>; Q64 =
// <__int128, ax::core::i256>. multi/moe stay <= Q32 by scope — no
// registered rung needs Q64 there.)

inline const char* const BIRTH_KEYS[11] = {"wq", "wk", "wv", "wo",
                                           "wg", "wu", "wd", "wh",
                                           "g1", "g2", "g3"};


/** Copyable running sha256 (FIPS 180-4), header-only twin of
    ib::detail::sha256 so the core stays free of intbirth.hpp; the
    i64 birth_impl digests are gated equal to the shipped class by
    the r2b drivers. */
struct sha256h {
  std::uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                        0xa54ff53a, 0x510e527f, 0x9b05688c,
                        0x1f83d9ab, 0x5be0cd19};
  std::uint8_t buf[64];
  std::uint64_t len = 0;
  std::size_t fill = 0;
  void blk(const std::uint8_t* p) {
    static const std::uint32_t K[64] = {
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
    const auto rot = [](std::uint32_t x, int n) {
      return (x >> n) | (x << (32 - n));
    };
    std::uint32_t w[64];
    for (int i = 0; i < 16; i++)
      w[i] = (std::uint32_t(p[4 * i]) << 24) |
             (std::uint32_t(p[4 * i + 1]) << 16) |
             (std::uint32_t(p[4 * i + 2]) << 8) |
             std::uint32_t(p[4 * i + 3]);
    for (int i = 16; i < 64; i++) {
      const std::uint32_t s0 =
          rot(w[i - 15], 7) ^ rot(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 =
          rot(w[i - 2], 17) ^ rot(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4],
                  f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; i++) {
      const std::uint32_t S1 = rot(e, 6) ^ rot(e, 11) ^ rot(e, 25);
      const std::uint32_t ch = (e & f) ^ (~e & g);
      const std::uint32_t t1 = hh + S1 + ch + K[i] + w[i];
      const std::uint32_t S0 = rot(a, 2) ^ rot(a, 13) ^ rot(a, 22);
      const std::uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t t2 = S0 + mj;
      hh = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  void update(const void* data, std::size_t n) {
    const std::uint8_t* p = static_cast<const std::uint8_t*>(data);
    len += n;
    while (n) {
      const std::size_t take = 64 - fill < n ? 64 - fill : n;
      std::memcpy(buf + fill, p, take);
      fill += take;
      p += take;
      n -= take;
      if (fill == 64) {
        blk(buf);
        fill = 0;
      }
    }
  }
  std::string hex() {
    const std::uint64_t bits = len * 8;
    const std::uint8_t pad = 0x80;
    update(&pad, 1);
    const std::uint8_t z = 0;
    while (fill != 56) update(&z, 1);
    std::uint8_t lb[8];
    for (int i = 0; i < 8; i++)
      lb[i] = std::uint8_t(bits >> (56 - 8 * i));
    update(lb, 8);
    char out[65];
    for (int i = 0; i < 8; i++)
      std::snprintf(out + 8 * i, 9, "%08x", h[i]);
    return std::string(out, 64);
  }
};

/** dims/scales from a contract-shaped bundle (kept as plain ints
    so this header stays independent of intbirth.hpp). */
struct birth_cfg_t {
  int T, D, DH, F, V, shift, precision;
  std::int64_t gboost, pq, act_clamp, eps32, lrn, lrd;
};

template <class Op, class Acc, class Round>
class birth_impl {
 public:
  using Vec = std::vector<Op>;
  using cfg = birth_cfg_t;

  birth_impl(const std::string& tables_bytes,
             const std::string& init_bytes, const cfg& c)
      : c_(c) {
    if (c.T <= 0 || c.D <= 0 || c.DH <= 0 || c.F <= 0 || c.V <= 0 ||
        c.DH % 2 || c.shift < 0 || c.gboost < 1 || c.pq < 1 ||
        c.act_clamp < 1 || c.lrd < 1 || c.lrn < 1 || c.eps32 < 1)
      throw std::runtime_error("intbirth: bad contract");
    tab_ = parse_axp3(tables_bytes);
    for (const char* k : {"silu.tab", "dsilu.tab", "exp.tab",
                          "rope.cos", "rope.sin"})
      if (!tab_.count(k))
        throw std::runtime_error(std::string("intbirth: missing ") +
                                 k);
    ts_ = (std::int64_t(tab_.at("silu.tab").size()) - 1) / 2;
    tse_ = std::int64_t(tab_.at("exp.tab").size()) - 1;
    if (std::int64_t(tab_.at("dsilu.tab").size()) != 2 * ts_ + 1 ||
        std::int64_t(tab_.at("rope.cos").size()) !=
            std::int64_t(c.T) * (c.DH / 2) ||
        std::int64_t(tab_.at("rope.sin").size()) !=
            std::int64_t(c.T) * (c.DH / 2))
      throw std::runtime_error("intbirth: table size mismatch");
    const int g = c.precision - 9;
    // frozen-grain attn scale: shipped isqrt, then re-embed
    std::int64_t s9 = 512 * 512 * std::int64_t(c.DH);
    {  // isqrt_round at shipped grain
      const std::int64_t r = isqrt_newton(s9);
      scale9_ = (s9 - r * r > (r + 1) * (r + 1) - s9) ? r + 1 : r;
    }
    // init: 11 KEYS tensors, x [T,D], tgt [T] — i64 LE at shipped
    // grain (the handoff convention at every rung)
    const auto sh = shapes(c.DH, c.D, c.F, c.V);
    std::size_t off = 0;
    const auto take = [&](std::size_t n) {
      if (off + n * 8 > init_bytes.size())
        throw std::runtime_error("intbirth: truncated init");
      std::vector<std::int64_t> m(n);
      std::memcpy(m.data(), init_bytes.data() + off, n * 8);
      off += n * 8;
      return m;
    };
    for (const char* k : BIRTH_KEYS) {
      const auto s = sh.at(k);
      const auto raw = take(std::size_t(s.r) * s.c);
      Vec& wv = w_[k];
      wv.resize(raw.size());
      for (std::size_t i = 0; i < raw.size(); i++)
        wv[i] = Op(raw[i]) << (c.shift + g);  // lift to rung Q_w
    }
    {
      const auto raw = take(std::size_t(c.T) * c.D);
      x_.resize(raw.size());
      for (std::size_t i = 0; i < raw.size(); i++)
        x_[i] = Op(raw[i]) << g;  // exact re-embed
    }
    tgt_ = take(std::size_t(c.T));
    if (off != init_bytes.size())
      throw std::runtime_error("intbirth: trailing init bytes");
    for (const std::int64_t t : tgt_)
      if (t < 0 || t >= c.V)
        throw std::runtime_error("intbirth: target out of vocab");
  }

  void run(int steps) {
    for (int i = 0; i < steps; i++) step_once();
  }
  void set_lr(std::int64_t lrn, std::int64_t lrd) {
    if (lrn < 1 || lrd < 1)
      throw std::runtime_error("intbirth: bad set_lr params");
    c_.lrn = lrn;
    c_.lrd = lrd;
  }
  int step_count() const { return step_; }
  /** milestone loss, de-grained to the shipped scale (declared) */
  std::int64_t last_loss() const {
    return std::int64_t(Round::to_grain(loss_, c_.precision - 9));
  }
  double nz_last() const { return nz_; }

  std::string mark() {
    if constexpr (std::is_trivially_copyable_v<Op>) {
      for (const char* k : BIRTH_KEYS)
        th_.update(w_.at(k).data(), w_.at(k).size() * sizeof(Op));
    } else {
      // non-POD scalar (the exact anchor): the DECLARED digest view
      // is the floor-to-shipped-grain i64 weights
      const auto g9 = weights_grain9();
      th_.update(g9.data(), g9.size() * 8);
    }
    return traj_sha();
  }
  std::string traj_sha() const {
    sha256h peek = th_;
    return peek.hex();
  }
  std::string weights_bytes() const {
    std::string out;
    if constexpr (std::is_trivially_copyable_v<Op>) {
      for (const char* k : BIRTH_KEYS) {
        const auto& w = w_.at(k);
        out.append(reinterpret_cast<const char*>(w.data()),
                   w.size() * sizeof(Op));
      }
    } else {
      const auto g9 = weights_grain9();
      out.append(reinterpret_cast<const char*>(g9.data()),
                 g9.size() * 8);
    }
    return out;
  }
  /** weights de-grained to shipped-scale i64 (for cross-rung
      divergence readouts; declared floor per the convention pin) */
  std::vector<std::int64_t> weights_grain9() const {
    std::vector<std::int64_t> out;
    for (const char* k : BIRTH_KEYS)
      for (const Op v : w_.at(k))
        out.push_back(std::int64_t(
            Round::to_grain(v, c_.precision - 9)));
    return out;
  }

 private:
  env<Op> make_env() const {
    env<Op> e;
    e.T = c_.T; e.D = c_.D; e.DH = c_.DH; e.F = c_.F; e.V = c_.V;
    e.E = 0;
    e.PQ = c_.pq;
    e.CL = Op(c_.act_clamp) << (c_.precision - 9);
    e.Q = Op(1) << c_.precision;
    e.scale = Op(scale9_) << (c_.precision - 9);
    e.eps32 = c_.eps32;
    e.ts = ts_; e.tse = tse_;
    e.tcos = &tab_.at("rope.cos"); e.tsin = &tab_.at("rope.sin");
    e.sil = &tab_.at("silu.tab"); e.dsl = &tab_.at("dsilu.tab");
    e.ex = &tab_.at("exp.tab");
    e.gshift = c_.precision - 9;
    return e;
  }

  void step_once() {
    const env<Op> e = make_env();
    const Op qc = e.Q;
    // Q-scale view of the wide weights (the matmul boundary)
    std::map<std::string, Vec> w;
    for (const char* k : BIRTH_KEYS) {
      w[k] = w_.at(k);
      for (auto& v : w[k]) v = Round::div(v, Op(1) << c_.shift);
    }
    cache_t<Op> bc;
    attn_fwd<Op, Acc, Round>(w, x_, bc, e);   // fills bc.x .. bc.x1
    const Vec x2 = ffn_fwd<Op, Acc, Round>(w, bc, e);
    const Vec logits = fwd_head<Op, Acc, Round>(w, x2, bc, e);
    const Vec pp = softmax_rows<Op, Acc, Round>(
        logits, c_.T, c_.V, qc, *e.ex, tse_, e.gshift);
    Op loss = 0;
    for (int t = 0; t < c_.T; t++)
      loss += qc - pp[std::size_t(t) * c_.V + tgt_[t]];
    loss_ = loss;
    Vec dlogits(std::size_t(c_.T) * c_.V);
    for (int t = 0; t < c_.T; t++)
      for (int vv = 0; vv < c_.V; vv++)
        dlogits[std::size_t(t) * c_.V + vv] =
            (pp[std::size_t(t) * c_.V + vv] -
             qc * Op(tgt_[t] == vv)) *
            Op(c_.gboost);
    std::map<std::string, Vec> G;
    const Vec dx2in =
        bwd_head<Op, Acc, Round>(w, dlogits, bc, G, e);
    {  // dense body backward (masks + residual chain, no dx0)
      Vec dx2 = dx2in;
      for (std::size_t i = 0; i < dx2.size(); i++) dx2[i] *= bc.m2[i];
      Vec dx1 = ffn_bwd<Op, Acc, Round>(w, dx2, bc, G, e);
      for (std::size_t i = 0; i < dx1.size(); i++)
        dx1[i] = (dx1[i] + dx2[i]) * bc.m1[i];
      attn_bwd<Op, Acc, Round>(w, dx1, bc, G, e);
    }
    // unboost + optimizer
    t_ += 1;
    big_mul(p10_, 10); big_mul(p9_, 9);
    big_mul(p1000_, 1000); big_mul(p999_, 999);
    BigV n1 = p10_, d1 = big_sub(p10_, p9_);
    BigV n2 = p1000_, d2 = big_sub(p1000_, p999_);
    big_norm30(n1, d1);
    big_norm30(n2, d2);
    const std::int64_t bc1n = big_i64(n1);
    const std::int64_t bc1d = std::max<std::int64_t>(big_i64(d1), 1);
    const std::int64_t bc2n = big_i64(n2);
    const std::int64_t bc2d = std::max<std::int64_t>(big_i64(d2), 1);
    if (m_.empty()) {
      for (const char* k : BIRTH_KEYS) {
        m_.emplace_back(w_.at(k).size(), Op(0));
        v_.emplace_back(w_.at(k).size(), Op(0));
      }
    }
    std::int64_t nz = 0, tot = 0;
    int j = 0;
    for (const char* k : BIRTH_KEYS) {
      Vec g = std::move(G.at(k));
      for (auto& v : g)  // unboost: one rdiv at qc*gboost
        v = narrow<Op, Acc>(Round::div(
            Acc(v), Acc(qc) * Acc(c_.gboost)));
      adamw_update<Op, Acc, Round>(
          w_.at(k), g, m_[j], v_[j], bc1n, bc1d, bc2n, bc2d, e.Q,
          c_.shift, c_.lrn, c_.lrd, e.gshift, nz, tot);
      j++;
    }
    nz_ = double(nz) / double(tot);
    step_ += 1;
  }

  cfg c_;
  std::map<std::string, std::vector<std::int64_t>> tab_;
  std::int64_t scale9_ = 0, ts_ = 0, tse_ = 0;
  std::map<std::string, Vec> w_;
  Vec x_;
  std::vector<std::int64_t> tgt_;
  std::vector<Vec> m_, v_;
  BigV p10_{1}, p9_{1}, p1000_{1}, p999_{1};
  int t_ = 0, step_ = 0;
  Op loss_ = 0;
  double nz_ = 0;
  sha256h th_;
};

}  // namespace ax::nn::ib::core
