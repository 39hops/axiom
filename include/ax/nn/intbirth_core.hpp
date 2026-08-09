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

template <class Op>
inline Op clampi(Op x, Op lo, Op hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

/** Round-half-away division at grain d (the program-wide rdiv),
    plus the frozen-grain seam helpers (spec §Convention pin):
    transcendental sites truncate rung scale -> shipped scale,
    apply the shipped convention, and shift back. Both are no-ops
    at gshift == 0 (the Q9 rung). */
template <class Op>
struct RoundHalfAway {
  static Op div(Op x, Op d) {  // verbatim shipped rdiv semantics
    const Op ax = x < 0 ? -x : x;
    const Op r = (ax + d / 2) / d;
    return x < 0 ? -r : r;
  }
  static Op to_grain(Op x, int gshift) {    // floor-truncate
    return gshift ? (x >> gshift) : x;
  }
  static Op from_grain(Op x, int gshift) {  // exact re-embed
    return gshift ? (x << gshift) : x;
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
      if (!xv) continue;
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
    the max-shifted logit truncated to the shipped grain; e and z
    live at shipped table scale (i64) at every rung; the output
    carries `scale` (a frozen carry, i64). */
template <class Op, class Acc, class Round>
std::vector<Op> softmax_rows(const std::vector<Op>& s, int rows,
                             int C, std::int64_t scale,
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
      i64 d = i64(RoundHalfAway<Op>::to_grain(
          s[std::size_t(t) * C + cc] - m, gshift));
      if (d < -tse - 1) d = -tse - 1;
      e[cc] = d < -tse ? 0 : ex[d + tse];
      z += e[cc];
    }
    for (int cc = 0; cc < C; cc++)
      p[std::size_t(t) * C + cc] =
          Op(RoundHalfAway<i64>::div(e[cc] * scale, z));
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
        narrow<i64, Acc>(((s2 / D) * (Acc(1) << 32)) /
                         (Acc(Q) * Acc(Q))) +
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
  c.q0 = gemm<Op, Acc>(c.h1, T, D, w.at("wq"), DH);
  c.k0 = gemm<Op, Acc>(c.h1, T, D, w.at("wk"), DH);
  c.v0 = gemm<Op, Acc>(c.h1, T, D, w.at("wv"), DH);
  rdiv_inplace<Op, Round>(c.q0, e.Q);
  rdiv_inplace<Op, Round>(c.k0, e.Q);
  rdiv_inplace<Op, Round>(c.v0, e.Q);
  c.qr = rope_apply<Op, Acc, Round>(c.q0, T, DH, *e.tcos, *e.tsin, 1);
  c.kr = rope_apply<Op, Acc, Round>(c.k0, T, DH, *e.tcos, *e.tsin, 1);
  std::vector<Op> s = gemm<Op, Acc>(c.qr, T, DH, c.kr, T);
  rdiv_inplace<Op, Round>(s, e.scale);
  const Op floor_v = Round::from_grain(Op(-(std::int64_t{1} << 40)),
                                       e.gshift);
  for (int t = 0; t < T; t++)
    for (int u = t + 1; u < T; u++)
      s[std::size_t(t) * T + u] = floor_v;  // causal
  c.p = softmax_rows<Op, Acc, Round>(s, T, T, e.PQ, *e.ex, e.tse,
                                     e.gshift);
  c.a = gemm_nt<Op, Acc>(c.p, T, T, c.v0, DH);
  rdiv_inplace<Op, Round>(c.a, Op(e.PQ));
  std::vector<Op> pre1 = gemm<Op, Acc>(c.a, T, DH, w.at("wo"), D);
  rdiv_inplace<Op, Round>(pre1, e.Q);
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
  Vec da = gemm_nt<Op, Acc>(dx1, T, D, w.at("wo"), DH);
  rdiv_inplace<Op, Round>(da, e.Q);
  G["wo"] = gemm_xty<Op, Acc>(dx1, T, D, c.a, DH);
  rdiv_inplace<Op, Round>(G["wo"], e.Q);
  Vec dp = gemm<Op, Acc>(da, T, DH, c.v0, T);
  rdiv_inplace<Op, Round>(dp, e.Q);
  Vec dv = gemm_xty<Op, Acc>(c.p, T, T, da, DH);
  rdiv_inplace<Op, Round>(dv, PQ);
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
  Vec dqr = gemm_nt<Op, Acc>(ds, T, T, c.kr, DH);
  rdiv_inplace<Op, Round>(dqr, e.scale);
  Vec dkr = gemm_xty<Op, Acc>(ds, T, T, c.qr, DH);
  rdiv_inplace<Op, Round>(dkr, e.scale);
  const Vec dq =
      rope_apply<Op, Acc, Round>(dqr, T, DH, *e.tcos, *e.tsin, -1);
  const Vec dk =
      rope_apply<Op, Acc, Round>(dkr, T, DH, *e.tcos, *e.tsin, -1);
  G["wq"] = gemm_xty<Op, Acc>(dq, T, DH, c.h1, D);
  rdiv_inplace<Op, Round>(G["wq"], e.Q);
  G["wk"] = gemm_xty<Op, Acc>(dk, T, DH, c.h1, D);
  rdiv_inplace<Op, Round>(G["wk"], e.Q);
  G["wv"] = gemm_xty<Op, Acc>(dv, T, DH, c.h1, D);
  rdiv_inplace<Op, Round>(G["wv"], e.Q);
  Vec dh1 = gemm_nt<Op, Acc>(dq, T, DH, w.at("wq"), D);
  {
    const Vec t2 = gemm_nt<Op, Acc>(dk, T, DH, w.at("wk"), D);
    const Vec t3 = gemm_nt<Op, Acc>(dv, T, DH, w.at("wv"), D);
    for (std::size_t i = 0; i < dh1.size(); i++)
      dh1[i] += t2[i] + t3[i];
  }
  rdiv_inplace<Op, Round>(dh1, e.Q);  // one rdiv after the 3-term sum
  return rms_bwd<Op, Acc, Round>(dh1, c.x, w.at("g1"), c.i1, G["g1"],
                                 T, D, e.Q);
}

}  // namespace ax::nn::ib::core
