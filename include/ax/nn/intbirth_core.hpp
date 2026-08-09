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
  /** Rung scale -> shipped scale: FLOOR (arithmetic shift toward
      -inf), not round-half-away and not toward-zero truncation —
      the frozen-grain seam is declared as exactly this shift. */
  static Op to_grain(Op x, int gshift) {
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
      if (!xv) continue;
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
    e, z, and the output all live at shipped/frozen scales at every
    rung — deliberately no Acc/Round params, so the freeze cannot
    be violated invisibly. `scale` must be a frozen carry (PQ
    class, << 2^40): e*scale is then always safe in i64. Row sums
    of the output are NOT exactly `scale` (per-element rounding,
    no residual assignment) — nothing downstream may assume it. */
template <class Op>
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
      i64 d = i64(RoundHalfAway<Op>::to_grain(  // floor to shipped
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
  c.p = softmax_rows<Op>(s, T, T, e.PQ, *e.ex, e.tse, e.gshift);
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
  c.pr = softmax_rows<Op>(c.r, T, E, e.PQ, *e.ex, e.tse, e.gshift);
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

}  // namespace ax::nn::ib::core
