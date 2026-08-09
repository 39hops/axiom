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
#include <stdexcept>
#include <vector>

namespace ax::nn::ib::core {

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

}  // namespace ax::nn::ib::core
