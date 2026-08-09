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

}  // namespace ax::nn::ib::core
