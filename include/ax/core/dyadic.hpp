#pragma once
/** @file dyadic.hpp anchor-v2 pin 3: arbitrary-precision dyadic
    interval [lo,hi]*2^e with OUTWARD rounding to `prec` mantissa
    bits — the certified shadow the RNS anchor consults at branch
    sites. gcd-free by construction: only shifts, add/mul, and
    directed bigint division ever run. Precision is a runtime knob
    (the growing-shadow pin: ~50 + 20*N bits for an N-step prefix);
    every rounding widens, so containment of the true value is an
    invariant, never an assumption. */
#include <algorithm>
#include <stdexcept>
#include <utility>

#include <ax/core/bigint.hpp>

namespace ax {

struct dyi {
  bigint lo, hi;  // value in [lo*2^e, hi*2^e], lo <= hi
  int e = 0;
  /** mantissa bit target (outward-rounded); driver-set per step */
  static inline int prec = 128;

  dyi() = default;
  dyi(long long v) : lo(v), hi(v) {}  // NOLINT — exact point
  dyi(bigint m, int ex) : lo(m), hi(std::move(m)), e(ex) {}

  // ---- helpers ------------------------------------------------
  static int bit_len(const bigint& x) {  // bits of |x|; 0 for 0
    if (x.is_zero()) return 0;
    const bigint a = ax::abs(x);
    int lo_ = 0, hi_ = 1;
    while (!(a >> unsigned(hi_)).is_zero()) { lo_ = hi_; hi_ *= 2; }
    while (lo_ + 1 < hi_) {
      const int mid = (lo_ + hi_) / 2;
      ((a >> unsigned(mid)).is_zero() ? hi_ : lo_) = mid;
    }
    return lo_ + 1;
  }
  static bigint shr_floor(const bigint& x, unsigned k) {
    const bigint q = x >> k;  // truncates toward zero
    if (x.is_negative() && !((q << k) == x)) return q - bigint(1);
    return q;
  }
  static bigint shr_ceil(const bigint& x, unsigned k) {
    return -shr_floor(-x, k);
  }
  static bigint div_floor(const bigint& n, const bigint& d) {
    auto [q, r] = bigint::divmod(n, d);
    if (!r.is_zero() && (n.is_negative() != d.is_negative()))
      return q - bigint(1);
    return q;
  }
  static bigint div_ceil(const bigint& n, const bigint& d) {
    return -div_floor(-n, d);
  }

  /** outward round to <= prec mantissa bits */
  void normalize() {
    const int b = std::max(bit_len(lo), bit_len(hi));
    if (b <= prec) return;
    const unsigned k = unsigned(b - prec);
    lo = shr_floor(lo, k);
    hi = shr_ceil(hi, k);
    e += int(k);
  }
  /** rewrite both operands to a common exponent (exact) */
  static void align(dyi& a, dyi& b) {
    if (a.e == b.e) return;
    if (a.e > b.e) {
      const unsigned k = unsigned(a.e - b.e);
      a.lo = a.lo << k; a.hi = a.hi << k; a.e = b.e;
    } else {
      const unsigned k = unsigned(b.e - a.e);
      b.lo = b.lo << k; b.hi = b.hi << k; b.e = a.e;
    }
  }

  // ---- ring ops -----------------------------------------------
  friend dyi operator+(dyi a, dyi b) {
    align(a, b);
    dyi r; r.e = a.e; r.lo = a.lo + b.lo; r.hi = a.hi + b.hi;
    r.normalize(); return r;
  }
  dyi operator-() const {
    dyi r; r.e = e; r.lo = -hi; r.hi = -lo; return r;
  }
  friend dyi operator-(const dyi& a, const dyi& b) { return a + (-b); }
  friend dyi operator*(const dyi& a, const dyi& b) {
    const bigint p1 = a.lo * b.lo, p2 = a.lo * b.hi,
                 p3 = a.hi * b.lo, p4 = a.hi * b.hi;
    dyi r; r.e = a.e + b.e; r.lo = p1; r.hi = p1;
    for (const bigint* p : {&p2, &p3, &p4}) {
      if (*p < r.lo) r.lo = *p;
      if (r.hi < *p) r.hi = *p;
    }
    r.normalize(); return r;
  }
  dyi& operator+=(const dyi& b) { return *this = *this + b; }
  dyi& operator-=(const dyi& b) { return *this = *this - b; }
  dyi& operator*=(const dyi& b) { return *this = *this * b; }
  friend dyi operator<<(dyi a, int n) { a.e += n; return a; }
  friend dyi operator>>(dyi a, int n) { a.e -= n; return a; }

  /** interval division; b must not contain 0 (throws — the anchor's
      divisors never straddle zero, so this is a soundness assert) */
  static dyi div(const dyi& a, const dyi& b) {
    if (b.contains_zero())
      throw std::runtime_error("dyi: divisor interval contains 0");
    dyi r; r.e = a.e - b.e - prec;
    bool first = true;
    for (const bigint* n : {&a.lo, &a.hi})
      for (const bigint* d : {&b.lo, &b.hi}) {
        const bigint ns = *n << unsigned(prec);
        const bigint qd = div_floor(ns, *d), qu = div_ceil(ns, *d);
        if (first) { r.lo = qd; r.hi = qu; first = false; }
        else {
          if (qd < r.lo) r.lo = qd;
          if (r.hi < qu) r.hi = qu;
        }
      }
    r.normalize(); return r;
  }

  // ---- branch queries -----------------------------------------
  bool is_point() const { return lo == hi; }
  bool contains_zero() const {  // lo <= 0 <= hi
    return (lo.is_negative() || lo.is_zero()) && !hi.is_negative();
  }
  /** -1 / 0 / +1 when certain, 2 when the interval straddles 0
      (0 only for the exact point zero) */
  int sign() const {
    if (hi.is_negative()) return -1;
    if (!lo.is_negative() && !lo.is_zero()) return 1;
    if (lo.is_zero() && hi.is_zero()) return 0;
    return 2;
  }
  /** certain a < b (intervals disjoint on that side) */
  static bool lt_certain(dyi a, dyi b) {
    align(a, b);
    return a.hi < b.lo;
  }
  /** floors of the two endpoints (negative-aware exact floors) */
  std::pair<bigint, bigint> floor_pair() const {
    if (e >= 0)
      return {lo << unsigned(e), hi << unsigned(e)};
    return {shr_floor(lo, unsigned(-e)), shr_floor(hi, unsigned(-e))};
  }
  /** exact integer value of a point interval (e >= 0 or divisible) */
  bigint point_int() const {
    if (!is_point()) throw std::runtime_error("dyi: not a point");
    if (e >= 0) return lo << unsigned(e);
    const bigint f = shr_floor(lo, unsigned(-e));
    if (!((f << unsigned(-e)) == lo))
      throw std::runtime_error("dyi: point not integral");
    return f;
  }
};

}  // namespace ax
