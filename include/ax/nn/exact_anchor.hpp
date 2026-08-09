#pragma once
/** @file exact_anchor.hpp ENGINE-EXACT-1 exact-prefix anchor: the
    templated birth loop instantiated with an exact-rational scalar
    and an Exact rounding policy — zero value-rounding at every ring
    site; the only quantizations are the frozen-grain seams the
    convention pin DECLARES as part of the function definitions
    (table indices, isqrt inputs), implemented as exact floors.

    SPEC CORRECTION (docs/specs/2026-08-08-engine-exact-ladder.md
    called this scalar "dyadic"): AdamW divides by 10/1000/100000,
    so the exact arm cannot stay dyadic — the scalar is a full
    rational (ax::rational on bigint) with a loud bit-ceiling guard.
    The Omega(2^steps) growth law says this only survives a short
    prefix; the guard makes the horizon observable, never silent. */
#include <compare>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include <ax/core/bigint.hpp>
#include <ax/core/rational.hpp>
#include <ax/nn/intbirth_core.hpp>

namespace ax::nn::ib::anchor {

/** Exact rational scalar with a bit-ceiling guard. */
struct exr {
  ax::rational v;
  /** loud-abort ceiling on the numerator magnitude (bits, approx —
      compared against 2^ceiling via bigint). Default 2^22. */
  static inline unsigned bit_ceiling = 1u << 22;

  // int64_t is `long long` on Darwin but `long` on Linux/LP64; all
  // three ctors are needed or gcc finds Op(int64_t) ambiguous.
  exr() = default;
  exr(int x) : v(ax::bigint(x)) {}                    // NOLINT
  exr(long x) : v(ax::bigint((long long)x)) {}        // NOLINT
  exr(long long x) : v(ax::bigint(x)) {}              // NOLINT
  explicit exr(ax::rational r) : v(std::move(r)) { guard(); }

  void guard() const {
    // cache keyed on the ceiling so runtime changes take effect
    static unsigned cached = 0;
    static ax::bigint ceil_big;
    if (cached != bit_ceiling) {
      ceil_big = ax::bigint(1) << bit_ceiling;
      cached = bit_ceiling;
    }
    if (ax::abs(v.num()) > ceil_big || v.den() > ceil_big)
      throw std::runtime_error("anchor: bit ceiling");
  }

  friend exr operator+(const exr& a, const exr& b) {
    return exr(a.v + b.v);
  }
  friend exr operator-(const exr& a, const exr& b) {
    return exr(a.v - b.v);
  }
  friend exr operator*(const exr& a, const exr& b) {
    return exr(a.v * b.v);
  }
  exr operator-() const { return exr(-v); }
  exr& operator+=(const exr& b) { return *this = *this + b; }
  exr& operator-=(const exr& b) { return *this = *this - b; }
  exr& operator*=(const exr& b) { return *this = *this * b; }
  friend bool operator==(const exr&, const exr&) = default;
  friend std::strong_ordering operator<=>(const exr& a,
                                           const exr& b) {
    return a.v <=> b.v;
  }
  friend exr operator<<(const exr& a, int n) {  // exact * 2^n
    return exr(ax::rational(a.v.num() << unsigned(n), a.v.den()));
  }
  friend exr operator>>(const exr& a, int n) {  // exact / 2^n
    return exr(ax::rational(a.v.num(), a.v.den() << unsigned(n)));
  }

  /** exact floor to bigint (negative-aware; bigint / truncates). */
  ax::bigint floor_big() const {
    const auto [q, r] = ax::bigint::divmod(v.num(), v.den());
    if (v.num().is_negative() && !r.is_zero()) return q - ax::bigint(1);
    return q;
  }
  /** DECLARED floor conversion (frozen-grain seams + de-grain).
      Same Darwin/Linux int64_t split as the ctors: both 64-bit
      builtin types need an operator. */
  explicit operator long long() const {
    const ax::bigint f = floor_big();
    // exact small-int extraction via decimal round-trip (cold path:
    // seams only, a handful per token per layer)
    return std::stoll(f.to_string());
  }
  explicit operator long() const {
    return (long)static_cast<long long>(*this);
  }
};

/** Zero value-rounding policy: div is exact; to_grain is the exact
    floor the convention pin declares at transcendental seams. */
struct Exact {
  static exr div(const exr& x, const exr& d) {
    return exr(x.v / d.v);
  }
  static exr div_trunc(const exr& x, const exr& d) {
    // exact — no truncation in the anchor. Seam experiment
    // (2026-08-09, tiny fixture): swapping this for shipped
    // trunc-to-integer semantics leaves anchor-vs-rung divergence
    // BIT-IDENTICAL (mean 20.285 both ways) — the seam is
    // exonerated; the residual is the frozen softmax-prob carry
    // (PQ-unit quantization, grain-independent by design).
    return exr(x.v / d.v);
  }
  static exr to_grain(const exr& x, int gshift) {
    const exr scaled = gshift ? (x >> gshift) : x;
    return exr(ax::rational(scaled.floor_big()));
  }
  static exr from_grain(const exr& x, int gshift) {
    return gshift ? (x << gshift) : x;
  }
};

/** The anchor loop: exact ring arithmetic over the same placement
    sites, same frozen-grain conventions, shipped-grain (p = 9). */
using anchor_birth = core::birth_impl<exr, exr, Exact>;

}  // namespace ax::nn::ib::anchor
