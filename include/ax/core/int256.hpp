#pragma once
/** @file int256.hpp Minimal fixed-width signed 256-bit integer for
    the ENGINE-EXACT-1 Q64 rung: the hot-loop accumulator where
    __int128 products of __int128 operands need headroom. Two's
    complement over four u64 limbs (little-endian). Only the ops the
    intbirth core's Acc concept uses: construct from integers,
    + - * << >>, comparisons, division (round toward zero, like the
    built-ins), unary -, %, and a checked narrow back to __int128.
    Cross-checked against ax::core::bigint by property tests. */
#include <array>
#include <cstdint>
#include <stdexcept>

namespace ax::core {

struct i256 {
  // little-endian limbs; l[3]'s MSB is the sign bit
  std::array<std::uint64_t, 4> l{0, 0, 0, 0};

  constexpr i256() = default;
  constexpr i256(int v) : i256(static_cast<__int128>(v)) {}
  constexpr i256(long v) : i256(static_cast<__int128>(v)) {}
  constexpr i256(long long v) : i256(static_cast<__int128>(v)) {}
  constexpr i256(__int128 v) {
    l[0] = std::uint64_t(v);
    l[1] = std::uint64_t(static_cast<unsigned __int128>(v) >> 64);
    const std::uint64_t ext = v < 0 ? ~0ull : 0ull;
    l[2] = ext;
    l[3] = ext;
  }

  bool negative() const { return l[3] >> 63; }

  friend i256 operator+(const i256& a, const i256& b) {
    i256 r;
    unsigned __int128 carry = 0;
    for (int i = 0; i < 4; i++) {
      const unsigned __int128 s =
          (unsigned __int128)a.l[i] + b.l[i] + carry;
      r.l[i] = std::uint64_t(s);
      carry = s >> 64;
    }
    return r;
  }
  i256 operator-() const {
    i256 r;
    for (int i = 0; i < 4; i++) r.l[i] = ~l[i];
    return r + i256(1);
  }
  friend i256 operator-(const i256& a, const i256& b) {
    return a + (-b);
  }
  i256& operator+=(const i256& b) { return *this = *this + b; }
  i256& operator-=(const i256& b) { return *this = *this - b; }

  friend bool operator==(const i256& a, const i256& b) {
    return a.l == b.l;
  }
  friend bool operator!=(const i256& a, const i256& b) {
    return !(a == b);
  }
  friend bool operator<(const i256& a, const i256& b) {
    if (a.negative() != b.negative()) return a.negative();
    for (int i = 3; i >= 0; i--)
      if (a.l[i] != b.l[i]) return a.l[i] < b.l[i];
    return false;
  }
  friend bool operator>(const i256& a, const i256& b) { return b < a; }
  friend bool operator<=(const i256& a, const i256& b) {
    return !(b < a);
  }
  friend bool operator>=(const i256& a, const i256& b) {
    return !(a < b);
  }

  /** unsigned magnitude helpers */
  i256 abs_u() const { return negative() ? -*this : *this; }

  friend i256 operator*(const i256& a, const i256& b) {
    // sign-magnitude schoolbook over u64 limbs, truncated to 256
    // bits (the engine never exceeds them; overflow beyond bit 255
    // is out of contract, matching built-in wrap semantics).
    const bool neg = a.negative() != b.negative();
    const i256 ua = a.abs_u(), ub = b.abs_u();
    i256 r;
    for (int i = 0; i < 4; i++) {
      unsigned __int128 carry = 0;
      for (int j = 0; i + j < 4; j++) {
        const unsigned __int128 cur =
            (unsigned __int128)ua.l[i] * ub.l[j] + r.l[i + j] + carry;
        r.l[i + j] = std::uint64_t(cur);
        carry = cur >> 64;
      }
    }
    return neg ? -r : r;
  }

  friend i256 operator<<(const i256& a, int n) {
    i256 r;
    const int limb = n / 64, bit = n % 64;
    for (int i = 3; i >= 0; i--) {
      std::uint64_t v = 0;
      if (i - limb >= 0) v = a.l[i - limb] << bit;
      if (bit && i - limb - 1 >= 0)
        v |= a.l[i - limb - 1] >> (64 - bit);
      r.l[i] = v;
    }
    return r;
  }
  friend i256 operator>>(const i256& a, int n) {  // arithmetic
    i256 r;
    const int limb = n / 64, bit = n % 64;
    const std::uint64_t ext = a.negative() ? ~0ull : 0ull;
    for (int i = 0; i < 4; i++) {
      std::uint64_t v = (i + limb < 4) ? a.l[i + limb] >> bit : ext;
      if (bit) {
        const std::uint64_t hi =
            (i + limb + 1 < 4) ? a.l[i + limb + 1] : ext;
        v |= hi << (64 - bit);
      }
      r.l[i] = v;
    }
    return r;
  }

  /** division truncated toward zero (built-in semantics); shift-
      subtract on magnitudes. d must be nonzero. */
  friend i256 operator/(const i256& a, const i256& b) {
    const bool neg = a.negative() != b.negative();
    i256 ua = a.abs_u();
    const i256 ub = b.abs_u();
    i256 q;
    if (!(ua < ub)) {
      int shift = ua.bits() - ub.bits();
      i256 d = ub << shift;
      for (; shift >= 0; shift--) {
        if (!(ua < d)) {
          ua -= d;
          q.l[shift / 64] |= (1ull << (shift % 64));
        }
        d = d >> 1;
      }
    }
    return neg ? -q : q;
  }
  friend i256 operator%(const i256& a, const i256& b) {
    return a - (a / b) * b;
  }

  int bits() const {  // magnitude bit length (this must be >= 0)
    for (int i = 3; i >= 0; i--)
      if (l[i]) {
        int b = 63;
        while (!(l[i] >> b)) b--;
        return i * 64 + b;
      }
    return 0;
  }

  /** explicit so accidental narrowing can't happen silently; the
      intbirth core's checked narrow<> is the intended path. Both
      64-bit builtin types get an operator because int64_t is
      `long long` on Darwin but `long` on Linux/LP64 — gcc will not
      route one through the other. */
  explicit operator __int128() const { return to_i128(); }
  explicit operator long long() const {
    const __int128 v = to_i128();
    return (long long)(v);
  }
  explicit operator long() const {
    const __int128 v = to_i128();
    return (long)(v);
  }

  __int128 to_i128() const {
    const std::uint64_t ext = negative() ? ~0ull : 0ull;
    if (l[2] != ext || l[3] != ext ||
        (std::uint64_t(l[1] >> 63) != (ext & 1)))
      throw std::runtime_error("i256: to_i128 overflow");
    return static_cast<__int128>(
        ((unsigned __int128)l[1] << 64) | l[0]);
  }
};

}  // namespace ax::core
