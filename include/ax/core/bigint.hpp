#pragma once
/** @file bigint.hpp Arbitrary-precision signed integer. */
#include <compare>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ax {

/**
 * Arbitrary-precision signed integer.
 * Representation: sign + little-endian 64-bit limbs, no leading zero limbs;
 * zero is the empty limb vector with non-negative sign.
 */
class bigint {
 public:
  /// Zero.
  bigint() = default;
  /// From built-in signed integer.
  bigint(long long v);  // NOLINT(google-explicit-constructor)
  /// From decimal string, optional leading '-'. Throws std::invalid_argument.
  explicit bigint(std::string_view dec);

  /// True iff value is zero.
  bool is_zero() const noexcept { return limbs_.empty(); }
  /// True iff value is negative.
  bool is_negative() const noexcept { return neg_; }
  /// Decimal representation.
  std::string to_string() const;

  /// Negation; -0 stays 0.
  bigint operator-() const;

  friend bool operator==(const bigint&, const bigint&) = default;
  friend std::strong_ordering operator<=>(const bigint& a, const bigint& b);
  friend bigint operator+(const bigint& a, const bigint& b);
  friend bigint operator-(const bigint& a, const bigint& b);
  friend bigint operator*(const bigint& a, const bigint& b);
  /// Magnitude shift left by bits; sign preserved.
  friend bigint operator<<(const bigint& a, unsigned bits);
  /// Magnitude shift right by bits (truncates toward zero on negatives).
  friend bigint operator>>(const bigint& a, unsigned bits);
  friend bigint operator/(const bigint& a, const bigint& b);
  friend bigint operator%(const bigint& a, const bigint& b);

  /** Quotient and remainder in one pass. Truncated division:
      q = trunc(a/b), r = a - q*b (r takes sign of a).
      Throws std::domain_error when b == 0. */
  static std::pair<bigint, bigint> divmod(const bigint& a, const bigint& b);

 private:
  bool neg_ = false;                 ///< Sign; false for zero.
  std::vector<std::uint64_t> limbs_; ///< Little-endian magnitude.
  void trim();                       ///< Drop leading zero limbs, fix -0.
};

/** Absolute value. */
bigint abs(const bigint& a);
/** Greatest common divisor of |a| and |b|; gcd(0,0) == 0. */
bigint gcd(bigint a, bigint b);
/** a raised to non-negative exponent e (square-and-multiply). */
bigint pow(const bigint& a, unsigned long long e);
/** (base^exp) mod m, exp >= 0, m > 0. Throws std::domain_error on m <= 0. */
bigint modpow(bigint base, bigint exp, const bigint& m);

}  // namespace ax
