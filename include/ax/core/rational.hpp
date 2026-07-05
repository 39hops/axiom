#pragma once
/** @file rational.hpp Exact rational number over bigint. */
#include <ax/core/bigint.hpp>

#include <compare>
#include <string>
#include <utility>

namespace ax {

/**
 * Exact rational, always normalized: gcd(num, den) == 1, den > 0,
 * zero is 0/1. Sign carried by numerator.
 */
class rational {
 public:
  /// Zero.
  rational() : num_{0}, den_{1} {}
  /// From integer.
  explicit rational(bigint n) : num_{std::move(n)}, den_{1} {}
  /// n/d, normalized. Throws std::domain_error if d == 0.
  rational(bigint n, bigint d);

  const bigint& num() const noexcept { return num_; }
  const bigint& den() const noexcept { return den_; }
  bool is_zero() const noexcept { return num_.is_zero(); }
  /// "n/d", or "n" when d == 1.
  std::string to_string() const;

  /// Negation.
  rational operator-() const;

  friend rational operator+(const rational& a, const rational& b);
  friend rational operator-(const rational& a, const rational& b);
  friend rational operator*(const rational& a, const rational& b);
  /// Throws std::domain_error when b == 0.
  friend rational operator/(const rational& a, const rational& b);

  friend bool operator==(const rational&, const rational&) = default;
  friend std::strong_ordering operator<=>(const rational& a, const rational& b);

 private:
  bigint num_, den_;
  void normalize();
};

}  // namespace ax
