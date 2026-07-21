#pragma once
/** @file series.hpp Dense truncated power series over exact rationals.

    A value represents sum_{k<N} c[k] x^k + O(x^N) where N == order().
    Invariant: coeffs().size() == order() (zeros kept, never trimmed —
    truncation order is information, unlike poly's degree). All arithmetic
    is exact; binary ops truncate to the smaller order. Immutable value
    type in the style of poly. */
#include <ax/core/rational.hpp>
#include <ax/sym/expr.hpp>

#include <vector>

namespace ax::sym {

class series {
 public:
  /** Empty series O(x^0): zero known coefficients. */
  series() = default;
  /** Zero series with n known coefficients: 0 + O(x^n). */
  explicit series(int order);
  /** Coefficients c[0] + c[1] x + ...; padded/truncated to order. */
  series(std::vector<rational> coeffs, int order);

  /** Maclaurin expansion of e (a function of x) to the given order.
      Supports numbers, x, add, mul, integer powers, and exp/sin/cos
      with u(0) == 0, log/sqrt with u(0) == 1. Throws std::domain_error
      on anything else (other symbols, shifted transcendentals) — the
      oracle maps that to UNDECIDED. */
  static series of_expr(const expr& e, const expr& x, int order);

  int order() const { return static_cast<int>(c_.size()); }
  /** Coefficient of x^k; std::out_of_range at or beyond order(). */
  const rational& coeff(std::size_t k) const;
  const std::vector<rational>& coeffs() const { return c_; }
  bool is_zero() const;

  friend series operator+(const series& a, const series& b);
  friend series operator-(const series& a, const series& b);
  series operator-() const;
  /** Cauchy product, truncated to min(order). O(N^2). */
  friend series operator*(const series& a, const series& b);
  friend bool operator==(const series& a, const series& b) = default;

  /** Termwise d/dx; order drops by one. */
  series derivative() const;
  /** Termwise antiderivative with constant 0; order grows by one. */
  series integrate() const;
  /** Multiplicative inverse; std::domain_error when c[0] == 0. */
  series inverse() const;
  /** Integer power (negative via inverse). pow(0) == 1 + O(x^order). */
  series pow_int(long long e) const;
  /** this(inner); std::domain_error unless inner c[0] == 0. */
  series compose(const series& inner) const;

  /** Partial-sum polynomial as an expr in x (for printing/chain rows). */
  expr to_expr(const expr& x) const;

 private:
  std::vector<rational> c_;
};

/** Maclaurin series of the named function applied to u: exp/sin/cos need
    u(0) == 0, log/sqrt need u(0) == 1. std::domain_error otherwise (and
    on unknown names) — same contract as series::of_expr. */
series apply_fn(const std::string& name, const series& u);

}  // namespace ax::sym
