#pragma once
/** @file poly.hpp Univariate polynomial over exact rationals (dense,
    lowest-degree-first coefficients, trailing zeros trimmed). */
#include <ax/core/rational.hpp>
#include <ax/sym/expr.hpp>

#include <utility>
#include <vector>

namespace ax::sym {

class poly {
 public:
  /** Zero polynomial. */
  poly() = default;
  /** c[0] + c[1] x + ... ; trailing zeros trimmed. */
  explicit poly(std::vector<rational> coeffs);

  /** Interpret e as a polynomial in symbol s. Throws std::invalid_argument
      if e contains other symbols, functions, or non-integer powers of s. */
  static poly from_expr(const expr& e, const expr& s);
  expr to_expr(const expr& s) const;

  int degree() const { return static_cast<int>(c_.size()) - 1; }
  /** Coefficient of x^k (zero beyond degree). */
  const rational& coeff(std::size_t k) const;

  friend poly operator+(const poly& a, const poly& b);
  friend poly operator-(const poly& a, const poly& b);
  friend poly operator*(const poly& a, const poly& b);
  friend bool operator==(const poly& a, const poly& b) = default;

  /** Euclidean division: {quotient, remainder}, deg(rem) < deg(d).
      Throws std::domain_error when d is zero. */
  std::pair<poly, poly> divmod(const poly& d) const;
  /** Monic greatest common divisor (gcd(0,0) = 0). */
  friend poly gcd(poly a, poly b);
  poly derivative() const;
  /** Square-free part p / gcd(p, p'), monic-normalized. */
  poly square_free() const;
  /** All rational roots (each listed once). Uses the rational root theorem;
      requires lead/const coefficients to fit in 64 bits after clearing
      denominators (std::overflow_error otherwise). */
  std::vector<rational> rational_roots() const;
  rational eval(const rational& x) const;

 private:
  std::vector<rational> c_;
  void trim();
};

}  // namespace ax::sym
