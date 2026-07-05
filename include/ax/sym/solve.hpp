#pragma once
/** @file solve.hpp Equation solving: polynomial roots (exact where radical
    forms stay sane, Durand-Kerner numeric fallback). */
#include <ax/sym/expr.hpp>

#include <complex>
#include <span>
#include <vector>

namespace ax::sym {

/** Roots of a univariate polynomial equation p == 0. */
struct poly_roots {
  std::vector<expr> exact;  ///< closed-form real roots, verified exact
  std::vector<std::complex<double>> approx;  ///< numeric roots (rest)
  bool complete_exact;  ///< true iff every root is in `exact`
};

/** Solve p(x) == 0 for rational-coefficient p (poly::from_expr must accept
    it — std::invalid_argument otherwise; also thrown for constant p).
    Exact ladder: rational-root deflation (with multiplicity) → quadratic
    formula → cubic via Cardano (radical form when one real root, trig form
    for three real roots) → biquadratic quartics. Roots with no real
    closed form (or degree > 4 residuals) land in approx via Durand-Kerner;
    nonreal roots are never in exact (expr is real-valued). */
poly_roots solve_poly(const expr& equation_lhs, const expr& x);

/** All complex roots of the polynomial with the given coefficients
    (lowest degree first, degree >= 1, nonzero leading coefficient —
    std::invalid_argument otherwise) by Durand-Kerner iteration from
    deterministic start points (0.4+0.9i)^k. */
std::vector<std::complex<double>> durand_kerner(std::span<const double> coeffs,
                                                double tol = 1e-12,
                                                int max_iter = 200);

}  // namespace ax::sym
