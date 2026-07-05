#pragma once
/** @file sf.hpp Special functions: lgamma, erf, incomplete gamma/beta. */

namespace ax::st::sf {

/** ln Γ(x) for x > 0 (Lanczos). Throws std::domain_error for x <= 0. */
double lgamma(double x);
/** Error function. */
double erf(double x);
/** Complementary error function, accurate (relative) for large x. */
double erfc(double x);
/** Inverse error function, |p| < 1. Throws std::domain_error otherwise. */
double erf_inv(double p);
/** Regularized lower incomplete gamma P(a,x), a > 0, x >= 0. */
double gamma_p(double a, double x);
/** Regularized upper incomplete gamma Q(a,x) = 1 - P(a,x). */
double gamma_q(double a, double x);
/** Regularized incomplete beta I_x(a,b), a,b > 0, x in [0,1]. */
double beta_inc(double a, double b, double x);
/** ln B(a,b) = lgamma(a) + lgamma(b) - lgamma(a+b). */
double log_beta(double a, double b);

}  // namespace ax::st::sf
