#pragma once
/** @file root.hpp Scalar root finding: bisection, Brent, Newton. */
#include <functional>

namespace ax::num {

struct root_result {
  double x;        ///< best root estimate
  double fx;       ///< f(x)
  int iters;       ///< iterations used
  bool converged;  ///< tolerance met within max_iter
};

/** Bisection on [a,b]. Requires f(a) f(b) <= 0 (std::invalid_argument). */
root_result bisect(const std::function<double(double)>& f, double a, double b,
                   double xtol = 1e-12, int max_iter = 200);

/** Brent's method (inverse quadratic / secant / bisection). Same bracket
    contract as bisect. */
root_result brent(const std::function<double(double)>& f, double a, double b,
                  double xtol = 1e-14, int max_iter = 200);

/** Newton-Raphson with user-supplied derivative. Throws std::runtime_error
    on zero derivative. converged=false if max_iter exhausted. */
root_result newton(const std::function<double(double)>& f,
                   const std::function<double(double)>& df, double x0,
                   double xtol = 1e-14, int max_iter = 100);

}  // namespace ax::num
