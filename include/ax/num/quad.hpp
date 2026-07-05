#pragma once
/** @file quad.hpp Numerical integration: adaptive Gauss-Kronrod G7-K15 and
    tanh-sinh (double-exponential) for endpoint singularities. */
#include <functional>

namespace ax::num {

struct quad_result {
  double value;      ///< integral estimate
  double error_est;  ///< estimated absolute error
  int evals;         ///< function evaluations
};

/** Adaptive G7-K15 on [a,b] (recursive bisection until the local Kronrod
    error estimate meets abstol/reltol). a > b integrates with flipped sign. */
quad_result integrate(const std::function<double(double)>& f, double a,
                      double b, double abstol = 1e-10, double reltol = 1e-10,
                      int max_depth = 50);

/** tanh-sinh quadrature on (a,b); tolerates integrable endpoint
    singularities. Levels double the node density until |change| < tol.
    Note: if the integrand internally computes distance to an endpoint by
    subtraction (e.g. 1-x near x=1), accuracy is limited to ~1e-7 by double
    rounding at the endpoint; a Boost-style f(x, xc) overload is a v2
    candidate if that matters. */
quad_result integrate_ts(const std::function<double(double)>& f, double a,
                         double b, double tol = 1e-10, int max_level = 12);

}  // namespace ax::num
