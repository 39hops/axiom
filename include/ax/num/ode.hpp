#pragma once
/** @file ode.hpp Adaptive Runge-Kutta 4(5) Dormand-Prince IVP solver. */
#include <ax/la/mat.hpp>

#include <functional>
#include <vector>

namespace ax::num {

struct ode_result {
  std::vector<double> t;      ///< accepted step times (t[0] = t0)
  std::vector<la::vec> y;     ///< states at t
  int steps = 0;              ///< accepted steps
  int rejected = 0;           ///< rejected step attempts
  bool converged = true;      ///< false if step budget exhausted
};

/** Integrate y' = f(t, y) from t0 to t1 (t1 > t0) with adaptive
    Dormand-Prince RK45. h0 = 0 picks (t1-t0)/100. */
ode_result solve_ivp(const std::function<la::vec(double, const la::vec&)>& f,
                     double t0, double t1, la::vec y0, double reltol = 1e-8,
                     double abstol = 1e-10, double h0 = 0.0);

}  // namespace ax::num
