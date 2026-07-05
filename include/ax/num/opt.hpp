#pragma once
/** @file opt.hpp Optimization: Brent 1-d, Nelder-Mead, BFGS. */
#include <ax/la/mat.hpp>

#include <functional>

namespace ax::num {

struct min_result_1d {
  double x, fx;
  int iters;
  bool converged;
};

/** Brent minimization (golden section + parabolic interpolation) on [a,b]. */
min_result_1d minimize(const std::function<double(double)>& f, double a,
                       double b, double xtol = 1e-10, int max_iter = 200);

struct min_result {
  la::vec x;
  double fx;
  int iters;
  bool converged;
};

/** Nelder-Mead downhill simplex. Converged when the simplex f-spread
    falls below ftol (relative). */
min_result nelder_mead(const std::function<double(const la::vec&)>& f,
                       la::vec x0, double ftol = 1e-10, int max_iter = 2000);

/** BFGS with user gradient; Armijo backtracking line search.
    Converged when max |g_i| < gtol. */
min_result bfgs(const std::function<double(const la::vec&)>& f,
                const std::function<la::vec(const la::vec&)>& grad,
                la::vec x0, double gtol = 1e-8, int max_iter = 500);

/** BFGS with central-difference numeric gradient. */
min_result bfgs(const std::function<double(const la::vec&)>& f, la::vec x0,
                double gtol = 1e-8, int max_iter = 500);

}  // namespace ax::num
