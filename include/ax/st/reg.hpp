#pragma once
/** @file reg.hpp Regression: OLS with inference (via QR). */
#include <ax/la/mat.hpp>

#include <cstddef>

namespace ax::st {

/** OLS fit output. When an intercept is fitted it is beta[0]. */
struct ols_result {
  la::vec beta;       ///< coefficients (intercept first if fitted)
  la::vec stderrs;    ///< standard errors
  la::vec t_stats;    ///< beta / stderr
  la::vec p_values;   ///< two-sided vs t(df_resid)
  la::vec residuals;  ///< y - X beta
  double sigma2;      ///< RSS / df_resid
  double r2;          ///< vs mean if intercept fitted, vs zero otherwise
  double adj_r2;
  std::size_t df_resid;
};

/** Fit y = X beta by least squares; a column of ones is prepended when
    intercept=true. Requires rows(x) == size(y) and more observations than
    parameters (std::invalid_argument); throws std::domain_error on rank
    deficiency. */
ols_result ols(const la::mat& x, const la::vec& y, bool intercept = true);

/** GLM families with canonical links: logistic (logit), poisson (log). */
enum class glm_family { logistic, poisson };

/** GLM fit output. */
struct glm_result {
  la::vec beta;
  la::vec stderrs;  ///< sqrt(diag((X'WX)^{-1})) at the final iterate
  double deviance;
  std::size_t iters;
  bool converged;
};

/** Fit a GLM with canonical link by iteratively reweighted least squares.
    logistic: y must be in {0,1}; poisson: y must be >= 0. A ones column is
    prepended when intercept=true. Stops when max |delta beta| < tol; if
    max_iter is hit first the result has converged=false (no throw) — perfect
    separation surfaces this way. Throws std::invalid_argument on bad inputs. */
glm_result glm_fit(const la::mat& x, const la::vec& y, glm_family family,
                   bool intercept = true, std::size_t max_iter = 50,
                   double tol = 1e-10);

}  // namespace ax::st
