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

}  // namespace ax::st
