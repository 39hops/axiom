#pragma once
/** @file htest.hpp Classical hypothesis tests: t (one/two-sample, Welch),
    chi-square (goodness of fit, independence), one-way ANOVA, one-sample
    Kolmogorov-Smirnov. All validate inputs with std::invalid_argument. */
#include <ax/la/mat.hpp>

#include <functional>
#include <span>
#include <vector>

namespace ax::st {

/** Alternative hypothesis direction for t tests. */
enum class alternative { two_sided, less, greater };

/** Common test output. */
struct test_result {
  double statistic;  ///< test statistic
  double df;         ///< degrees of freedom (df1 for F; NaN where n/a)
  double df2;        ///< second df (F/ANOVA only; NaN otherwise)
  double p_value;
};

/** One-sample t test of H0: mean == mu0. Requires n >= 2. */
test_result t_test(std::span<const double> xs, double mu0,
                   alternative alt = alternative::two_sided);

/** Two-sample t test of H0: mean(xs) == mean(ys). welch=true (default) uses
    the Welch-Satterthwaite approximation; welch=false pools variances
    (df = n1+n2-2). Each sample requires n >= 2. */
test_result t_test(std::span<const double> xs, std::span<const double> ys,
                   alternative alt = alternative::two_sided, bool welch = true);

/** Chi-square goodness of fit. observed: counts; expected_p: cell
    probabilities (renormalized to sum 1). Sizes must match, >= 2 cells,
    every expected count > 0. df = k-1. */
test_result chi2_gof(std::span<const double> observed,
                     std::span<const double> expected_p);

/** Chi-square test of independence on an r x c table of counts.
    Requires r,c >= 2 and positive row/column sums. df = (r-1)(c-1). */
test_result chi2_independence(const la::mat& table);

/** One-way ANOVA. Requires >= 2 groups, each with >= 2 observations.
    statistic = F, df = k-1, df2 = N-k. */
test_result anova_oneway(std::span<const std::vector<double>> groups);

/** One-sample Kolmogorov-Smirnov against a continuous CDF.
    statistic = D_n; p-value from the asymptotic Kolmogorov distribution
    Q(sqrt(n) * D) — approximate for small n (good for n >= ~10). df = NaN.
    Requires n >= 1. */
test_result ks_test(std::span<const double> xs,
                    const std::function<double(double)>& cdf);

}  // namespace ax::st
