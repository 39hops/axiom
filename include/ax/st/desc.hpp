#pragma once
/** @file desc.hpp Descriptive statistics: one-pass moments, quantiles,
    covariance/correlation. */
#include <ax/la/mat.hpp>

#include <cstddef>
#include <span>
#include <vector>

namespace ax::st {

/** One-pass (Welford) accumulator for mean and central moments 2-4. */
struct moments {
  std::size_t n = 0;
  double mean = 0.0;
  double m2 = 0.0;  ///< sum of squared deviations
  double m3 = 0.0;
  double m4 = 0.0;

  void push(double x);
  double var() const;       ///< sample variance (n-1); throws domain_error n<2
  double sd() const;
  double skewness() const;  ///< g1; throws domain_error n<2 or zero variance
  double kurtosis() const;  ///< excess g2; same domain requirements
};

/** Accumulate all of xs. */
moments describe(std::span<const double> xs);

/** R type-7 quantile. p in [0,1], xs non-empty (std::invalid_argument). */
double quantile(std::vector<double> xs, double p);
/** Batch quantiles (single sort). */
std::vector<double> quantiles(std::vector<double> xs,
                              std::span<const double> ps);

/** Sample covariance matrix; rows = observations, cols = variables.
    Requires >= 2 rows (std::invalid_argument). */
la::mat covariance(const la::mat& x);
/** Pearson correlation matrix. */
la::mat correlation(const la::mat& x);

}  // namespace ax::st
