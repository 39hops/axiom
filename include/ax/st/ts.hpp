#pragma once
/** @file ts.hpp Time series: ACF/PACF, AR (Yule-Walker) and ARMA (CSS)
    fitting, periodogram. */
#include <cstddef>
#include <span>
#include <vector>

namespace ax::st {

/** Sample autocorrelation at lags 0..max_lag (acf[0] == 1). Biased
    estimator (denominator n). Requires n > max_lag
    (std::invalid_argument). */
std::vector<double> acf(std::span<const double> xs, std::size_t max_lag);

/** Partial autocorrelation at lags 1..max_lag via Durbin-Levinson.
    Requires n > max_lag >= 1. */
std::vector<double> pacf(std::span<const double> xs, std::size_t max_lag);

/** AR(p) fit. */
struct ar_result {
  std::vector<double> phi;  ///< AR coefficients, lags 1..p
  double intercept;         ///< c in x_t = c + sum phi_i x_{t-i} + e_t
  double sigma2;            ///< innovation variance
};

/** Fit AR(p) by Yule-Walker on the demeaned series. Requires n > p >= 1. */
ar_result ar_fit(std::span<const double> xs, std::size_t p);

/** ARMA(p,q) fit. */
struct arma_result {
  std::vector<double> phi, theta;
  double intercept;
  double sigma2;
  bool converged;
};

/** Fit ARMA(p,q) by minimizing the conditional sum of squares (pre-sample
    values zero) with Nelder-Mead; AR part initialized from Yule-Walker,
    MA from zeros. Requires n > p + q, p + q >= 1. */
arma_result arma_fit(std::span<const double> xs, std::size_t p,
                     std::size_t q);

/** Raw periodogram. */
struct periodogram_result {
  std::vector<double> freq;   ///< cycles per observation, in (0, 0.5]
  std::vector<double> power;  ///< |X_k|^2 / n
};

/** Periodogram of the demeaned series via FFT. The series is zero-padded to
    the next power of two m; frequencies are k/m for k = 1..m/2. Requires a
    non-empty input. */
periodogram_result periodogram(std::span<const double> xs);

}  // namespace ax::st
