#pragma once
/** @file mcmc.hpp Markov chain Monte Carlo: random-walk Metropolis and an
    effective-sample-size diagnostic. */
#include <ax/la/mat.hpp>
#include <ax/st/rng.hpp>

#include <cstddef>
#include <functional>
#include <span>

namespace ax::st {

/** Options for metropolis(). */
struct mh_options {
  std::size_t n_samples = 1000;  ///< kept samples (post burn-in, post thin)
  std::size_t burn_in = 1000;
  std::size_t thin = 1;  ///< keep every thin-th draw (>= 1)
  double step = 1.0;     ///< isotropic gaussian proposal standard deviation
};

/** Metropolis chain output. */
struct mh_result {
  la::mat samples;         ///< n_samples x dim
  double acceptance_rate;  ///< accepted / proposed over the whole run
};

/** Random-walk Metropolis with N(0, step^2 I) proposals.
    log_density is the unnormalized log target; it may return -infinity for
    points outside the support (such proposals are always rejected).
    Throws std::invalid_argument if log_density(x0) is not finite, or if
    thin == 0, n_samples == 0, step <= 0, or x0 is empty. */
mh_result metropolis(const std::function<double(const la::vec&)>& log_density,
                     const la::vec& x0, const mh_options& opt, rng& g);

/** Effective sample size of a chain component: n / (1 + 2 sum rho_k) with
    the sum truncated by Geyer's initial positive sequence rule (adjacent
    autocorrelation pairs kept while their sum is positive).
    Requires n >= 2 (std::invalid_argument). */
double ess(std::span<const double> chain);

}  // namespace ax::st
