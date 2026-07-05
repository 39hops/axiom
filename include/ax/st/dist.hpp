#pragma once
/** @file dist.hpp Probability distributions: pdf/cdf/quantile/sample.
    All constructors validate parameters (std::invalid_argument).
    quantile(p) requires p in (0,1) (std::invalid_argument).
    mean()/var() throw std::domain_error where undefined. */
#include <ax/st/rng.hpp>

namespace ax::st {

struct normal_dist {
  double mu, sigma;
  normal_dist(double mu, double sigma);
  double pdf(double x) const;
  double cdf(double x) const;
  double quantile(double p) const;
  double sample(rng& g) const;
  double mean() const;
  double var() const;
};

struct t_dist {
  double nu;
  explicit t_dist(double nu);
  double pdf(double x) const;
  double cdf(double x) const;
  double quantile(double p) const;
  double sample(rng& g) const;
  double mean() const;  ///< requires nu > 1
  double var() const;   ///< requires nu > 2
};

struct chi2_dist {
  double k;
  explicit chi2_dist(double k);
  double pdf(double x) const;
  double cdf(double x) const;
  double quantile(double p) const;
  double sample(rng& g) const;
  double mean() const;
  double var() const;
};

struct f_dist {
  double d1, d2;
  f_dist(double d1, double d2);
  double pdf(double x) const;
  double cdf(double x) const;
  double quantile(double p) const;
  double sample(rng& g) const;
  double mean() const;  ///< requires d2 > 2
  double var() const;   ///< requires d2 > 4
};

struct gamma_dist {
  double shape, scale;
  gamma_dist(double shape, double scale);
  double pdf(double x) const;
  double cdf(double x) const;
  double quantile(double p) const;
  double sample(rng& g) const;
  double mean() const;
  double var() const;
};

struct beta_dist {
  double a, b;
  beta_dist(double a, double b);
  double pdf(double x) const;
  double cdf(double x) const;
  double quantile(double p) const;
  double sample(rng& g) const;
  double mean() const;
  double var() const;
};

struct exponential_dist {
  double lambda;
  explicit exponential_dist(double lambda);
  double pdf(double x) const;
  double cdf(double x) const;
  double quantile(double p) const;
  double sample(rng& g) const;
  double mean() const;
  double var() const;
};

struct uniform_dist {
  double a, b;
  uniform_dist(double a, double b);
  double pdf(double x) const;
  double cdf(double x) const;
  double quantile(double p) const;
  double sample(rng& g) const;
  double mean() const;
  double var() const;
};

struct lognormal_dist {
  double mu, sigma;
  lognormal_dist(double mu, double sigma);
  double pdf(double x) const;
  double cdf(double x) const;
  double quantile(double p) const;
  double sample(rng& g) const;
  double mean() const;
  double var() const;
};

struct weibull_dist {
  double k, lambda;
  weibull_dist(double k, double lambda);
  double pdf(double x) const;
  double cdf(double x) const;
  double quantile(double p) const;
  double sample(rng& g) const;
  double mean() const;
  double var() const;
};

struct cauchy_dist {
  double x0, gamma;
  cauchy_dist(double x0, double gamma);
  double pdf(double x) const;
  double cdf(double x) const;
  double quantile(double p) const;
  double sample(rng& g) const;
  double mean() const;  ///< undefined — always throws std::domain_error
  double var() const;   ///< undefined — always throws std::domain_error
};

struct laplace_dist {
  double mu, b;
  laplace_dist(double mu, double b);
  double pdf(double x) const;
  double cdf(double x) const;
  double quantile(double p) const;
  double sample(rng& g) const;
  double mean() const;
  double var() const;
};

struct binomial_dist {
  int n;
  double p;
  binomial_dist(int n, double p);  ///< n >= 0, p in (0,1)
  double pmf(int k) const;
  double cdf(int k) const;
  int quantile(double p) const;  ///< smallest k with cdf(k) >= p
  int sample(rng& g) const;
  double mean() const;
  double var() const;
};

struct poisson_dist {
  double lambda;
  explicit poisson_dist(double lambda);  ///< lambda > 0
  double pmf(int k) const;
  double cdf(int k) const;
  int quantile(double p) const;
  int sample(rng& g) const;
  double mean() const;
  double var() const;
};

/** Number of failures before the r-th success, success probability p. */
struct negbinom_dist {
  int r;
  double p;
  negbinom_dist(int r, double p);  ///< r >= 1, p in (0,1)
  double pmf(int k) const;
  double cdf(int k) const;
  int quantile(double p) const;
  int sample(rng& g) const;
  double mean() const;
  double var() const;
};

}  // namespace ax::st
