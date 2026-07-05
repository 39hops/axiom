#include <ax/st/dist.hpp>
#include <ax/st/sf.hpp>

#include <cmath>
#include <stdexcept>

namespace ax::st {

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kSqrt2 = 1.41421356237309504880168872420969808;

void check_p(double p) {
  if (!(p > 0.0 && p < 1.0))
    throw std::invalid_argument("quantile: p must be in (0,1)");
}

/** Invert a monotone cdf by bracket expansion + bisection.
    lo/hi: initial bracket. lo_open/hi_open: whether the support extends
    beyond the initial endpoint (expand geometrically if so). */
template <typename F>
double invert_cdf(F cdf, double p, double lo, double hi, bool lo_open,
                  bool hi_open) {
  double width = hi - lo;
  while (lo_open && cdf(lo) > p) {
    lo -= width;
    width *= 2.0;
  }
  width = hi - lo;
  while (hi_open && cdf(hi) < p) {
    hi += width;
    width *= 2.0;
  }
  for (int i = 0; i < 200; ++i) {
    const double mid = 0.5 * (lo + hi);
    if (mid == lo || mid == hi) break;  // full double precision reached
    if (cdf(mid) < p)
      lo = mid;
    else
      hi = mid;
  }
  return 0.5 * (lo + hi);
}

}  // namespace

// ---------------------------------------------------------------- normal

normal_dist::normal_dist(double mu_, double sigma_) : mu(mu_), sigma(sigma_) {
  if (!(sigma > 0.0))
    throw std::invalid_argument("normal_dist: sigma must be > 0");
}
double normal_dist::pdf(double x) const {
  const double z = (x - mu) / sigma;
  return std::exp(-0.5 * z * z) / (sigma * std::sqrt(2.0 * kPi));
}
double normal_dist::cdf(double x) const {
  return 0.5 * sf::erfc(-(x - mu) / (sigma * kSqrt2));
}
double normal_dist::quantile(double p) const {
  check_p(p);
  return mu + sigma * kSqrt2 * sf::erf_inv(2.0 * p - 1.0);
}
double normal_dist::sample(rng& g) const { return g.normal(mu, sigma); }
double normal_dist::mean() const { return mu; }
double normal_dist::var() const { return sigma * sigma; }

// ---------------------------------------------------------------- t

t_dist::t_dist(double nu_) : nu(nu_) {
  if (!(nu > 0.0)) throw std::invalid_argument("t_dist: nu must be > 0");
}
double t_dist::pdf(double x) const {
  const double lc = sf::lgamma(0.5 * (nu + 1.0)) - sf::lgamma(0.5 * nu) -
                    0.5 * std::log(nu * kPi);
  return std::exp(lc - 0.5 * (nu + 1.0) * std::log1p(x * x / nu));
}
double t_dist::cdf(double x) const {
  const double ib = sf::beta_inc(0.5 * nu, 0.5, nu / (nu + x * x));
  return x >= 0.0 ? 1.0 - 0.5 * ib : 0.5 * ib;
}
double t_dist::quantile(double p) const {
  check_p(p);
  return invert_cdf([this](double x) { return cdf(x); }, p, -1.0, 1.0, true,
                    true);
}
double t_dist::sample(rng& g) const {
  return g.normal() / std::sqrt(g.gamma(0.5 * nu, 2.0) / nu);
}
double t_dist::mean() const {
  if (!(nu > 1.0)) throw std::domain_error("t_dist::mean: requires nu > 1");
  return 0.0;
}
double t_dist::var() const {
  if (!(nu > 2.0)) throw std::domain_error("t_dist::var: requires nu > 2");
  return nu / (nu - 2.0);
}

// ---------------------------------------------------------------- chi2

chi2_dist::chi2_dist(double k_) : k(k_) {
  if (!(k > 0.0)) throw std::invalid_argument("chi2_dist: k must be > 0");
}
double chi2_dist::pdf(double x) const {
  if (x < 0.0) return 0.0;
  if (x == 0.0) return k < 2.0 ? 0.0 : (k == 2.0 ? 0.5 : 0.0);
  const double h = 0.5 * k;
  return std::exp((h - 1.0) * std::log(x) - 0.5 * x - h * std::log(2.0) -
                  sf::lgamma(h));
}
double chi2_dist::cdf(double x) const {
  if (x <= 0.0) return 0.0;
  return sf::gamma_p(0.5 * k, 0.5 * x);
}
double chi2_dist::quantile(double p) const {
  check_p(p);
  return invert_cdf([this](double x) { return cdf(x); }, p, 0.0, k + 1.0,
                    false, true);
}
double chi2_dist::sample(rng& g) const { return g.gamma(0.5 * k, 2.0); }
double chi2_dist::mean() const { return k; }
double chi2_dist::var() const { return 2.0 * k; }

// ---------------------------------------------------------------- f

f_dist::f_dist(double d1_, double d2_) : d1(d1_), d2(d2_) {
  if (!(d1 > 0.0) || !(d2 > 0.0))
    throw std::invalid_argument("f_dist: d1, d2 must be > 0");
}
double f_dist::pdf(double x) const {
  if (x < 0.0) return 0.0;
  if (x == 0.0) return 0.0;
  const double h1 = 0.5 * d1, h2 = 0.5 * d2;
  return std::exp(h1 * std::log(d1 / d2) + (h1 - 1.0) * std::log(x) -
                  (h1 + h2) * std::log1p(d1 * x / d2) - sf::log_beta(h1, h2));
}
double f_dist::cdf(double x) const {
  if (x <= 0.0) return 0.0;
  return sf::beta_inc(0.5 * d1, 0.5 * d2, d1 * x / (d1 * x + d2));
}
double f_dist::quantile(double p) const {
  check_p(p);
  return invert_cdf([this](double x) { return cdf(x); }, p, 0.0, 2.0, false,
                    true);
}
double f_dist::sample(rng& g) const {
  const double u = g.gamma(0.5 * d1, 2.0) / d1;
  const double v = g.gamma(0.5 * d2, 2.0) / d2;
  return u / v;
}
double f_dist::mean() const {
  if (!(d2 > 2.0)) throw std::domain_error("f_dist::mean: requires d2 > 2");
  return d2 / (d2 - 2.0);
}
double f_dist::var() const {
  if (!(d2 > 4.0)) throw std::domain_error("f_dist::var: requires d2 > 4");
  const double m = d2 - 2.0;
  return 2.0 * d2 * d2 * (d1 + d2 - 2.0) / (d1 * m * m * (d2 - 4.0));
}

// ---------------------------------------------------------------- gamma

gamma_dist::gamma_dist(double shape_, double scale_)
    : shape(shape_), scale(scale_) {
  if (!(shape > 0.0) || !(scale > 0.0))
    throw std::invalid_argument("gamma_dist: shape, scale must be > 0");
}
double gamma_dist::pdf(double x) const {
  if (x < 0.0) return 0.0;
  if (x == 0.0) return 0.0;
  return std::exp((shape - 1.0) * std::log(x) - x / scale -
                  shape * std::log(scale) - sf::lgamma(shape));
}
double gamma_dist::cdf(double x) const {
  if (x <= 0.0) return 0.0;
  return sf::gamma_p(shape, x / scale);
}
double gamma_dist::quantile(double p) const {
  check_p(p);
  return invert_cdf([this](double x) { return cdf(x); }, p, 0.0,
                    shape * scale + scale, false, true);
}
double gamma_dist::sample(rng& g) const { return g.gamma(shape, scale); }
double gamma_dist::mean() const { return shape * scale; }
double gamma_dist::var() const { return shape * scale * scale; }

// ---------------------------------------------------------------- beta

beta_dist::beta_dist(double a_, double b_) : a(a_), b(b_) {
  if (!(a > 0.0) || !(b > 0.0))
    throw std::invalid_argument("beta_dist: a, b must be > 0");
}
double beta_dist::pdf(double x) const {
  if (x <= 0.0 || x >= 1.0) return 0.0;
  return std::exp((a - 1.0) * std::log(x) + (b - 1.0) * std::log1p(-x) -
                  sf::log_beta(a, b));
}
double beta_dist::cdf(double x) const {
  if (x <= 0.0) return 0.0;
  if (x >= 1.0) return 1.0;
  return sf::beta_inc(a, b, x);
}
double beta_dist::quantile(double p) const {
  check_p(p);
  return invert_cdf([this](double x) { return cdf(x); }, p, 0.0, 1.0, false,
                    false);
}
double beta_dist::sample(rng& g) const {
  const double x = g.gamma(a, 1.0);
  const double y = g.gamma(b, 1.0);
  return x / (x + y);
}
double beta_dist::mean() const { return a / (a + b); }
double beta_dist::var() const {
  const double s = a + b;
  return a * b / (s * s * (s + 1.0));
}

// ---------------------------------------------------------------- exponential

exponential_dist::exponential_dist(double lambda_) : lambda(lambda_) {
  if (!(lambda > 0.0))
    throw std::invalid_argument("exponential_dist: lambda must be > 0");
}
double exponential_dist::pdf(double x) const {
  return x < 0.0 ? 0.0 : lambda * std::exp(-lambda * x);
}
double exponential_dist::cdf(double x) const {
  return x <= 0.0 ? 0.0 : 1.0 - std::exp(-lambda * x);
}
double exponential_dist::quantile(double p) const {
  check_p(p);
  return -std::log1p(-p) / lambda;
}
double exponential_dist::sample(rng& g) const {
  return g.exponential(lambda);
}
double exponential_dist::mean() const { return 1.0 / lambda; }
double exponential_dist::var() const { return 1.0 / (lambda * lambda); }

// ---------------------------------------------------------------- uniform

uniform_dist::uniform_dist(double a_, double b_) : a(a_), b(b_) {
  if (!(a < b)) throw std::invalid_argument("uniform_dist: requires a < b");
}
double uniform_dist::pdf(double x) const {
  return (x < a || x > b) ? 0.0 : 1.0 / (b - a);
}
double uniform_dist::cdf(double x) const {
  if (x <= a) return 0.0;
  if (x >= b) return 1.0;
  return (x - a) / (b - a);
}
double uniform_dist::quantile(double p) const {
  check_p(p);
  return a + p * (b - a);
}
double uniform_dist::sample(rng& g) const { return g.uniform(a, b); }
double uniform_dist::mean() const { return 0.5 * (a + b); }
double uniform_dist::var() const {
  const double w = b - a;
  return w * w / 12.0;
}

// ---------------------------------------------------------------- lognormal

lognormal_dist::lognormal_dist(double mu_, double sigma_)
    : mu(mu_), sigma(sigma_) {
  if (!(sigma > 0.0))
    throw std::invalid_argument("lognormal_dist: sigma must be > 0");
}
double lognormal_dist::pdf(double x) const {
  if (x <= 0.0) return 0.0;
  const double z = (std::log(x) - mu) / sigma;
  return std::exp(-0.5 * z * z) / (x * sigma * std::sqrt(2.0 * kPi));
}
double lognormal_dist::cdf(double x) const {
  if (x <= 0.0) return 0.0;
  return 0.5 * sf::erfc(-(std::log(x) - mu) / (sigma * kSqrt2));
}
double lognormal_dist::quantile(double p) const {
  check_p(p);
  return std::exp(mu + sigma * kSqrt2 * sf::erf_inv(2.0 * p - 1.0));
}
double lognormal_dist::sample(rng& g) const {
  return std::exp(g.normal(mu, sigma));
}
double lognormal_dist::mean() const {
  return std::exp(mu + 0.5 * sigma * sigma);
}
double lognormal_dist::var() const {
  const double s2 = sigma * sigma;
  return (std::exp(s2) - 1.0) * std::exp(2.0 * mu + s2);
}

// ---------------------------------------------------------------- weibull

weibull_dist::weibull_dist(double k_, double lambda_) : k(k_), lambda(lambda_) {
  if (!(k > 0.0) || !(lambda > 0.0))
    throw std::invalid_argument("weibull_dist: k, lambda must be > 0");
}
double weibull_dist::pdf(double x) const {
  if (x < 0.0) return 0.0;
  if (x == 0.0) return k < 1.0 ? 0.0 : (k == 1.0 ? 1.0 / lambda : 0.0);
  const double z = x / lambda;
  return k / lambda * std::pow(z, k - 1.0) * std::exp(-std::pow(z, k));
}
double weibull_dist::cdf(double x) const {
  if (x <= 0.0) return 0.0;
  return 1.0 - std::exp(-std::pow(x / lambda, k));
}
double weibull_dist::quantile(double p) const {
  check_p(p);
  return lambda * std::pow(-std::log1p(-p), 1.0 / k);
}
double weibull_dist::sample(rng& g) const {
  return lambda * std::pow(-std::log(1.0 - g.next_double()), 1.0 / k);
}
double weibull_dist::mean() const {
  return lambda * std::exp(sf::lgamma(1.0 + 1.0 / k));
}
double weibull_dist::var() const {
  const double g1 = std::exp(sf::lgamma(1.0 + 1.0 / k));
  const double g2 = std::exp(sf::lgamma(1.0 + 2.0 / k));
  return lambda * lambda * (g2 - g1 * g1);
}

// ---------------------------------------------------------------- cauchy

cauchy_dist::cauchy_dist(double x0_, double gamma_) : x0(x0_), gamma(gamma_) {
  if (!(gamma > 0.0))
    throw std::invalid_argument("cauchy_dist: gamma must be > 0");
}
double cauchy_dist::pdf(double x) const {
  const double z = (x - x0) / gamma;
  return 1.0 / (kPi * gamma * (1.0 + z * z));
}
double cauchy_dist::cdf(double x) const {
  return 0.5 + std::atan((x - x0) / gamma) / kPi;
}
double cauchy_dist::quantile(double p) const {
  check_p(p);
  return x0 + gamma * std::tan(kPi * (p - 0.5));
}
double cauchy_dist::sample(rng& g) const {
  return x0 + gamma * std::tan(kPi * (g.next_double() - 0.5));
}
double cauchy_dist::mean() const {
  throw std::domain_error("cauchy_dist::mean: undefined");
}
double cauchy_dist::var() const {
  throw std::domain_error("cauchy_dist::var: undefined");
}

// ---------------------------------------------------------------- laplace

laplace_dist::laplace_dist(double mu_, double b_) : mu(mu_), b(b_) {
  if (!(b > 0.0)) throw std::invalid_argument("laplace_dist: b must be > 0");
}
double laplace_dist::pdf(double x) const {
  return 0.5 / b * std::exp(-std::abs(x - mu) / b);
}
double laplace_dist::cdf(double x) const {
  if (x < mu) return 0.5 * std::exp((x - mu) / b);
  return 1.0 - 0.5 * std::exp(-(x - mu) / b);
}
double laplace_dist::quantile(double p) const {
  check_p(p);
  if (p < 0.5) return mu + b * std::log(2.0 * p);
  return mu - b * std::log(2.0 * (1.0 - p));
}
double laplace_dist::sample(rng& g) const {
  const double u = g.next_double();
  if (u < 0.5) return mu + b * std::log(2.0 * u + 1e-300);
  return mu - b * std::log(2.0 * (1.0 - u));
}
double laplace_dist::mean() const { return mu; }
double laplace_dist::var() const { return 2.0 * b * b; }

}  // namespace ax::st
