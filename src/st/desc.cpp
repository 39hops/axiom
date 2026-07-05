#include <ax/st/desc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ax::st {

void moments::push(double x) {
  const double n1 = static_cast<double>(n);
  ++n;
  const double nn = static_cast<double>(n);
  const double delta = x - mean;
  const double delta_n = delta / nn;
  const double delta_n2 = delta_n * delta_n;
  const double term1 = delta * delta_n * n1;
  mean += delta_n;
  m4 += term1 * delta_n2 * (nn * nn - 3.0 * nn + 3.0) +
        6.0 * delta_n2 * m2 - 4.0 * delta_n * m3;
  m3 += term1 * delta_n * (nn - 2.0) - 3.0 * delta_n * m2;
  m2 += term1;
}

double moments::var() const {
  if (n < 2) throw std::domain_error("moments::var: requires n >= 2");
  return m2 / static_cast<double>(n - 1);
}

double moments::sd() const { return std::sqrt(var()); }

double moments::skewness() const {
  if (n < 2 || m2 == 0.0)
    throw std::domain_error("moments::skewness: undefined");
  const double nn = static_cast<double>(n);
  return (m3 / nn) / std::pow(m2 / nn, 1.5);
}

double moments::kurtosis() const {
  if (n < 2 || m2 == 0.0)
    throw std::domain_error("moments::kurtosis: undefined");
  const double nn = static_cast<double>(n);
  return nn * m4 / (m2 * m2) - 3.0;
}

moments describe(std::span<const double> xs) {
  moments m;
  for (const double x : xs) m.push(x);
  return m;
}

double quantile(std::vector<double> xs, double p) {
  if (xs.empty()) throw std::invalid_argument("quantile: empty data");
  if (!(p >= 0.0 && p <= 1.0))
    throw std::invalid_argument("quantile: p must be in [0,1]");
  std::sort(xs.begin(), xs.end());
  const double h = (static_cast<double>(xs.size()) - 1.0) * p;
  const std::size_t lo = static_cast<std::size_t>(h);
  if (lo + 1 >= xs.size()) return xs.back();
  const double frac = h - static_cast<double>(lo);
  return xs[lo] + frac * (xs[lo + 1] - xs[lo]);
}

std::vector<double> quantiles(std::vector<double> xs,
                              std::span<const double> ps) {
  if (xs.empty()) throw std::invalid_argument("quantiles: empty data");
  std::sort(xs.begin(), xs.end());
  std::vector<double> out;
  out.reserve(ps.size());
  for (const double p : ps) {
    if (!(p >= 0.0 && p <= 1.0))
      throw std::invalid_argument("quantiles: p must be in [0,1]");
    const double h = (static_cast<double>(xs.size()) - 1.0) * p;
    const std::size_t lo = static_cast<std::size_t>(h);
    if (lo + 1 >= xs.size()) {
      out.push_back(xs.back());
    } else {
      const double frac = h - static_cast<double>(lo);
      out.push_back(xs[lo] + frac * (xs[lo + 1] - xs[lo]));
    }
  }
  return out;
}

la::mat covariance(const la::mat& x) {
  const std::size_t n = x.rows(), d = x.cols();
  if (n < 2) throw std::invalid_argument("covariance: requires >= 2 rows");
  std::vector<double> mean(d, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < d; ++j) mean[j] += x(i, j);
  for (double& m : mean) m /= static_cast<double>(n);
  la::mat c(d, d);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t a = 0; a < d; ++a) {
      const double da = x(i, a) - mean[a];
      for (std::size_t b = a; b < d; ++b)
        c(a, b) += da * (x(i, b) - mean[b]);
    }
  }
  const double denom = static_cast<double>(n - 1);
  for (std::size_t a = 0; a < d; ++a)
    for (std::size_t b = a; b < d; ++b) {
      c(a, b) /= denom;
      c(b, a) = c(a, b);
    }
  return c;
}

la::mat correlation(const la::mat& x) {
  la::mat c = covariance(x);
  const std::size_t d = c.rows();
  std::vector<double> sd(d);
  for (std::size_t i = 0; i < d; ++i) sd[i] = std::sqrt(c(i, i));
  la::mat r(d, d);
  for (std::size_t a = 0; a < d; ++a)
    for (std::size_t b = 0; b < d; ++b) r(a, b) = c(a, b) / (sd[a] * sd[b]);
  return r;
}

}  // namespace ax::st
