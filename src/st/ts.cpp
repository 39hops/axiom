#include <ax/core/fft.hpp>
#include <ax/num/opt.hpp>
#include <ax/st/ts.hpp>

#include <cmath>
#include <complex>
#include <stdexcept>

namespace ax::st {

namespace {

double sample_mean(std::span<const double> xs) {
  double m = 0.0;
  for (double x : xs) m += x;
  return m / static_cast<double>(xs.size());
}

/** Biased sample autocovariances c_0..c_max_lag (denominator n). */
std::vector<double> autocov(std::span<const double> xs, std::size_t max_lag) {
  const std::size_t n = xs.size();
  const double m = sample_mean(xs);
  std::vector<double> c(max_lag + 1, 0.0);
  for (std::size_t k = 0; k <= max_lag; ++k) {
    double s = 0.0;
    for (std::size_t i = 0; i + k < n; ++i)
      s += (xs[i] - m) * (xs[i + k] - m);
    c[k] = s / static_cast<double>(n);
  }
  return c;
}

/** Durbin-Levinson on rho[1..m]. Returns per-order reflection coefficients
    (== PACF) and leaves the final AR coefficients in phi, the final
    normalized innovation variance in v. */
void durbin_levinson(std::span<const double> rho, std::size_t m,
                     std::vector<double>& pacf_out, std::vector<double>& phi,
                     double& v) {
  pacf_out.assign(m, 0.0);
  phi.assign(m, 0.0);
  v = 1.0;
  if (m == 0) return;
  phi[0] = rho[1];
  pacf_out[0] = rho[1];
  v = 1.0 - rho[1] * rho[1];
  std::vector<double> prev(m, 0.0);
  for (std::size_t k = 2; k <= m; ++k) {
    for (std::size_t j = 0; j < k - 1; ++j) prev[j] = phi[j];
    double num = rho[k];
    for (std::size_t j = 0; j < k - 1; ++j) num -= prev[j] * rho[k - 1 - j];
    const double a = num / v;
    for (std::size_t j = 0; j < k - 1; ++j)
      phi[j] = prev[j] - a * prev[k - 2 - j];
    phi[k - 1] = a;
    pacf_out[k - 1] = a;
    v *= (1.0 - a * a);
  }
}

/** Conditional sum of squares of an ARMA(p,q) on the demeaned series. */
double css(std::span<const double> x, std::span<const double> phi,
           std::span<const double> theta) {
  const std::size_t n = x.size(), p = phi.size(), q = theta.size();
  std::vector<double> e(n, 0.0);
  double s = 0.0;
  for (std::size_t t = 0; t < n; ++t) {
    double pred = 0.0;
    for (std::size_t i = 0; i < p; ++i)
      if (t > i) pred += phi[i] * x[t - 1 - i];
    for (std::size_t j = 0; j < q; ++j)
      if (t > j) pred += theta[j] * e[t - 1 - j];
    e[t] = x[t] - pred;
    s += e[t] * e[t];
  }
  return s;
}

}  // namespace

std::vector<double> acf(std::span<const double> xs, std::size_t max_lag) {
  if (xs.size() <= max_lag)
    throw std::invalid_argument("acf: need n > max_lag");
  auto c = autocov(xs, max_lag);
  if (c[0] <= 0.0) throw std::invalid_argument("acf: zero variance");
  std::vector<double> a(max_lag + 1);
  for (std::size_t k = 0; k <= max_lag; ++k) a[k] = c[k] / c[0];
  return a;
}

std::vector<double> pacf(std::span<const double> xs, std::size_t max_lag) {
  if (max_lag == 0) throw std::invalid_argument("pacf: max_lag >= 1");
  auto rho = acf(xs, max_lag);  // validates n > max_lag
  std::vector<double> out, phi;
  double v;
  durbin_levinson(rho, max_lag, out, phi, v);
  return out;
}

ar_result ar_fit(std::span<const double> xs, std::size_t p) {
  if (p == 0) throw std::invalid_argument("ar_fit: p >= 1");
  if (xs.size() <= p) throw std::invalid_argument("ar_fit: need n > p");
  auto c = autocov(xs, p);
  if (c[0] <= 0.0) throw std::invalid_argument("ar_fit: zero variance");
  std::vector<double> rho(p + 1);
  for (std::size_t k = 0; k <= p; ++k) rho[k] = c[k] / c[0];
  std::vector<double> pk, phi;
  double v;
  durbin_levinson(rho, p, pk, phi, v);
  const double m = sample_mean(xs);
  double phisum = 0.0;
  for (double f : phi) phisum += f;
  return {std::move(phi), m * (1.0 - phisum), c[0] * v};
}

arma_result arma_fit(std::span<const double> xs, std::size_t p,
                     std::size_t q) {
  if (p + q == 0) throw std::invalid_argument("arma_fit: p + q >= 1");
  if (xs.size() <= p + q)
    throw std::invalid_argument("arma_fit: need n > p + q");
  const double m = sample_mean(xs);
  std::vector<double> x(xs.size());
  for (std::size_t i = 0; i < xs.size(); ++i) x[i] = xs[i] - m;

  // initial point: Yule-Walker AR part, zero MA part
  la::vec x0(p + q, 0.0);
  if (p > 0) {
    auto init = ar_fit(xs, p);
    for (std::size_t i = 0; i < p; ++i) x0[i] = init.phi[i];
  }
  auto objective = [&](const la::vec& theta_all) {
    std::vector<double> phi(p), th(q);
    for (std::size_t i = 0; i < p; ++i) phi[i] = theta_all[i];
    for (std::size_t j = 0; j < q; ++j) th[j] = theta_all[p + j];
    return css(x, phi, th);
  };
  auto r = num::nelder_mead(objective, x0);

  std::vector<double> phi(p), th(q);
  for (std::size_t i = 0; i < p; ++i) phi[i] = r.x[i];
  for (std::size_t j = 0; j < q; ++j) th[j] = r.x[p + j];
  double phisum = 0.0;
  for (double f : phi) phisum += f;
  const double sigma2 = r.fx / static_cast<double>(x.size());
  return {std::move(phi), std::move(th), m * (1.0 - phisum), sigma2,
          r.converged};
}

periodogram_result periodogram(std::span<const double> xs) {
  if (xs.empty()) throw std::invalid_argument("periodogram: empty input");
  const double mean = sample_mean(xs);
  std::size_t m = 1;
  while (m < xs.size()) m <<= 1;
  std::vector<std::complex<double>> a(m, 0.0);
  for (std::size_t i = 0; i < xs.size(); ++i) a[i] = xs[i] - mean;
  ax::fft(a, /*invert=*/false);
  const double n = static_cast<double>(xs.size());
  periodogram_result r;
  r.freq.reserve(m / 2);
  r.power.reserve(m / 2);
  for (std::size_t k = 1; k <= m / 2; ++k) {
    r.freq.push_back(static_cast<double>(k) / static_cast<double>(m));
    r.power.push_back(std::norm(a[k]) / n);
  }
  return r;
}

}  // namespace ax::st
