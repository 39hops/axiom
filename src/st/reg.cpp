#include <ax/la/decomp.hpp>
#include <ax/st/dist.hpp>
#include <ax/st/reg.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ax::st {

namespace {

/** Prepend a ones column when intercept is requested. */
la::mat design_matrix(const la::mat& x, bool intercept) {
  if (!intercept) return x;
  la::mat d(x.rows(), x.cols() + 1);
  for (std::size_t i = 0; i < x.rows(); ++i) {
    d(i, 0) = 1.0;
    for (std::size_t j = 0; j < x.cols(); ++j) d(i, j + 1) = x(i, j);
  }
  return d;
}

}  // namespace

ols_result ols(const la::mat& x, const la::vec& y, bool intercept) {
  if (x.rows() != y.size())
    throw std::invalid_argument("ols: rows(x) != size(y)");
  const la::mat d = design_matrix(x, intercept);
  const std::size_t n = d.rows(), p = d.cols();
  if (n <= p) throw std::invalid_argument("ols: need more rows than params");

  la::vec beta = la::lstsq(d, y);  // throws domain_error on rank deficiency

  la::vec resid(n);
  double rss = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    double fit = 0.0;
    for (std::size_t j = 0; j < p; ++j) fit += d(i, j) * beta[j];
    resid[i] = y[i] - fit;
    rss += resid[i] * resid[i];
  }
  const std::size_t df = n - p;
  const double sigma2 = rss / static_cast<double>(df);

  // diag((X'X)^{-1}) = rows of R^{-1} squared-summed, R from QR of design.
  const la::mat r = la::qr_decompose(d).r;
  // invert the upper-triangular p x p block by back-substitution per column
  la::mat rinv(p, p);
  for (std::size_t col = 0; col < p; ++col) {
    for (std::size_t ii = p; ii-- > 0;) {
      double s = (ii == col) ? 1.0 : 0.0;
      for (std::size_t k = ii + 1; k < p; ++k) s -= r(ii, k) * rinv(k, col);
      rinv(ii, col) = s / r(ii, ii);
    }
  }
  la::vec stderrs(p), t_stats(p), p_values(p);
  const t_dist tdf(static_cast<double>(df));
  for (std::size_t j = 0; j < p; ++j) {
    double diag = 0.0;
    for (std::size_t k = j; k < p; ++k) diag += rinv(j, k) * rinv(j, k);
    stderrs[j] = std::sqrt(sigma2 * diag);
    t_stats[j] = beta[j] / stderrs[j];
    const double c = tdf.cdf(t_stats[j]);
    p_values[j] = 2.0 * std::min(c, 1.0 - c);
  }

  // total sum of squares: about the mean iff an intercept was fitted
  double tss = 0.0;
  if (intercept) {
    double ym = 0.0;
    for (std::size_t i = 0; i < n; ++i) ym += y[i];
    ym /= static_cast<double>(n);
    for (std::size_t i = 0; i < n; ++i) tss += (y[i] - ym) * (y[i] - ym);
  } else {
    for (std::size_t i = 0; i < n; ++i) tss += y[i] * y[i];
  }
  const double r2 = (tss > 0.0) ? 1.0 - rss / tss : 1.0;
  const double adj_r2 =
      1.0 - (1.0 - r2) * static_cast<double>(n - (intercept ? 1 : 0)) /
                static_cast<double>(df);

  return {std::move(beta), std::move(stderrs), std::move(t_stats),
          std::move(p_values), std::move(resid), sigma2, r2, adj_r2, df};
}

namespace {

constexpr double kmu_eps = 1e-10;  ///< clamp for fitted means (stability)

double glm_deviance(const la::vec& y, const la::vec& mu, glm_family family) {
  double dev = 0.0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    if (family == glm_family::logistic) {
      if (y[i] > 0.5)
        dev += -2.0 * std::log(mu[i]);
      else
        dev += -2.0 * std::log(1.0 - mu[i]);
    } else {  // poisson: 2 [ y log(y/mu) - (y - mu) ], y=0 term -> 2 mu
      if (y[i] > 0.0)
        dev += 2.0 * (y[i] * std::log(y[i] / mu[i]) - (y[i] - mu[i]));
      else
        dev += 2.0 * mu[i];
    }
  }
  return dev;
}

}  // namespace

glm_result glm_fit(const la::mat& x, const la::vec& y, glm_family family,
                   bool intercept, std::size_t max_iter, double tol) {
  if (x.rows() != y.size())
    throw std::invalid_argument("glm_fit: rows(x) != size(y)");
  for (std::size_t i = 0; i < y.size(); ++i) {
    if (family == glm_family::logistic && y[i] != 0.0 && y[i] != 1.0)
      throw std::invalid_argument("glm_fit: logistic response must be 0/1");
    if (family == glm_family::poisson && y[i] < 0.0)
      throw std::invalid_argument("glm_fit: poisson response must be >= 0");
  }
  const la::mat d = design_matrix(x, intercept);
  const std::size_t n = d.rows(), p = d.cols();
  if (p == 0) throw std::invalid_argument("glm_fit: no parameters");
  if (n < p) throw std::invalid_argument("glm_fit: need rows >= params");

  la::vec beta(p, 0.0);
  la::vec mu(n), w(n);
  bool converged = false;
  std::size_t iter = 0;
  la::mat xtwx(p, p);
  while (iter < max_iter) {
    ++iter;
    // eta, mu, weights under the canonical link
    la::vec z(n);  // working response
    for (std::size_t i = 0; i < n; ++i) {
      double eta = 0.0;
      for (std::size_t j = 0; j < p; ++j) eta += d(i, j) * beta[j];
      if (family == glm_family::logistic) {
        mu[i] = 1.0 / (1.0 + std::exp(-eta));
        mu[i] = std::clamp(mu[i], kmu_eps, 1.0 - kmu_eps);
        w[i] = std::max(mu[i] * (1.0 - mu[i]), kmu_eps);
      } else {
        mu[i] = std::max(std::exp(eta), kmu_eps);
        w[i] = mu[i];
      }
      z[i] = eta + (y[i] - mu[i]) / w[i];
    }
    // normal equations of the weighted least squares step
    la::vec xtwz(p);
    for (std::size_t a = 0; a < p; ++a) {
      for (std::size_t b = a; b < p; ++b) {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += d(i, a) * w[i] * d(i, b);
        xtwx(a, b) = xtwx(b, a) = s;
      }
      double s = 0.0;
      for (std::size_t i = 0; i < n; ++i) s += d(i, a) * w[i] * z[i];
      xtwz[a] = s;
    }
    const la::mat l = la::cholesky(xtwx);
    la::vec beta_new = la::cholesky_solve(l, xtwz);
    double delta = 0.0;
    for (std::size_t j = 0; j < p; ++j)
      delta = std::max(delta, std::abs(beta_new[j] - beta[j]));
    beta = std::move(beta_new);
    if (delta < tol) {
      converged = true;
      break;
    }
  }

  // stderrs from (X'WX)^{-1} at the final iterate
  la::vec stderrs(p);
  const la::mat cov = la::inverse(xtwx);
  for (std::size_t j = 0; j < p; ++j) stderrs[j] = std::sqrt(cov(j, j));

  const double dev = glm_deviance(y, mu, family);
  return {std::move(beta), std::move(stderrs), dev, iter, converged};
}

}  // namespace ax::st
