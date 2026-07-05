#include <ax/la/decomp.hpp>
#include <ax/st/dist.hpp>
#include <ax/st/reg.hpp>

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

}  // namespace ax::st
