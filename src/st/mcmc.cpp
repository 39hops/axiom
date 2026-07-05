#include <ax/st/mcmc.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace ax::st {

mh_result metropolis(const std::function<double(const la::vec&)>& log_density,
                     const la::vec& x0, const mh_options& opt, rng& g) {
  if (x0.size() == 0) throw std::invalid_argument("metropolis: empty x0");
  if (opt.thin == 0) throw std::invalid_argument("metropolis: thin == 0");
  if (opt.n_samples == 0)
    throw std::invalid_argument("metropolis: n_samples == 0");
  if (!(opt.step > 0.0)) throw std::invalid_argument("metropolis: step <= 0");
  double lp = log_density(x0);
  if (!std::isfinite(lp))
    throw std::invalid_argument("metropolis: log_density(x0) not finite");

  const std::size_t dim = x0.size();
  la::vec x = x0, prop(dim);
  la::mat out(opt.n_samples, dim);
  std::size_t accepted = 0, proposed = 0, kept = 0;
  const std::size_t total = opt.burn_in + opt.n_samples * opt.thin;
  for (std::size_t it = 0; it < total; ++it) {
    for (std::size_t j = 0; j < dim; ++j)
      prop[j] = x[j] + opt.step * g.normal();
    const double lp_new = log_density(prop);
    ++proposed;
    // accept with probability min(1, exp(lp_new - lp)); -inf never accepted
    if (lp_new - lp >= 0.0 || g.next_double() < std::exp(lp_new - lp)) {
      x = prop;
      lp = lp_new;
      ++accepted;
    }
    if (it >= opt.burn_in && (it - opt.burn_in) % opt.thin == 0) {
      for (std::size_t j = 0; j < dim; ++j) out(kept, j) = x[j];
      ++kept;
    }
  }
  return {std::move(out),
          static_cast<double>(accepted) / static_cast<double>(proposed)};
}

double ess(std::span<const double> chain) {
  const std::size_t n = chain.size();
  if (n < 2) throw std::invalid_argument("ess: need n >= 2");
  double m = 0.0;
  for (double x : chain) m += x;
  m /= static_cast<double>(n);
  double c0 = 0.0;
  for (double x : chain) c0 += (x - m) * (x - m);
  c0 /= static_cast<double>(n);
  if (c0 <= 0.0) return static_cast<double>(n);  // constant chain

  auto autocov = [&](std::size_t k) {
    double s = 0.0;
    for (std::size_t i = 0; i + k < n; ++i)
      s += (chain[i] - m) * (chain[i + k] - m);
    return s / static_cast<double>(n);
  };

  // Geyer initial positive sequence: sum rho over adjacent pairs while the
  // pair sum stays positive.
  double sum = 0.0;
  for (std::size_t k = 1; k + 1 < n; k += 2) {
    const double pair = (autocov(k) + autocov(k + 1)) / c0;
    if (pair <= 0.0) break;
    sum += pair;
  }
  const double denom = 1.0 + 2.0 * sum;
  const double e = static_cast<double>(n) / denom;
  return std::min(e, static_cast<double>(n));
}

}  // namespace ax::st
