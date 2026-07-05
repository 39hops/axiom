#include <ax/st/dist.hpp>
#include <ax/st/htest.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ax::st {

namespace {

constexpr double knan = std::numeric_limits<double>::quiet_NaN();

struct mean_var {
  double mean, var;  // sample variance (n-1)
  std::size_t n;
};

mean_var summarize(std::span<const double> xs, const char* what) {
  if (xs.size() < 2)
    throw std::invalid_argument(std::string(what) + ": need n >= 2");
  double m = 0.0;
  for (double x : xs) m += x;
  m /= static_cast<double>(xs.size());
  double s2 = 0.0;
  for (double x : xs) s2 += (x - m) * (x - m);
  s2 /= static_cast<double>(xs.size() - 1);
  return {m, s2, xs.size()};
}

/** Map a t statistic to a p-value under the given alternative. */
double t_p_value(double t, double df, alternative alt) {
  const double c = t_dist(df).cdf(t);
  switch (alt) {
    case alternative::less:
      return c;
    case alternative::greater:
      return 1.0 - c;
    case alternative::two_sided:
    default:
      return 2.0 * std::min(c, 1.0 - c);
  }
}

double chi2_p_value(double x2, double df) {
  return 1.0 - chi2_dist(df).cdf(x2);
}

}  // namespace

test_result t_test(std::span<const double> xs, double mu0, alternative alt) {
  auto s = summarize(xs, "t_test");
  const double se = std::sqrt(s.var / static_cast<double>(s.n));
  const double df = static_cast<double>(s.n - 1);
  const double t = (s.mean - mu0) / se;
  return {t, df, knan, t_p_value(t, df, alt)};
}

test_result t_test(std::span<const double> xs, std::span<const double> ys,
                   alternative alt, bool welch) {
  auto a = summarize(xs, "t_test");
  auto b = summarize(ys, "t_test");
  const double n1 = static_cast<double>(a.n), n2 = static_cast<double>(b.n);
  double t, df;
  if (welch) {
    const double v1 = a.var / n1, v2 = b.var / n2;
    const double se2 = v1 + v2;
    t = (a.mean - b.mean) / std::sqrt(se2);
    df = se2 * se2 / (v1 * v1 / (n1 - 1.0) + v2 * v2 / (n2 - 1.0));
  } else {
    const double sp2 =
        ((n1 - 1.0) * a.var + (n2 - 1.0) * b.var) / (n1 + n2 - 2.0);
    t = (a.mean - b.mean) / std::sqrt(sp2 * (1.0 / n1 + 1.0 / n2));
    df = n1 + n2 - 2.0;
  }
  return {t, df, knan, t_p_value(t, df, alt)};
}

test_result chi2_gof(std::span<const double> observed,
                     std::span<const double> expected_p) {
  if (observed.size() != expected_p.size())
    throw std::invalid_argument("chi2_gof: size mismatch");
  if (observed.size() < 2)
    throw std::invalid_argument("chi2_gof: need >= 2 cells");
  double total = 0.0, psum = 0.0;
  for (double o : observed) total += o;
  for (double p : expected_p) psum += p;
  if (total <= 0.0 || psum <= 0.0)
    throw std::invalid_argument("chi2_gof: non-positive totals");
  double x2 = 0.0;
  for (std::size_t i = 0; i < observed.size(); ++i) {
    const double e = total * expected_p[i] / psum;
    if (e <= 0.0)
      throw std::invalid_argument("chi2_gof: expected count <= 0");
    const double d = observed[i] - e;
    x2 += d * d / e;
  }
  const double df = static_cast<double>(observed.size() - 1);
  return {x2, df, knan, chi2_p_value(x2, df)};
}

test_result chi2_independence(const la::mat& table) {
  const std::size_t r = table.rows(), c = table.cols();
  if (r < 2 || c < 2)
    throw std::invalid_argument("chi2_independence: need >= 2x2 table");
  std::vector<double> row(r, 0.0), col(c, 0.0);
  double total = 0.0;
  for (std::size_t i = 0; i < r; ++i)
    for (std::size_t j = 0; j < c; ++j) {
      row[i] += table(i, j);
      col[j] += table(i, j);
      total += table(i, j);
    }
  for (double s : row)
    if (s <= 0.0)
      throw std::invalid_argument("chi2_independence: zero row sum");
  for (double s : col)
    if (s <= 0.0)
      throw std::invalid_argument("chi2_independence: zero column sum");
  double x2 = 0.0;
  for (std::size_t i = 0; i < r; ++i)
    for (std::size_t j = 0; j < c; ++j) {
      const double e = row[i] * col[j] / total;
      const double d = table(i, j) - e;
      x2 += d * d / e;
    }
  const double df = static_cast<double>((r - 1) * (c - 1));
  return {x2, df, knan, chi2_p_value(x2, df)};
}

test_result anova_oneway(std::span<const std::vector<double>> groups) {
  if (groups.size() < 2)
    throw std::invalid_argument("anova_oneway: need >= 2 groups");
  std::size_t total_n = 0;
  double grand = 0.0;
  for (const auto& g : groups) {
    if (g.size() < 2)
      throw std::invalid_argument("anova_oneway: group needs n >= 2");
    for (double x : g) grand += x;
    total_n += g.size();
  }
  grand /= static_cast<double>(total_n);
  double ssb = 0.0, ssw = 0.0;
  for (const auto& g : groups) {
    double m = 0.0;
    for (double x : g) m += x;
    m /= static_cast<double>(g.size());
    ssb += static_cast<double>(g.size()) * (m - grand) * (m - grand);
    for (double x : g) ssw += (x - m) * (x - m);
  }
  const double df1 = static_cast<double>(groups.size() - 1);
  const double df2 = static_cast<double>(total_n - groups.size());
  const double f = (ssb / df1) / (ssw / df2);
  const double p = 1.0 - f_dist(df1, df2).cdf(f);
  return {f, df1, df2, p};
}

test_result ks_test(std::span<const double> xs,
                    const std::function<double(double)>& cdf) {
  if (xs.empty()) throw std::invalid_argument("ks_test: empty sample");
  std::vector<double> s(xs.begin(), xs.end());
  std::sort(s.begin(), s.end());
  const double n = static_cast<double>(s.size());
  double d = 0.0;
  for (std::size_t i = 0; i < s.size(); ++i) {
    const double f = cdf(s[i]);
    const double lo = static_cast<double>(i) / n;
    const double hi = static_cast<double>(i + 1) / n;
    d = std::max({d, hi - f, f - lo});
  }
  // Asymptotic Kolmogorov tail: Q(x) = 2 sum_{k>=1} (-1)^{k-1} exp(-2k^2x^2).
  const double x = std::sqrt(n) * d;
  double q = 0.0;
  for (int k = 1; k <= 100; ++k) {
    const double term = 2.0 * std::exp(-2.0 * k * k * x * x);
    q += (k % 2 == 1) ? term : -term;
    if (term < 1e-12) break;
  }
  q = std::clamp(q, 0.0, 1.0);
  return {d, knan, knan, q};
}

}  // namespace ax::st
