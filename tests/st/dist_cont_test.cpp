#include <gtest/gtest.h>
#include <ax/st/dist.hpp>

#include <cmath>
#include <stdexcept>

using namespace ax::st;

namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;

/** quantile(cdf(x)) == x and cdf(quantile(p)) == p over a p-grid. */
template <typename D>
void check_quantile_roundtrip(const D& d, double tol = 1e-9) {
  for (const double p : {0.01, 0.25, 0.5, 0.9, 0.999}) {
    const double x = d.quantile(p);
    EXPECT_NEAR(d.cdf(x), p, tol) << "p=" << p;
  }
}

/** Sample n draws; mean/var within rtol of analytic. */
template <typename D>
void check_sample_moments(const D& d, int n = 100000, double rtol = 0.03) {
  rng g{2026};
  double s1 = 0, s2 = 0;
  for (int i = 0; i < n; ++i) {
    const double x = d.sample(g);
    s1 += x;
    s2 += x * x;
  }
  const double mean = s1 / n;
  const double var = s2 / n - mean * mean;
  const double scale_m = std::max(std::abs(d.mean()), std::sqrt(d.var()));
  const double scale_v = std::max(d.var(), 0.1);
  EXPECT_NEAR(mean, d.mean(), rtol * scale_m);
  EXPECT_NEAR(var, d.var(), 3.0 * rtol * scale_v);
}
}  // namespace

TEST(dist_normal, pdf_cdf_quantile) {
  const normal_dist d{0.0, 1.0};
  EXPECT_NEAR(d.pdf(0.0), 1.0 / std::sqrt(2.0 * kPi), 1e-14);
  EXPECT_NEAR(d.cdf(0.0), 0.5, 1e-14);
  EXPECT_NEAR(d.cdf(1.959963984540054), 0.975, 1e-10);
  EXPECT_NEAR(d.quantile(0.975), 1.959963984540054, 1e-9);
  check_quantile_roundtrip(d);
  check_sample_moments(d);
  EXPECT_THROW(normal_dist(0.0, 0.0), std::invalid_argument);
  EXPECT_THROW(d.quantile(0.0), std::invalid_argument);
}

TEST(dist_t, closed_form_low_dof) {
  const t_dist t1{1.0};  // == Cauchy(0,1)
  EXPECT_NEAR(t1.pdf(0.0), 1.0 / kPi, 1e-13);
  EXPECT_NEAR(t1.cdf(1.0), 0.75, 1e-13);
  const t_dist t2{2.0};  // cdf(x) = 1/2 + x / (2 sqrt(2 + x^2))
  EXPECT_NEAR(t2.cdf(1.0), 0.5 + 1.0 / (2.0 * std::sqrt(3.0)), 1e-13);
  const t_dist t5{5.0};
  check_quantile_roundtrip(t5);
  check_sample_moments(t5);  // mean 0, var nu/(nu-2)
  EXPECT_THROW(t_dist(0.0), std::invalid_argument);
}

TEST(dist_chi2, exponential_special_case) {
  const chi2_dist d{2.0};  // == exponential(1/2)
  EXPECT_NEAR(d.cdf(3.0), 1.0 - std::exp(-1.5), 1e-13);
  EXPECT_NEAR(d.pdf(1.0), 0.5 * std::exp(-0.5), 1e-13);
  const chi2_dist d5{5.0};
  check_quantile_roundtrip(d5);
  check_sample_moments(d5);  // mean k, var 2k
  EXPECT_THROW(chi2_dist(-1.0), std::invalid_argument);
}

TEST(dist_f, closed_form_d1_2) {
  const f_dist d{2.0, 4.0};  // cdf(x) = 1 - (1 + x/2)^{-2}
  EXPECT_NEAR(d.cdf(3.0), 1.0 - std::pow(2.5, -2.0), 1e-13);
  const f_dist d2{3.0, 10.0};
  check_quantile_roundtrip(d2);
  check_sample_moments(d2, 200000, 0.05);  // heavy-ish tails
  EXPECT_THROW(f_dist(0.0, 1.0), std::invalid_argument);
}

TEST(dist_gamma, closed_form_integer_shape) {
  const gamma_dist d{3.0, 2.0};  // cdf(x) = P(3, x/2)
  const double y = 3.0 / 2.0;
  EXPECT_NEAR(d.cdf(3.0),
              1.0 - std::exp(-y) * (1.0 + y + 0.5 * y * y), 1e-13);
  check_quantile_roundtrip(d);
  check_sample_moments(d);  // mean 6, var 12
  EXPECT_THROW(gamma_dist(1.0, -1.0), std::invalid_argument);
}

TEST(dist_beta, polynomial_cdf) {
  const beta_dist d{2.0, 3.0};  // I_x(2,3) = x^2 (6 - 8x + 3x^2)
  const double x = 0.4;
  EXPECT_NEAR(d.cdf(x), x * x * (6.0 - 8.0 * x + 3.0 * x * x), 1e-13);
  check_quantile_roundtrip(d);
  check_sample_moments(d);
  EXPECT_THROW(beta_dist(0.0, 1.0), std::invalid_argument);
}

TEST(dist_exponential, closed_forms) {
  const exponential_dist d{2.0};
  EXPECT_NEAR(d.cdf(0.5), 1.0 - std::exp(-1.0), 1e-14);
  EXPECT_NEAR(d.quantile(0.5), std::log(2.0) / 2.0, 1e-12);
  check_quantile_roundtrip(d);
  check_sample_moments(d);
}

TEST(dist_uniform, closed_forms) {
  const uniform_dist d{2.0, 5.0};
  EXPECT_NEAR(d.cdf(3.5), 0.5, 1e-14);
  EXPECT_NEAR(d.quantile(0.25), 2.75, 1e-14);
  EXPECT_EQ(d.cdf(1.0), 0.0);
  EXPECT_EQ(d.cdf(6.0), 1.0);
  check_sample_moments(d);
  EXPECT_THROW(uniform_dist(5.0, 2.0), std::invalid_argument);
}

TEST(dist_lognormal, via_normal) {
  const lognormal_dist d{0.5, 0.75};
  const normal_dist n{0.5, 0.75};
  EXPECT_NEAR(d.cdf(2.0), n.cdf(std::log(2.0)), 1e-13);
  EXPECT_NEAR(d.quantile(0.5), std::exp(0.5), 1e-10);  // median = e^mu
  check_quantile_roundtrip(d);
  check_sample_moments(d, 200000, 0.05);
}

TEST(dist_weibull, closed_forms) {
  const weibull_dist d{2.0, 3.0};  // k=2, lambda=3
  EXPECT_NEAR(d.cdf(3.0), 1.0 - std::exp(-1.0), 1e-14);
  EXPECT_NEAR(d.quantile(1.0 - std::exp(-1.0)), 3.0, 1e-10);
  check_quantile_roundtrip(d);
  check_sample_moments(d);
}

TEST(dist_cauchy, closed_forms_no_moments) {
  const cauchy_dist d{1.0, 2.0};
  EXPECT_NEAR(d.cdf(1.0), 0.5, 1e-14);
  EXPECT_NEAR(d.cdf(3.0), 0.5 + std::atan(1.0) / kPi, 1e-13);
  EXPECT_NEAR(d.quantile(0.5), 1.0, 1e-12);
  check_quantile_roundtrip(d);
  EXPECT_THROW(d.mean(), std::domain_error);
  EXPECT_THROW(d.var(), std::domain_error);
  // sampling sanity: median of draws near x0
  rng g{5};
  int below = 0;
  for (int i = 0; i < 20000; ++i)
    if (d.sample(g) < 1.0) ++below;
  EXPECT_NEAR(below / 20000.0, 0.5, 0.02);
}

TEST(dist_laplace, closed_forms) {
  const laplace_dist d{1.0, 2.0};
  EXPECT_NEAR(d.cdf(1.0), 0.5, 1e-14);
  EXPECT_NEAR(d.cdf(3.0), 1.0 - 0.5 * std::exp(-1.0), 1e-13);
  EXPECT_NEAR(d.quantile(0.5), 1.0, 1e-12);
  check_quantile_roundtrip(d);
  check_sample_moments(d);  // mean mu, var 2 b^2
}
