// Metropolis-Hastings. Fixed seeds -> deterministic; statistical bounds have
// ~3x slack over expected sampling error at the stated chain lengths.
#include <ax/st/mcmc.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

using namespace ax::st;
using ax::la::vec;

namespace {
constexpr double kinf = std::numeric_limits<double>::infinity();

double std_normal_logpdf(const vec& x) { return -0.5 * x[0] * x[0]; }
}  // namespace

TEST(metropolis, standard_normal_moments) {
  rng g{42};
  mh_options opt;
  opt.n_samples = 20000;
  opt.burn_in = 2000;
  opt.step = 2.4;  // near-optimal for 1-d normal
  auto r = metropolis(std_normal_logpdf, vec{0.0}, opt, g);
  ASSERT_EQ(r.samples.rows(), 20000u);
  ASSERT_EQ(r.samples.cols(), 1u);
  double m = 0.0;
  for (std::size_t i = 0; i < r.samples.rows(); ++i) m += r.samples(i, 0);
  m /= 20000.0;
  double v = 0.0;
  for (std::size_t i = 0; i < r.samples.rows(); ++i)
    v += (r.samples(i, 0) - m) * (r.samples(i, 0) - m);
  v /= 19999.0;
  EXPECT_LT(std::abs(m), 0.05);
  EXPECT_GT(v, 0.9);
  EXPECT_LT(v, 1.1);
}

TEST(metropolis, correlated_gaussian_2d) {
  // target: zero-mean bivariate normal, rho = 0.8 (unnormalized)
  const double rho = 0.8, c = 1.0 / (1.0 - rho * rho);
  auto logp = [&](const vec& x) {
    return -0.5 * c * (x[0] * x[0] - 2.0 * rho * x[0] * x[1] + x[1] * x[1]);
  };
  rng g{99};
  mh_options opt;
  opt.n_samples = 30000;
  opt.burn_in = 3000;
  opt.step = 1.0;
  auto r = metropolis(logp, vec{0.0, 0.0}, opt, g);
  double m0 = 0, m1 = 0;
  const double n = static_cast<double>(r.samples.rows());
  for (std::size_t i = 0; i < r.samples.rows(); ++i) {
    m0 += r.samples(i, 0);
    m1 += r.samples(i, 1);
  }
  m0 /= n;
  m1 /= n;
  double s00 = 0, s11 = 0, s01 = 0;
  for (std::size_t i = 0; i < r.samples.rows(); ++i) {
    const double a = r.samples(i, 0) - m0, b = r.samples(i, 1) - m1;
    s00 += a * a;
    s11 += b * b;
    s01 += a * b;
  }
  const double corr = s01 / std::sqrt(s00 * s11);
  EXPECT_GT(corr, 0.7);
  EXPECT_LT(corr, 0.9);
}

TEST(metropolis, bounded_support_respected) {
  auto logp = [](const vec& x) {
    return (x[0] > 0.0 && x[0] < 1.0) ? 0.0 : -kinf;
  };
  rng g{5};
  mh_options opt;
  opt.n_samples = 5000;
  opt.burn_in = 500;
  opt.step = 0.5;
  auto r = metropolis(logp, vec{0.5}, opt, g);
  for (std::size_t i = 0; i < r.samples.rows(); ++i) {
    EXPECT_GT(r.samples(i, 0), 0.0);
    EXPECT_LT(r.samples(i, 0), 1.0);
  }
}

TEST(metropolis, acceptance_rate_scales_with_step) {
  rng g1{1}, g2{2};
  mh_options tiny;
  tiny.n_samples = 5000;
  tiny.step = 0.01;
  mh_options huge;
  huge.n_samples = 5000;
  huge.step = 50.0;
  auto a = metropolis(std_normal_logpdf, vec{0.0}, tiny, g1);
  auto b = metropolis(std_normal_logpdf, vec{0.0}, huge, g2);
  EXPECT_GT(a.acceptance_rate, 0.9);
  EXPECT_LT(b.acceptance_rate, 0.2);
  EXPECT_GT(a.acceptance_rate, 0.0);
  EXPECT_LT(a.acceptance_rate, 1.0);
}

TEST(metropolis, deterministic_given_seed) {
  mh_options opt;
  opt.n_samples = 100;
  opt.burn_in = 10;
  rng g1{7}, g2{7};
  auto a = metropolis(std_normal_logpdf, vec{0.0}, opt, g1);
  auto b = metropolis(std_normal_logpdf, vec{0.0}, opt, g2);
  for (std::size_t i = 0; i < a.samples.rows(); ++i)
    EXPECT_DOUBLE_EQ(a.samples(i, 0), b.samples(i, 0));
  EXPECT_DOUBLE_EQ(a.acceptance_rate, b.acceptance_rate);
}

TEST(metropolis, thinning_reduces_kept_samples_correctly) {
  mh_options opt;
  opt.n_samples = 200;
  opt.burn_in = 50;
  opt.thin = 5;
  rng g{3};
  auto r = metropolis(std_normal_logpdf, vec{0.0}, opt, g);
  EXPECT_EQ(r.samples.rows(), 200u);  // n_samples kept regardless of thin
}

TEST(metropolis, infinite_start_throws) {
  auto logp = [](const vec& x) {
    return x[0] > 0.0 ? 0.0 : -kinf;
  };
  rng g{1};
  mh_options opt;
  EXPECT_THROW((void)metropolis(logp, vec{-1.0}, opt, g),
               std::invalid_argument);
}

TEST(ess_diag, iid_chain_ess_near_n) {
  rng g{21};
  std::vector<double> chain(4000);
  for (auto& x : chain) x = g.normal();
  const double e = ess(chain);
  EXPECT_GT(e, 2000.0);  // within factor 2 of n
  EXPECT_LE(e, 4000.0 + 1e-9);
}

TEST(ess_diag, autocorrelated_chain_ess_small) {
  rng g{22};
  const double phi = 0.95, se = std::sqrt(1.0 - phi * phi);
  std::vector<double> chain(4000);
  chain[0] = g.normal();
  for (std::size_t i = 1; i < chain.size(); ++i)
    chain[i] = phi * chain[i - 1] + se * g.normal();
  EXPECT_LT(ess(chain), 4000.0 / 5.0);
}
