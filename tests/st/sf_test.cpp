#include <gtest/gtest.h>
#include <ax/st/sf.hpp>

#include <cmath>
#include <stdexcept>

namespace sf = ax::st::sf;

TEST(sf_lgamma, known_values) {
  EXPECT_NEAR(sf::lgamma(0.5), 0.5723649429247001, 1e-13);
  EXPECT_NEAR(sf::lgamma(1.0), 0.0, 1e-14);
  EXPECT_NEAR(sf::lgamma(2.0), 0.0, 1e-14);
  EXPECT_NEAR(sf::lgamma(10.0), 12.801827480081469, 1e-12);
  EXPECT_NEAR(sf::lgamma(0.1), 2.252712651734206, 1e-12);
}

TEST(sf_lgamma, domain) {
  EXPECT_THROW(sf::lgamma(0.0), std::domain_error);
  EXPECT_THROW(sf::lgamma(-1.5), std::domain_error);
}

TEST(sf_erf, known_values) {
  EXPECT_NEAR(sf::erf(1.0), 0.8427007929497149, 1e-13);
  EXPECT_NEAR(sf::erf(-1.0), -0.8427007929497149, 1e-13);
  EXPECT_NEAR(sf::erf(0.0), 0.0, 1e-15);
  EXPECT_NEAR(sf::erf(0.2), 0.22270258921047845, 1e-13);
  EXPECT_NEAR(sf::erfc(2.0), 0.004677734981063127, 1e-13);
  EXPECT_NEAR(sf::erfc(0.5), 0.4795001221869535, 1e-13);
  // large-x erfc stays accurate (relative)
  EXPECT_NEAR(sf::erfc(5.0) / 1.5374597944280349e-12, 1.0, 1e-10);
}

TEST(sf_erf_inv, roundtrip) {
  for (const double x : {-2.0, -0.5, 0.1, 1.5}) {
    EXPECT_NEAR(sf::erf_inv(sf::erf(x)), x, 1e-12);
  }
  EXPECT_THROW(sf::erf_inv(1.0), std::domain_error);
  EXPECT_THROW(sf::erf_inv(-1.5), std::domain_error);
}

TEST(sf_gamma_inc, known_values) {
  // P(3,x) = 1 - e^{-x}(1 + x + x^2/2), x = 2.5
  EXPECT_NEAR(sf::gamma_p(3.0, 2.5),
              1.0 - std::exp(-2.5) * (1.0 + 2.5 + 3.125), 1e-13);
  EXPECT_NEAR(sf::gamma_q(0.5, 1.0), 0.15729920705028513, 1e-13);
  EXPECT_NEAR(sf::gamma_p(1.0, 1.0), 1.0 - std::exp(-1.0), 1e-13);
  EXPECT_NEAR(sf::gamma_p(10.0, 3.0) + sf::gamma_q(10.0, 3.0), 1.0, 1e-13);
  EXPECT_EQ(sf::gamma_p(2.0, 0.0), 0.0);
}

TEST(sf_beta_inc, known_values) {
  EXPECT_NEAR(sf::beta_inc(2.0, 3.0, 0.4), 0.5248, 1e-12);
  // I_x(1/2,1/2) = (2/pi) asin(sqrt(x))
  EXPECT_NEAR(sf::beta_inc(0.5, 0.5, 0.3),
              2.0 / 3.141592653589793 * std::asin(std::sqrt(0.3)), 1e-12);
  EXPECT_EQ(sf::beta_inc(1.0, 1.0, 0.0), 0.0);
  EXPECT_EQ(sf::beta_inc(1.0, 1.0, 1.0), 1.0);
  // symmetry I_x(a,b) = 1 - I_{1-x}(b,a)
  EXPECT_NEAR(sf::beta_inc(2.5, 4.0, 0.35),
              1.0 - sf::beta_inc(4.0, 2.5, 0.65), 1e-13);
  EXPECT_THROW(sf::beta_inc(2.0, 3.0, -0.1), std::domain_error);
  EXPECT_THROW(sf::beta_inc(2.0, 3.0, 1.1), std::domain_error);
}

TEST(sf_log_beta, known_values) {
  // B(2,3) = 1/12
  EXPECT_NEAR(sf::log_beta(2.0, 3.0), std::log(1.0 / 12.0), 1e-13);
  EXPECT_NEAR(sf::log_beta(0.5, 0.5), std::log(3.141592653589793), 1e-13);
}
