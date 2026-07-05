#include <gtest/gtest.h>
#include <ax/la/blas.hpp>
#include <ax/la/decomp.hpp>

#include <random>
#include <stdexcept>

using ax::la::cholesky;
using ax::la::cholesky_solve;
using ax::la::mat;
using ax::la::vec;

namespace {
mat random_spd(std::size_t n, std::mt19937_64& rng) {
  // A = M^T M + n*I is symmetric positive definite
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  mat m(n, n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) m(i, j) = u(rng);
  mat a = m.transposed() * m;
  for (std::size_t i = 0; i < n; ++i) a(i, i) += static_cast<double>(n);
  return a;
}
}  // namespace

TEST(cholesky, known_3x3) {
  const mat a{{4, 12, -16}, {12, 37, -43}, {-16, -43, 98}};
  const mat expected{{2, 0, 0}, {6, 1, 0}, {-8, 5, 3}};
  EXPECT_TRUE(approx_equal(cholesky(a), expected, 1e-12));
}

TEST(cholesky, reconstructs_random_spd) {
  std::mt19937_64 rng{21};
  const std::size_t n = 40;
  const mat a = random_spd(n, rng);
  const mat l = cholesky(a);
  EXPECT_TRUE(approx_equal(l * l.transposed(), a, 1e-9 * static_cast<double>(n)));
  // L lower triangular
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i + 1; j < n; ++j) EXPECT_EQ(l(i, j), 0.0);
}

TEST(cholesky, solve_residual) {
  std::mt19937_64 rng{22};
  const std::size_t n = 30;
  const mat a = random_spd(n, rng);
  vec b(n);
  for (std::size_t i = 0; i < n; ++i) b[i] = static_cast<double>(i) - 10.0;
  const vec x = cholesky_solve(cholesky(a), b);
  EXPECT_LE((a * x - b).norm(), 1e-9);
}

TEST(cholesky, non_spd_throws) {
  const mat neg{{1, 0}, {0, -1}};
  EXPECT_THROW(cholesky(neg), std::domain_error);
  const mat zero(3, 3);
  EXPECT_THROW(cholesky(zero), std::domain_error);
}

TEST(cholesky, non_square_throws) {
  EXPECT_THROW(cholesky(mat(2, 3)), std::invalid_argument);
}

TEST(cholesky, solve_size_mismatch_throws) {
  const mat l = cholesky(mat{{4, 0}, {0, 9}});
  EXPECT_THROW(cholesky_solve(l, vec(3, 1.0)), std::invalid_argument);
}
