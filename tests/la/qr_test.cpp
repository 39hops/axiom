#include <gtest/gtest.h>
#include <ax/la/blas.hpp>
#include <ax/la/decomp.hpp>

#include <cmath>
#include <random>
#include <stdexcept>

using ax::la::lstsq;
using ax::la::lu_decompose;
using ax::la::lu_solve;
using ax::la::mat;
using ax::la::qr_decompose;
using ax::la::vec;

namespace {
mat random_mat(std::size_t m, std::size_t n, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  mat a(m, n);
  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t j = 0; j < n; ++j) a(i, j) = u(rng);
  return a;
}
}  // namespace

TEST(qr, orthogonal_and_reconstructs) {
  std::mt19937_64 rng{31};
  const std::size_t m = 30, n = 20;
  const mat a = random_mat(m, n, rng);
  const auto [q, r] = qr_decompose(a);
  EXPECT_TRUE(approx_equal(q.transposed() * q, mat::identity(m),
                           1e-10 * static_cast<double>(m)));
  EXPECT_TRUE(approx_equal(q * r, a, 1e-10 * static_cast<double>(m)));
  // R upper trapezoidal
  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t j = 0; j < n && j < i; ++j)
      EXPECT_LT(std::abs(r(i, j)), 1e-12);
}

TEST(qr, square_solve_matches_lu) {
  std::mt19937_64 rng{32};
  const std::size_t n = 25;
  mat a = random_mat(n, n, rng);
  for (std::size_t i = 0; i < n; ++i) a(i, i) += static_cast<double>(n);
  vec b(n);
  for (std::size_t i = 0; i < n; ++i) b[i] = static_cast<double>(i);
  const vec x_qr = lstsq(a, b);
  const vec x_lu = lu_solve(lu_decompose(a), b);
  EXPECT_LE((x_qr - x_lu).norm(), 1e-9);
}

TEST(qr, overdetermined_line_fit) {
  // y = 2x + 1, exact — lstsq must recover slope/intercept
  const std::size_t m = 50;
  mat a(m, 2);
  vec b(m);
  for (std::size_t i = 0; i < m; ++i) {
    const double x = static_cast<double>(i) * 0.1;
    a(i, 0) = x;
    a(i, 1) = 1.0;
    b[i] = 2.0 * x + 1.0;
  }
  const vec c = lstsq(a, b);
  EXPECT_NEAR(c[0], 2.0, 1e-10);
  EXPECT_NEAR(c[1], 1.0, 1e-10);
}

TEST(qr, underdetermined_throws) {
  EXPECT_THROW(qr_decompose(mat(2, 3)), std::invalid_argument);
  EXPECT_THROW(lstsq(mat(2, 3), vec(2, 1.0)), std::invalid_argument);
}

TEST(qr, rank_deficient_lstsq_throws) {
  // second column = 2 * first column
  mat a(4, 2);
  for (std::size_t i = 0; i < 4; ++i) {
    a(i, 0) = static_cast<double>(i + 1);
    a(i, 1) = 2.0 * static_cast<double>(i + 1);
  }
  EXPECT_THROW(lstsq(a, vec(4, 1.0)), std::domain_error);
}

TEST(qr, lstsq_size_mismatch_throws) {
  EXPECT_THROW(lstsq(mat(3, 2), vec(4, 1.0)), std::invalid_argument);
}
