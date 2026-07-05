#include <gtest/gtest.h>
#include <ax/la/blas.hpp>
#include <ax/la/decomp.hpp>

#include <random>
#include <stdexcept>

using ax::la::det;
using ax::la::inverse;
using ax::la::lu_decompose;
using ax::la::lu_solve;
using ax::la::mat;
using ax::la::vec;

namespace {
mat random_spd_ish(std::size_t n, std::mt19937_64& rng) {
  // well-conditioned: random + n on the diagonal
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  mat m(n, n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) m(i, j) = u(rng);
  for (std::size_t i = 0; i < n; ++i) m(i, i) += static_cast<double>(n);
  return m;
}
}  // namespace

TEST(lu, known_determinants) {
  EXPECT_NEAR(det(mat{{1, 2}, {3, 4}}), -2.0, 1e-12);
  EXPECT_NEAR(det(mat{{6, 1, 1}, {4, -2, 5}, {2, 8, 7}}), -306.0, 1e-10);
  EXPECT_NEAR(det(mat::identity(5)), 1.0, 1e-12);
}

TEST(lu, solve_random_well_conditioned) {
  std::mt19937_64 rng{11};
  const std::size_t n = 50;
  const mat a = random_spd_ish(n, rng);
  vec b(n);
  for (std::size_t i = 0; i < n; ++i) b[i] = static_cast<double>(i) - 20.0;
  const vec x = lu_solve(lu_decompose(a), b);
  EXPECT_LE((a * x - b).norm(), 1e-9);
}

TEST(lu, inverse_roundtrip) {
  std::mt19937_64 rng{12};
  const mat a = random_spd_ish(20, rng);
  EXPECT_TRUE(approx_equal(a * inverse(a), mat::identity(20), 1e-9));
}

TEST(lu, singular_detected) {
  mat s{{1, 2, 3}, {2, 4, 6}, {1, 0, 1}};  // row2 = 2*row1
  EXPECT_NEAR(det(s), 0.0, 1e-12);
  EXPECT_THROW(lu_solve(lu_decompose(s), vec(3, 1.0)), std::domain_error);
  EXPECT_THROW(inverse(s), std::domain_error);
}

TEST(lu, non_square_throws) {
  EXPECT_THROW(lu_decompose(mat(2, 3)), std::invalid_argument);
}
