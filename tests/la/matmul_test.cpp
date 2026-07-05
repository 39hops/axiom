#include <gtest/gtest.h>
#include <ax/la/blas.hpp>

#include <random>
#include <stdexcept>

using ax::la::approx_equal;
using ax::la::mat;
using ax::la::matmul;

namespace {
mat random_mat(std::size_t r, std::size_t c, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  mat m(r, c);
  for (std::size_t i = 0; i < r; ++i)
    for (std::size_t j = 0; j < c; ++j) m(i, j) = u(rng);
  return m;
}
}  // namespace

TEST(matmul, known_small) {
  mat a{{1, 2, 3}, {4, 5, 6}};
  mat b{{7, 8}, {9, 10}, {11, 12}};
  const mat c = a * b;
  EXPECT_DOUBLE_EQ(c(0, 0), 58.0);
  EXPECT_DOUBLE_EQ(c(0, 1), 64.0);
  EXPECT_DOUBLE_EQ(c(1, 0), 139.0);
  EXPECT_DOUBLE_EQ(c(1, 1), 154.0);
}

TEST(matmul, identity_and_shape_throw) {
  std::mt19937_64 rng{1};
  const mat a = random_mat(17, 23, rng);
  EXPECT_TRUE(approx_equal(matmul(a, mat::identity(23)), a, 1e-14));
  EXPECT_THROW(matmul(a, mat(5, 5)), std::invalid_argument);
}

TEST(matmul, parallel_matches_serial_odd_shapes) {
  std::mt19937_64 rng{2};
  const mat a = random_mat(65, 37, rng);
  const mat b = random_mat(37, 53, rng);
  ax::thread_pool pool{4};
  EXPECT_TRUE(approx_equal(matmul(a, b, pool), matmul(a, b), 1e-12 * 37));
}

TEST(matmul, parallel_matches_serial_200) {
  std::mt19937_64 rng{3};
  const mat a = random_mat(200, 200, rng);
  const mat b = random_mat(200, 200, rng);
  ax::thread_pool pool;
  EXPECT_TRUE(approx_equal(matmul(a, b, pool), matmul(a, b), 1e-12 * 200));
}
