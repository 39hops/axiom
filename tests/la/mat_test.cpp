#include <gtest/gtest.h>
#include <ax/la/mat.hpp>

#include <stdexcept>

using ax::la::approx_equal;
using ax::la::mat;
using ax::la::vec;

TEST(vec_basic, construction_and_ops) {
  vec v{1.0, 2.0, 3.0};
  EXPECT_EQ(v.size(), 3u);
  EXPECT_DOUBLE_EQ(v[1], 2.0);
  vec w{4.0, 5.0, 6.0};
  EXPECT_DOUBLE_EQ(dot(v, w), 32.0);
  EXPECT_DOUBLE_EQ((v + w)[0], 5.0);
  EXPECT_DOUBLE_EQ((w - v)[2], 3.0);
  EXPECT_DOUBLE_EQ((2.0 * v)[2], 6.0);
  EXPECT_NEAR((vec{3.0, 4.0}).norm(), 5.0, 1e-15);
  EXPECT_THROW(v + vec{1.0}, std::invalid_argument);
}

TEST(mat_basic, construction) {
  mat a{{1, 2, 3}, {4, 5, 6}};
  EXPECT_EQ(a.rows(), 2u);
  EXPECT_EQ(a.cols(), 3u);
  EXPECT_DOUBLE_EQ(a(1, 2), 6.0);
  EXPECT_THROW((mat{{1, 2}, {3}}), std::invalid_argument);
  const mat i3 = mat::identity(3);
  EXPECT_DOUBLE_EQ(i3(1, 1), 1.0);
  EXPECT_DOUBLE_EQ(i3(0, 2), 0.0);
}

TEST(mat_basic, checked_access) {
  mat a(2, 2);
  EXPECT_NO_THROW(a.at(1, 1) = 7.0);
  EXPECT_DOUBLE_EQ(a.at(1, 1), 7.0);
  EXPECT_THROW(a.at(2, 0), std::out_of_range);
  EXPECT_THROW(a.at(0, 2), std::out_of_range);
}

TEST(mat_basic, arithmetic_and_transpose) {
  mat a{{1, 2}, {3, 4}};
  mat b{{5, 6}, {7, 8}};
  EXPECT_DOUBLE_EQ((a + b)(0, 0), 6.0);
  EXPECT_DOUBLE_EQ((b - a)(1, 1), 4.0);
  EXPECT_DOUBLE_EQ((3.0 * a)(1, 0), 9.0);
  const mat t = a.transposed();
  EXPECT_DOUBLE_EQ(t(0, 1), 3.0);
  EXPECT_THROW(a + mat(3, 3), std::invalid_argument);
}

TEST(mat_basic, mat_vec_product) {
  mat a{{1, 2, 3}, {4, 5, 6}};
  vec x{1.0, 1.0, 1.0};
  const vec y = a * x;
  EXPECT_DOUBLE_EQ(y[0], 6.0);
  EXPECT_DOUBLE_EQ(y[1], 15.0);
  EXPECT_THROW(a * vec{1.0}, std::invalid_argument);
}

TEST(mat_basic, approx_equal_helper) {
  mat a{{1, 2}, {3, 4}};
  mat b = a;
  b(0, 0) += 1e-12;
  EXPECT_TRUE(approx_equal(a, b, 1e-10));
  EXPECT_FALSE(approx_equal(a, b, 1e-14));
  EXPECT_FALSE(approx_equal(a, mat(2, 3), 1.0));  // shape mismatch
}
