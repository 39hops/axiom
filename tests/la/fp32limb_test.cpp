/** @file fp32limb_test.cpp
 *  R1 oracle invariants (PRE-REG FP32LIMB-METAL): decode/reference GEMM
 *  exactness against ax::bigint ground truth. Never fp-vs-fp.
 */
#include <gtest/gtest.h>

#include <ax/la/fp32limb.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

using namespace ax::la::fp32limb;
using ax::bigint;

TEST(fp32limb_decode, known_values) {
  // 1.5 = 3 * 2^-1
  EXPECT_TRUE(dyadic_eq(decode(1.5f), dyadic{bigint(3), -1}));
  EXPECT_TRUE(dyadic_eq(decode(-0.75f), dyadic{bigint(-3), -2}));
  EXPECT_TRUE(dyadic_eq(decode(0.0f), dyadic{bigint(0), 0}));
  // denormal: 2^-140
  EXPECT_TRUE(dyadic_eq(decode(std::ldexp(1.0f, -140)), dyadic{bigint(1), -140}));
  EXPECT_THROW(decode(std::numeric_limits<float>::infinity()),
               std::runtime_error);
  EXPECT_THROW(decode(std::numeric_limits<float>::quiet_NaN()),
               std::runtime_error);
}

TEST(fp32limb_decode, acc_aligns_exactly) {
  dyadic r{bigint(0), 0};
  acc(r, bigint(3), -2);   // 0.75
  acc(r, bigint(1), 1);    // + 2
  EXPECT_TRUE(dyadic_eq(r, dyadic{bigint(11), -2}));  // 2.75
}

TEST(fp32limb_ref, known_2x2) {
  matf a(2, 2), b(2, 2);
  a.at(0, 0) = 1; a.at(0, 1) = 2; a.at(1, 0) = 3; a.at(1, 1) = 4;
  b.at(0, 0) = 5; b.at(0, 1) = 6; b.at(1, 0) = 7; b.at(1, 1) = 8;
  const auto c = gemm_ref(a, b);
  ASSERT_EQ(c.size(), 4u);
  EXPECT_TRUE(dyadic_eq(c[0], dyadic{bigint(19), 0}));
  EXPECT_TRUE(dyadic_eq(c[1], dyadic{bigint(22), 0}));
  EXPECT_TRUE(dyadic_eq(c[2], dyadic{bigint(43), 0}));
  EXPECT_TRUE(dyadic_eq(c[3], dyadic{bigint(50), 0}));
}

TEST(fp32limb_ref, rejects_nonfinite) {
  matf a(1, 1), b(1, 1);
  a.at(0, 0) = std::numeric_limits<float>::infinity();
  b.at(0, 0) = 1.0f;
  EXPECT_THROW(gemm_ref(a, b), std::runtime_error);
}

#include <ax/st/rng.hpp>

namespace {
// Exact dyadic sum of a list of floats — bigint ground truth for EFT tests.
dyadic exact_sum(const std::vector<float>& xs) {
  dyadic r{bigint(0), 0};
  for (float x : xs) {
    const dyadic d = decode(x);
    acc(r, d.m, d.e);
  }
  return r;
}
}  // namespace

TEST(fp32limb_eft, two_sum_exact_10k_random_pairs) {
  ax::st::rng g(20260810);
  for (int t = 0; t < 10000; ++t) {
    // spread exponents so hi/lo splits are exercised
    const float a =
        static_cast<float>(g.uniform(-1.0, 1.0) * std::ldexp(1.0, int(g.below(40)) - 20));
    const float b =
        static_cast<float>(g.uniform(-1.0, 1.0) * std::ldexp(1.0, int(g.below(40)) - 20));
    const f2 sr = two_sum(a, b);
    EXPECT_TRUE(dyadic_eq(exact_sum({a, b}), exact_sum({sr.s, sr.r})))
        << "a=" << a << " b=" << b;
  }
}

TEST(fp32limb_eft, expansion_sums_1000_random_exactly) {
  ax::st::rng g(7);
  std::vector<float> xs;
  for (int t = 0; t < 1000; ++t)
    xs.push_back(static_cast<float>(
        g.uniform(-1.0, 1.0) * std::ldexp(1.0, int(g.below(30)) - 15)));
  std::vector<float> e{0.0f};
  for (float x : xs) e = exp_add(std::move(e), x);
  EXPECT_TRUE(dyadic_eq(exact_sum(xs), exact_sum(e)));
}

TEST(fp32limb_eft, capped_expansion_triple_double_ok_and_overflow_throws) {
  // three well-separated components: fits in cap 3
  std::vector<float> e{0.0f};
  for (float x : {std::ldexp(1.0f, 60), std::ldexp(1.0f, 30), 1.0f})
    e = exp_add_capped(std::move(e), x, 3);
  EXPECT_TRUE(dyadic_eq(
      exact_sum({std::ldexp(1.0f, 60), std::ldexp(1.0f, 30), 1.0f}),
      exact_sum(e)));
  // a fourth separated component must be a loud reject, not silent rounding
  EXPECT_THROW(exp_add_capped(std::move(e), std::ldexp(1.0f, -30), 3),
               std::runtime_error);
}
