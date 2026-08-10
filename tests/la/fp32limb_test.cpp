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

namespace {
// reconstruct element k of a sliced segment as an exact dyadic
dyadic reconstruct(const sliced& s, int k) {
  dyadic r{bigint(0), 0};
  for (std::size_t p = 0; p < s.sl.size(); ++p) {
    const dyadic d = decode(s.sl[p][k]);
    acc(r, d.m, d.e - SLICE_W * (int(p) + 1) + s.e_align);
  }
  return r;
}
}  // namespace

TEST(fp32limb_slice, reconstructs_random_window_exactly) {
  ax::st::rng g(11);
  std::vector<float> x(BLOCK);
  for (auto& v : x) v = static_cast<float>(g.uniform(-1.0, 1.0));
  const sliced s = slice_row(x.data(), BLOCK);
  for (int k = 0; k < BLOCK; ++k)
    EXPECT_TRUE(dyadic_eq(reconstruct(s, k), decode(x[k]))) << "k=" << k;
}

TEST(fp32limb_slice, uniform_exponent_needs_few_slices) {
  // all elements in [1, 2): 24 mantissa bits / 7 -> 4 slices suffice
  ax::st::rng g(12);
  std::vector<float> x(BLOCK);
  for (auto& v : x) v = static_cast<float>(g.uniform(1.0, 2.0));
  EXPECT_LE(slice_row(x.data(), BLOCK).sl.size(), 4u);
}

TEST(fp32limb_slice, zero_window_and_short_window) {
  std::vector<float> z(BLOCK, 0.0f);
  const sliced s = slice_row(z.data(), 4);
  ASSERT_GE(s.sl.size(), 1u);
  for (int k = 0; k < 4; ++k) EXPECT_EQ(s.sl[0][k], 0.0f);
}

TEST(fp32limb_slice, wide_exponent_spread_is_loud_envelope_reject) {
  // full-mantissa element at spread 40: lowest bit lands at 2^-64, below
  // the MAX_SLICES*SLICE_W = 56-bit envelope -> must throw, in Release
  // builds too (fence is a throw, not an assert). Note a SINGLE-bit
  // element at 2^-40 is exactly representable and rightly accepted.
  std::vector<float> x(BLOCK, 0.0f);
  x[0] = 1.0f;
  x[1] = std::ldexp(1.0f + std::ldexp(1.0f, -23), -40);
  EXPECT_NO_THROW(slice_row(x.data(), 1));  // alone it aligns fine
  EXPECT_THROW(slice_row(x.data(), BLOCK), std::runtime_error);
}

namespace {
matf random_matf(int r, int c, ax::st::rng& g, bool scaled_normal) {
  matf m(r, c);
  for (auto& v : m.v)
    v = scaled_normal ? static_cast<float>(g.normal() * 0.05)
                      : static_cast<float>(g.uniform(-1.0, 1.0));
  return m;
}
}  // namespace

// P-ENVELOPE-EXACT: fp32-limb GEMM equals the bigint reference exactly,
// inside the envelope, across sizes straddling the K-block boundary.
TEST(fp32limb_gemm, p_envelope_exact) {
  for (int n : {8, 33, 64}) {
    for (std::uint64_t seed : {1, 2, 3, 4, 5}) {
      for (bool cls : {false, true}) {
        ax::st::rng g(seed * 1000 + static_cast<std::uint64_t>(n) + cls);
        const matf a = random_matf(n, n, g, cls);
        const matf b = random_matf(n, n, g, cls);
        const auto ours = gemm_fp32limb(a, b);
        const auto ref = gemm_ref(a, b);
        ASSERT_EQ(ours.size(), ref.size());
        for (std::size_t t = 0; t < ref.size(); ++t)
          ASSERT_TRUE(dyadic_eq(ours[t], ref[t]))
              << "n=" << n << " seed=" << seed << " cls=" << cls << " t=" << t;
      }
    }
  }
}
