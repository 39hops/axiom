#include <gtest/gtest.h>
#include <ax/core/bigint.hpp>
#include <ax/core/fft.hpp>

#include <random>
#include <string>
#include <vector>

TEST(fft, roundtrip) {
  std::vector<std::complex<double>> v{{1, 0}, {2, 0}, {3, 0}, {4, 0}};
  auto f = v;
  ax::fft(f, false);
  ax::fft(f, true);
  for (std::size_t i = 0; i < v.size(); ++i)
    EXPECT_NEAR(f[i].real(), v[i].real(), 1e-9);
}

TEST(fft, convolution_matches_naive) {
  std::vector<double> a{1, 2, 3}, b{4, 5, 6, 7};
  auto c = ax::convolve(a, b);
  const std::vector<double> want{4, 13, 28, 34, 32, 21};
  ASSERT_EQ(c.size(), want.size());
  for (std::size_t i = 0; i < c.size(); ++i) EXPECT_NEAR(c[i], want[i], 1e-6);
}

TEST(ntt, exact_convolution) {
  std::vector<std::uint64_t> a{123456789, 987654321, 555};
  std::vector<std::uint64_t> b{111111111, 222222222};
  auto c = ax::ntt_convolve(a, b);
  std::vector<std::uint64_t> want(a.size() + b.size() - 1, 0);
  for (std::size_t i = 0; i < a.size(); ++i)
    for (std::size_t j = 0; j < b.size(); ++j)
      want[i + j] = (want[i + j] + a[i] * b[j]) % ax::ntt_mod;
  EXPECT_EQ(c, want);
}

TEST(ntt, huge_bigint_mul_still_correct) {
  // forces NTT path in bigint mul (>= ntt threshold limbs)
  std::mt19937_64 rng{7};
  std::string sa{"1"}, sb{"2"};
  // 100k digits ~ 5200 limbs, safely above the 4096-limb NTT threshold
  for (int i = 0; i < 100000; ++i)
    sa.push_back(static_cast<char>('0' + rng() % 10));
  for (int i = 0; i < 100000; ++i)
    sb.push_back(static_cast<char>('0' + rng() % 10));
  ax::bigint a{sa}, b{sb};
  const ax::bigint p = a * b;
  EXPECT_EQ(p / b, a);
  EXPECT_TRUE((p % b).is_zero());
}
