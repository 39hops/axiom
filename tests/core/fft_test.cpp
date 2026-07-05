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

namespace {

/// Random n-limb positive bigint assembled via shifts (fast; avoids the
/// O(n^2) decimal parse that dominates at this size).
ax::bigint random_wide(std::size_t limbs, std::mt19937_64& rng) {
  ax::bigint r;
  for (std::size_t i = 0; i < limbs; ++i) {
    // 63-bit chunks keep the long long constructor happy
    r = (r << 63) + ax::bigint{static_cast<long long>(rng() >> 1)};
  }
  return r;
}

}  // namespace

TEST(ntt, huge_bigint_mul_still_correct) {
  // forces NTT path in bigint mul: ~13000 limbs, above the 12288-limb NTT
  // threshold. Independent oracle: split b in half by shifting so both
  // partial products stay on the Karatsuba path (min operand ~6500 limbs),
  // then compare against the NTT product.
  std::mt19937_64 rng{7};
  const ax::bigint a = random_wide(13000, rng);
  const ax::bigint b = random_wide(13000, rng);
  constexpr unsigned split_bits = 6500 * 64;
  const ax::bigint b_hi = b >> split_bits;
  const ax::bigint b_lo = b - (b_hi << split_bits);
  EXPECT_EQ(a * b, ((a * b_hi) << split_bits) + a * b_lo);
}
