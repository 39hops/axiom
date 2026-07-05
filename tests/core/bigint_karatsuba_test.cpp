#include <gtest/gtest.h>
#include <ax/core/bigint.hpp>

#include <random>
#include <string>

using ax::bigint;

namespace {
bigint random_bigint(std::size_t decimal_digits, std::mt19937_64& rng) {
  std::string s;
  s.push_back(static_cast<char>('1' + rng() % 9));
  for (std::size_t i = 1; i < decimal_digits; ++i)
    s.push_back(static_cast<char>('0' + rng() % 10));
  return bigint{s};
}
}  // namespace

TEST(bigint_karatsuba, matches_identity_on_large_operands) {
  std::mt19937_64 rng{12345};
  for (int iter = 0; iter < 10; ++iter) {
    // ~2000 decimal digits => ~104 limbs, above Karatsuba threshold
    bigint a = random_bigint(2000, rng);
    bigint b = random_bigint(2000, rng);
    bigint c = random_bigint(2000, rng);
    EXPECT_EQ(a * (b + c), a * b + a * c);
    EXPECT_EQ((a * b) / b, a);  // exercises division against big products
  }
}

TEST(bigint_karatsuba, asymmetric_sizes) {
  std::mt19937_64 rng{99};
  bigint a = random_bigint(5000, rng);
  bigint b = random_bigint(37, rng);
  EXPECT_EQ((a * b) / b, a);
}
