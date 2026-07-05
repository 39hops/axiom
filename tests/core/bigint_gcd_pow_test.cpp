#include <gtest/gtest.h>
#include <ax/core/bigint.hpp>

using ax::bigint;

TEST(bigint_gcd, basics) {
  EXPECT_EQ(ax::gcd(bigint{12}, bigint{18}).to_string(), "6");
  EXPECT_EQ(ax::gcd(bigint{0}, bigint{5}).to_string(), "5");
  EXPECT_EQ(ax::gcd(bigint{-12}, bigint{18}).to_string(), "6");
  EXPECT_EQ(ax::gcd(bigint{17}, bigint{13}).to_string(), "1");
}

TEST(bigint_pow, basics) {
  EXPECT_EQ(ax::pow(bigint{2}, 10).to_string(), "1024");
  EXPECT_EQ(ax::pow(bigint{10}, 30).to_string(),
            "1000000000000000000000000000000");
  EXPECT_EQ(ax::pow(bigint{7}, 0).to_string(), "1");
  EXPECT_EQ(ax::pow(bigint{-2}, 3).to_string(), "-8");
}

TEST(bigint_modpow, fermat_little) {
  // a^(p-1) == 1 mod p for prime p, gcd(a,p)=1
  bigint p{"1000000007"};
  EXPECT_EQ(ax::modpow(bigint{123456}, p - bigint{1}, p).to_string(), "1");
  // 2^1000 mod 1000000007 == 688423210
  EXPECT_EQ(ax::modpow(bigint{2}, bigint{1000}, p).to_string(), "688423210");
}
