#include <gtest/gtest.h>
#include <ax/core/nt.hpp>

#include <stdexcept>
#include <vector>

using ax::bigint;

TEST(nt, is_prime_u64_known_values) {
  EXPECT_FALSE(ax::is_prime(0ull));
  EXPECT_FALSE(ax::is_prime(1ull));
  EXPECT_TRUE(ax::is_prime(2ull));
  EXPECT_TRUE(ax::is_prime(3ull));
  EXPECT_FALSE(ax::is_prime(4ull));
  EXPECT_TRUE(ax::is_prime(1000000007ull));
  EXPECT_TRUE(ax::is_prime(18446744073709551557ull));  // largest u64 prime
  // strong pseudoprime to bases 10,15; composite = 151 * 21291191:
  EXPECT_FALSE(ax::is_prime(3215031751ull));
}

TEST(nt, is_prime_bigint_probabilistic) {
  EXPECT_TRUE(
      ax::is_prime(bigint{"170141183460469231731687303715884105727"}));  // 2^127-1
  EXPECT_FALSE(
      ax::is_prime(bigint{"170141183460469231731687303715884105725"}));
}

TEST(nt, factor_u64) {
  auto f = ax::factor(600851475143ull);  // = 71 * 839 * 1471 * 6857
  ASSERT_EQ(f.size(), 4u);
  EXPECT_EQ(f[0], 71ull);
  EXPECT_EQ(f[1], 839ull);
  EXPECT_EQ(f[2], 1471ull);
  EXPECT_EQ(f[3], 6857ull);
  auto p2 = ax::factor(1024ull);
  EXPECT_EQ(p2, (std::vector<std::uint64_t>(10, 2ull)));
  EXPECT_TRUE(ax::factor(1ull).empty());
}

TEST(nt, modinv) {
  // 3 * 4 = 12 == 1 mod 11
  EXPECT_EQ(ax::modinv(bigint{3}, bigint{11}).to_string(), "4");
  EXPECT_THROW(ax::modinv(bigint{4}, bigint{8}), std::domain_error);
}

TEST(nt, crt) {
  // x == 2 mod 3, x == 3 mod 5, x == 2 mod 7 -> x = 23 mod 105
  auto x = ax::crt({{bigint{2}, bigint{3}},
                    {bigint{3}, bigint{5}},
                    {bigint{2}, bigint{7}}});
  EXPECT_EQ(x.first.to_string(), "23");
  EXPECT_EQ(x.second.to_string(), "105");
  EXPECT_THROW(ax::crt({}), std::domain_error);
  EXPECT_THROW(ax::crt({{bigint{1}, bigint{4}}, {bigint{1}, bigint{6}}}),
               std::domain_error);
}
