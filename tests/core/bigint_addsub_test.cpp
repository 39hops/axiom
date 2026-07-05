#include <gtest/gtest.h>
#include <ax/core/bigint.hpp>

using ax::bigint;

TEST(bigint_addsub, small_values) {
  EXPECT_EQ((bigint{2} + bigint{3}).to_string(), "5");
  EXPECT_EQ((bigint{2} - bigint{3}).to_string(), "-1");
  EXPECT_EQ((bigint{-2} + bigint{-3}).to_string(), "-5");
  EXPECT_EQ((bigint{-2} - bigint{-3}).to_string(), "1");
}

TEST(bigint_addsub, carry_across_limbs) {
  bigint a{"18446744073709551615"};  // 2^64 - 1
  EXPECT_EQ((a + bigint{1}).to_string(), "18446744073709551616");
  EXPECT_EQ((bigint{"18446744073709551616"} - bigint{1}).to_string(),
            "18446744073709551615");
}

TEST(bigint_addsub, algebraic_identities) {
  // (a + b) - b == a on many-limb values
  bigint a{"98765432109876543210987654321098765432109876543210"};
  bigint b{"12345678901234567890123456789012345678901234567890"};
  EXPECT_EQ(((a + b) - b), a);
  EXPECT_EQ((a - a).to_string(), "0");
  EXPECT_EQ((a + (-a)).to_string(), "0");
}

TEST(bigint_addsub, unary_minus) {
  EXPECT_EQ((-bigint{5}).to_string(), "-5");
  EXPECT_EQ((-bigint{0}).to_string(), "0");
}
