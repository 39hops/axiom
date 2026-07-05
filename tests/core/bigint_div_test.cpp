#include <gtest/gtest.h>
#include <ax/core/bigint.hpp>

#include <stdexcept>

using ax::bigint;

TEST(bigint_div, small) {
  EXPECT_EQ((bigint{42} / bigint{7}).to_string(), "6");
  EXPECT_EQ((bigint{43} / bigint{7}).to_string(), "6");
  EXPECT_EQ((bigint{43} % bigint{7}).to_string(), "1");
  EXPECT_EQ((bigint{-43} / bigint{7}).to_string(), "-6");  // trunc toward 0
  EXPECT_EQ((bigint{-43} % bigint{7}).to_string(), "-1");  // sign of dividend
}

TEST(bigint_div, division_by_zero_throws) {
  EXPECT_THROW(bigint{1} / bigint{0}, std::domain_error);
  EXPECT_THROW(bigint{1} % bigint{0}, std::domain_error);
}

TEST(bigint_div, euclid_identity_large) {
  bigint a{"123456789012345678901234567890123456789012345678901234567890"};
  bigint b{"98765432109876543210987654321"};
  bigint q = a / b, r = a % b;
  EXPECT_EQ(q * b + r, a);
  EXPECT_LT(r, b);
  EXPECT_FALSE(r.is_negative());
}

TEST(bigint_div, quotient_digit_correction_case) {
  // exercise rare q-hat overestimate path: divisor of all-ones limbs
  bigint b = (bigint{1} << 64) - bigint{1};
  bigint a = (b << 128) + (b << 64) + b;
  bigint q = a / b, r = a % b;
  EXPECT_EQ(q * b + r, a);
}
