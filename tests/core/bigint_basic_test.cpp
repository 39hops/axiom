#include <gtest/gtest.h>
#include <ax/core/bigint.hpp>

#include <cstdint>
#include <stdexcept>

using ax::bigint;

TEST(bigint_basic, default_is_zero) {
  bigint z;
  EXPECT_TRUE(z.is_zero());
  EXPECT_EQ(z.to_string(), "0");
}

TEST(bigint_basic, from_int64_roundtrip) {
  EXPECT_EQ(bigint{42}.to_string(), "42");
  EXPECT_EQ(bigint{-7}.to_string(), "-7");
  EXPECT_EQ(bigint{INT64_MIN}.to_string(), "-9223372036854775808");
}

TEST(bigint_basic, from_string_roundtrip_large) {
  const std::string s = "123456789012345678901234567890123456789";
  EXPECT_EQ(bigint{s}.to_string(), s);
  EXPECT_EQ(bigint{"-" + s}.to_string(), "-" + s);
  EXPECT_EQ(bigint{"000123"}.to_string(), "123");
  EXPECT_EQ(bigint{"-0"}.to_string(), "0");
}

TEST(bigint_basic, from_string_rejects_garbage) {
  EXPECT_THROW(bigint{""}, std::invalid_argument);
  EXPECT_THROW(bigint{"12a3"}, std::invalid_argument);
  EXPECT_THROW(bigint{"-"}, std::invalid_argument);
}

TEST(bigint_basic, ordering) {
  EXPECT_LT(bigint{-5}, bigint{3});
  EXPECT_LT(bigint{2}, bigint{10});
  EXPECT_LT(bigint{-10}, bigint{-2});
  EXPECT_EQ(bigint{"999999999999999999999"}, bigint{"999999999999999999999"});
  EXPECT_GT(bigint{"10000000000000000000000"}, bigint{"9999999999999999999999"});
}
