#include <gtest/gtest.h>
#include <ax/core/bigint.hpp>

using ax::bigint;

TEST(bigint_mul, small) {
  EXPECT_EQ((bigint{6} * bigint{7}).to_string(), "42");
  EXPECT_EQ((bigint{-6} * bigint{7}).to_string(), "-42");
  EXPECT_EQ((bigint{-6} * bigint{-7}).to_string(), "42");
  EXPECT_EQ((bigint{0} * bigint{7}).to_string(), "0");
}

TEST(bigint_mul, known_large_product) {
  bigint a{"123456789012345678901234567890"};
  bigint b{"987654321098765432109876543210"};
  EXPECT_EQ((a * b).to_string(),
            "121932631137021795226185032733622923332237463801111263526900");
}

TEST(bigint_mul, distributive_law) {
  bigint a{"31415926535897932384626433832795028841971"};
  bigint b{"27182818284590452353602874713526624977572"};
  bigint c{"16180339887498948482045868343656381177203"};
  EXPECT_EQ(a * (b + c), a * b + a * c);
}

TEST(bigint_mul, shifts) {
  EXPECT_EQ((bigint{1} << 64).to_string(), "18446744073709551616");
  EXPECT_EQ((bigint{"18446744073709551616"} >> 64).to_string(), "1");
  EXPECT_EQ((bigint{"12345"} << 3).to_string(), "98760");
  EXPECT_EQ((bigint{"98765"} >> 2).to_string(), "24691");
}
