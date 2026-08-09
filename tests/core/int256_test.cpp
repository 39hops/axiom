/** @file int256_test.cpp i256 cross-checked against ax::core::bigint
    (the codebase's oracle-by-cross-check doctrine): random product /
    quotient round-trips through decimal strings. */
#include <gtest/gtest.h>

#include <ax/core/bigint.hpp>
#include <ax/core/int256.hpp>

#include <random>
#include <string>

using ax::bigint;
using ax::core::i256;

namespace {
std::string i128_str(__int128 v) {
  if (v == 0) return "0";
  const bool neg = v < 0;
  unsigned __int128 u = neg ? -(unsigned __int128)v : v;
  std::string s;
  while (u) {
    s.push_back(char('0' + int(u % 10)));
    u /= 10;
  }
  if (neg) s.push_back('-');
  return {s.rbegin(), s.rend()};
}
std::string i256_str(const i256& v) {
  // via to_i128 when it fits, else limb-decompose through bigint
  bigint b(0);
  const bigint two64("18446744073709551616");
  const i256 m = v.negative() ? -v : v;
  for (int i = 3; i >= 0; i--)
    b = b * two64 + bigint((long long)(m.l[i] >> 32)) *
                        bigint(4294967296LL) +
        bigint((long long)(m.l[i] & 0xffffffffull));
  return (v.negative() ? bigint(0) - b : b).to_string();
}
}  // namespace

TEST(Int256, MulDivRoundtripVsBigint) {
  std::mt19937_64 rng(613);
  for (int it = 0; it < 2000; it++) {
    __int128 a = ((__int128)rng() << 64) | rng();
    __int128 b = ((__int128)rng() << 40) | rng();
    a >>= (rng() % 120);  // vary magnitudes
    b >>= (rng() % 100);
    if (rng() & 1) a = -a;
    if (rng() & 1) b = -b;
    if (b == 0) continue;
    const i256 p = i256(a) * i256(b);
    EXPECT_EQ(i256_str(p),
              (bigint(i128_str(a)) * bigint(i128_str(b))).to_string())
        << "iter " << it;
    EXPECT_EQ((p / i256(b)).to_i128(), a) << "iter " << it;
  }
}

TEST(Int256, ShiftAndCompare) {
  const i256 one(1);
  EXPECT_EQ(((one << 200) >> 200).to_i128(), __int128(1));
  EXPECT_TRUE(i256(-5) < i256(3));
  EXPECT_TRUE(i256(3) > i256(-5));
  EXPECT_EQ((i256(-40) / i256(7)).to_i128(), __int128(-5));  // trunc
  EXPECT_EQ((i256(-40) % i256(7)).to_i128(), __int128(-5));
}

TEST(Int256, ToI128OverflowThrows) {
  EXPECT_THROW((i256(1) << 130).to_i128(), std::runtime_error);
}
