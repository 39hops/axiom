/** @file rns_test.cpp anchor-v2 ring core: pinned primes, CRT,
    rational reconstruction with held-out-prime verification. The
    load-bearing property is LOUDNESS: an under-budgeted modulus must
    report ok=false, never return a plausible wrong rational. */
#include <gtest/gtest.h>

#include <ax/core/nt.hpp>
#include <ax/core/rational.hpp>
#include <ax/core/rns.hpp>

#include <random>

using ax::bigint;
using ax::rational;
namespace rns = ax::rns;

namespace {
bigint rand_big(std::mt19937_64& rng, int bits) {
  bigint x(0);
  for (int i = 0; i < bits; i += 32)
    x = (x << 32u) + bigint((long long)(rng() & 0xffffffffu));
  return x;
}
}  // namespace

TEST(Rns, PinnedPrimesArePrimeAndDistinct) {
  const auto c = rns::ctx::make(8);
  ASSERT_EQ(c.P.size(), 8u);
  for (std::size_t i = 0; i < c.P.size(); ++i) {
    EXPECT_TRUE(ax::is_prime(c.P[i]));
    EXPECT_LT(c.P[i], std::uint64_t(1) << 61);
    for (std::size_t j = i + 1; j < c.P.size(); ++j)
      EXPECT_NE(c.P[i], c.P[j]);
  }
}

TEST(Rns, RoundTripRandomRationals) {
  const auto c = rns::ctx::make(20);
  std::mt19937_64 rng(613);
  for (int it = 0; it < 200; ++it) {
    bigint n = rand_big(rng, 200);
    if (rng() & 1) n = -n;
    bigint d = rand_big(rng, 200) + bigint(1);
    const rational v(n, d);
    const auto [res, ok] = rns::to_res(v, c);
    const auto rec = rns::reconstruct(res, ok, c);
    ASSERT_TRUE(rec.ok) << it;
    EXPECT_EQ(rec.v.num(), v.num());
    EXPECT_EQ(rec.v.den(), v.den());
  }
}

TEST(Rns, InsufficientModulusIsLoud) {
  const auto c = rns::ctx::make(4);
  std::mt19937_64 rng(7);
  int loud = 0;
  for (int it = 0; it < 50; ++it) {
    const rational v(rand_big(rng, 600), rand_big(rng, 600) + bigint(1));
    const auto [res, ok] = rns::to_res(v, c);
    const auto rec = rns::reconstruct(res, ok, c);
    // either loud, or (astronomically unlikely) actually correct
    if (!rec.ok) { loud++; continue; }
    EXPECT_EQ(rec.v.num(), v.num());
    EXPECT_EQ(rec.v.den(), v.den());
  }
  EXPECT_GE(loud, 45);
}

TEST(Rns, PoleDropAndRecover) {
  const auto c = rns::ctx::make(12);
  // denominator exactly P[0]: residue at P[0] is a pole
  const rational v(bigint(12345), bigint((long long)c.P[0]));
  const auto [res, ok] = rns::to_res(v, c);
  EXPECT_EQ(ok[0], 0);
  const auto rec = rns::reconstruct(res, ok, c);
  ASSERT_TRUE(rec.ok);
  EXPECT_EQ(rec.v.num(), v.num());
  EXPECT_EQ(rec.v.den(), v.den());
}
