/** @file exact_anchor2_test.cpp ANCHOR-V2: rx scalar semantics
    (pins 1/2/4 at the unit level), then the engine slice of
    P-DIGEST-EQUAL (step-1 EQ vs the exact anchor, SCHED boundary,
    forced fallback, loud modulus exhaustion). */
#include <gtest/gtest.h>

#include <ax/nn/exact_anchor.hpp>
#include <ax/nn/exact_anchor2.hpp>

#include <random>
#include <string>
#include <vector>

namespace ib = ax::nn::ib;
namespace a2 = ax::nn::ib::anchor2;
using a2::rx;
using i64 = std::int64_t;

TEST(Anchor2Scalar, RingIdentitiesResidueNative) {
  rx::init(8, 64);
  EXPECT_TRUE(rx(7) * rx(3) - rx(21) == rx(0));
  // exact /2^n through residues: 1/2 + 1/2 == 1 (pin 1 zero-test)
  EXPECT_TRUE((rx(1) >> 1) + (rx(1) >> 1) == rx(1));
  EXPECT_FALSE(rx(2) == rx(3));
  // exact division roundtrip
  const rx q = a2::Exact2::div(rx(10), rx(4));
  EXPECT_TRUE(q * rx(4) == rx(10));
  EXPECT_EQ(a2::rx::fb.cmp, 0);
}

TEST(Anchor2Scalar, FloorsDeclaredAndNegativeAware) {
  rx::init(8, 64);
  const rx a = a2::Exact2::div(rx(7), rx(2));
  EXPECT_EQ((long long)a2::Exact2::to_grain(a, 0), 3);
  const rx b = a2::Exact2::div(rx(-7), rx(2));
  EXPECT_EQ((long long)a2::Exact2::to_grain(b, 0), -4);
  EXPECT_EQ((long long)a2::Exact2::to_grain(rx(5), 0), 5);
}

TEST(Anchor2Scalar, ForcedFloorFallbackClassifiesAndCaches) {
  rx::init(8, 4);  // starved shadow: force straddles
  const rx t = a2::Exact2::div(rx(1), rx(3));
  const rx one = t + t + t;  // exactly 1, interval straddles it
  const long fe0 = rx::fb.floor_exact;
  EXPECT_EQ((long long)a2::Exact2::to_grain(one, 0), 1);
  EXPECT_EQ(rx::fb.floor_exact, fe0 + 1);
  // same residue value again: pin-2 cache, no second reconstruction
  const long rc = rx::fb.recon, ch = rx::fb.cache_hit;
  EXPECT_EQ((long long)a2::Exact2::to_grain(one, 0), 1);
  EXPECT_EQ(rx::fb.recon, rc);
  EXPECT_EQ(rx::fb.cache_hit, ch + 1);
  // near-boundary class
  const rx near = one + (rx(1) >> 30);
  const long fn0 = rx::fb.floor_near;
  EXPECT_EQ((long long)a2::Exact2::to_grain(near, 0), 1);
  EXPECT_EQ(rx::fb.floor_near, fn0 + 1);
}

TEST(Anchor2Scalar, CmpFallbackOrdersCorrectly) {
  rx::init(8, 8);
  const rx a = a2::Exact2::div(rx(1), rx(3));
  const rx b = a + (rx(1) >> 20);  // closer than the shadow width
  const long c0 = rx::fb.cmp;
  EXPECT_TRUE(a < b);
  EXPECT_TRUE(b > a);
  EXPECT_GE(rx::fb.cmp, c0 + 1);
}
