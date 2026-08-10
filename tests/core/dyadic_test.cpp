/** @file dyadic_test.cpp anchor-v2 pin 3: growing dyadic interval —
    outward-rounded containment is the whole certification. */
#include <gtest/gtest.h>

#include <ax/core/dyadic.hpp>
#include <ax/core/rational.hpp>

#include <random>

using ax::bigint;
using ax::dyi;

namespace {
// ground truth: does [lo,hi]*2^e contain p/q?  (q > 0)
bool contains(const dyi& d, const bigint& p, const bigint& q) {
  // lo*2^e <= p/q  <=>  lo*q*2^e <= p (careful with e sign)
  const auto scaled = [&](const bigint& m) {
    return d.e >= 0 ? (m * q) << unsigned(d.e) : m * q;
  };
  const bigint ps = d.e >= 0 ? p : p << unsigned(-d.e);
  return scaled(d.lo) <= ps && ps <= scaled(d.hi);
}
}  // namespace

TEST(Dyadic, PointAndFloorNegativeAware) {
  dyi a(-7);
  a = a >> 1;  // -7/2
  const auto [fl, fh] = a.floor_pair();
  EXPECT_EQ(fl.to_string(), "-4");
  EXPECT_EQ(fh.to_string(), "-4");
  dyi b(7);
  b = b >> 1;
  EXPECT_EQ(b.floor_pair().first.to_string(), "3");
  EXPECT_TRUE(dyi(5).is_point());
  EXPECT_EQ(dyi(5).point_int().to_string(), "5");
}

TEST(Dyadic, ArithmeticContainsGroundTruth) {
  const int saved = dyi::prec;
  dyi::prec = 96;
  std::mt19937_64 rng(613);
  for (int it = 0; it < 200; ++it) {
    const long long pn = (long long)(rng() % 200001) - 100000;
    const long long qn = (long long)(rng() % 99991) + 1;
    const long long rn = (long long)(rng() % 200001) - 100000;
    const long long sn = (long long)(rng() % 99991) + 1;
    // x = pn/qn, y = rn/sn via interval division
    const dyi x = dyi::div(dyi(pn), dyi(qn));
    const dyi y = dyi::div(dyi(rn), dyi(sn));
    // sum: (pn*sn + rn*qn) / (qn*sn)
    const dyi s = x + y;
    EXPECT_TRUE(contains(s, bigint(pn) * bigint(sn) + bigint(rn) * bigint(qn),
                         bigint(qn) * bigint(sn)));
    const dyi m = x * y;
    EXPECT_TRUE(contains(m, bigint(pn) * bigint(rn),
                         bigint(qn) * bigint(sn)));
    const dyi d2 = x - y;
    EXPECT_TRUE(contains(d2, bigint(pn) * bigint(sn) - bigint(rn) * bigint(qn),
                         bigint(qn) * bigint(sn)));
    if (rn != 0) {
      const dyi q = dyi::div(x, y);
      EXPECT_TRUE(contains(q, bigint(pn) * bigint(sn),
                           bigint(qn) * bigint(rn) *
                               bigint(rn < 0 ? -1 : 1)) ||
                  rn < 0);  // sign handled below
      if (rn < 0)
        EXPECT_TRUE(contains(q, bigint(-1) * bigint(pn) * bigint(sn),
                             bigint(qn) * bigint(-rn)));
    }
  }
  dyi::prec = saved;
}

TEST(Dyadic, NormalizeRoundsOutward) {
  const int saved = dyi::prec;
  dyi::prec = 8;
  // 1/3 at 8 bits: interval must still contain 1/3 and be non-point
  const dyi t = dyi::div(dyi(1), dyi(3));
  EXPECT_TRUE(contains(t, bigint(1), bigint(3)));
  EXPECT_FALSE(t.is_point());
  // width bounded: hi-lo <= 2 ulps at prec
  EXPECT_TRUE(t.hi - t.lo <= bigint(4));
  dyi::prec = saved;
}

TEST(Dyadic, ComparesAndZero) {
  const dyi a = dyi::div(dyi(1), dyi(3));
  const dyi b = dyi::div(dyi(1), dyi(2));
  EXPECT_TRUE(dyi::lt_certain(a, b));
  EXPECT_FALSE(dyi::lt_certain(b, a));
  EXPECT_TRUE((a - a).contains_zero());
  EXPECT_FALSE(a.contains_zero());
  EXPECT_EQ(dyi(0).sign(), 0);
  EXPECT_EQ(dyi(-3).sign(), -1);
  EXPECT_EQ(b.sign(), 1);
}
