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
  const long fe0 = rx::fb.floor_exact, rc0 = rx::fb.recon;
  EXPECT_EQ((long long)a2::Exact2::to_grain(one, 0), 1);
  EXPECT_EQ(rx::fb.floor_exact, fe0 + 1);
  EXPECT_EQ(rx::fb.recon, rc0);  // residue-native: NO reconstruction
  // near-boundary class reconstructs once, then hits the pin-2 cache
  const rx near = one + (rx(1) >> 30);
  const long fn0 = rx::fb.floor_near;
  EXPECT_EQ((long long)a2::Exact2::to_grain(near, 0), 1);
  EXPECT_EQ(rx::fb.floor_near, fn0 + 1);
  const long rc = rx::fb.recon, ch = rx::fb.cache_hit;
  EXPECT_EQ((long long)a2::Exact2::to_grain(near, 0), 1);
  EXPECT_EQ(rx::fb.recon, rc);
  EXPECT_EQ(rx::fb.cache_hit, ch + 1);
}

namespace {
// same tiny fixture as exact_anchor_test (duplicated: test files
// are independent translation units in this suite)
void put_u16(std::string& b, std::uint16_t v) {
  b.append(reinterpret_cast<const char*>(&v), 2);
}
void put_u32(std::string& b, std::uint32_t v) {
  b.append(reinterpret_cast<const char*>(&v), 4);
}
void put_u64(std::string& b, std::uint64_t v) {
  b.append(reinterpret_cast<const char*>(&v), 8);
}
void put_tensor(std::string& b, const std::string& name,
                const std::vector<i64>& v) {
  put_u16(b, std::uint16_t(name.size()));
  b.append(name);
  b.push_back(char(1));
  put_u64(b, v.size());
  b.append(reinterpret_cast<const char*>(v.data()), v.size() * 8);
}
ib::core::birth_cfg_t tiny_cfg() {
  return {4, 8, 4, 16, 8, 12, 9, 256, 8192, 16384, 42950, 1, 1000};
}
std::string tiny_tables(int T, int DH) {
  const i64 ts = 100, tse = 50, RS = i64{1} << 14;
  std::vector<i64> sil(2 * ts + 1), dsl(2 * ts + 1), ex(tse + 1);
  for (i64 i = 0; i < i64(sil.size()); i++) sil[i] = (i - ts) / 2;
  for (i64 i = 0; i < i64(dsl.size()); i++) dsl[i] = 256;
  for (i64 i = 0; i < i64(ex.size()); i++) ex[i] = 1 + i * 7;
  std::vector<i64> rc(std::size_t(T) * (DH / 2), RS);
  std::vector<i64> rs(std::size_t(T) * (DH / 2), 0);
  std::string b("AXP3", 4);
  put_u32(b, 5);
  put_tensor(b, "silu.tab", sil);
  put_tensor(b, "dsilu.tab", dsl);
  put_tensor(b, "exp.tab", ex);
  put_tensor(b, "rope.cos", rc);
  put_tensor(b, "rope.sin", rs);
  return b;
}
std::string tiny_init(const ib::core::birth_cfg_t& c) {
  std::mt19937_64 rng(613);
  const auto d = [](std::mt19937_64& r) { return i64(r() % 513) - 256; };
  const std::size_t sizes[11] = {
      std::size_t(c.DH) * c.D, std::size_t(c.DH) * c.D,
      std::size_t(c.DH) * c.D, std::size_t(c.D) * c.DH,
      std::size_t(c.F) * c.D,  std::size_t(c.F) * c.D,
      std::size_t(c.D) * c.F,  std::size_t(c.V) * c.D,
      std::size_t(c.D),        std::size_t(c.D),
      std::size_t(c.D)};
  std::string b;
  for (std::size_t n : sizes)
    for (std::size_t i = 0; i < n; i++)
      put_u64(b, std::uint64_t(d(rng)));
  for (int i = 0; i < c.T * c.D; i++)
    put_u64(b, std::uint64_t(d(rng)));
  for (int t = 0; t < c.T; t++) put_u64(b, std::uint64_t(t % c.V));
  return b;
}
}  // namespace

TEST(Anchor2Engine, StepOneEqualsExactAnchorExactly) {
  const auto c = tiny_cfg();
  const std::string tb = tiny_tables(c.T, c.DH), in = tiny_init(c);
  rx::init(24, 70);
  a2::anchor2_birth b2(tb, in, c);
  b2.run(1);
  const i64 loss2 = b2.last_loss();
  const std::string dig2 = b2.mark();
  ib::anchor::anchor_birth an(tb, in, c);
  an.run(1);
  EXPECT_EQ(loss2, an.last_loss());   // EQ, not NEAR: both exact
  EXPECT_EQ(dig2, an.mark());
}

TEST(Anchor2Engine, TwelveStepsSchedBoundaryDeterministic) {
  const auto c = tiny_cfg();
  const std::string tb = tiny_tables(c.T, c.DH), in = tiny_init(c);
  const auto run12 = [&] {
    rx::init(24, 70);
    a2::anchor2_birth b(tb, in, c);
    std::string d;
    for (int s = 1; s <= 12; ++s) {
      ax::dyi::prec = 50 + 20 * s;  // pin 3: growing shadow
      if (s == 7) b.set_lr(2 * c.lrn, c.lrd);  // SCHED boundary
      b.run(1);
      d = b.mark();
    }
    return d;
  };
  const std::string d1 = run12();
  const a2::fb_counters fb1 = rx::fb;
  const std::string d2 = run12();
  EXPECT_EQ(d1, d2);
  EXPECT_EQ(fb1.cmp, rx::fb.cmp);  // counters reproduce too
  EXPECT_EQ(fb1.floor_exact, rx::fb.floor_exact);
}

TEST(Anchor2Engine, ForcedFallbackDigestInvariant) {
  const auto c = tiny_cfg();
  const std::string tb = tiny_tables(c.T, c.DH), in = tiny_init(c);
  const auto run1 = [&](int prec) {
    rx::init(96, prec);
    a2::anchor2_birth b(tb, in, c);
    ax::dyi::prec = prec;
    b.run(1);
    return b.mark();
  };
  const std::string rich = run1(70);
  const std::string starved = run1(34);  // forces fallbacks
  EXPECT_GT(rx::fb.cmp + rx::fb.floor_exact + rx::fb.floor_near, 0);
  EXPECT_EQ(rich, starved);  // fallbacks are inside the certified surface
}

TEST(Anchor2Engine, ModulusExhaustionIsLoud) {
  const auto c = tiny_cfg();
  rx::init(3, 40);  // starved prime budget
  a2::anchor2_birth b(tiny_tables(c.T, c.DH), tiny_init(c), c);
  try {
    b.run(3);
    // if the tiny fixture never needed a reconstruction this deep,
    // force one: floor of a straddling non-integer value
    rx t = a2::Exact2::div(rx(1), rx(3));
    for (int i = 0; i < 40; ++i) t = t * t + t;  // blow up bits
    (void)a2::Exact2::to_grain(t, 0);
    FAIL() << "expected anchor2 modulus exhaustion";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("anchor2"), std::string::npos);
  }
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
