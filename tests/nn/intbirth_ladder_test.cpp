/** @file intbirth_ladder_test.cpp ENGINE-EXACT-1 Q32/Q64 rungs.
    The Q9 rung is certified by the pinned-digest drivers; these
    tests cover rung acceptance, determinism, and de-shift sanity
    against Q9 on a tiny in-test fixture. Real per-rung reference
    digests get pinned by the anchor/driver tooling. */
#include <gtest/gtest.h>

#include <ax/nn/intbirth.hpp>

#include <cstring>
#include <random>
#include <string>

namespace ib = ax::nn::ib;
using ib::i64;

namespace {

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
  b.push_back(char(1));  // 1-d
  put_u64(b, v.size());
  b.append(reinterpret_cast<const char*>(v.data()), v.size() * 8);
}

ib::contract tiny_contract(int precision) {
  ib::contract c;
  c.T = 4; c.D = 8; c.DH = 4; c.F = 16; c.V = 8;
  c.precision = precision;
  return c;
}

/** Shipped-format tables: identity rope (cos=RS, sin=0), linear
    silu ramp, positive exp ramp. Values at shipped scale — the
    same bytes feed every rung (tables are opaque, never move). */
std::string tiny_tables(const ib::contract& c) {
  const i64 ts = 100, tse = 50, RS = i64{1} << 14;
  std::vector<i64> sil(2 * ts + 1), dsl(2 * ts + 1), ex(tse + 1);
  for (i64 i = 0; i < i64(sil.size()); i++) sil[i] = (i - ts) / 2;
  for (i64 i = 0; i < i64(dsl.size()); i++) dsl[i] = 256;
  for (i64 i = 0; i < i64(ex.size()); i++) ex[i] = 1 + i * 7;
  std::vector<i64> rc(std::size_t(c.T) * (c.DH / 2), RS);
  std::vector<i64> rs(std::size_t(c.T) * (c.DH / 2), 0);
  std::string b("AXP3", 4);
  put_u32(b, 5);
  put_tensor(b, "silu.tab", sil);
  put_tensor(b, "dsilu.tab", dsl);
  put_tensor(b, "exp.tab", ex);
  put_tensor(b, "rope.cos", rc);
  put_tensor(b, "rope.sin", rs);
  return b;
}

/** init: 11 KEYS tensors at shipped Q9 scale, then x [T,D], then
    tgt [T] (int64 LE) — the full_birth ctor convention. */
std::string tiny_init(const ib::contract& c) {
  std::mt19937_64 rng(613);
  std::uniform_int_distribution<i64> d(-256, 256);
  const std::size_t sizes[11] = {
      std::size_t(c.DH) * c.D, std::size_t(c.DH) * c.D,
      std::size_t(c.DH) * c.D, std::size_t(c.D) * c.DH,
      std::size_t(c.F) * c.D,  std::size_t(c.F) * c.D,
      std::size_t(c.D) * c.F,  std::size_t(c.V) * c.D,
      std::size_t(c.D),        std::size_t(c.D),
      std::size_t(c.D)};
  std::string b;
  for (std::size_t n : sizes)
    for (std::size_t i = 0; i < n; i++) put_u64(b, std::uint64_t(d(rng)));
  for (int i = 0; i < c.T * c.D; i++) put_u64(b, std::uint64_t(d(rng)));
  for (int t = 0; t < c.T; t++) put_u64(b, std::uint64_t(t % c.V));
  return b;
}

}  // namespace

TEST(IntbirthLadder, Q32AcceptedRunsAndIsDeterministic) {
  const auto c = tiny_contract(32);
  const std::string tb = tiny_tables(c), in = tiny_init(c);
  ib::full_birth fb1(tb, in, c), fb2(tb, in, c);
  fb1.run(3);
  fb2.run(3);
  EXPECT_EQ(fb1.mark(), fb2.mark());
  EXPECT_NE(fb1.last_loss(), 0);
}

TEST(IntbirthLadder, Q32LossDeShiftTracksQ9) {
  const auto c9 = tiny_contract(9), c32 = tiny_contract(32);
  const std::string tb = tiny_tables(c9), in = tiny_init(c9);
  ib::full_birth f9(tb, in, c9), f32(tb, in, c32);
  f9.run(1);
  f32.run(1);
  // same trajectory at two grains: de-shifted step-1 loss agrees to
  // within rounding-grain slack (loss is T * (Q - p[tgt]) ~ T*Q).
  const i64 l9 = f9.last_loss();
  const i64 l32 = f32.last_loss() >> 23;
  EXPECT_NEAR(double(l9), double(l32), double(l9) * 0.02 + 4.0);
}

TEST(IntbirthLadder, Q32ReferenceDigest) {
  // Pinned Q32 rung reference on the tiny fixture (40 steps).
  // Self-referenced until llmopt counter-books it (relay pending);
  // any drift in the Q32 arithmetic path breaks this loudly.
  const auto c = tiny_contract(32);
  ib::full_birth fb(tiny_tables(c), tiny_init(c), c);
  fb.run(40);
  EXPECT_EQ(
      fb.mark(),
      "cd62c4624ed1a6cc3eb069089f6ab02064f060d7468fad84dfd147fb23bad99d");
}

TEST(IntbirthLadder, UnwiredPrecisionStillRefuses) {
  const auto c = tiny_contract(48);
  EXPECT_THROW(ib::full_birth(tiny_tables(c), tiny_init(c), c),
               std::runtime_error);
}
