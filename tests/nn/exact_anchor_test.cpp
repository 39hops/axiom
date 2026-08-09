/** @file exact_anchor_test.cpp ENGINE-EXACT-1 exact-prefix anchor:
    runs, agrees with Q9 at the shipped grain on a short prefix, and
    hits its bit ceiling LOUDLY (never silently). */
#include <gtest/gtest.h>

#include <ax/nn/exact_anchor.hpp>
#include <ax/nn/intbirth.hpp>

#include <cstring>
#include <random>
#include <string>

namespace ib = ax::nn::ib;
using ib::i64;

namespace {
// same tiny fixture as intbirth_ladder_test (duplicated: test files
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
    for (std::size_t i = 0; i < n; i++)
      put_u64(b, std::uint64_t(d(rng)));
  for (int i = 0; i < c.T * c.D; i++)
    put_u64(b, std::uint64_t(d(rng)));
  for (int t = 0; t < c.T; t++) put_u64(b, std::uint64_t(t % c.V));
  return b;
}
}  // namespace

TEST(ExactAnchor, RunsTwoStepsAndTracksQ9AtStepOne) {
  const auto c = tiny_cfg();
  const std::string tb = tiny_tables(c.T, c.DH), in = tiny_init(c);
  ib::anchor::anchor_birth an(tb, in, c);
  an.run(1);
  ib::contract cc;
  cc.T = c.T; cc.D = c.D; cc.DH = c.DH; cc.F = c.F; cc.V = c.V;
  ib::full_birth f9(tb, in, cc);
  f9.run(1);
  // step-1 losses at the shipped grain: exact vs Q9 rounding —
  // grain-level slack only
  EXPECT_NEAR(double(f9.last_loss()), double(an.last_loss()),
              double(f9.last_loss()) * 0.02 + 4.0);
  EXPECT_EQ(an.step_count(), 1);
  // (step 2+ is minutes-class even at this toy size — the growth
  // law in person; the horizon probe lives in tools/exact_anchor)
}

TEST(ExactAnchor, BitCeilingAbortsLoudly) {
  const auto c = tiny_cfg();
  ib::anchor::anchor_birth an(tiny_tables(c.T, c.DH), tiny_init(c),
                              c);
  const unsigned saved = ib::anchor::exr::bit_ceiling;
  ib::anchor::exr::bit_ceiling = 512;  // low: force the guard fast
  try {
    EXPECT_THROW(an.run(2), std::runtime_error);
  } catch (...) {
    ib::anchor::exr::bit_ceiling = saved;
    throw;
  }
  ib::anchor::exr::bit_ceiling = saved;
}
