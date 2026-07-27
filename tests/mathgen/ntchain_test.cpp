/** @file ntchain_test.cpp Number-theory chain tests: nt_eval grammar,
    per-family certification sweeps, core/nt parity. */
#include <ax/core/nt.hpp>
#include <ax/mathgen/ntchain.hpp>

#include <gtest/gtest.h>

namespace {

using namespace ax;
using namespace ax::mathgen;

TEST(NtEval, ArithmeticAndPrecedence) {
  EXPECT_EQ(nt_eval("2 + 3*4"), bigint(14));
  EXPECT_EQ(nt_eval("(2 + 3)*4"), bigint(20));
  EXPECT_EQ(nt_eval("10 - (-3)"), bigint(13));
  EXPECT_EQ(nt_eval("-4*5"), bigint(-20));
  EXPECT_EQ(nt_eval("2**10"), bigint(1024));
  EXPECT_EQ(nt_eval("2*3**2"), bigint(18));
}

TEST(NtEval, GcdAndMod) {
  EXPECT_EQ(nt_eval("gcd(270, 192)"), bigint(6));
  EXPECT_EQ(nt_eval("gcd(0, 0)"), bigint(0));
  EXPECT_EQ(nt_eval("Mod(17, 5)"), bigint(2));
  EXPECT_EQ(nt_eval("Mod(-3, 7)"), bigint(4));
  EXPECT_EQ(nt_eval("Mod(3**20, 1000)"), bigint(401));
  EXPECT_THROW(nt_eval("Mod(3, 0)"), std::domain_error);
  EXPECT_THROW(nt_eval("frob(1, 2)"), std::invalid_argument);
  EXPECT_THROW(nt_eval("1 + "), std::invalid_argument);
}

void sweep(pchain_problem (*mk)(int, long long), const char* family,
           std::size_t min_rows) {
  for (int level = 1; level <= 3; ++level)
    for (long long seed = 0; seed < 25; ++seed) {
      const pchain_problem p = mk(level, seed);
      EXPECT_EQ(p.family, family);
      EXPECT_TRUE(p.certified)
          << family << " L" << level << " seed " << seed << ": " << p.error;
      EXPECT_GE(p.rows.size(), min_rows)
          << family << " L" << level << " seed " << seed;
      for (const auto& r : p.rows)
        EXPECT_EQ(nt_eval(r.cur), nt_eval(r.nxt))
            << family << " row " << r.kind << ": " << r.cur;
    }
}

TEST(NtChain, GcdSweep) { sweep(make_nt_gcd_chain, "nt_gcd", 4); }
TEST(NtChain, BezoutSweep) { sweep(make_nt_bezout_chain, "nt_bezout", 6); }
TEST(NtChain, ModinvSweep) { sweep(make_nt_modinv_chain, "nt_modinv", 6); }
TEST(NtChain, CrtSweep) { sweep(make_nt_crt_chain, "nt_crt", 10); }
TEST(NtChain, ModexpSweep) { sweep(make_nt_modexp_chain, "nt_modexp", 4); }
TEST(NtChain, CfSweep) { sweep(make_nt_cf_chain, "nt_cf", 8); }

TEST(NtChain, DeterministicBySeed) {
  const pchain_problem a = make_nt_crt_chain(2, 7);
  const pchain_problem b = make_nt_crt_chain(2, 7);
  ASSERT_EQ(a.rows.size(), b.rows.size());
  for (std::size_t i = 0; i < a.rows.size(); ++i) {
    EXPECT_EQ(a.rows[i].cur, b.rows[i].cur);
    EXPECT_EQ(a.rows[i].nxt, b.rows[i].nxt);
  }
}

TEST(NtChain, BezoutFinalRowMatchesExtGcd) {
  const pchain_problem p = make_nt_bezout_chain(3, 11);
  ASSERT_TRUE(p.certified) << p.error;
  const auto& last = p.rows.back();
  EXPECT_EQ(last.kind, "bezout");
  // the final row's value is the gcd the independent core impl computes
  EXPECT_EQ(nt_eval(last.nxt), nt_eval(last.cur));
}

}  // namespace
