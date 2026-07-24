/** Tests for the poly-algebra and physics chain makers: every problem
    across a seed sweep must certify, and every row must stay inside
    its vocab (x-charset for poly, t-charset for physics). */
#include <ax/mathgen/physchain.hpp>
#include <ax/mathgen/polychain.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <string>

namespace {

bool in_vocab(const std::string& s, char sym) {
  for (char c : s)
    if (!(std::isdigit(static_cast<unsigned char>(c)) || c == sym ||
          c == '+' || c == '-' || c == '*' || c == '/' || c == '(' ||
          c == ')' || c == ',' || c == ' ' || c == 'O'))
      return false;
  return true;
}

void check_problem(const ax::mathgen::pchain_problem& p, char sym) {
  EXPECT_TRUE(p.certified) << p.family << " L" << p.level << " seed "
                           << p.seed << ": " << p.error;
  EXPECT_FALSE(p.rows.empty());
  for (const auto& r : p.rows) {
    EXPECT_TRUE(in_vocab(r.cur, sym)) << r.cur;
    EXPECT_TRUE(in_vocab(r.nxt, sym)) << r.nxt;
  }
}

TEST(DomainChain, GcdChainsCertifyAcrossSeeds) {
  for (int level = 1; level <= 3; ++level)
    for (long long seed = 0; seed < 10; ++seed)
      check_problem(ax::mathgen::make_gcd_chain(level, seed), 'x');
}

TEST(DomainChain, GcdChainEndsAtEngineGcd) {
  const auto p = ax::mathgen::make_gcd_chain(1, 0);
  ASSERT_TRUE(p.certified) << p.error;
  EXPECT_EQ(p.rows.front().kind, "pmul");  // division steps are trees now
}

TEST(DomainChain, PartialFractionsCertifyAcrossSeeds) {
  for (int level = 1; level <= 3; ++level)
    for (long long seed = 0; seed < 10; ++seed)
      check_problem(ax::mathgen::make_pf_chain(level, seed), 'x');
}

TEST(DomainChain, PartialFractionRowsAreOnePrimitiveEach) {
  const auto p = ax::mathgen::make_pf_chain(1, 0);
  ASSERT_TRUE(p.certified) << p.error;
  int res = 0, pmul = 0;
  for (const auto& r : p.rows) {
    if (r.kind == "res") ++res;
    if (r.kind == "pmul") ++pmul;
    if (r.kind == "mul" || r.kind == "add" || r.kind == "sub")
      // one binary operator per constant-primitive row
      EXPECT_LE(std::count_if(r.cur.begin(), r.cur.end(),
                              [](char c) {
                                return c == '*' || c == '+';
                              }),
                2)
          << r.cur;
  }
  EXPECT_EQ(res, 2);   // one residue per root
  EXPECT_EQ(pmul, 2);  // one recombination product per root (2 roots)
  EXPECT_EQ(p.rows.back().kind, "assemble");
}

TEST(DomainChain, BridgeChainsCertifyAcrossSeeds) {
  for (int level = 1; level <= 3; ++level)
    for (long long seed = 0; seed < 10; ++seed) {
      const auto p = ax::mathgen::make_bridge_chain(level, seed);
      EXPECT_TRUE(p.certified) << p.family << " L" << level << " seed "
                               << seed << ": " << p.error;
      int ibridge = 0, icancel = 0, iclose = 0, close = 0;
      for (const auto& r : p.rows) {
        if (r.kind == "ibridge") ++ibridge;
        if (r.kind == "icancel") ++icancel;
        if (r.kind == "iclose") ++iclose;
        if (r.kind == "close") ++close;
      }
      const int nroots = level >= 2 ? 3 : 2;
      EXPECT_EQ(ibridge, nroots - 1);  // one residue split per row
      EXPECT_EQ(icancel, nroots - 1);  // one factor cancellation per row
      EXPECT_EQ(close, nroots);        // one piece folded per row
      EXPECT_EQ(iclose, nroots);
      // one-integral discipline: an ibridge row introduces exactly one
      // NEW Integral literal in nxt vs cur; icancel rewrites one to one
      for (const auto& r : p.rows) {
        if (r.kind != "ibridge" && r.kind != "icancel") continue;
        const auto count = [](const std::string& s) {
          int n = 0;
          for (std::size_t pos = 0;
               (pos = s.find("Integral(", pos)) != std::string::npos;
               pos += 9)
            ++n;
          return n;
        };
        if (r.kind == "ibridge")
          EXPECT_EQ(count(r.nxt), count(r.cur) + 1) << r.cur;
        else
          EXPECT_EQ(count(r.nxt), count(r.cur)) << r.cur;
      }
    }
}

TEST(DomainChain, KinematicsChainsCertifyAcrossSeeds) {
  for (int level = 1; level <= 3; ++level)
    for (long long seed = 0; seed < 10; ++seed)
      check_problem(ax::mathgen::make_kin_chain(level, seed), 't');
}

TEST(DomainChain, ShmChainsCertifyAcrossSeeds) {
  for (int level = 1; level <= 3; ++level)
    for (long long seed = 0; seed < 10; ++seed)
      check_problem(ax::mathgen::make_shm_chain(level, seed, 8), 't');
}

TEST(DomainChain, EnergyChainsCertifyAcrossSeeds) {
  for (int level = 1; level <= 3; ++level)
    for (long long seed = 0; seed < 10; ++seed)
      check_problem(ax::mathgen::make_energy_chain(level, seed, 8), 't');
}

TEST(DomainChain, EnergyOrdersEndOnZeroRows) {
  const auto p = ax::mathgen::make_energy_chain(2, 0, 8);
  ASSERT_TRUE(p.certified) << p.error;
  int zeros = 0;
  for (const auto& r : p.rows)
    if (r.kind == "zero") {
      ++zeros;
      EXPECT_EQ(r.nxt, "0") << r.cur;
    }
  EXPECT_GE(zeros, 2);  // several orders must vanish visibly
}

TEST(DomainChain, ShmRowsCarryTNotX) {
  const auto p = ax::mathgen::make_shm_chain(2, 0, 8);
  ASSERT_TRUE(p.certified) << p.error;
  bool saw_append = false;
  for (const auto& r : p.rows)
    if (r.kind == "append") {
      saw_append = true;
      EXPECT_EQ(r.nxt.find('x'), std::string::npos) << r.nxt;
      EXPECT_NE(r.nxt.find('t'), std::string::npos) << r.nxt;
    }
  EXPECT_TRUE(saw_append);
}

}  // namespace
