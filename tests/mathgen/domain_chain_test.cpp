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
