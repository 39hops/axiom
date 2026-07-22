/** Tests for the poly-algebra and physics chain makers: every problem
    across a seed sweep must certify, and every row must stay inside
    its vocab (x-charset for poly, t-charset for physics). */
#include <ax/mathgen/physchain.hpp>
#include <ax/mathgen/polychain.hpp>

#include <gtest/gtest.h>

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
  EXPECT_EQ(p.rows.front().kind, "divstep");
}

TEST(DomainChain, PartialFractionsCertifyAcrossSeeds) {
  for (int level = 1; level <= 3; ++level)
    for (long long seed = 0; seed < 10; ++seed)
      check_problem(ax::mathgen::make_pf_chain(level, seed), 'x');
}

TEST(DomainChain, PartialFractionRowOrderIsNumDenResPerRoot) {
  const auto p = ax::mathgen::make_pf_chain(1, 0);
  ASSERT_TRUE(p.certified) << p.error;
  ASSERT_EQ(p.rows.size(), 7u);  // 2 roots x (num, den, res) + assemble
  EXPECT_EQ(p.rows[0].kind, "num");
  EXPECT_EQ(p.rows[1].kind, "den");
  EXPECT_EQ(p.rows[2].kind, "res");
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
