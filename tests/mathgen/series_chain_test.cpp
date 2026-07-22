/** Tests for the series derivation-row speller (L10 rung 6): every
    emitted arithmetic row must fold, via sym::parse -> canonical,
    byte-exactly to its nxt, and stay inside the vocab-40 charset. */
#include <ax/mathgen/ode.hpp>
#include <ax/mathgen/series_chain.hpp>
#include <ax/mathgen/series_solve.hpp>
#include <ax/sym/parse.hpp>
#include <ax/sym/print_sstr.hpp>

#include <gtest/gtest.h>

#include <cctype>
#include <string>

namespace {

using ax::bigint;
using ax::rational;

rational q(long long n, long long d = 1) {
  return rational(bigint(n), bigint(d));
}

std::string fold(const std::string& s) {
  return ax::sym::to_sstr(ax::sym::parse(s));
}

bool vocab40(const std::string& s) {
  for (char c : s)
    if (!(std::isdigit(static_cast<unsigned char>(c)) || c == 'x' ||
          c == '+' || c == '-' || c == '*' || c == '/' || c == '(' ||
          c == ')' || c == ',' || c == ' '))
      return false;
  return true;
}

TEST(SeriesChain, LitParenthesizesNegativesAndFractions) {
  EXPECT_EQ(ax::mathgen::chain_lit(q(6)), "6");
  EXPECT_EQ(ax::mathgen::chain_lit(q(-2)), "(-2)");
  EXPECT_EQ(ax::mathgen::chain_lit(q(3, 4)), "(3/4)");
  EXPECT_EQ(ax::mathgen::chain_lit(q(-3, 4)), "(-3/4)");
}

TEST(SeriesChain, SingleTermStepSpellsTheRelayExample) {
  // y' = -2y, a_1 = -6: a_2 = (0 - (-2)*(-6))/2 = -6
  ax::mathgen::series_step st;
  st.n = 2;
  st.a_n = q(-6);
  st.q_n = q(0);
  st.divisor = q(2);
  st.terms = {{q(-2), q(1), q(-6)}};
  const auto d = ax::mathgen::derivation_rows(st);
  EXPECT_FALSE(d.sum_cur.has_value());
  EXPECT_EQ(d.solve_cur, "(0 - (-2)*(-6))/2");
  EXPECT_EQ(d.solve_nxt, "-6");
  EXPECT_EQ(fold(d.solve_cur), d.solve_nxt);
}

TEST(SeriesChain, MultiTermStepEmitsSumRowThenSolveRow) {
  ax::mathgen::series_step st;
  st.n = 3;
  st.a_n = q(-1, 2);
  st.q_n = q(1);
  st.divisor = q(6);
  st.terms = {{q(2), q(1), q(1)}, {q(1, 2), q(2), q(2)}};
  const auto d = ax::mathgen::derivation_rows(st);
  ASSERT_TRUE(d.sum_cur.has_value());
  EXPECT_EQ(*d.sum_cur, "2 + (1/2)*2*2");
  EXPECT_EQ(d.sum_nxt, "4");
  EXPECT_EQ(fold(*d.sum_cur), d.sum_nxt);
  EXPECT_EQ(d.solve_cur, "(1 - 4)/6");
  EXPECT_EQ(fold(d.solve_cur), d.solve_nxt);
  EXPECT_EQ(d.solve_nxt, "-1/2");
}

TEST(SeriesChain, EveryRowOfEveryFamilyCertifiesAndStaysInVocab) {
  using namespace ax::mathgen;
  const int order = 8;
  for (int level = 1; level <= 3; ++level)
    for (long long seed = 0; seed < 5; ++seed)
      for (int fam = 0; fam < 3; ++fam) {
        const ode_problem p =
            fam == 0   ? make_linear_first_order(level, seed)
            : fam == 1 ? make_second_order_cc(level, seed)
                       : make_separable_growth(level, seed);
        const auto sol = series_solve(p, order);
        for (const auto& st : sol.steps) {
          const auto d = derivation_rows(st);
          if (d.sum_cur) {
            EXPECT_TRUE(vocab40(*d.sum_cur)) << *d.sum_cur;
            EXPECT_EQ(fold(*d.sum_cur), d.sum_nxt) << *d.sum_cur;
          }
          EXPECT_TRUE(vocab40(d.solve_cur)) << d.solve_cur;
          EXPECT_EQ(fold(d.solve_cur), d.solve_nxt) << d.solve_cur;
          EXPECT_EQ(d.solve_nxt, ax::sym::to_sstr(
                                     ax::sym::parse(st.a_n.to_string())));
        }
      }
}

}  // namespace
