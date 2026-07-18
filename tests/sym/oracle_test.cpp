#include <ax/sym/oracle.hpp>

#include <ax/sym/calc.hpp>
#include <ax/sym/expand.hpp>
#include <ax/sym/expr.hpp>
#include <ax/sym/parse.hpp>
#include <ax/sym/print.hpp>

#include <gtest/gtest.h>

#include <string>

namespace {

using ax::sym::canonical;
using ax::sym::diff;
using ax::sym::equivalent;
using ax::sym::equivalent_mod_const;
using ax::sym::expr;
using ax::sym::parse;
using ax::sym::to_string;
using ax::sym::verdict;

const expr x = expr::symbol("x");

// ---------------------------------------------------------------- canonical

TEST(Canonical, PolyQuotientCancels) {
  const expr e = canonical(parse("(x**2 - 1)/(x - 1)"), x);
  EXPECT_TRUE(e.same(parse("x + 1"))) << to_string(e);
}

TEST(Canonical, FnFactorCancels) {
  const expr e = canonical(parse("(2*x - 3)*sin(x)/(2*x - 3)"), x);
  EXPECT_TRUE(e.same(parse("sin(x)"))) << to_string(e);
}

TEST(Canonical, ExpandCollectToZero) {
  const expr e = canonical(parse("(x + 1)**2 - x**2 - 2*x - 1"), x);
  EXPECT_TRUE(e.same(expr::num(0))) << to_string(e);
}

TEST(Canonical, CommonDenominatorZero) {
  // 1/(x-1) - 1/(x-1) style, but non-trivially split:
  const expr e =
      canonical(parse("x/(x**2 - 1) - 1/(2*(x - 1)) - 1/(2*(x + 1))"), x);
  EXPECT_TRUE(e.same(expr::num(0))) << to_string(e);
}

TEST(Canonical, QuotientDiffRecombines) {
  const expr q = parse("(x**2 + 1)/(x - 1)");
  const expr d = diff(q, x);
  // Hand-derived: ((2x)(x-1) - (x^2+1)) / (x-1)^2 = (x^2 - 2x - 1)/(x-1)^2
  const expr want = parse("(x**2 - 2*x - 1)/(x - 1)**2");
  EXPECT_TRUE(canonical(d, x).same(canonical(want, x)))
      << to_string(canonical(d, x));
}

TEST(Canonical, Idempotent) {
  const std::string cases[] = {
      "(x**2 - 1)/(x - 1)",
      "sin(x)**2 + cos(x)**2",
      "5*((3 - 4*x)*sin(log(x*(2*x - 3))) + "
      "(2*x - 3)*cos(log(x*(2*x - 3))))/(2*x - 3)",
      "x**(1/2) + exp(-x)*x",
  };
  for (const auto& s : cases) {
    const expr c = canonical(parse(s), x);
    EXPECT_TRUE(canonical(c, x).same(c)) << s << " -> " << to_string(c);
  }
}

TEST(Canonical, LeavesTrigIdentityAlone) {
  // Documented incompleteness: no trig rewriting.
  const expr e = canonical(parse("sin(x)**2 + cos(x)**2"), x);
  EXPECT_FALSE(e.same(expr::num(1)));
}

// --------------------------------------------------------------- equivalent

TEST(Equivalent, StructuralProof) {
  EXPECT_EQ(equivalent(parse("2*x + 2"), parse("2*(x + 1)"), x),
            verdict::equivalent);
  EXPECT_EQ(equivalent(parse("(x**2 - 1)/(x - 1)"), parse("x + 1"), x),
            verdict::equivalent);
}

TEST(Equivalent, NumericWitness) {
  EXPECT_EQ(equivalent(parse("x**2"), parse("x**3"), x),
            verdict::not_equivalent);
  EXPECT_EQ(equivalent(parse("sin(x)"), parse("cos(x)"), x),
            verdict::not_equivalent);
}

TEST(Equivalent, TrigIdentityIsUndecidedNeverGuessed) {
  // Numerics agree everywhere but there is no structural proof: the sound
  // answer is undecided. This test is the soundness sentinel.
  EXPECT_EQ(equivalent(parse("sin(x)**2 + cos(x)**2"), parse("1"), x),
            verdict::undecided);
}

TEST(Equivalent, DisjointDomainsUndecided) {
  // log(x) and log(-x) share no valid sample points: no witness is
  // obtainable and there is no structural proof -> undecided, not a crash.
  EXPECT_EQ(equivalent(parse("log(x)"), parse("log(-x)"), x),
            verdict::undecided);
}

TEST(Equivalent, OtherSymbolsAreParameters) {
  const expr a = expr::symbol("a");
  EXPECT_EQ(equivalent(parse("a*x + a"), parse("a*(x + 1)"), x),
            verdict::equivalent);
}

// ----------------------------------------------------- equivalent_mod_const

TEST(EquivModConst, CorrectAntiderivativeAnyConstant) {
  EXPECT_EQ(equivalent_mod_const(parse("x**2/2 + 7"), parse("x"), x),
            verdict::equivalent);
  EXPECT_EQ(equivalent_mod_const(parse("-cos(2*x + 1)/2 - 3/4"),
                                 parse("sin(2*x + 1)"), x),
            verdict::equivalent);
}

TEST(EquivModConst, WrongCandidateRejected) {
  EXPECT_EQ(equivalent_mod_const(parse("x**2"), parse("x"), x),
            verdict::not_equivalent);
}

TEST(EquivModConst, LlmoptShapedPair) {
  // Candidate F(x) = 5*x*sin(log(x*(2*x-3)))/(2*x-3)... too clever; use the
  // honest construction: pick F, let the integrand be ax's own diff of an
  // *independently parsed* equal form, exercising cancel across the quotient.
  const expr big_f = parse("5*x*sin(log(x*(2*x - 3)))/(2*x - 3)");
  const expr integrand = diff(big_f + parse("13/9"), x);
  EXPECT_EQ(equivalent_mod_const(big_f, integrand, x), verdict::equivalent);
}

}  // namespace
