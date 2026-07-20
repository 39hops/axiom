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
using ax::sym::check_ic;
using ax::sym::check_odesol;
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

// ------------------------------------------------- sqrt-as-algebraic-atom

TEST(CanonicalSqrt, SqrtTimesSqrtIsArgument) {
  const expr e = canonical(parse("sqrt(2 - x)*sqrt(2 - x)"), x);
  EXPECT_TRUE(e.same(parse("2 - x"))) << to_string(e);
}

TEST(CanonicalSqrt, ReciprocalSqrtRecombines) {
  // sqrt(x)/x == 1/sqrt(x) on the domain of sqrt.
  const expr e = canonical(parse("1/sqrt(x) - sqrt(x)/x"), x);
  EXPECT_TRUE(e.same(expr::num(0))) << to_string(e);
}

TEST(CanonicalSqrt, TaxRowShapeCombines) {
  // Real parity-tax shape: split form vs combined over 2*sqrt(2-x).
  // (checked numerically: x=1 gives 22.5 both sides)
  const expr a =
      parse("45*sqrt(2 - x) + 90/sqrt(2 - x) - (45*x + 180)/(2*sqrt(2 - x))");
  const expr b = parse("(180 - 135*x)/(2*sqrt(2 - x))");
  EXPECT_EQ(equivalent(a, b, x), verdict::equivalent);
}

TEST(CanonicalSqrt, SqrtX2DerivativeMatchesSympyForm) {
  // axiom's d/dx 8*sqrt(x**2) is 8*x*(x**2)**(-1/2); sympy prints
  // 8*sqrt(x**2)/x. Equal via (x**2)**(1/2)*(x**2)**(1/2) -> x**2.
  EXPECT_EQ(equivalent(diff(parse("8*sqrt(x**2)"), x),
                       parse("8*sqrt(x**2)/x"), x),
            verdict::equivalent);
}

TEST(CanonicalSqrt, IntegerPowerOfSqrtFolds) {
  const expr e = canonical(parse("sqrt(x + 1)**4"), x);
  EXPECT_TRUE(e.same(canonical(parse("(x + 1)**2"), x))) << to_string(e);
}

TEST(CanonicalSqrt, StillHonestOnDifferentRadicands) {
  // sqrt(4*x+2) vs sqrt(2)*sqrt(2*x+1): true, but needs radicand
  // factorization we don't do -> undecided, never guessed.
  EXPECT_EQ(equivalent(parse("sqrt(4*x + 2)"), parse("sqrt(2)*sqrt(2*x + 1)"),
                       x),
            verdict::undecided);
}

}  // namespace

// ------------------------------------------------------ L9: check_odesol

TEST(OdeSol, FirstOrderLinear) {
  // y' + 3y = e^{2x}; y = e^{2x}/5 + C1*e^{-3x} solves it for ALL C1
  // (C1 stays symbolic: parameter_env binds it at sample points).
  const expr eq = parse("Eq(Derivative(y(x), x) + 3*y(x), exp(2*x))");
  const expr sol = parse("exp(2*x)/5 + C1*exp(-3*x)");
  EXPECT_EQ(check_odesol(eq, sol, x), verdict::equivalent);
  // wrong coefficient: must be caught, not UNDECIDED
  const expr bad = parse("exp(2*x)/4 + C1*exp(-3*x)");
  EXPECT_EQ(check_odesol(eq, bad, x), verdict::not_equivalent);
}

TEST(OdeSol, SecondOrderConstantCoeff) {
  // y'' - 3y' + 2y = 0; y = C1*e^x + C2*e^{2x}
  const expr eq =
      parse("Eq(Derivative(y(x), (x, 2)) - 3*Derivative(y(x), x) + 2*y(x), 0)");
  EXPECT_EQ(check_odesol(eq, parse("C1*exp(x) + C2*exp(2*x)"), x),
            verdict::equivalent);
  EXPECT_EQ(check_odesol(eq, parse("C1*exp(x) + C2*exp(3*x)"), x),
            verdict::not_equivalent);
}

TEST(OdeSol, SeparableGrowth) {
  // y' = 2*x*y; y = C1*exp(x**2)
  const expr eq = parse("Eq(Derivative(y(x), x), 2*x*y(x))");
  EXPECT_EQ(check_odesol(eq, parse("C1*exp(x**2)"), x),
            verdict::equivalent);
}

TEST(OdeSol, PinnedSentinels) {
  // llmopt's fixture ask: wrong constant-binding is the phantom-monomial
  // class (wrong-without-erroring inside an exact procedure). Sentinels:
  // (1) a candidate whose correctness DEPENDS on a specific C value must
  //     NOT pass with C symbolic
  const expr eq = parse("Eq(Derivative(y(x), x), y(x))");
  //     y = C1*e^x passes; y = e^x + C1 solves only for C1 = 0
  EXPECT_EQ(check_odesol(eq, parse("C1*exp(x)"), x), verdict::equivalent);
  EXPECT_EQ(check_odesol(eq, parse("exp(x) + C1"), x),
            verdict::not_equivalent);
  // (2) the trig-identity class stays honestly UNDECIDED, never valid.
  //     (NOT y' = 0 with candidate sin^2+cos^2 — diff cancels that
  //     exactly, so it is a GENUINE solution; the identity must sit in
  //     the residual itself.)
  const expr eq2 = parse("Eq(y(x), 1)");
  EXPECT_EQ(check_odesol(eq2, parse("sin(x)**2 + cos(x)**2"), x),
            verdict::undecided);
  // (3) a candidate still mentioning y is unsubstitutable: UNDECIDED
  EXPECT_EQ(check_odesol(eq2, parse("y(x)"), x), verdict::undecided);
}

TEST(OdeSol, InitialConditionCheck) {
  // pinned solution of y' + 3y = e^{2x} with y(0) = 1: C1 = 4/5
  const expr sol = parse("exp(2*x)/5 + 4*exp(-3*x)/5");
  EXPECT_TRUE(check_ic(sol, x, 0.0, 1.0, 0));
  EXPECT_FALSE(check_ic(sol, x, 0.0, 2.0, 0));
  // derivative IC (order 1): y'(0) = 2/5 - 12/5 = -2
  EXPECT_TRUE(check_ic(sol, x, 0.0, -2.0, 1));
}
