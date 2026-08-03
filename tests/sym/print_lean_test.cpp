#include <ax/sym/print_lean.hpp>

#include <ax/sym/expr.hpp>
#include <ax/sym/harness.hpp>

#include <algorithm>
#include <sstream>
#include <gtest/gtest.h>

namespace {

using ax::sym::expr;
using ax::sym::lean_cert;
using ax::sym::to_lean;

const expr x = expr::symbol("x");

// ------------------------------------------------------------- eligibility

TEST(LeanCert, PolynomialIsRingClosable) {
  const lean_cert c = to_lean("(x + 1)**2", "x**2 + 2*x + 1", x);
  ASSERT_TRUE(c.eligible);
  EXPECT_EQ(c.tactic, "ring");
  EXPECT_EQ(c.statement,
            "example (x : ℝ) : (x + 1)^2 = 2*x + x^2 + 1 := by ring");
  EXPECT_TRUE(c.atoms.empty());
}

TEST(LeanCert, FnSubtermsGeneralizeToAtoms) {
  // sin(x)*(x + sin(x)) == x*sin(x) + sin(x)**2, rational in the atom.
  const lean_cert c =
      to_lean("sin(x)*(x + sin(x))", "x*sin(x) + sin(x)**2", x);
  ASSERT_TRUE(c.eligible);
  ASSERT_EQ(c.atoms.size(), 1u);
  EXPECT_EQ(c.atoms[0].first, "a1");
  EXPECT_EQ(c.atoms[0].second, "sin(x)");
  EXPECT_EQ(c.statement,
            "example (a1 x : ℝ) : (x + a1)*a1 = x*a1 + a1^2 := by ring");
}

TEST(LeanCert, DistinctAtomsGetDistinctNames) {
  const lean_cert c =
      to_lean("sin(x) + exp(x)", "exp(x) + sin(x)", x);
  ASSERT_TRUE(c.eligible);
  EXPECT_EQ(c.atoms.size(), 2u);
}

TEST(LeanCert, SqrtAtRingLevelIsFencedOut) {
  const lean_cert c = to_lean("sqrt(x)*sqrt(x)", "x", x);
  EXPECT_FALSE(c.eligible);
}

TEST(LeanCert, FractionalPowAtRingLevelIsFencedOut) {
  const lean_cert c = to_lean("x**(1/2)*x**(1/2)", "x", x);
  EXPECT_FALSE(c.eligible);
}

TEST(LeanCert, SymbolicExponentIsFencedOut) {
  const lean_cert c = to_lean("x**k*x", "x**(k + 1)", x);
  EXPECT_FALSE(c.eligible);
}

TEST(LeanCert, SqrtInsideFnArgumentIsFrozenAndEligible) {
  // The sqrt lives inside the opaque atom sin(sqrt(x)); never touches ring.
  const lean_cert c =
      to_lean("2*sin(sqrt(x))", "sin(sqrt(x)) + sin(sqrt(x))", x);
  ASSERT_TRUE(c.eligible);
  ASSERT_EQ(c.atoms.size(), 1u);
  EXPECT_EQ(c.atoms[0].second, "sin(sqrt(x))");
}

// -------------------------------------------------------------- divisions

TEST(LeanCert, DivisionEmitsNonzeroHypothesesAndFieldSimp) {
  const lean_cert c = to_lean("(x**2 - 1)/(x - 1)", "x + 1", x);
  ASSERT_TRUE(c.eligible);
  EXPECT_EQ(c.tactic, "field_simp; ring");
  EXPECT_EQ(c.statement,
            "example (x : ℝ) (h1 : x - 1 ≠ 0) : "
            "(x^2 - 1)/(x - 1) = x + 1 := by field_simp; ring");
}

TEST(LeanCert, NumericDenominatorNeedsNoHypothesis) {
  const lean_cert c = to_lean("x/2 + x/2", "x", x);
  ASSERT_TRUE(c.eligible);
  EXPECT_EQ(c.tactic, "ring");
}

// ------------------------------------------------------------- sidecar row

TEST(LeanCert, SidecarLineIsWellFormedJson) {
  const lean_cert c =
      to_lean("sin(x)*(x + sin(x))", "x*sin(x) + sin(x)**2", x);
  const std::string line = sidecar_line("row7", c);
  EXPECT_NE(line.find("\"id\":\"row7\""), std::string::npos);
  EXPECT_NE(line.find("\"tactic\":\"ring\""), std::string::npos);
  EXPECT_NE(line.find("\"atoms\":{\"a1\":\"sin(x)\"}"), std::string::npos);
  EXPECT_NE(line.find("\"lean\":\"example"), std::string::npos);
}

// --------------------------------------------------------- harness sidecar

TEST(LeanSidecar, HarnessEmitsEligibleRowsAndCountsFenced) {
  std::istringstream in(
      "{\"id\":\"r1\",\"task\":\"equiv\",\"var\":\"x\","
      "\"lhs\":\"(x + 1)**2\",\"rhs\":\"x**2 + 2*x + 1\"}\n"
      "{\"id\":\"r2\",\"task\":\"equiv\",\"var\":\"x\","
      "\"lhs\":\"sqrt(x)*sqrt(x)\",\"rhs\":\"x\"}\n"
      "{\"id\":\"r3\",\"task\":\"equiv\",\"var\":\"x\","
      "\"lhs\":\"x\",\"rhs\":\"x + 1\"}\n");
  std::ostringstream out, lean;
  ax::sym::lean_stats stats;
  ax::sym::run_oracle(in, out, &lean, &stats);
  // r1: EQUIVALENT + eligible -> one sidecar line. r2: EQUIVALENT but
  // fenced (sqrt at ring level) -> counted. r3: NOT_EQUIVALENT -> ignored.
  EXPECT_EQ(stats.emitted, 1);
  EXPECT_EQ(stats.fenced, 1);
  const std::string side = lean.str();
  EXPECT_NE(side.find("\"id\":\"r1\""), std::string::npos);
  EXPECT_EQ(side.find("\"id\":\"r2\""), std::string::npos);
  EXPECT_EQ(std::count(side.begin(), side.end(), '\n'), 1);
}

}  // namespace
