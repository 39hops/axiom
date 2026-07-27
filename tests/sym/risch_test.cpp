/** @file risch_test.cpp Magic boards r2 certificates. Every 'dead'
    case is a textbook non-elementary integral; every 'alive' case is
    either provably elementary (the ODE solves) or outside the
    certified families (undecided must NEVER mask). */
#include <ax/sym/risch.hpp>

#include <ax/sym/parse.hpp>

#include <gtest/gtest.h>

namespace {

using ax::sym::dead_state;
using ax::sym::dead_state_mask;
using ax::sym::parse;
using ax::sym::risch_certify;

TEST(Risch, ClassicNonElementaryAtomsCertify) {
  EXPECT_EQ(risch_certify(parse("Integral(exp(x**2), x)")).reason,
            "risch-exp-poly");
  EXPECT_EQ(risch_certify(parse("Integral(sin(x**2), x)")).reason,
            "risch-trig-poly");  // Fresnel
  EXPECT_EQ(risch_certify(parse("Integral(cos(x**2), x)")).reason,
            "risch-trig-poly");
  EXPECT_EQ(risch_certify(parse("Integral(exp(x)/x, x)")).reason,
            "risch-exp-laurent");  // Ei
  EXPECT_EQ(risch_certify(parse("Integral(sin(x)/x, x)")).reason,
            "risch-trig-laurent");  // Si
  EXPECT_EQ(risch_certify(parse("Integral(cos(x)/x**3, x)")).reason,
            "risch-trig-laurent");
  EXPECT_EQ(risch_certify(parse("Integral(1/log(x), x)")).reason,
            "risch-log");  // li
  EXPECT_EQ(risch_certify(parse("Integral(x**2/log(x), x)")).reason,
            "risch-log");
  EXPECT_EQ(risch_certify(parse("Integral(x**2*exp(x**2), x)")).reason,
            "risch-exp-poly");
  EXPECT_EQ(risch_certify(parse("Integral(exp(-x**2), x)")).reason,
            "risch-exp-poly");  // Gauss
  EXPECT_EQ(risch_certify(parse("Integral(7*exp(x**3 + x), x)")).reason,
            "risch-exp-poly");
}

TEST(Risch, ElementaryShapesInTheSameFamiliesStayAlive) {
  // the ODE solves: these are elementary and must NOT certify
  EXPECT_FALSE(risch_certify(parse("Integral(x*exp(x**2), x)")).dead);
  EXPECT_FALSE(
      risch_certify(parse("Integral((2*x**2 + 1)*exp(x**2), x)")).dead);
  EXPECT_FALSE(risch_certify(
                   parse("Integral(exp(x)*(1/x - 1/x**2), x)"))
                   .dead);  // d/dx (exp(x)/x)
  EXPECT_FALSE(risch_certify(parse("Integral(x**3*exp(x), x)")).dead);
  EXPECT_FALSE(risch_certify(parse("Integral(x*sin(x), x)")).dead);
  EXPECT_FALSE(risch_certify(parse("Integral(x**2*log(x), x)")).dead);
  EXPECT_FALSE(risch_certify(parse("Integral(2*x*sin(x**2), x)")).dead);
}

TEST(Risch, OutsideFamiliesUndecidedNeverMasks) {
  EXPECT_FALSE(risch_certify(parse("Integral(exp(x**2)*sin(x), x)")).dead);
  EXPECT_FALSE(risch_certify(parse("Integral(exp(exp(x)), x)")).dead);
  EXPECT_FALSE(risch_certify(parse("Integral(x**2 + 3, x)")).dead);
  EXPECT_FALSE(risch_certify(parse("Integral(log(x + 1), x)")).dead);
  EXPECT_FALSE(risch_certify(parse("Integral(1/(x*log(x)), x)")).dead);
}

TEST(Risch, StateLevelLiftConditions) {
  // dead atom + inert context: masked
  EXPECT_TRUE(dead_state(parse("Integral(exp(x**2), x) + 5")).dead);
  EXPECT_TRUE(
      dead_state(parse("3*Integral(sin(x**2), x) - x**3 + 1")).dead);
  // the core appears OUTSIDE the atom: a value-preserving move could
  // recombine them, so the lift must refuse
  EXPECT_FALSE(
      dead_state(parse("x*exp(x**2) + Integral(exp(x**2), x)")).dead);
  // solved / benign states: never masked
  EXPECT_FALSE(dead_state(parse("x**2 + sin(x)")).dead);
  EXPECT_FALSE(dead_state(parse("Integral(x**2, x)")).dead);
}

TEST(Risch, WaveGroupsCombineSameArgumentAtoms) {
  // measured on stuck_states_p1: individually both atoms certify, but
  // the SUM is d/dx[8*(2x+1)*cos(x**2+x+1)] — elementary. sin and cos
  // of one argument share the e^{i*arg} monomial, so the certificate
  // must run on the combined component, never per-atom.
  EXPECT_FALSE(dead_state(parse("Integral(-8*(2*x + 1)**2*sin(x**2 + x + 1), x)"
                                " + Integral(16*cos(x**2 + x + 1), x)"))
                   .dead);
  EXPECT_FALSE(dead_state(parse("Integral(-8*(3*x - 1)**2*sin(3*x**2 - 2*x + 4), x)"
                                " + Integral(12*cos(3*x**2 - 2*x + 4), x)"))
                   .dead);
  // a genuinely dead wave: combined component stays unsolvable
  EXPECT_TRUE(dead_state(parse("Integral(sin(x**2), x)"
                               " + Integral(cos(x**2), x)"))
                  .dead);
  // per-atom the same integrands DO certify (the wave fix must not
  // weaken single-atom certificates)
  EXPECT_TRUE(
      risch_certify(parse("Integral(-8*(2*x + 1)**2*sin(x**2 + x + 1), x)"))
          .dead);
  // constant-shifted twin arguments: undecided, never masked
  EXPECT_FALSE(dead_state(parse("Integral(exp(x**2), x)"
                                " + Integral(exp(x**2 + 1), x)"))
                   .dead);
}

TEST(Risch, BatchMaskMatchesScalar) {
  const std::vector states = {
      parse("Integral(exp(x**2), x)"), parse("Integral(x*exp(x**2), x)"),
      parse("Integral(sin(x)/x, x) + 4"), parse("x + 1")};
  const auto mask = dead_state_mask(states);
  ASSERT_EQ(mask.size(), states.size());
  EXPECT_TRUE(mask[0]);
  EXPECT_FALSE(mask[1]);
  EXPECT_TRUE(mask[2]);
  EXPECT_FALSE(mask[3]);
}

}  // namespace
