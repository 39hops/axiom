/** @file inverse_test.cpp S7 inverse-move enumeration round-trips.

    The contract under test: for every forward edge (rule, p) -> t the
    engine emits, predecessors(t) must contain (rule, p) exactly. Each
    case runs the REAL forward engine (default_rules, verify_p = 1) so
    the test can never drift from successors() semantics. */
#include <ax/search/inverse.hpp>

#include <ax/sym/parse.hpp>
#include <ax/sym/print_sstr.hpp>

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

using ax::search::default_rules;
using ax::search::inverse_options;
using ax::search::predecessors;
using ax::search::state;
using ax::search::successor_options;
using ax::search::successors;
using ax::sym::parse;
using ax::sym::to_sstr;

/** Forward-expand p, then require every (rule, child) edge to be
    recovered by predecessors(child). Returns a diagnostic string of
    missed edges ("" = all recovered). */
std::string round_trip_misses(const std::string& parent_sstr,
                              bool use_macros = true) {
  const auto p = parse(parent_sstr);
  successor_options sopt;
  sopt.use_macros = use_macros;
  inverse_options iopt;
  iopt.use_macros = use_macros;
  std::string misses;
  for (const auto& [rule, child] :
       successors(state{p}, default_rules(), sopt)) {
    bool found = false;
    for (const auto& pred : predecessors(child.e, default_rules(), iopt))
      found = found || (pred.rule == rule && pred.p.same(p));
    if (!found)
      misses += rule + " :: " + to_sstr(child.e) + "\n";
  }
  return misses;
}

TEST(Inverse, LinearityFamily) {
  EXPECT_EQ(round_trip_misses("Integral(x**2 + sin(x), x)"), "");
  EXPECT_EQ(round_trip_misses("Integral(3*x**2, x)"), "");
  EXPECT_EQ(round_trip_misses("Integral(5, x)"), "");
}

TEST(Inverse, ClosersMergedIntoContext) {
  // the reassociation problem: closer outputs merge into surrounding
  // add/mul, so subtree sites alone cannot recover these
  EXPECT_EQ(round_trip_misses("x + Integral(x, x)"), "");
  EXPECT_EQ(round_trip_misses("2*Integral(cos(x), x) + 7"), "");
  EXPECT_EQ(round_trip_misses("Integral(log(x), x) - x"), "");
}

TEST(Inverse, TableAndPower) {
  EXPECT_EQ(round_trip_misses("Integral(sin(x), x)"), "");
  EXPECT_EQ(round_trip_misses("Integral(x**3, x)"), "");
  EXPECT_EQ(round_trip_misses("Integral(x**-1, x)"), "");
  EXPECT_EQ(round_trip_misses("Integral(exp(x), x)"), "");
}

TEST(Inverse, PartsAndUsub) {
  EXPECT_EQ(round_trip_misses("Integral(x*cos(x), x)"), "");
  EXPECT_EQ(round_trip_misses("Integral(2*x*exp(x**2), x)"), "");
}

TEST(Inverse, DerivativeFamily) {
  EXPECT_EQ(round_trip_misses("Derivative(x**2 + sin(x), x)"), "");
  EXPECT_EQ(round_trip_misses("Derivative(sin(x**2), x)"), "");
  EXPECT_EQ(round_trip_misses("Derivative(5*x**3, x)"), "");
}

TEST(Inverse, NestedValueIsolated) {
  const auto cur =
      parse("(14 - 39*x)*Integral(x, x) - Integral(-39*Integral(x, x), x)");
  const auto nxt = parse("x**2*(14 - 39*x)/2 - Integral(-39*x**2/2, x)");
  bool cand_seen = false, pair_found = false;
  for (const auto& pred : predecessors(nxt, default_rules(), {})) {
    if (pred.p.same(cur)) {
      cand_seen = true;
      if (pred.rule == "i_power") pair_found = true;
    }
  }
  EXPECT_TRUE(pair_found) << "cand_seen=" << cand_seen;
}

TEST(Inverse, NestedMultiOccurrenceValues) {
  // i_parts chains: the closed value replaces the carrier both bare and
  // inside another Integral's integrand (forward replace-all semantics)
  EXPECT_EQ(round_trip_misses(
                "(14 - 39*x)*Integral(x, x) - Integral(-39*Integral(x, x), x)"),
            "");
}

TEST(Inverse, SoundnessEveryPairForwardVerifies) {
  // every returned (rule, p) must actually reach t through the engine
  const auto t = parse("Integral(x, x) + Integral(sin(x), x)");
  for (const auto& pred : predecessors(t, default_rules(), {})) {
    bool reaches = false;
    for (const auto& [rule, child] :
         successors(state{pred.p}, default_rules(), {}))
      reaches = reaches || (rule == pred.rule && child.e.same(t));
    EXPECT_TRUE(reaches) << pred.rule << " :: " << to_sstr(pred.p);
  }
}

TEST(Inverse, DeadlineReturnsPartial) {
  const auto t = parse("x**4/4 + x**3/3 + x**2/2 + sin(x) + exp(x)");
  inverse_options iopt;
  iopt.deadline = std::chrono::steady_clock::now();  // already expired
  // must return promptly and empty-or-partial, never throw
  (void)predecessors(t, default_rules(), iopt);
}

}  // namespace
