/** Phase D: chain emission (farm-shard rows). D1 replay_chain,
    D2 annotate (syndrome hints + ansatz think traces). */
#include <ax/search/search.hpp>

#include <ax/sym/calc.hpp>
#include <ax/sym/oracle.hpp>
#include <ax/sym/parse.hpp>
#include <ax/sym/print_sstr.hpp>

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

using ax::search::beam_options;
using ax::search::beam_search;
using ax::search::default_rules;
using ax::search::replay_chain;
using ax::sym::expr;
using ax::sym::parse;

const expr x = expr::symbol("x");

TEST(EmitD1, ReplayChainMatchesBeam) {
  const expr root = parse("Integral(2*x*exp(x**2), x)");
  beam_options opt;
  opt.width = 8;
  opt.max_plies = 12;
  opt.max_nodes = 200;
  opt.use_macros = true;
  const auto res = beam_search(root, default_rules(), opt);
  ASSERT_TRUE(res.solved);
  const auto chain = replay_chain(root, res.best.history, default_rules());
  ASSERT_TRUE(chain.has_value());
  // states: root .. answer, one per history step plus the root
  ASSERT_EQ(chain->size(), res.best.history.size() + 1);
  EXPECT_TRUE(chain->front().e.same(root));
  EXPECT_TRUE(chain->back().e.same(res.best.e));
  // every replayed edge carries the corresponding history label
  for (std::size_t i = 0; i + 1 < chain->size(); ++i)
    EXPECT_EQ((*chain)[i + 1].history.back(), res.best.history[i]);
}

TEST(EmitD1, CorruptedHistoryReturnsNullopt) {
  const expr root = parse("Integral(2*x*exp(x**2), x)");
  const std::vector<std::string> bogus{"i_usub", "no_such_rule"};
  EXPECT_FALSE(replay_chain(root, bogus, default_rules()).has_value());
}

TEST(EmitD2, HintsAreFiringRuleNames) {
  const expr root = parse("Integral(2*x*exp(x**2), x)");
  const auto ann = ax::search::annotate(root, default_rules());
  // i_usub must be among the firing rules on this classic
  bool has_usub = false;
  for (const auto& h : ann.hints) has_usub = has_usub || h == "i_usub";
  EXPECT_TRUE(has_usub);
  EXPECT_FALSE(ann.hints.empty());
}

TEST(EmitD2, ThinkNullForNonAnsatzRules) {
  const expr root = parse("Integral(2*x*exp(x**2), x)");
  const auto ann =
      ax::search::annotate(root, default_rules(), "i_usub");
  EXPECT_FALSE(ann.think.has_value());
}

TEST(EmitD2, ThinkVerbalizedForSqrtBasis) {
  // f = d/dx[(x**2+1)*sqrt(x**2+1)]-shaped integrand: i_sqrt_basis
  // poly branch fires and the trace must match llmopt's _trace string.
  const expr F = parse("(x**2 + 1)*sqrt(x**2 + 1)");
  const expr f = ax::sym::diff(F, x);
  const expr root = expr::integral(f, x);
  const auto ann =
      ax::search::annotate(root, default_rules(), "i_sqrt_basis");
  ASSERT_TRUE(ann.think.has_value());
  EXPECT_NE(ann.think->find("ansatz A(x)*sqrt("), std::string::npos)
      << *ann.think;
  EXPECT_NE(ann.think->find(
                "match 2*A'*P + A*P' = 2*f*sqrt(P) in the poly ring"),
            std::string::npos)
      << *ann.think;
}

TEST(EmitD2, ThinkVerbalizedForLinearBasis) {
  const expr root = parse("Integral(x*exp(x)*sin(x), x)");
  const auto ann =
      ax::search::annotate(root, default_rules(), "i_linear_basis");
  ASSERT_TRUE(ann.think.has_value());
  EXPECT_NE(ann.think->find("ansatz sum of c_ij * x^j * m_i over basis {"),
            std::string::npos)
      << *ann.think;
  EXPECT_NE(ann.think->find("differentiate and equate coefficients"),
            std::string::npos)
      << *ann.think;
}

}  // namespace
