#include <ax/search/search.hpp>

#include <algorithm>

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
using ax::search::hce;
using ax::search::is_solved;
using ax::search::replay_verify;
using ax::search::rule_set;
using ax::search::state;
using ax::search::successor_options;
using ax::search::successors;
using ax::search::verify_edge;
using ax::sym::expr;
using ax::sym::parse;
using ax::sym::to_sstr;

const expr x = expr::symbol("x");

/** Mini rule set for chassis tests: sum split, constant factor, a tiny
    integral table, and the power rule — enough to solve small roots. */
rule_set test_rules() {
  rule_set r;
  const auto unpack = [](const expr& node)
      -> std::optional<std::pair<expr, expr>> {
    if (!node.is_fn() || node.name() != "Integral" ||
        node.args().size() != 2)
      return std::nullopt;
    return std::make_pair(node.args()[0], node.args()[1]);
  };
  r.integral.emplace_back("i_sum", [unpack](const expr& node) {
    std::vector<expr> out;
    if (const auto p = unpack(node); p && p->first.is_add()) {
      expr split = expr::num(0);
      for (const expr& t : p->first.args())
        split = split + expr::integral(t, p->second);
      out.push_back(split);
    }
    return out;
  });
  r.integral.emplace_back("i_const_factor", [unpack](const expr& node) {
    std::vector<expr> out;
    if (const auto p = unpack(node);
        p && p->first.is_mul() && p->first.args()[0].is_num()) {
      const expr c = p->first.args()[0];
      expr rest = p->first.args()[1];
      for (std::size_t i = 2; i < p->first.args().size(); ++i)
        rest = rest * p->first.args()[i];
      out.push_back(c * expr::integral(rest, p->second));
    }
    return out;
  });
  r.integral.emplace_back("i_table", [unpack](const expr& node) {
    std::vector<expr> out;
    if (const auto p = unpack(node)) {
      if (p->first.is_fn() && p->first.name() == "sin" &&
          p->first.args()[0].same(p->second))
        out.push_back(-expr::fn("cos", p->second));
      if (p->first.same(p->second))
        out.push_back(p->second.pow(expr::num(2)) / expr::num(2));
    }
    return out;
  });
  r.integral.emplace_back("i_power", [unpack](const expr& node) {
    std::vector<expr> out;
    if (const auto p = unpack(node);
        p && p->first.is_pow() && p->first.args()[0].same(p->second) &&
        p->first.args()[1].is_num()) {
      const auto n = p->first.args()[1].value();
      const auto n1 = n + ax::rational(ax::bigint(1));
      out.push_back(p->second.pow(expr::num(n1)) / expr::num(n1));
    }
    return out;
  });
  // wrong-by-construction rule for the soundness test
  r.integral.emplace_back("i_wrong", [unpack](const expr& node) {
    std::vector<expr> out;
    if (const auto p = unpack(node);
        p && p->first.is_fn() && p->first.name() == "sin")
      out.push_back(expr::fn("cos", p->second));  // missing the minus
    return out;
  });
  return r;
}

TEST(VerifyEdge, AcceptsTrueIntegralStepModConstant) {
  const expr parent = parse("Integral(sin(x), x)");
  const expr child = parse("-cos(x)");
  EXPECT_TRUE(verify_edge(parent, child, {}));
  // and with an added constant (equality mod constant)
  EXPECT_TRUE(verify_edge(parent, parse("-cos(x) + 7"), {}));
}

TEST(VerifyEdge, RejectsWrongRewrite) {
  EXPECT_FALSE(verify_edge(parse("Integral(sin(x), x)"), parse("cos(x)"), {}));
  EXPECT_FALSE(
      verify_edge(parse("Integral(x**2, x)"), parse("x**3"), {}));
}

TEST(VerifyEdge, DerivativeEdgesResolveCarriers) {
  // d_sum-style rewrite: Derivative(x**2 + sin(x), x) ->
  // Derivative(x**2, x) + Derivative(sin(x), x)
  const expr parent = parse("Derivative(x**2 + sin(x), x)");
  const expr child = parse("Derivative(x**2, x) + Derivative(sin(x), x)");
  EXPECT_TRUE(verify_edge(parent, child, {}));
  EXPECT_FALSE(verify_edge(parent, parse("Derivative(x**2, x)"), {}));
}

TEST(Successors, SplitsVerifiesAndLabelsBareNames) {
  const state s{parse("Integral(sin(x) + x**2, x)")};
  const auto kids = successors(s, test_rules(), {});
  ASSERT_FALSE(kids.empty());
  bool found_sum = false;
  for (const auto& [name, child] : kids) {
    EXPECT_EQ(name.find('@'), std::string::npos);  // bare names
    if (name == "i_sum") found_sum = true;
  }
  EXPECT_TRUE(found_sum);
  // the wrong rule's rewrite must have been rejected by verify_edge
  for (const auto& [name, child] : kids) EXPECT_NE(name, "i_wrong");
}

TEST(Beam, SolvesSmallRootAndReplayVerifies) {
  const expr root = parse("Integral(3*sin(x) + x**2, x)");
  beam_options opt;
  opt.width = 4;
  opt.max_plies = 8;
  const auto res = beam_search(root, test_rules(), opt);
  ASSERT_TRUE(res.solved) << to_sstr(res.best.e);
  EXPECT_TRUE(is_solved(res.best));
  // the emitted answer differentiates back to the integrand
  EXPECT_EQ(ax::sym::equivalent(ax::sym::diff(res.best.e, x),
                                parse("3*sin(x) + x**2"), x),
            ax::sym::verdict::equivalent);
  // soundness boundary: the chain replays under full verification
  EXPECT_TRUE(replay_verify(root, res.best.history, test_rules()));
}

TEST(Beam, Deterministic) {
  const expr root = parse("Integral(3*sin(x) + x**2, x)");
  const auto a = beam_search(root, test_rules(), {});
  const auto b = beam_search(root, test_rules(), {});
  EXPECT_EQ(a.solved, b.solved);
  EXPECT_TRUE(a.best.e.same(b.best.e));
  EXPECT_EQ(a.best.history, b.best.history);
  EXPECT_EQ(a.nodes, b.nodes);
}

TEST(Beam, ProposeKNeverGuillotinesSolvedKid) {
  // proposer that puts the solving move last; propose_k=1 must keep it
  const expr root = parse("Integral(sin(x), x)");
  beam_options opt;
  opt.propose_k = 1;
  opt.proposer = [](const state&,
                    std::vector<std::pair<std::string, state>> kids) {
    std::stable_sort(kids.begin(), kids.end(),
                     [](const auto& a, const auto& b) {
                       return static_cast<int>(is_solved(a.second)) <
                              static_cast<int>(is_solved(b.second));
                     });
    return kids;
  };
  const auto res = beam_search(root, test_rules(), opt);
  EXPECT_TRUE(res.solved);
}

TEST(Beam, QualRootSmoke) {
  // a couple of L1 qualification roots solve with the mini rules
  const char* roots[] = {
      "Integral(x**2 + x, x)",
      "Integral(5*sin(x) + 2*x, x)",
  };
  for (const char* r : roots) {
    const auto res = beam_search(parse(r), test_rules(), {});
    EXPECT_TRUE(res.solved) << r;
  }
}

}  // namespace
