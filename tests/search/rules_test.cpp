#include <ax/search/search.hpp>

#include <ax/sym/calc.hpp>
#include <ax/sym/jsonl.hpp>
#include <ax/sym/oracle.hpp>
#include <ax/sym/parse.hpp>
#include <ax/sym/print_sstr.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <fstream>
#include <map>
#include <string>

namespace {

using ax::search::beam_options;
using ax::search::beam_search;
using ax::search::default_rules;
using ax::search::replay_verify;
using ax::sym::expr;
using ax::sym::parse;

const expr x = expr::symbol("x");

TEST(RulesT1, IndividualRulesFire) {
  const auto& rs = default_rules();
  const auto find = [&](const std::string& n) {
    for (const auto& [name, fn] : rs.integral)
      if (name == n) return fn;
    for (const auto& [name, fn] : rs.core)
      if (name == n) return fn;
    for (const auto& [name, fn] : rs.macros)
      if (name == n) return fn;
    return ax::search::rule_fn{};
  };
  // i_usub on the classic 2x*exp(x^2)
  const auto usub = find("i_usub")(parse("Integral(2*x*exp(x**2), x)"));
  ASSERT_FALSE(usub.empty());
  bool found = false;
  for (const expr& r : usub)
    found = found || ax::sym::to_sstr(r).find("Subs(Integral(exp(u_), u_)") !=
                         std::string::npos;
  EXPECT_TRUE(found);
  // i_parts on x*cos(x): u=x branch exists
  const auto parts = find("i_parts")(parse("Integral(x*cos(x), x)"));
  EXPECT_GE(parts.size(), 2u);  // one branch per differentiable factor
  // d_chain_table on sin(x**2)
  const auto chain =
      find("d_chain_table")(parse("Derivative(sin(x**2), x)"));
  ASSERT_EQ(chain.size(), 1u);
  // the rule emits cos(x**2)*Derivative(x**2, x) — carrier unevaluated,
  // exactly as rules.py; resolve it before comparing
  EXPECT_EQ(ax::sym::equivalent(ax::search::doit_no_integrals(chain[0]),
                                parse("2*x*cos(x**2)"), x),
            ax::sym::verdict::equivalent);
}

TEST(RulesT1, QualRootsSmokeSlice) {
  // Informal solve-count gate on the contamination-clean qualification
  // roots (official parity reference comes from llmopt's sympy run).
  // Every emitted chain must be oracle-valid: diff-back equivalence +
  // full replay under verify_p=1.
  std::ifstream in(std::string(AX_SOURCE_DIR) + "/data/llmopt/axiom_qual_roots.jsonl");
  if (!in.good()) GTEST_SKIP() << "qual roots not present";
  std::map<int, int> solved, total;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const auto row = ax::sym::jsonl::parse_line(line);
    const int level = std::stoi(row.at("level"));
    if (level > 2) continue;  // full L1-L8 gate = axiom-qual-gate tool (Release)
    if (total[level] >= 10) continue;  // smoke slice; keeps the suite fast
    ++total[level];
    const expr root = parse(row.at("root"));
    beam_options opt;
    opt.width = 8;
    opt.max_plies = 12;
    opt.max_nodes = 200;
    opt.use_macros = true;
    const auto res = beam_search(root, default_rules(), opt);
    if (!res.solved) continue;
    // oracle validity, both directions of the contract
    const expr integrand = root.args()[0];
    ASSERT_EQ(ax::sym::equivalent(ax::sym::diff(res.best.e, x), integrand, x),
              ax::sym::verdict::equivalent)
        << row.at("id");
    ASSERT_TRUE(replay_verify(root, res.best.history, default_rules()))
        << row.at("id");
    ++solved[level];
  }
  for (const auto& [lvl, tot] : total)
    std::cout << "L" << lvl << ": " << solved[lvl] << "/" << tot << "\n";
  // conservative floors; the official gate number is llmopt's reference
  EXPECT_GE(solved[1], 9);
  EXPECT_GE(solved[2], 7);
}

TEST(RulesT3, PathologyRootsBoundedUnderDeadline) {
  // llmopt-style pathology collection: roots that once wedged the gate.
  // Contract: with a short deadline the search RETURNS (solved or not)
  // within a small wall — no single rule fire may hold the search
  // hostage. Guards the "budget lives inside child generation" lesson.
  std::ifstream in(std::string(AX_SOURCE_DIR) +
                   "/tests/search/fixtures/pathology_roots.jsonl");
  ASSERT_TRUE(in.good());
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const auto row = ax::sym::jsonl::parse_line(line);
    const expr root = parse(row.at("root"));
    beam_options opt;
    opt.width = 8;
    opt.max_plies = 12;
    opt.max_nodes = 200;
    opt.use_macros = true;
    opt.deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    const auto t0 = std::chrono::steady_clock::now();
    (void)beam_search(root, default_rules(), opt);
    const double dt =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    // generous 4x margin over the deadline for the in-flight fire
    EXPECT_LT(dt, 20.0) << row.at("id");
  }
}

}  // namespace
