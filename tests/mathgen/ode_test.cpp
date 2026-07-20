/** L9 rung 3 parity gate: byte-exact sstr equality per (family, level,
    seed) against fixtures generated from llmopt's odes.py, plus the
    self-consistency bar — every generated row's stored solution must
    pass the native check_odesol + IC checks. */
#include <ax/mathgen/ode.hpp>

#include <ax/pyrand/pyrand.hpp>

#include <ax/sym/jsonl.hpp>
#include <ax/sym/oracle.hpp>
#include <ax/sym/print_sstr.hpp>

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <string>

#include <cstdlib>

namespace {

using ax::mathgen::ode_problem;
using ax::sym::expr;

double rat_d(const ax::rational& r) {
  return std::strtod(r.num().to_string().c_str(), nullptr) /
         std::strtod(r.den().to_string().c_str(), nullptr);
}

ode_problem dispatch(const std::string& family, int level, long long seed) {
  if (family == "ode_linear1")
    return ax::mathgen::make_linear_first_order(level, seed);
  if (family == "ode_cc2")
    return ax::mathgen::make_second_order_cc(level, seed);
  return ax::mathgen::make_separable_growth(level, seed);
}

TEST(OdeGen, FixtureParityByteExact) {
  std::ifstream in(std::string(AX_SOURCE_DIR) +
                   "/tests/mathgen/fixtures/ode_fixture.jsonl");
  ASSERT_TRUE(in.good());
  std::string line;
  int rows = 0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const auto row = ax::sym::jsonl::parse_line(line);
    const auto p = dispatch(row.at("family"), std::stoi(row.at("level")),
                            std::stoll(row.at("seed")));
    EXPECT_EQ(ax::sym::to_sstr(p.eq), row.at("eq"))
        << row.at("family") << " L" << row.at("level") << " s"
        << row.at("seed");
    EXPECT_EQ(ax::sym::to_sstr(p.sol), row.at("sol"))
        << row.at("family") << " L" << row.at("level") << " s"
        << row.at("seed");
    EXPECT_EQ(p.y0.to_string(), row.at("y0"))
        << row.at("family") << " L" << row.at("level") << " s"
        << row.at("seed");
    if (row.count("yp0") && row.at("yp0") != "null" &&
        !row.at("yp0").empty()) {
      ASSERT_TRUE(p.yp0.has_value());
      EXPECT_EQ(p.yp0->to_string(), row.at("yp0"));
    }
    ++rows;
  }
  EXPECT_EQ(rows, 90);
}

TEST(OdeGen, EveryRowSelfVerifies) {
  const expr x = expr::symbol("x");
  const char* fams[] = {"ode_linear1", "ode_cc2", "ode_separable"};
  for (const char* fam : fams)
    for (int level = 1; level <= 3; ++level)
      for (long long seed = 0; seed < 10; ++seed) {
        const auto p = dispatch(fam, level, seed);
        EXPECT_EQ(ax::sym::check_odesol(p.eq, p.sol, x),
                  ax::sym::verdict::equivalent)
            << fam << " L" << level << " s" << seed;
        EXPECT_TRUE(ax::sym::check_ic(
            p.sol, x, static_cast<double>(p.x0), rat_d(p.y0), 0))
            << fam << " L" << level << " s" << seed;
        if (p.yp0)
          EXPECT_TRUE(ax::sym::check_ic(p.sol, x,
                                        static_cast<double>(p.x0),
                                        rat_d(*p.yp0), 1))
              << fam << " L" << level << " s" << seed;
      }
}

}  // namespace
