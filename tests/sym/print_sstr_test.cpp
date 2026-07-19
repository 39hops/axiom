#include <ax/sym/print_sstr.hpp>

#include <ax/sym/parse.hpp>
#include <ax/sym/oracle.hpp>

#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <vector>

namespace {

using ax::sym::parse;
using ax::sym::to_sstr;

TEST(PrintSstr, FixtureParity) {
  // Gate: parse each input with axiom, print with to_sstr, must equal
  // sympy's sstr byte-for-byte. Fixture from scripts/gen_sstr_fixture.py.
  std::ifstream in(std::string(AX_SOURCE_DIR) +
                   "/tests/sym/fixtures/sstr_fixture.tsv");
  ASSERT_TRUE(in.good()) << "run scripts/gen_sstr_fixture.py";
  std::string line;
  int cases = 0, failures = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    const auto tab = line.find('\t');
    ASSERT_NE(tab, std::string::npos);
    const std::string input = line.substr(0, tab);
    const std::string want = line.substr(tab + 1);
    ++cases;
    const std::string got = to_sstr(parse(input));
    EXPECT_EQ(got, want) << "input: " << input;
    if (got != want) ++failures;
  }
  EXPECT_GE(cases, 50);
  EXPECT_EQ(failures, 0) << failures << "/" << cases << " diverge";
}

TEST(PrintSstr, RoundTripsThroughOwnParser) {
  const char* inputs[] = {
      "x**3 + 2*x + 5",
      "4*exp(x) - 7*sin(x) + 6*cos(x) + 5",
      "1/(2*sqrt(2 - x))",
      "5*((3 - 4*x)*sin(log(x*(2*x - 3))) + "
      "(2*x - 3)*cos(log(x*(2*x - 3))))/(2*x - 3)",
  };
  // sympy's auto-eval can produce a value-equal but structurally different
  // form (e.g. it distributes the 5 in the monster where axiom does not),
  // so the printer's round-trip contract is oracle equivalence, not
  // hash-cons identity.
  const auto x = ax::sym::expr::symbol("x");
  for (const char* s : inputs) {
    const auto e = parse(s);
    EXPECT_EQ(ax::sym::equivalent(parse(to_sstr(e)), e, x),
              ax::sym::verdict::equivalent)
        << s << " -> " << to_sstr(e);
  }
}

}  // namespace
