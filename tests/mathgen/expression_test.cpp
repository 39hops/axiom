#include <ax/mathgen/problems.hpp>

#include <ax/pyrand/pyrand.hpp>
#include <ax/sym/print_sstr.hpp>

#include <gtest/gtest.h>

#include <fstream>
#include <string>

namespace {

using ax::mathgen::expression;
using ax::mathgen::seed_string;
using ax::pyrand::python_random;
using ax::sym::to_sstr;

TEST(MathgenExpression, L1L3ExactSstrParity) {
  // Gate: byte-exact equality with llmopt's sympy generator per
  // (level, seed). Fixture from the authentic problems.py module.
  int cases = 0, failures = 0;
  for (const char* fixture : {"expression_l1_3.tsv", "expression_l4.tsv", "expression_l5_8.tsv"}) {
  std::ifstream in(std::string(AX_SOURCE_DIR) +
                   "/tests/mathgen/fixtures/" + fixture);
  ASSERT_TRUE(in.good()) << "regenerate via WSL llmopt venv (see plan)";
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    const auto t1 = line.find('\t');
    const auto t2 = line.find('\t', t1 + 1);
    const int level = std::stoi(line.substr(0, t1));
    const long long seed = std::stoll(line.substr(t1 + 1, t2 - t1 - 1));
    const std::string want = line.substr(t2 + 1);
    ++cases;
    python_random rng(seed_string("diff", level, seed));
    const std::string got = to_sstr(expression(rng, level));
    EXPECT_EQ(got, want) << "level " << level << " seed " << seed;
    if (got != want) ++failures;
  }
  }
  EXPECT_EQ(cases, 1050);
  EXPECT_EQ(failures, 0) << failures << "/" << cases << " diverge";
}

TEST(MathgenExpression, TenThousandPairGate) {
  // The Phase B gate proper: 10^4 (level, seed) pairs, byte-exact.
  std::ifstream in(std::string(AX_SOURCE_DIR) +
                   "/tests/mathgen/fixtures/expression_10k.tsv");
  ASSERT_TRUE(in.good()) << "regenerate via WSL llmopt venv (see plan)";
  std::string line;
  int cases = 0, failures = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    const auto t1 = line.find('	');
    const auto t2 = line.find('	', t1 + 1);
    const int level = std::stoi(line.substr(0, t1));
    const long long seed = std::stoll(line.substr(t1 + 1, t2 - t1 - 1));
    const std::string want = line.substr(t2 + 1);
    ++cases;
    python_random rng(seed_string("diff", level, seed));
    const std::string got = to_sstr(expression(rng, level));
    if (got != want) {
      ++failures;
      if (failures <= 10)
        ADD_FAILURE() << "level " << level << " seed " << seed
                      << " got " << got << " want " << want;
    }
  }
  EXPECT_EQ(cases, 10000);
  EXPECT_EQ(failures, 0) << failures << "/" << cases << " diverge";
}

}  // namespace
