#include <ax/sym/count_ops.hpp>

#include <ax/sym/calc.hpp>
#include <ax/sym/parse.hpp>
#include <ax/sym/print_sstr.hpp>

#include <gtest/gtest.h>

#include <fstream>
#include <string>

namespace {

using ax::sym::count_ops;
using ax::sym::diff;
using ax::sym::expr;
using ax::sym::parse;
using ax::sym::to_sstr;

TEST(CountOps, FixtureParity) {
  std::ifstream in(std::string(AX_SOURCE_DIR) +
                   "/tests/sym/fixtures/count_ops_fixture.tsv");
  ASSERT_TRUE(in.good()) << "run scripts/gen_count_ops_fixture.py";
  std::string line;
  int cases = 0, failures = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    const auto tab = line.rfind('\t');
    const std::string src = line.substr(0, tab);
    const long long want = std::stoll(line.substr(tab + 1));
    ++cases;
    const long long got = count_ops(parse(src));
    EXPECT_EQ(got, want) << src;
    if (got != want) ++failures;
  }
  EXPECT_GE(cases, 400);
  EXPECT_EQ(failures, 0) << failures << "/" << cases << " diverge";
}

// ------------------------------------------------------------- carriers

TEST(Carriers, SstrRoundTrip) {
  const char* cases[] = {
      "Integral(x*cos(x), x)",
      "Integral(x*cos(x), x, x)",
      "Derivative(x**3 + 2*x, x)",
      "Subs(Integral(sin(_u), _u), _u, x**2)",
      "Integral(x*exp(x), x) + Integral(sin(x), x)",
  };
  for (const char* s : cases) {
    const expr e = parse(s);
    EXPECT_EQ(to_sstr(e), s);
    EXPECT_TRUE(parse(to_sstr(e)).same(e)) << s;
  }
}

TEST(Carriers, DiffOfIntegralIsIntegrand) {
  const expr x = expr::symbol("x");
  const expr f = parse("x*cos(x)");
  // the verify identity: d/dx Integral(f, x) == f, no integration anywhere
  EXPECT_TRUE(diff(expr::integral(f, x), x).same(f));
  // nested: d/dx Integral(f, x, x) == Integral(f, x)
  const expr nested = parse("Integral(x*cos(x), x, x)");
  EXPECT_TRUE(diff(nested, x).same(expr::integral(f, x)));
}

TEST(Carriers, OpaqueToCanonicalizationAndEvalThrows) {
  const expr x = expr::symbol("x");
  const expr i = expr::integral(parse("sin(x)"), x);
  // structural cancellation through Add: I - I == 0
  EXPECT_TRUE((i - i).same(expr::num(0)));
  EXPECT_THROW(i.eval({{"x", 1.0}}), std::logic_error);
}

}  // namespace
