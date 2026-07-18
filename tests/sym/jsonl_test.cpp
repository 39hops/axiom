#include <ax/sym/harness.hpp>
#include <ax/sym/jsonl.hpp>

#include <ax/sym/parse.hpp>
#include <ax/sym/print.hpp>

#include <gtest/gtest.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace jl = ax::sym::jsonl;
using ax::sym::run_oracle;

// ------------------------------------------------------------------- jsonl

TEST(Jsonl, ParseFlatObject) {
  const auto o = jl::parse_line(
      R"js({"id": "r1", "task": "diff", "var": "x", "expr": "x**2"})js");
  EXPECT_EQ(o.at("id"), "r1");
  EXPECT_EQ(o.at("task"), "diff");
  EXPECT_EQ(o.at("expr"), "x**2");
}

TEST(Jsonl, BareTokensCapturedRaw) {
  const auto o = jl::parse_line(R"js({"id": 17, "ok": true, "z": null})js");
  EXPECT_EQ(o.at("id"), "17");
  EXPECT_EQ(o.at("ok"), "true");
  EXPECT_EQ(o.at("z"), "null");
}

TEST(Jsonl, EscapeRoundTrip) {
  const std::string nasty = "a\"b\\c\nd\te\x01";
  const std::string line = jl::write_line({{"k", nasty}});
  const auto back = jl::parse_line(line);
  EXPECT_EQ(back.at("k"), nasty);
}

TEST(Jsonl, MalformedThrows) {
  EXPECT_THROW(jl::parse_line("not json"), std::runtime_error);
  EXPECT_THROW(jl::parse_line(R"js({"a": )js"), std::runtime_error);
  EXPECT_THROW(jl::parse_line(R"js({"a": {"nested": 1}})js"), std::runtime_error);
  EXPECT_THROW(jl::parse_line(""), std::runtime_error);
}

TEST(Jsonl, WriteOrderPreserved) {
  EXPECT_EQ(jl::write_line({{"b", "1"}, {"a", "2"}}),
            R"js({"b":"1","a":"2"})js");
}

// ----------------------------------------------------------------- harness

std::vector<jl::object> run_rows(const std::string& in_text) {
  std::istringstream in(in_text);
  std::ostringstream out;
  run_oracle(in, out);
  std::vector<jl::object> rows;
  std::istringstream lines(out.str());
  std::string line;
  while (std::getline(lines, line))
    if (!line.empty()) rows.push_back(jl::parse_line(line));
  return rows;
}

TEST(Harness, EndToEndAllTasksPlusPoisonRow) {
  const std::string in =
      R"js({"id": "d1", "task": "diff", "var": "x", "expr": "x**3"})js"
      "\n"
      R"js({"id": "e1", "task": "equiv", "var": "x", "lhs": "2*x + 2", "rhs": "2*(x + 1)"})js"
      "\n"
      R"js({"id": "e2", "task": "equiv", "var": "x", "lhs": "x**2", "rhs": "x**3"})js"
      "\n"
      R"js({"id": "e3", "task": "equiv", "var": "x", "lhs": "sin(x)**2 + cos(x)**2", "rhs": "1"})js"
      "\n"
      R"js({"id": "m1", "task": "equiv_mod_const", "var": "x", "candidate": "x**2/2 + 5", "integrand": "x"})js"
      "\n"
      R"js({"id": "p1", "task": "diff", "var": "x", "expr": "foo(x)"})js"
      "\n"
      "this line is not json\n"
      R"js({"id": "u1", "task": "no_such_task"})js"
      "\n";
  const auto rows = run_rows(in);
  ASSERT_EQ(rows.size(), 8u);

  EXPECT_EQ(rows[0].at("id"), "d1");
  EXPECT_EQ(rows[0].at("status"), "ok");
  // Result must be parseable back and be d/dx x^3.
  EXPECT_TRUE(ax::sym::parse(rows[0].at("result"))
                  .same(ax::sym::parse("3*x**2")));

  EXPECT_EQ(rows[1].at("verdict"), "EQUIVALENT");
  EXPECT_EQ(rows[2].at("verdict"), "NOT_EQUIVALENT");
  EXPECT_EQ(rows[3].at("verdict"), "UNDECIDED");
  EXPECT_EQ(rows[4].at("verdict"), "EQUIVALENT");

  EXPECT_EQ(rows[5].at("id"), "p1");
  EXPECT_EQ(rows[5].at("status"), "error");
  EXPECT_NE(rows[5].at("error").find("unknown function"), std::string::npos);

  EXPECT_EQ(rows[6].at("status"), "error");  // non-json line
  EXPECT_EQ(rows[6].at("id"), "line:7");     // fallback id

  EXPECT_EQ(rows[7].at("id"), "u1");
  EXPECT_EQ(rows[7].at("status"), "error");
}

TEST(Harness, DiffResultUsesSstrPow) {
  const auto rows = run_rows(
      R"js({"id": "d", "task": "diff", "var": "x", "expr": "x**5"})js"
      "\n");
  ASSERT_EQ(rows.size(), 1u);
  // sympy must be able to sympify the result: '**', never '^'.
  EXPECT_EQ(rows[0].at("result").find('^'), std::string::npos);
  EXPECT_NE(rows[0].at("result").find("**"), std::string::npos);
}

TEST(Harness, MissingFieldIsRowError) {
  const auto rows = run_rows(
      R"js({"id": "d", "task": "diff", "expr": "x**5"})js"
      "\n");
  ASSERT_EQ(rows.size(), 1u);
  EXPECT_EQ(rows[0].at("status"), "error");
}

TEST(Harness, BulkRowSmoke) {
  // 200 monster-shaped rows; throughput itself is measured on the Release
  // build during the parity run, this only proves the loop doesn't leak
  // state or die mid-stream.
  std::ostringstream in;
  for (int i = 0; i < 200; ++i) {
    in << R"js({"id": ")js" << i
       << R"js(", "task": "equiv_mod_const", "var": "x", )js"
       << R"js("candidate": "5*x*sin(log(x*(2*x - 3)))/(2*x - 3) + )js" << i
       << R"js(", "integrand": "x**)js" << (i % 7)
       << R"js("})js"
       << "\n";
  }
  const auto rows = run_rows(in.str());
  ASSERT_EQ(rows.size(), 200u);
  for (const auto& r : rows) EXPECT_EQ(r.at("status"), "ok");
}

}  // namespace
