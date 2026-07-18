#include <ax/sym/harness.hpp>

#include <ax/sym/calc.hpp>
#include <ax/sym/jsonl.hpp>
#include <ax/sym/oracle.hpp>
#include <ax/sym/parse.hpp>
#include <ax/sym/print.hpp>

#include <stdexcept>
#include <string>

namespace ax::sym {

namespace {

/** axiom's printer spells pow `^`; sympy sstr needs `**`. `^` occurs only
    as the pow operator in printer output, so plain replacement is exact. */
std::string to_sstr(const expr& e) {
  std::string s = to_string(e);
  std::string out;
  out.reserve(s.size());
  for (const char c : s) {
    if (c == '^')
      out += "**";
    else
      out += c;
  }
  return out;
}

const std::string& field(const jsonl::object& row, const std::string& key) {
  const auto it = row.find(key);
  if (it == row.end())
    throw std::runtime_error("missing field '" + key + "'");
  return it->second;
}

const char* verdict_name(verdict v) {
  switch (v) {
    case verdict::equivalent: return "EQUIVALENT";
    case verdict::not_equivalent: return "NOT_EQUIVALENT";
    case verdict::undecided: return "UNDECIDED";
  }
  return "UNDECIDED";
}

jsonl::fields process_row(const jsonl::object& row, const std::string& id) {
  const std::string& task = field(row, "task");
  const expr x = expr::symbol(field(row, "var"));
  if (task == "diff") {
    const expr d = diff(parse(field(row, "expr")), x);
    return {{"id", id}, {"status", "ok"}, {"result", to_sstr(d)}};
  }
  if (task == "equiv") {
    const verdict v =
        equivalent(parse(field(row, "lhs")), parse(field(row, "rhs")), x);
    return {{"id", id}, {"status", "ok"}, {"verdict", verdict_name(v)}};
  }
  if (task == "equiv_mod_const") {
    const verdict v = equivalent_mod_const(parse(field(row, "candidate")),
                                           parse(field(row, "integrand")), x);
    return {{"id", id}, {"status", "ok"}, {"verdict", verdict_name(v)}};
  }
  throw std::runtime_error("unknown task '" + task + "'");
}

}  // namespace

int run_oracle(std::istream& in, std::ostream& out) {
  int errors = 0;
  std::string line;
  long long lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    if (line.empty()) continue;
    std::string id = "line:" + std::to_string(lineno);
    try {
      const jsonl::object row = jsonl::parse_line(line);
      if (const auto it = row.find("id"); it != row.end()) id = it->second;
      out << jsonl::write_line(process_row(row, id)) << "\n";
    } catch (const std::exception& ex) {
      ++errors;
      out << jsonl::write_line(
                 {{"id", id}, {"status", "error"}, {"error", ex.what()}})
          << "\n";
    }
  }
  return errors;
}

}  // namespace ax::sym
