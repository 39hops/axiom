#include <ax/sym/harness.hpp>

#include <ax/sym/calc.hpp>
#include <ax/sym/jsonl.hpp>
#include <ax/sym/oracle.hpp>
#include <ax/sym/parse.hpp>
#include <ax/sym/print.hpp>
#include <ax/sym/print_lean.hpp>

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

/** Emit a certificate sidecar line when the EQUIVALENT verdict falls in
    the eligible subset; count fenced rows either way. */
void emit_lean(const std::string& id, const std::string& lhs,
               const std::string& rhs, const expr& x, std::ostream* lean_out,
               lean_stats* stats) {
  if (!lean_out) return;
  const lean_cert c = to_lean(lhs, rhs, x);
  if (!c.eligible) {
    if (stats) ++stats->fenced;
    return;
  }
  *lean_out << sidecar_line(id, c) << "\n";
  if (stats) ++stats->emitted;
}

jsonl::fields process_row(const jsonl::object& row, const std::string& id,
                          std::ostream* lean_out, lean_stats* stats) {
  const std::string& task = field(row, "task");
  const expr x = expr::symbol(field(row, "var"));
  if (task == "diff") {
    const expr d = diff(parse(field(row, "expr")), x);
    return {{"id", id}, {"status", "ok"}, {"result", to_sstr(d)}};
  }
  if (task == "equiv") {
    const std::string& lhs = field(row, "lhs");
    const std::string& rhs = field(row, "rhs");
    const verdict v = equivalent(parse(lhs), parse(rhs), x);
    if (v == verdict::equivalent) emit_lean(id, lhs, rhs, x, lean_out, stats);
    return {{"id", id}, {"status", "ok"}, {"verdict", verdict_name(v)}};
  }
  if (task == "equiv_mod_const") {
    const expr candidate = parse(field(row, "candidate"));
    const verdict v =
        equivalent_mod_const(candidate, parse(field(row, "integrand")), x);
    // The verdict is equivalent(diff(candidate, x), integrand); certify
    // that derived identity — diff's output is what the judge compared.
    if (v == verdict::equivalent)
      emit_lean(id, to_sstr(diff(candidate, x)), field(row, "integrand"), x,
                lean_out, stats);
    return {{"id", id}, {"status", "ok"}, {"verdict", verdict_name(v)}};
  }
  throw std::runtime_error("unknown task '" + task + "'");
}

}  // namespace

int run_oracle(std::istream& in, std::ostream& out, std::ostream* lean_out,
               lean_stats* stats) {
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
      out << jsonl::write_line(process_row(row, id, lean_out, stats)) << "\n";
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
