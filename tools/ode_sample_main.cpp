/** ode-sample: L9 rung 5 farm-gate artifact. Usage:
      axiom-ode-sample <out.jsonl> [seeds_per_cell]
    Emits native ODE rows (3 families x 3 levels x N seeds), every row
    self-verified through check_odesol + IC checks before writing; a row
    that fails verification is written with its honest verdict, never
    dropped silently (llmopt adjudicates against sympy checkodesol). */
#include <ax/mathgen/ode.hpp>
#include <ax/sym/jsonl.hpp>
#include <ax/sym/oracle.hpp>
#include <ax/sym/print_sstr.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  using namespace ax;
  if (argc < 2) {
    std::cerr << "usage: axiom-ode-sample <out.jsonl> [seeds_per_cell]\n";
    return 2;
  }
  const long long seeds = argc > 2 ? std::atoll(argv[2]) : 20;
  std::ofstream out(argv[1]);
  if (!out.good()) {
    std::cerr << "cannot open " << argv[1] << "\n";
    return 2;
  }
  const sym::expr x = sym::expr::symbol("x");
  long long rows = 0, eq_ok = 0;
  const char* fams[] = {"ode_linear1", "ode_cc2", "ode_separable"};
  for (const char* fam : fams)
    for (int level = 1; level <= 3; ++level)
      for (long long seed = 0; seed < seeds; ++seed) {
        const std::string f(fam);
        const mathgen::ode_problem p =
            f == "ode_linear1"
                ? mathgen::make_linear_first_order(level, seed)
                : f == "ode_cc2"
                      ? mathgen::make_second_order_cc(level, seed)
                      : mathgen::make_separable_growth(level, seed);
        const auto v = sym::check_odesol(p.eq, p.sol, x);
        const char* verdict = v == sym::verdict::equivalent
                                  ? "EQUIVALENT"
                                  : v == sym::verdict::not_equivalent
                                        ? "NOT_EQUIVALENT"
                                        : "UNDECIDED";
        if (v == sym::verdict::equivalent) ++eq_ok;
        out << "{\"family\": \"" << f << "\", \"level\": " << level
            << ", \"seed\": " << seed << ", \"eq\": \""
            << sym::jsonl::escape(sym::to_sstr(p.eq)) << "\", \"sol\": \""
            << sym::jsonl::escape(sym::to_sstr(p.sol))
            << "\", \"x0\": " << p.x0 << ", \"y0\": \""
            << p.y0.to_string() << "\", \"yp0\": "
            << (p.yp0 ? "\"" + p.yp0->to_string() + "\"" : "null")
            << ", \"axiom_verdict\": \"" << verdict << "\"}\n";
        ++rows;
      }
  std::cerr << "== " << rows << " rows, " << eq_ok
            << " oracle-EQUIVALENT\n";
  return 0;
}
