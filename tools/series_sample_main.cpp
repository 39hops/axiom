/** series-sample: L10 farm-gate emitter for the series ODE method.
    Usage:
      axiom-series-sample <out.jsonl> [seeds_per_cell] [order]
    For each certified L9b row (3 families x 3 levels x N seeds): run the
    coefficient recurrence to the truncation order and emit ONE CHAIN ROW
    PER COEFFICIENT STEP (a_{n+k} from predecessors, exact rationals).
    Adjudication contract: "a_n" is byte-exact vs sympy's series()
    coefficients; cur/nxt are partial-sum sstr + textual O() marker
    (training-row shape). Every problem is verified two independent ways
    before its rows are written — recurrence coefficients must equal the
    Maclaurin coefficients of the drawn solution, and the residual oracle
    (check_odesol_series) must return EQUIVALENT_TO_ORDER. A failing
    problem is written as a single row with its honest verdict, never
    dropped silently. */
#include <ax/mathgen/ode.hpp>
#include <ax/mathgen/series_solve.hpp>
#include <ax/sym/jsonl.hpp>
#include <ax/sym/print_sstr.hpp>
#include <ax/sym/series_oracle.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::string partial_sstr(const std::vector<ax::rational>& a, int upto,
                         const ax::sym::expr& x) {
  const ax::sym::series s(
      std::vector<ax::rational>(a.begin(), a.begin() + upto), upto);
  return ax::sym::to_sstr(s.to_expr(x)) + " + O(x**" +
         std::to_string(upto) + ")";
}

}  // namespace

int main(int argc, char** argv) {
  using namespace ax;
  if (argc < 2) {
    std::cerr << "usage: axiom-series-sample <out.jsonl> [seeds_per_cell] "
                 "[order]\n";
    return 2;
  }
  const long long seeds = argc > 2 ? std::atoll(argv[2]) : 20;
  const int order = argc > 3 ? std::atoi(argv[3]) : 8;
  std::ofstream out(argv[1]);
  if (!out.good()) {
    std::cerr << "cannot open " << argv[1] << "\n";
    return 2;
  }
  const sym::expr x = sym::expr::symbol("x");
  long long problems = 0, ok = 0, rows = 0;
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
        ++problems;
        const std::string head =
            "{\"family\": \"" + f + "\", \"level\": " +
            std::to_string(level) + ", \"seed\": " + std::to_string(seed) +
            ", \"order\": " + std::to_string(order);
        std::string verdict = "UNDECIDED";
        int residual_order = 0;
        try {
          const auto sol = mathgen::series_solve(p, order);
          const auto chk = sym::check_odesol_series(p.eq, sol.y, x);
          residual_order = chk.order;
          const bool coeffs_match =
              sol.y == sym::series::of_expr(p.sol, x, order);
          verdict =
              chk.v == sym::series_verdict::equivalent_to_order
                  ? (coeffs_match ? "EQUIVALENT_TO_ORDER" : "MISMATCH")
                  : chk.v == sym::series_verdict::not_equivalent
                        ? "NOT_EQUIVALENT"
                        : "UNDECIDED";
          for (const auto& st : sol.steps) {
            out << head << ", \"ode_order\": " << sol.ode_order
                << ", \"n\": " << st.n << ", \"a_n\": \""
                << st.a_n.to_string() << "\", \"cur\": \""
                << sym::jsonl::escape(
                       partial_sstr(sol.y.coeffs(), st.n, x))
                << "\", \"nxt\": \""
                << sym::jsonl::escape(
                       partial_sstr(sol.y.coeffs(), st.n + 1, x))
                << "\", \"source\": \"axiom-series\", \"verdict\": \""
                << verdict << "\", \"residual_order\": " << residual_order
                << "}\n";
            ++rows;
          }
        } catch (const std::exception& ex) {
          out << head << ", \"ode_order\": null, \"n\": null, \"a_n\": "
              << "null, \"cur\": null, \"nxt\": null, \"source\": "
              << "\"axiom-series\", \"verdict\": \"UNDECIDED\", "
              << "\"residual_order\": 0, \"error\": \""
              << sym::jsonl::escape(ex.what()) << "\"}\n";
          ++rows;
        }
        if (verdict == "EQUIVALENT_TO_ORDER") ++ok;
      }
  std::cerr << "== " << problems << " problems, " << ok
            << " EQUIVALENT_TO_ORDER (coeffs cross-checked), " << rows
            << " rows\n";
  return 0;
}
