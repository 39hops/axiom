/** series-chain: L10 rung 6 emitter — coefficient derivation as explicit
    certified rewrite rows (llmopt relay 2026-07-22: single-shot arithmetic
    is simulation; decomposed steps train up).
    Usage:
      axiom-series-chain <out.jsonl> [seeds_per_cell] [order] [cc2_seeds]
    For each certified L9b problem, every recurrence step a_m emits rows,
    all inside the vocab-40 charset (digits x + - * / ( ) , space; no
    letters beyond function atoms, no '='). Rung 7: the recurrence sum is
    a pairwise reduction tree, one primitive per emission:
      kind "mul"    cur = one binary product        nxt = its value
      kind "add"    cur = one binary addition       nxt = its value
      kind "solve"  cur = (q_n - S)/divisor, everything a literal
                    nxt = a_m
      kind "append" cur = partial sum through a_{m-1} + O() marker
                    nxt = partial sum through a_m   + O() marker
    cc2_seeds (default = seeds_per_cell) oversamples the ode_cc2 family
    where two-back placement lags.
    Certification: each arithmetic row's cur is re-parsed by the engine
    (sym::parse -> canonical) and must fold byte-exactly to nxt; nxt is
    printed through to_sstr(expr::num(...)) so it IS the canonical
    spelling. Problem-level verification is unchanged from series-sample:
    recurrence coefficients must equal the Maclaurin coefficients of the
    drawn solution AND the residual oracle must say EQUIVALENT_TO_ORDER.
    Failures are written with honest verdicts, never dropped. */
#include <ax/mathgen/ode.hpp>
#include <ax/mathgen/series_chain.hpp>
#include <ax/mathgen/series_solve.hpp>
#include <ax/sym/jsonl.hpp>
#include <ax/sym/parse.hpp>
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

/** Engine certification of an arithmetic row: cur must parse and fold
    byte-exactly to nxt (which is already canonical sstr). */
bool folds_to(const std::string& cur, const std::string& nxt) {
  try {
    return ax::sym::to_sstr(ax::sym::parse(cur)) == nxt;
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace

int main(int argc, char** argv) {
  using namespace ax;
  if (argc < 2) {
    std::cerr << "usage: axiom-series-chain <out.jsonl> [seeds_per_cell] "
                 "[order]\n";
    return 2;
  }
  const long long seeds = argc > 2 ? std::atoll(argv[2]) : 20;
  const int order = argc > 3 ? std::atoi(argv[3]) : 8;
  const long long cc2_seeds = argc > 4 ? std::atoll(argv[4]) : seeds;
  std::ofstream out(argv[1]);
  if (!out.good()) {
    std::cerr << "cannot open " << argv[1] << "\n";
    return 2;
  }
  const sym::expr x = sym::expr::symbol("x");
  long long problems = 0, ok = 0, rows = 0, cert_fail = 0;
  const char* fams[] = {"ode_linear1", "ode_cc2", "ode_separable"};
  for (const char* fam : fams)
    for (int level = 1; level <= 3; ++level) {
      const std::string f(fam);
      const long long fam_seeds = f == "ode_cc2" ? cc2_seeds : seeds;
      for (long long seed = 0; seed < fam_seeds; ++seed) {
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
          const auto emit = [&](int n, const char* kind,
                                const rational& a_n, const std::string& cur,
                                const std::string& nxt, bool certified) {
            if (!certified) ++cert_fail;
            out << head << ", \"ode_order\": " << sol.ode_order
                << ", \"n\": " << n << ", \"kind\": \"" << kind
                << "\", \"a_n\": \"" << a_n.to_string() << "\", \"cur\": \""
                << sym::jsonl::escape(cur) << "\", \"nxt\": \""
                << sym::jsonl::escape(nxt)
                << "\", \"source\": \"axiom-series-chain\", \"verdict\": \""
                << (certified ? verdict : "MISMATCH")
                << "\", \"residual_order\": " << residual_order << "}\n";
            ++rows;
          };
          for (const auto& st : sol.steps) {
            const auto d = mathgen::derivation_rows(st);
            for (const auto& r : d.reduction)
              emit(st.n, r.kind.c_str(), st.a_n, r.cur, r.nxt,
                   folds_to(r.cur, r.nxt));
            emit(st.n, "solve", st.a_n, d.solve_cur, d.solve_nxt,
                 folds_to(d.solve_cur, d.solve_nxt));
            emit(st.n, "append", st.a_n,
                 partial_sstr(sol.y.coeffs(), st.n, x),
                 partial_sstr(sol.y.coeffs(), st.n + 1, x), true);
          }
        } catch (const std::exception& ex) {
          out << head << ", \"ode_order\": null, \"n\": null, \"kind\": "
              << "null, \"a_n\": null, \"cur\": null, \"nxt\": null, "
              << "\"source\": \"axiom-series-chain\", \"verdict\": "
              << "\"UNDECIDED\", \"residual_order\": 0, \"error\": \""
              << sym::jsonl::escape(ex.what()) << "\"}\n";
          ++rows;
        }
        if (verdict == "EQUIVALENT_TO_ORDER") ++ok;
      }
    }
  std::cerr << "== " << problems << " problems, " << ok
            << " EQUIVALENT_TO_ORDER, " << rows << " rows, " << cert_fail
            << " cert failures\n";
  return cert_fail == 0 ? 0 : 1;
}
