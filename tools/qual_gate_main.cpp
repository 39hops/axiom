/** qual-gate: run the native solver over qualification roots and report
    per-level solve counts + per-root timing. Usage:
      axiom-qual-gate <roots.jsonl> [max_level] [budget] [plies] [width]
    Every reported solve is oracle-valid: diff-back equivalence + full
    replay verification. Chains and misses stream to stdout as TSV. */
#include <ax/search/search.hpp>
#include <ax/sym/calc.hpp>
#include <ax/sym/jsonl.hpp>
#include <ax/sym/oracle.hpp>
#include <ax/sym/parse.hpp>
#include <ax/sym/print_sstr.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

int main(int argc, char** argv) {
  using namespace ax;
  if (argc < 2) {
    std::cerr << "usage: axiom-qual-gate <roots.jsonl> [max_level] [budget] [plies] [width]\n";
    return 2;
  }
  const int max_level = argc > 2 ? std::atoi(argv[2]) : 8;
  const long long budget = argc > 3 ? std::atoll(argv[3]) : 200;
  const int plies = argc > 4 ? std::atoi(argv[4]) : 12;
  const int width = argc > 5 ? std::atoi(argv[5]) : 8;

  std::ifstream in(argv[1]);
  if (!in.good()) {
    std::cerr << "cannot open " << argv[1] << "\n";
    return 2;
  }
  const sym::expr x = sym::expr::symbol("x");
  std::map<int, int> solved, total;
  std::string line;
  double wall_total = 0.0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const auto row = sym::jsonl::parse_line(line);
    const int level = std::stoi(row.at("level"));
    if (level > max_level) continue;
    ++total[level];
    const sym::expr root = sym::parse(row.at("root"));
    search::beam_options opt;
    opt.width = width;
    opt.max_plies = plies;
    opt.max_nodes = budget;
    opt.use_macros = true;
    const auto t0 = std::chrono::steady_clock::now();
    const auto res = search::beam_search(root, search::default_rules(), opt);
    const double dt =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    wall_total += dt;
    bool ok = res.solved;
    std::string why = ok ? "solved" : "unsolved";
    if (ok) {
      const sym::expr integrand = root.args()[0];
      const auto v = sym::equivalent(sym::diff(res.best.e, x), integrand, x);
      if (v == sym::verdict::not_equivalent) {
        ok = false;
        why = "DIFFBACK-NOTEQ";  // a real soundness bug if it ever fires
      } else if (v == sym::verdict::undecided) {
        ok = false;
        why = "DIFFBACK-UNDECIDED";  // oracle incompleteness, not unsound
      } else if (!search::replay_verify(root, res.best.history,
                                        search::default_rules())) {
        ok = false;
        why = "REPLAY-FAIL";
      }
    }
    if (ok) ++solved[level];
    std::cout << row.at("id") << "\t" << level << "\t" << why << "\t"
              << res.nodes << "\t" << dt << "\t"
              << (res.solved ? sym::to_sstr(res.best.e) : "-") << "\n";
  }
  std::cerr << "== per-level (budget " << budget << ", wall "
            << wall_total << "s):\n";
  for (const auto& [lvl, tot] : total)
    std::cerr << "  L" << lvl << ": " << solved[lvl] << "/" << tot << "\n";
  return 0;
}
