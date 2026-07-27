/** inverse-gate: S7 acceptance runner (relay 2026-07-27-0 ask 1).
    Usage:
      axiom-inverse-gate <roots.jsonl> [max_edges] [budget] [plies]
                         [width] [prior.tsv] [deadline_s]

    Builds replay-labeled edges the same way the farm does (beam solve +
    replay_chain, verify_p = 1), then for each edge (cur, rule, nxt)
    runs predecessors(nxt) and checks containment of the true (rule,
    cur). Reports containment rate, per-rule misses, and the wall
    distribution (median / p90 / max, ms). Acceptance bar: containment
    >= 95%, median wall <= 100 ms; every returned pair forward-verifies
    by construction (successors runs verify_edge). Exit 0 iff both bars
    hold. */
#include <ax/search/inverse.hpp>
#include <ax/sym/jsonl.hpp>
#include <ax/sym/parse.hpp>
#include <ax/sym/print_sstr.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  using namespace ax;
  if (argc < 2) {
    std::cerr << "usage: axiom-inverse-gate <roots.jsonl> [max_edges] "
                 "[budget] [plies] [width] [prior.tsv] [deadline_s]\n";
    return 2;
  }
  const long long max_edges = argc > 2 ? std::atoll(argv[2]) : 200;
  const long long budget = argc > 3 ? std::atoll(argv[3]) : 200;
  const int plies = argc > 4 ? std::atoi(argv[4]) : 24;
  const int width = argc > 5 ? std::atoi(argv[5]) : 3;
  std::optional<search::markov_prior> prior;
  if (argc > 6 && argv[6][0] != '\0' && std::string(argv[6]) != "-")
    prior = search::markov_prior::load_tsv(argv[6]);
  const int deadline_s = argc > 7 ? std::atoi(argv[7]) : 20;

  std::ifstream in(argv[1]);
  if (!in.good()) {
    std::cerr << "cannot open " << argv[1] << "\n";
    return 2;
  }
  const auto& rules = search::default_rules();

  long long edges = 0, contained = 0, root_errors = 0;
  std::vector<double> walls_ms;
  std::map<std::string, std::pair<long long, long long>> by_rule;  // hit, n
  std::string line;
  while (std::getline(in, line) && edges < max_edges) {
    if (line.empty()) continue;
    try {
      const auto row = sym::jsonl::parse_line(line);
      const sym::expr root = sym::parse(row.at("root"));
      search::beam_options opt;
      opt.deadline = std::chrono::steady_clock::now() +
                     std::chrono::seconds(deadline_s);
      opt.width = width;
      opt.max_plies = plies;
      opt.max_nodes = budget;
      opt.use_macros = true;
      if (prior) {
        opt.proposer = prior->proposer();
        opt.propose_k = 3;
      }
      const auto res = search::beam_search(root, rules, opt);
      if (!res.solved) continue;
      const auto chain =
          search::replay_chain(root, res.best.history, rules);
      if (!chain) continue;
      for (std::size_t i = 0; i + 1 < chain->size() && edges < max_edges;
           ++i) {
        const sym::expr& cur = (*chain)[i].e;
        const sym::expr& nxt = (*chain)[i + 1].e;
        const std::string& rule = res.best.history[i];
        const auto t0 = std::chrono::steady_clock::now();
        const auto preds = search::predecessors(nxt, rules);
        const double ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0)
                .count();
        walls_ms.push_back(ms);
        bool hit = false;
        for (const auto& pr : preds)
          hit = hit || (pr.rule == rule && pr.p.same(cur));
        ++edges;
        auto& [h, n] = by_rule[rule];
        ++n;
        if (hit) {
          ++h;
          ++contained;
        } else {
          std::cerr << "[miss] " << rule << " :: cur "
                    << sym::to_sstr(cur) << " :: nxt " << sym::to_sstr(nxt)
                    << "\n";
        }
      }
    } catch (const std::exception& ex) {
      ++root_errors;
      std::cerr << "[root-error] " << ex.what() << "\n";
    }
  }

  if (edges == 0) {
    std::cerr << "no edges produced\n";
    return 2;
  }
  std::sort(walls_ms.begin(), walls_ms.end());
  const auto pct = [&](double q) {
    return walls_ms[std::min(walls_ms.size() - 1,
                             static_cast<std::size_t>(
                                 q * static_cast<double>(walls_ms.size())))];
  };
  const double rate =
      100.0 * static_cast<double>(contained) / static_cast<double>(edges);
  std::cout << "edges " << edges << "  contained " << contained << " ("
            << rate << "%)  wall median " << pct(0.5) << " ms  p90 "
            << pct(0.9) << " ms  max " << walls_ms.back() << " ms  "
            << "root_errors " << root_errors << "\n";
  for (const auto& [rule, hn] : by_rule)
    std::cout << "  " << rule << " " << hn.first << "/" << hn.second
              << "\n";
  const bool pass = rate >= 95.0 && pct(0.5) <= 100.0;
  std::cout << (pass ? "PASS" : "FAIL") << "\n";
  return pass ? 0 : 1;
}
