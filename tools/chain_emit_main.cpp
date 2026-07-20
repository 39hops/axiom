/** chain-emit: Phase D farm-shard chain emission. Usage:
      axiom-chain-emit <roots.jsonl> <out.jsonl> [budget] [plies] [width]
                       [prior.tsv] [deadline_s] [max_roots]
    For each solved root: replay the winning history recording the state
    chain, annotate every step (hints = rule-fire syndrome on the largest
    Integral node; think = ansatz verbalized derivation), and write rows
    in llmopt's farm_v22 shard schema:
      {"cur", "nxt", "level", "source": "axiom-chain"|"axiom-oneply",
       "hints": [...], "think": ...|null}
    Every (cur, nxt) pair is re-verified through the oracle before being
    written (diff-back mod const on Integral edges — same soundness bar
    as the gate); a pair that fails is dropped and counted. */
#include <ax/search/search.hpp>
#include <ax/sym/calc.hpp>
#include <ax/sym/jsonl.hpp>
#include <ax/sym/oracle.hpp>
#include <ax/sym/parse.hpp>
#include <ax/sym/print_sstr.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <set>
#include <string>

int main(int argc, char** argv) {
  using namespace ax;
  if (argc < 3) {
    std::cerr << "usage: axiom-chain-emit <roots.jsonl> <out.jsonl> "
                 "[budget] [plies] [width] [prior.tsv] [deadline_s] "
                 "[max_roots]\n";
    return 2;
  }
  const long long budget = argc > 3 ? std::atoll(argv[3]) : 200;
  const int plies = argc > 4 ? std::atoi(argv[4]) : 24;
  const int width = argc > 5 ? std::atoi(argv[5]) : 3;
  std::optional<search::markov_prior> prior;
  if (argc > 6 && argv[6][0] != '\0' && std::string(argv[6]) != "-")
    prior = search::markov_prior::load_tsv(argv[6]);
  const int deadline_s = argc > 7 ? std::atoi(argv[7]) : 20;
  const long long max_roots = argc > 8 ? std::atoll(argv[8]) : -1;

  std::ifstream in(argv[1]);
  std::ofstream out(argv[2]);
  if (!in.good() || !out.good()) {
    std::cerr << "cannot open " << argv[1] << " / " << argv[2] << "\n";
    return 2;
  }
  const sym::expr x = sym::expr::symbol("x");
  const auto& rules = search::default_rules();
  std::set<std::pair<std::string, std::string>> seen;
  long long emitted_roots = 0, rows = 0, dropped_pairs = 0,
            replay_failures = 0;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (max_roots >= 0 && emitted_roots >= max_roots) break;
    const auto row = sym::jsonl::parse_line(line);
    const int level = std::stoi(row.at("level"));
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
    const auto chain = search::replay_chain(root, res.best.history, rules);
    if (!chain) {
      ++replay_failures;  // booked, never silently skipped
      std::cerr << "[replay-failure] " << row.at("id") << "\n";
      continue;
    }
    const bool oneply = res.best.history.size() == 1;
    const char* source = oneply ? "axiom-oneply" : "axiom-chain";
    bool any = false;
    for (std::size_t i = 0; i + 1 < chain->size(); ++i) {
      const sym::expr& cur = (*chain)[i].e;
      const sym::expr& nxt = (*chain)[i + 1].e;
      // oracle re-verification of the emitted pair (training rows share
      // the search's soundness bar; a rejected pair is dropped, counted)
      if (!search::verify_edge(cur, nxt, rules.external)) {
        ++dropped_pairs;
        continue;
      }
      const std::string cur_s = sym::to_sstr(cur);
      const std::string nxt_s = sym::to_sstr(nxt);
      if (!seen.insert({cur_s, nxt_s}).second) continue;
      const std::string& label = res.best.history[i];
      const auto ann = search::annotate(cur, rules,
                                        label.substr(0, label.find('@')));
      std::string j = "{\"cur\": \"" + sym::jsonl::escape(cur_s) +
                      "\", \"nxt\": \"" + sym::jsonl::escape(nxt_s) +
                      "\", \"level\": " + std::to_string(level) +
                      ", \"source\": \"" + source + "\", \"hints\": [";
      for (std::size_t h = 0; h < ann.hints.size(); ++h) {
        if (h) j += ", ";
        j += "\"" + sym::jsonl::escape(ann.hints[h]) + "\"";
      }
      j += "], \"think\": ";
      j += ann.think ? "\"" + sym::jsonl::escape(*ann.think) + "\""
                     : "null";
      j += "}";
      out << j << "\n";
      ++rows;
      any = true;
    }
    if (any) ++emitted_roots;
  }
  std::cerr << "== emitted " << rows << " rows from " << emitted_roots
            << " roots; dropped pairs " << dropped_pairs
            << "; replay failures " << replay_failures << "\n";
  return 0;
}
