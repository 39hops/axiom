/** @file nn_moe_greedy_main.cpp merged-crystal greedy driver (relay
    2026-07-31 ask). Float AXNN path (double accumulation), not the
    exact integer path — the acceptance bar is token-identical greedy
    streams against the torch merged model.
    Usage:
      axiom-nn-moe-greedy <model.axnn> <prompts.txt> <n_new>
    prompts.txt: one row per line, space-separated token ids. Output:
    one line of generated ids per row on stdout. */
#include <ax/nn/model.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: axiom-nn-moe-greedy <model.axnn> <prompts.txt> "
                 "<n_new>\n";
    return 2;
  }
  const auto m = ax::nn::model::load(argv[1]);
  std::ifstream in(argv[2]);
  if (!in.good()) {
    std::cerr << "cannot open " << argv[2] << "\n";
    return 1;
  }
  const int n_new = std::atoi(argv[3]);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    std::vector<int> ids;
    int t;
    while (ss >> t) ids.push_back(t);
    for (int j = 0; j < n_new; ++j) {
      const auto lg = m.logits_last(ids);
      int best = 0;
      for (std::size_t v = 1; v < lg.size(); ++v)
        if (lg[v] > lg[best]) best = static_cast<int>(v);
      ids.push_back(best);
      std::cout << best << (j + 1 == n_new ? "" : " ");
    }
    std::cout << "\n";
  }
  return 0;
}
