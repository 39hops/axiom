/** nn-logits: AXNN micro-inference CLI (cross-check harness surface).
    Usage:
      axiom-nn-logits <model.axnn> <prompts.txt>
    prompts.txt: one prompt per line, space-separated token ids.
    Output: one line per prompt — last-position logits, %.9e,
    space-separated (the scorer/readout shape the acceptance gate
    compares against torch). */
#include <ax/nn/model.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: axiom-nn-logits <model.axnn> <prompts.txt>\n";
    return 2;
  }
  try {
    const auto m = ax::nn::model::load(argv[1]);
    std::ifstream in(argv[2]);
    if (!in.good()) {
      std::cerr << "cannot open " << argv[2] << "\n";
      return 2;
    }
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      std::istringstream ss(line);
      std::vector<int> toks;
      int t;
      while (ss >> t) toks.push_back(t);
      const auto lg = m.logits_last(toks);
      std::string out;
      char buf[32];
      for (std::size_t i = 0; i < lg.size(); ++i) {
        std::snprintf(buf, sizeof buf, "%.9e", lg[i]);
        if (i) out += ' ';
        out += buf;
      }
      std::cout << out << "\n";
    }
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
  return 0;
}
