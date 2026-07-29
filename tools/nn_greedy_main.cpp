/** @file nn_greedy_main.cpp E3 paired-gate driver (relay 2026-07-29-2).
    Usage:
      axiom-nn-greedy <model.axnn> <prompts.txt> <expected.txt>
    prompts.txt / expected.txt: one row per line, space-separated token
    ids; row i of expected sets the emit length for prompt i. Output:
    one line of generated ids per row, then "E3 PASS 50/50"-style
    verdict (first divergence as "DIVERGE row=<r> pos=<p> got=<g>
    want=<w>") on stderr so stdout stays a clean id file. */
#include <ax/nn/exact.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::vector<std::vector<int>> read_rows(const char* path) {
  std::ifstream in(path);
  if (!in.good()) {
    std::cerr << "cannot open " << path << "\n";
    std::exit(1);
  }
  std::vector<std::vector<int>> rows;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    std::vector<int> row;
    int t;
    while (ss >> t) row.push_back(t);
    rows.push_back(std::move(row));
  }
  return rows;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: axiom-nn-greedy <model.axnn> <prompts.txt> "
                 "<expected.txt>\n";
    return 2;
  }
  const auto m = ax::nn::exact_model::load(argv[1]);
  const std::string cert = m.certify_tables();
  if (!cert.empty()) {
    std::cerr << "table certification failed: " << cert << "\n";
    return 1;
  }
  const auto prompts = read_rows(argv[2]);
  const auto expected = read_rows(argv[3]);
  if (prompts.size() != expected.size()) {
    std::cerr << "row count mismatch: " << prompts.size() << " prompts, "
              << expected.size() << " expected\n";
    return 1;
  }
  long long pass = 0;
  bool diverged = false;
  for (std::size_t r = 0; r < prompts.size(); ++r) {
    const auto gen =
        m.generate(prompts[r], static_cast<int>(expected[r].size()));
    for (std::size_t i = 0; i < gen.size(); ++i)
      std::cout << (i ? " " : "") << gen[i];
    std::cout << "\n";
    bool row_ok = gen.size() == expected[r].size();
    for (std::size_t i = 0; row_ok && i < gen.size(); ++i)
      row_ok = gen[i] == expected[r][i];
    if (row_ok) {
      ++pass;
    } else if (!diverged) {
      diverged = true;
      std::size_t p = 0;
      while (p < gen.size() && p < expected[r].size() &&
             gen[p] == expected[r][p])
        ++p;
      std::cerr << "DIVERGE row=" << r << " pos=" << p << " got="
                << (p < gen.size() ? std::to_string(gen[p]) : "<short>")
                << " want="
                << (p < expected[r].size() ? std::to_string(expected[r][p])
                                           : "<short>")
                << "\n";
    }
  }
  std::cerr << "E3 " << (pass == static_cast<long long>(prompts.size())
                             ? "PASS "
                             : "FAIL ")
            << pass << "/" << prompts.size() << "\n";
  return pass == static_cast<long long>(prompts.size()) ? 0 : 1;
}
