/** nn-exact: FX-V1 exact inference CLI (the cross-platform hash
    surface). Usage:
      axiom-nn-exact <model.axnn> <prompts.txt>
    prompts.txt: one prompt per line, space-separated token ids.
    Output: one line per prompt: "<fnv1a64-hex> <argmax>", then a
    final line "BATTERY <fnv1a64-hex>" over the per-prompt hashes in
    file order. Bit-identical across platforms by construction. */
#include <ax/nn/exact.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: axiom-nn-exact <model.axnn> <prompts.txt>\n";
    return 2;
  }
  try {
    const auto m = ax::nn::exact_model::load(argv[1]);
    std::ifstream in(argv[2]);
    if (!in.good()) {
      std::cerr << "cannot open " << argv[2] << "\n";
      return 2;
    }
    std::uint64_t battery = 14695981039346656037ull;
    std::string line;
    char buf[64];
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      std::istringstream ss(line);
      std::vector<int> toks;
      int t;
      while (ss >> t) toks.push_back(t);
      const std::uint64_t h = m.logits_hash(toks);
      for (int i = 0; i < 8; ++i) {
        battery ^= (h >> (8 * i)) & 0xff;
        battery *= 1099511628211ull;
      }
      std::snprintf(buf, sizeof buf, "%016llx %d",
                    static_cast<unsigned long long>(h), m.argmax(toks));
      std::cout << buf << "\n";
    }
    std::snprintf(buf, sizeof buf, "BATTERY %016llx",
                  static_cast<unsigned long long>(battery));
    std::cout << buf << "\n";
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
  return 0;
}
