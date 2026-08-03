/** axiom-oracle: JSONL verification-oracle harness for the llmopt parity
    audit. Usage: axiom-oracle <in.jsonl> <out.jsonl> [--lean-cert <path>]
    ('-' = stdin/stdout). Exit code 0 even when rows error — row failures
    are data, not crashes. --lean-cert writes one certificate sidecar line
    per eligible EQUIVALENT verdict and reports emitted/fenced counts. */
#include <ax/sym/harness.hpp>

#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc != 3 && !(argc == 5 && std::string(argv[3]) == "--lean-cert")) {
    std::cerr << "usage: axiom-oracle <in.jsonl|-> <out.jsonl|->"
                 " [--lean-cert <path>]\n";
    return 2;
  }
  const std::string in_path = argv[1];
  const std::string out_path = argv[2];

  std::ifstream fin;
  if (in_path != "-") {
    fin.open(in_path);
    if (!fin) {
      std::cerr << "axiom-oracle: cannot open " << in_path << "\n";
      return 2;
    }
  }
  std::ofstream fout;
  if (out_path != "-") {
    fout.open(out_path);
    if (!fout) {
      std::cerr << "axiom-oracle: cannot open " << out_path << "\n";
      return 2;
    }
  }
  std::istream& in = in_path == "-" ? std::cin : fin;
  std::ostream& out = out_path == "-" ? std::cout : fout;

  std::ofstream flean;
  ax::sym::lean_stats stats;
  if (argc == 5) {
    flean.open(argv[4]);
    if (!flean) {
      std::cerr << "axiom-oracle: cannot open " << argv[4] << "\n";
      return 2;
    }
  }
  const int errors = ax::sym::run_oracle(in, out, argc == 5 ? &flean : nullptr,
                                         argc == 5 ? &stats : nullptr);
  std::cerr << "axiom-oracle: done, " << errors << " row error(s)\n";
  if (argc == 5)
    std::cerr << "axiom-oracle: lean certs " << stats.emitted << " emitted, "
              << stats.fenced << " fenced\n";
  return 0;
}
