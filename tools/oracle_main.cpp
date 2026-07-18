/** axiom-oracle: JSONL verification-oracle harness for the llmopt parity
    audit. Usage: axiom-oracle <in.jsonl> <out.jsonl>  ('-' = stdin/stdout).
    Exit code 0 even when rows error — row failures are data, not crashes. */
#include <ax/sym/harness.hpp>

#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: axiom-oracle <in.jsonl|-> <out.jsonl|->\n";
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

  const int errors = ax::sym::run_oracle(in, out);
  std::cerr << "axiom-oracle: done, " << errors << " row error(s)\n";
  return 0;
}
