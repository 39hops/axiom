/** poly-chain: poly-algebra pilot emitter (llmopt GO 2026-07-23).
    Usage:
      axiom-poly-chain <out.jsonl> [seeds_per_cell]
    Families poly_gcd (Euclidean division chains) and poly_pf (partial
    fractions over distinct linear factors), 3 levels each, existing
    vocab. Row kinds and certification: see polychain.hpp. Problems
    whose rows fail certification are written with verdict UNDECIDED
    and their error, never dropped. */
#include <ax/mathgen/polychain.hpp>
#include <ax/sym/jsonl.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
  using namespace ax;
  if (argc < 2) {
    std::cerr << "usage: axiom-poly-chain <out.jsonl> [seeds_per_cell]\n";
    return 2;
  }
  const long long seeds = argc > 2 ? std::atoll(argv[2]) : 20;
  std::ofstream out(argv[1]);
  if (!out.good()) {
    std::cerr << "cannot open " << argv[1] << "\n";
    return 2;
  }
  long long problems = 0, ok = 0, rows = 0;
  for (int fam = 0; fam < 2; ++fam)
    for (int level = 1; level <= 3; ++level)
      for (long long seed = 0; seed < seeds; ++seed) {
        const mathgen::pchain_problem p =
            fam == 0 ? mathgen::make_gcd_chain(level, seed)
                     : mathgen::make_pf_chain(level, seed);
        ++problems;
        if (p.certified) ++ok;
        int n = 0;
        for (const auto& r : p.rows) {
          out << "{\"family\": \"" << p.family
              << "\", \"level\": " << p.level << ", \"seed\": " << seed
              << ", \"n\": " << n++ << ", \"kind\": \"" << r.kind
              << "\", \"cur\": \"" << sym::jsonl::escape(r.cur)
              << "\", \"nxt\": \"" << sym::jsonl::escape(r.nxt)
              << "\", \"source\": \"axiom-poly-chain\", \"verdict\": \""
              << (p.certified ? "CERTIFIED" : "UNDECIDED") << "\"";
          if (!p.certified)
            out << ", \"error\": \"" << sym::jsonl::escape(p.error) << "\"";
          out << "}\n";
          ++rows;
        }
      }
  std::cerr << "== " << problems << " problems, " << ok << " certified, "
            << rows << " rows\n";
  return problems == ok ? 0 : 1;
}
