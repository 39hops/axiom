/** phys-chain: physics rung 1 emitter (llmopt GO 2026-07-23).
    Usage:
      axiom-phys-chain <out.jsonl> [seeds_per_cell] [order] [family]
    Families phys_kin (a(t) -> v(t) -> x(t) with ICs), phys_shm
    (y'' + w^2 y = 0, the certified cc2 recurrence relabeled), and
    phys_energy (rung 2: conservation as vanishing arithmetic), emitted
    with t throughout (vocab-41; the math-expert rename t->x is the
    router's side of the interface). Certification: see physchain.hpp.
    Failing problems are written with verdict UNDECIDED, never
    dropped. */
#include <ax/mathgen/physchain.hpp>
#include <ax/sym/jsonl.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
  using namespace ax;
  if (argc < 2) {
    std::cerr << "usage: axiom-phys-chain <out.jsonl> [seeds_per_cell] "
                 "[order]\n";
    return 2;
  }
  const long long seeds = argc > 2 ? std::atoll(argv[2]) : 20;
  const int order = argc > 3 ? std::atoi(argv[3]) : 8;
  std::ofstream out(argv[1]);
  if (!out.good()) {
    std::cerr << "cannot open " << argv[1] << "\n";
    return 2;
  }
  long long problems = 0, ok = 0, rows = 0;
  const char* only = argc > 4 ? argv[4] : nullptr;  // family filter
  for (int fam = 0; fam < 3; ++fam)
    for (int level = 1; level <= 3; ++level)
      for (long long seed = 0; seed < seeds; ++seed) {
        const mathgen::pchain_problem p =
            fam == 0   ? mathgen::make_kin_chain(level, seed)
            : fam == 1 ? mathgen::make_shm_chain(level, seed, order)
                       : mathgen::make_energy_chain(level, seed, order);
        if (only && p.family != only) continue;
        ++problems;
        if (p.certified) ++ok;
        int n = 0;
        for (const auto& r : p.rows) {
          out << "{\"family\": \"" << p.family
              << "\", \"level\": " << p.level << ", \"seed\": " << seed
              << ", \"n\": " << n++ << ", \"kind\": \"" << r.kind
              << "\", \"cur\": \"" << sym::jsonl::escape(r.cur)
              << "\", \"nxt\": \"" << sym::jsonl::escape(r.nxt)
              << "\", \"source\": \"axiom-phys-chain\", \"verdict\": \""
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
