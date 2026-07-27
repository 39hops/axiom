/** nt-chain: number-theory chain emitter (llmopt relay 2026-07-27 -9,
    priority 1 — exact-stack infrastructure: CRT/modular is the RNS
    exact-GEMM machinery, continued fractions the best-rational snap).
    Usage:
      axiom-nt-chain <out.jsonl> [seeds_per_cell] [family]
    Families nt_gcd, nt_bezout, nt_modinv, nt_crt, nt_modexp, nt_cf,
    3 levels each. Row schema (farm_v22 style):
      {"family", "level", "seed", "n", "kind", "cur", "nxt",
       "source": "axiom-nt-chain", "verdict": CERTIFIED | UNDECIDED}
    Certification: every row re-evaluated exactly from its emitted
    strings (nt_eval), problem cross-checked against core/nt; failing
    problems are written UNDECIDED with an error, never dropped. Rows
    stream incrementally (killed workers keep their rows).
    VOCAB_EXTRA (atom list, per the serialization-spec fence): digits,
    '+', '-', '*', '**', '(', ')', ',', ' ', names "gcd", "Mod". A
    sidecar <out>.config.json records atoms, families, and counts. */
#include <ax/mathgen/ntchain.hpp>
#include <ax/sym/jsonl.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  using namespace ax;
  if (argc < 2) {
    std::cerr << "usage: axiom-nt-chain <out.jsonl> [seeds_per_cell] "
                 "[family]\n";
    return 2;
  }
  const long long seeds = argc > 2 ? std::atoll(argv[2]) : 40;
  const char* only = argc > 3 ? argv[3] : nullptr;
  std::ofstream out(argv[1]);
  if (!out.good()) {
    std::cerr << "cannot open " << argv[1] << "\n";
    return 2;
  }
  using maker = mathgen::pchain_problem (*)(int, long long);
  const maker makers[] = {
      mathgen::make_nt_gcd_chain,    mathgen::make_nt_bezout_chain,
      mathgen::make_nt_modinv_chain, mathgen::make_nt_crt_chain,
      mathgen::make_nt_modexp_chain, mathgen::make_nt_cf_chain};
  long long problems = 0, ok = 0, rows = 0;
  for (const maker mk : makers)
    for (int level = 1; level <= 3; ++level)
      for (long long seed = 0; seed < seeds; ++seed) {
        const mathgen::pchain_problem p = mk(level, seed);
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
              << "\", \"source\": \"axiom-nt-chain\", \"verdict\": \""
              << (p.certified ? "CERTIFIED" : "UNDECIDED") << "\"";
          if (!p.certified)
            out << ", \"error\": \"" << sym::jsonl::escape(p.error) << "\"";
          out << "}\n";
          ++rows;
        }
        out.flush();  // incremental-stream fence
      }
  std::ofstream cfg(std::string(argv[1]) + ".config.json");
  cfg << "{\"tool\": \"axiom-nt-chain\", \"seeds_per_cell\": " << seeds
      << ", \"families\": [\"nt_gcd\", \"nt_bezout\", \"nt_modinv\", "
         "\"nt_crt\", \"nt_modexp\", \"nt_cf\"], \"levels\": 3, "
         "\"problems\": "
      << problems << ", \"certified\": " << ok << ", \"rows\": " << rows
      << ", \"vocab_extra\": {\"chars\": \"0123456789+-*(), \", "
         "\"names\": [\"gcd\", \"Mod\"], \"ops\": [\"**\"]}}\n";
  std::cerr << "== " << problems << " problems, " << ok << " certified, "
            << rows << " rows\n";
  return problems == ok ? 0 : 1;
}
