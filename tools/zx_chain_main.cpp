/** zx-chain: ZX row-factory emitter (llmopt relay 2026-07-26).
    Usage:
      axiom-zx-chain <out.jsonl> [seeds_per_cell]
    3 size classes x seeds_per_cell problems; one row per applied move
    in the farm_v22-style schema:
      {"family": "zx", "level": <initial T-count bucket>, "size", "seed",
       "n", "kind": fuse|id|lcomp|pivot, "site": "<labels>",
       "cur", "nxt", "tcount", "spiders", "source": "axiom-zx-chain"}
    Soundness is by construction per move; semantic adjudication is
    llmopt-side (pyzx replay at the named site + compare_tensors).
    Default seeds_per_cell 120 lands near the 10k-row sample batch. */
#include <ax/mathgen/zxchain.hpp>
#include <ax/sym/jsonl.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
  using namespace ax;
  if (argc < 2) {
    std::cerr << "usage: axiom-zx-chain <out.jsonl> [seeds_per_cell]\n";
    return 2;
  }
  const long long seeds = argc > 2 ? std::atoll(argv[2]) : 120;
  std::ofstream out(argv[1]);
  if (!out.good()) {
    std::cerr << "cannot open " << argv[1] << "\n";
    return 2;
  }
  long long problems = 0, rows = 0;
  long long kind_counts[4] = {0, 0, 0, 0};  // fuse id lcomp pivot
  for (int size = 1; size <= 3; ++size)
    for (long long seed = 0; seed < seeds; ++seed) {
      const mathgen::zx_problem p = mathgen::make_zx_chain(size, seed);
      ++problems;
      int n = 0;
      for (const auto& r : p.rows) {
        out << "{\"family\": \"zx\", \"level\": " << p.level
            << ", \"size\": " << p.size << ", \"seed\": " << seed
            << ", \"n\": " << n++ << ", \"kind\": \"" << r.kind
            << "\", \"site\": \"" << r.site << "\", \"cur\": \""
            << sym::jsonl::escape(r.cur) << "\", \"nxt\": \""
            << sym::jsonl::escape(r.nxt)
            << "\", \"tcount\": " << r.tcount
            << ", \"spiders\": " << r.spiders
            << ", \"source\": \"axiom-zx-chain\"}\n";
        ++rows;
        kind_counts[r.kind == "fuse"    ? 0
                    : r.kind == "id"    ? 1
                    : r.kind == "lcomp" ? 2
                                        : 3]++;
      }
    }
  std::cerr << "== " << problems << " problems, " << rows
            << " rows (fuse " << kind_counts[0] << ", id "
            << kind_counts[1] << ", lcomp " << kind_counts[2]
            << ", pivot " << kind_counts[3] << ")\n";
  return 0;
}
