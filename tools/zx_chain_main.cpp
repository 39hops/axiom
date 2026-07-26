/** zx-chain: ZX row-factory emitter (llmopt relay 2026-07-26 +
    addendum: color-change move five, balanced-kind batch).
    Usage:
      axiom-zx-chain <out.jsonl> [seeds_per_cell]
    3 size classes x seeds_per_cell problems; one row per applied move
    in the farm_v22-style schema:
      {"family": "zx", "level": <initial T-count bucket>, "size", "seed",
       "n", "kind": fuse|id|lcomp|pivot|color, "site": "<labels>",
       "cur", "nxt", "tcount", "spiders", "source": "axiom-zx-chain"}
    Diet ration at the farm (vm-asm 1a scar: a 63% one-kind diet
    manufactured a fake verdict): any kind above 50% of rows is
    subsampled down to the size of all other kinds combined, drawn by
    string-seeded shuffle, original row order preserved. Soundness is
    by construction per move; semantic adjudication is llmopt-side
    (pyzx replay at the named site + compare_tensors). */
#include <ax/mathgen/zxchain.hpp>
#include <ax/pyrand/pyrand.hpp>
#include <ax/sym/jsonl.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

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
  struct pending_row {
    std::string kind;
    std::string json;
  };
  std::vector<pending_row> rows;
  long long problems = 0;
  for (int size = 1; size <= 3; ++size)
    for (long long seed = 0; seed < seeds; ++seed) {
      const mathgen::zx_problem p = mathgen::make_zx_chain(size, seed);
      ++problems;
      int n = 0;
      for (const auto& r : p.rows) {
        std::string j = "{\"family\": \"zx\", \"level\": " +
                        std::to_string(p.level) +
                        ", \"size\": " + std::to_string(p.size) +
                        ", \"seed\": " + std::to_string(seed) +
                        ", \"n\": " + std::to_string(n++) +
                        ", \"kind\": \"" + r.kind + "\", \"site\": \"" +
                        r.site + "\", \"cur\": \"" +
                        sym::jsonl::escape(r.cur) + "\", \"nxt\": \"" +
                        sym::jsonl::escape(r.nxt) + "\", \"tcount\": " +
                        std::to_string(r.tcount) + ", \"spiders\": " +
                        std::to_string(r.spiders) +
                        ", \"source\": \"axiom-zx-chain\"}";
        rows.push_back({r.kind, std::move(j)});
      }
    }
  // diet ration: cap any kind at 50% by subsampling it down to the
  // combined size of the other kinds (string-seeded, order-preserving)
  std::map<std::string, long long> count;
  for (const auto& r : rows) ++count[r.kind];
  std::vector<bool> keep(rows.size(), true);
  for (const auto& [kind, c] : count) {
    const long long others = static_cast<long long>(rows.size()) - c;
    if (c <= others) continue;
    std::vector<std::size_t> idx;
    for (std::size_t i = 0; i < rows.size(); ++i)
      if (rows[i].kind == kind) idx.push_back(i);
    pyrand::python_random rng("zx_ration|" + kind + "|" +
                              std::to_string(seeds));
    rng.shuffle(idx);
    for (std::size_t i = static_cast<std::size_t>(others); i < idx.size();
         ++i)
      keep[idx[i]] = false;
  }
  long long emitted = 0;
  std::map<std::string, long long> emitted_by_kind;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (!keep[i]) continue;
    out << rows[i].json << "\n";
    ++emitted;
    ++emitted_by_kind[rows[i].kind];
  }
  std::cerr << "== " << problems << " problems, " << rows.size()
            << " rows farmed, " << emitted << " emitted after ration (";
  bool first = true;
  for (const auto& [kind, c] : emitted_by_kind) {
    if (!first) std::cerr << ", ";
    first = false;
    std::cerr << kind << " " << c;
  }
  std::cerr << ")\n";
  return 0;
}
