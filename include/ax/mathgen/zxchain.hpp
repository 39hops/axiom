#pragma once
/** @file zxchain.hpp ZX chain emitter (llmopt relay 2026-07-26).

    Random Clifford+T circuit -> ZX diagram -> greedy/random descent by
    the four local moves, one row per applied move (cur -> nxt) in the
    farm_v22-style schema: kind = move name, level = initial T-count
    bucket. Both cur and nxt of a row are serialized under the SAME
    per-row random label permutation (string-seeded), so the site named
    in the row addresses both sides and pyzx can replay the move.
    T-count and spider count of cur ride along as row metadata for the
    rarity/mass instruments. */
#include <ax/zx/zx.hpp>

#include <string>
#include <vector>

namespace ax::mathgen {

struct zx_row {
  std::string kind;  ///< fuse | id | lcomp | pivot
  std::string site;  ///< move vertices as emitted labels, space-joined
  std::string cur;
  std::string nxt;
  int tcount = 0;   ///< T-count of cur
  int spiders = 0;  ///< spider count of cur
};

struct zx_problem {
  int size = 1;        ///< generator size class 1..3
  long long seed = 0;
  int level = 1;       ///< initial T-count bucket: 0-4 -> 1, 5-9 -> 2,
                       ///< 10+ -> 3
  int qubits = 0;
  int gates = 0;
  std::vector<zx_row> rows;
};

/** Build a random Clifford+T circuit diagram (rng seeded
    "zx_chain|size|seed") and descend: fuse/id moves are preferred
    while any fire (greedy tier), then lcomp/pivot; the move inside a
    tier is drawn uniformly. Descent stops when no move checks or at
    an internal ply cap. */
zx_problem make_zx_chain(int size, long long seed);

}  // namespace ax::mathgen
