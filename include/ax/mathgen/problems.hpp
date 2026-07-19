#pragma once
/** @file problems.hpp Port of llmopt/mathgen/problems.py expression
    generators (Phase B). Every rng call mirrors the Python original
    call-for-call — the parity gate is byte-exact sstr equality against
    the sympy generator per (level, seed), fixtures generated from
    llmopt's actual module (tests/mathgen/fixtures/). */
#include <ax/pyrand/pyrand.hpp>
#include <ax/sym/expr.hpp>

namespace ax::mathgen {

/** problems._atom(rng, level). */
sym::expr atom(pyrand::python_random& rng, int level);

/** problems._expression(rng, level); levels 1-3 in v1 (L4-L8 follow). */
sym::expr expression(pyrand::python_random& rng, int level);

/** The generator's seed string for a task family, e.g.
    seed_string("diff", 3, 17) == "diff-3-17". */
std::string seed_string(const std::string& kind, int level, long long seed);

}  // namespace ax::mathgen
