#pragma once
/** @file harness.hpp JSONL oracle harness entry (the axiom-oracle tool).

    Protocol (docs/specs/2026-07-18-llmopt-oracle.md): one task object per
    input line, exactly one result object per line out, same order. Any
    per-row failure emits a status:"error" row; the run never dies on a row. */
#include <istream>
#include <ostream>

namespace ax::sym {

/** Process all rows; returns the number of rows that ended in error. */
int run_oracle(std::istream& in, std::ostream& out);

}  // namespace ax::sym
