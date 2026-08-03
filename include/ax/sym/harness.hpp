#pragma once
/** @file harness.hpp JSONL oracle harness entry (the axiom-oracle tool).

    Protocol (docs/specs/2026-07-18-llmopt-oracle.md): one task object per
    input line, exactly one result object per line out, same order. Any
    per-row failure emits a status:"error" row; the run never dies on a row. */
#include <istream>
#include <ostream>

namespace ax::sym {

/** Lean-sidecar counters: emitted certificates vs EQUIVALENT rows fenced
    out of the tactic-closable subset (so the closable fraction is
    measured, never silently capped). */
struct lean_stats {
  long long emitted = 0;
  long long fenced = 0;
};

/** Process all rows; returns the number of rows that ended in error.
    When lean_out is non-null, every EQUIVALENT verdict on the eligible
    subset additionally writes one certificate sidecar line to it. */
int run_oracle(std::istream& in, std::ostream& out,
               std::ostream* lean_out = nullptr, lean_stats* stats = nullptr);

}  // namespace ax::sym
