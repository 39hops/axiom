#pragma once
/** @file print_lean.hpp Lean 4 certificate emitter for EQUIVALENT verdicts.

    Emits, for the tactic-closable subset, a kernel-checkable statement of
    the GENERALIZED identity: every distinct fn-subterm of the parsed
    lhs/rhs maps to a fresh universally-quantified real (the atom table);
    the statement is rational in x and the atoms. Eligibility rule (relay
    2026-08-03-1): no fractional exponents at ring level — sqrt and
    non-integer/symbolic pow exponents outside fn-atoms are fenced out;
    anything frozen inside an opaque fn argument is eligible. Atom identity
    is structural identity of the RAW PARSED subterm — canonical() (and its
    intra-argument sqrt->pow(1/2) merging) is never consulted, so a verdict
    that leaned on an unsound merge yields a loudly-failing certificate,
    not a quietly-matching one. Divisions quantify a nonzero hypothesis per
    syntactic denominator and close by field_simp; ring. */
#include <ax/sym/expr.hpp>

#include <string>
#include <utility>
#include <vector>

namespace ax::sym {

struct lean_cert {
  bool eligible = false;
  std::string statement;  // full `example ... := by <tactic>` line
  std::string tactic;     // "ring" or "field_simp; ring"
  std::string lhs;        // sstr of the parsed lhs (sidecar provenance)
  std::string rhs;
  /** atom name -> sstr of the generalized subterm, first-appearance order. */
  std::vector<std::pair<std::string, std::string>> atoms;
};

/** Build the certificate for lhs == rhs (identity in x; all other free
    symbols are universally quantified alongside the atoms). Takes the RAW
    sstr strings, not exprs: expr construction already merges fractional
    pows (x**(1/2)*x**(1/2) -> x), so the fence must run lexically on the
    input before parsing or it can never fire. */
lean_cert to_lean(const std::string& lhs_raw, const std::string& rhs_raw,
                  const expr& x);

/** One sidecar JSONL row: {"id","lhs","rhs","lean","tactic","atoms":{...}}. */
std::string sidecar_line(const std::string& id, const lean_cert& c);

}  // namespace ax::sym
