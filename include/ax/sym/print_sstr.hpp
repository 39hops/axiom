#pragma once
/** @file print_sstr.hpp sympy-canonical sstr printer.

    Prints an expr byte-identically to sympy 1.14's `sstr` on the mathgen
    expression zoo: Add terms in sympy's printing order (as_ordered_terms:
    descending-lex monomial vectors over default_sort_key-sorted gens,
    ascending-coefficient tie-break, plus the positive-number/negative-Mul
    two-term special case), Mul factors via as_ordered_factors (sort_key
    ascending, numeric coefficient first), numerator/denominator splitting,
    sqrt spelling, `**` pow. Scoped to the zoo — not sympy-general; the
    fixture gate is tests/sym/fixtures/sstr_fixture.tsv. */
#include <ax/sym/expr.hpp>

#include <string>

namespace ax::sym {

std::string to_sstr(const expr& e);

}  // namespace ax::sym
