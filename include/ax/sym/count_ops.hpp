#pragma once
/** @file count_ops.hpp sympy-exact count_ops (Phase C task 1).

    HCE and successor node-ordering in the solver kernel depend on
    sympy-identical operation counts, so this transcribes sympy 1.14's
    count_ops(visual=False) worklist: NEG/DIV/SUB accounting, Mul
    fraction-splitting, E counting as EXP, sqrt as POW+DIV, carriers
    (Integral/Derivative) as one op each, Subs as zero. Fixture gate:
    tests/sym/fixtures/count_ops_fixture.tsv. */
#include <ax/sym/expr.hpp>

namespace ax::sym {

long long count_ops(const expr& e);

}  // namespace ax::sym
