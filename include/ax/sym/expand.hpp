#pragma once
/** @file expand.hpp Polynomial expansion: distribute products over sums and
    expand positive-integer powers of sums. */
#include <ax/sym/expr.hpp>

namespace ax::sym {

/** Expanded form of e: products are distributed over sums and integer powers
    of sums (exponent 2..64) are multiplied out; canonicalization then merges
    like terms. Non-polynomial subtrees (functions, symbolic or fractional
    exponents) are kept, with their arguments expanded. */
expr expand(const expr& e);

}  // namespace ax::sym
