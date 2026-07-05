#pragma once
/** @file integrate.hpp Symbolic integration: table + linearity + power rule,
    u-substitution, integration by parts, partial fractions ("Risch-lite"). */
#include <ax/sym/expr.hpp>

#include <optional>

namespace ax::sym {

/** Antiderivative of e with respect to x (no +C), or nullopt when no rule
    applies — never a wrong answer: any returned value satisfies
    d(result)/dx == e (the test oracle). x must be a symbol
    (std::invalid_argument). */
std::optional<expr> integrate(const expr& e, const expr& x);

}  // namespace ax::sym
