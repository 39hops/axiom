#pragma once
/** @file print.hpp Expression printers: plain text and LaTeX. */
#include <ax/sym/expr.hpp>

#include <string>

namespace ax::sym {

/** Deterministic plain-text form, e.g. "2*x + y^2 - 3". */
std::string to_string(const expr& e);
/** LaTeX form, e.g. "2 x + y^{2} - 3", \frac{2}{3}, \sin\left(x\right). */
std::string to_latex(const expr& e);

}  // namespace ax::sym
