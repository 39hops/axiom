#pragma once
/** @file calc.hpp Symbolic differentiation. */
#include <ax/sym/expr.hpp>

namespace ax::sym {

/** d e / d s, where s is a symbol. Throws std::invalid_argument if s is not
    a symbol, std::logic_error on an unknown function name. */
expr diff(const expr& e, const expr& s);

}  // namespace ax::sym
