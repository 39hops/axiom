#pragma once
/** @file parse.hpp Parser for sympy `sstr` expression text.

    Accepts the llmopt expression zoo: integers, decimal floats (converted
    exactly to rational), symbols, pi/E, the unary functions
    sin cos tan exp log sqrt atan asin acos, and + - * / ** with sympy
    precedence (`**` right-associative, binding tighter than unary minus).
    `^` is accepted as a synonym for `**` so axiom's own printer output
    round-trips. Unknown function names and unrepresentable atoms
    (oo, zoo, nan, I) are rejected. */
#include <ax/sym/expr.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ax::sym {

/** Parse failure; carries the byte offset of the offending token. */
class parse_error : public std::runtime_error {
 public:
  parse_error(const std::string& what, std::size_t offset)
      : std::runtime_error(what + " (at offset " + std::to_string(offset) +
                           ")"),
        offset_(offset) {}
  std::size_t offset() const { return offset_; }

 private:
  std::size_t offset_;
};

/** Parse sympy sstr text into a canonicalized expr. Throws parse_error. */
expr parse(std::string_view src);

}  // namespace ax::sym
