#pragma once
/** @file series_chain.hpp L10 rung 6: spell one recurrence step's
    coefficient arithmetic as explicit rewrite rows in the vocab-40
    charset (digits, x, + - * / ( ) space — no letters, no '=').
    Each row's cur must fold, via sym::parse -> canonical, byte-exactly
    to its nxt; the emitter certifies that before writing. */
#include <ax/mathgen/series_solve.hpp>

#include <optional>
#include <string>

namespace ax::mathgen {

/** Rational literal for use inside a larger expression: parenthesized
    when negative or non-integer so "a*(-2)*(3/4)" parses unambiguously. */
std::string chain_lit(const rational& r);

/** The rewrite rows derived from one recurrence step. sum_cur is absent
    when the recurrence sum has fewer than two terms (nothing to fold). */
struct derivation {
  std::optional<std::string> sum_cur;  ///< "c1*f1*a1 + c2*f2*a2 + ..."
  std::string sum_nxt;                 ///< folded sum, canonical sstr
  std::string solve_cur;               ///< "(q - S)/divisor", literals only
  std::string solve_nxt;               ///< a_n, canonical sstr
};

derivation derivation_rows(const series_step& st);

}  // namespace ax::mathgen
