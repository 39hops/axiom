#pragma once
/** @file series_chain.hpp L10 rung 6/7: spell one recurrence step's
    coefficient arithmetic as explicit rewrite rows in the vocab-40
    charset (digits, x, + - * / ( ) space — no letters, no '=').
    Rung 7 (llmopt relay 2026-07-22 night): multi-operand sums train
    poorly in one emission, so the recurrence sum is emitted as a binary
    reduction tree — one product per row, one add per row. Each row's
    cur must fold, via sym::parse -> canonical, byte-exactly to its nxt;
    the emitter certifies that before writing. */
#include <ax/mathgen/series_solve.hpp>

#include <string>
#include <vector>

namespace ax::mathgen {

/** Rational literal for use inside a larger expression: parenthesized
    when negative or non-integer so "a*(-2)*(3/4)" parses unambiguously. */
std::string chain_lit(const rational& r);

/** One primitive rewrite row: kind is "mul" (one binary product) or
    "add" (one binary addition). */
struct chain_row {
  std::string kind;
  std::string cur;
  std::string nxt;
};

/** The rewrite rows derived from one recurrence step. reduction holds
    the pairwise tree that folds the recurrence sum: first each term's
    factors as left-folded binary products, then the term values as
    left-folded binary adds (empty when the sum is a single bare value —
    the solve row carries it inline, which trains well). */
struct derivation {
  std::vector<chain_row> reduction;
  std::string solve_cur;  ///< "(q - S)/divisor", literals only
  std::string solve_nxt;  ///< a_n, canonical sstr
};

derivation derivation_rows(const series_step& st);

}  // namespace ax::mathgen
