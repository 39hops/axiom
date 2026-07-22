/** @file series_chain.cpp Derivation-row spelling for series recurrence
    steps (see series_chain.hpp). */
#include <ax/mathgen/series_chain.hpp>

#include <ax/sym/expr.hpp>
#include <ax/sym/print_sstr.hpp>

namespace ax::mathgen {

namespace {

const rational kOne(bigint(1));

/** Canonical sstr of a rational value (the byte-exact nxt spelling). */
std::string value_sstr(const rational& r) {
  return sym::to_sstr(sym::expr::num(r));
}

/** One recurrence product c*fall*a with unit factors dropped. */
std::string term_str(const series_term& t) {
  std::string s;
  for (const rational* f : {&t.c, &t.fall, &t.a}) {
    if (*f == kOne) continue;
    if (!s.empty()) s += "*";
    s += chain_lit(*f);
  }
  return s.empty() ? "1" : s;
}

}  // namespace

std::string chain_lit(const rational& r) {
  const std::string s = r.to_string();
  const bool needs_parens =
      s.find('-') != std::string::npos || s.find('/') != std::string::npos;
  return needs_parens ? "(" + s + ")" : s;
}

derivation derivation_rows(const series_step& st) {
  derivation d;
  rational sum;
  std::string sum_expr;
  for (const series_term& t : st.terms) {
    sum = sum + t.c * t.fall * t.a;
    if (!sum_expr.empty()) sum_expr += " + ";
    sum_expr += term_str(t);
  }
  if (st.terms.empty()) sum_expr = "0";
  d.sum_nxt = value_sstr(sum);
  if (st.terms.size() >= 2) d.sum_cur = sum_expr;
  // solve row folds the sum first when a sum row precedes it, so each
  // emission carries exactly one new arithmetic fact.
  const std::string s_lit =
      st.terms.size() >= 2 ? chain_lit(sum) : sum_expr;
  d.solve_cur = "(" + st.q_n.to_string() + " - " + s_lit + ")/" +
                chain_lit(st.divisor);
  d.solve_nxt = value_sstr(st.a_n);
  return d;
}

}  // namespace ax::mathgen
