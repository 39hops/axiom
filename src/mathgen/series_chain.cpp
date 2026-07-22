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

/** The non-unit factors of one recurrence product c*fall*a. */
std::vector<rational> factors_of(const series_term& t) {
  std::vector<rational> f;
  for (const rational* p : {&t.c, &t.fall, &t.a})
    if (!(*p == kOne)) f.push_back(*p);
  return f;
}

/** One recurrence product spelled inline (single-term solve rows). */
std::string term_str(const series_term& t) {
  std::string s;
  for (const rational& f : factors_of(t)) {
    if (!s.empty()) s += "*";
    s += chain_lit(f);
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
  std::vector<rational> values;
  // single-term sums stay inline in the solve row (that shape trained
  // well); only multi-term sums get the reduction tree.
  const bool reduce = st.terms.size() >= 2;
  for (const series_term& t : st.terms) {
    const std::vector<rational> f = factors_of(t);
    rational v = f.empty() ? kOne : f[0];
    // left-folded binary products: one multiplication fact per row
    for (std::size_t i = 1; i < f.size(); ++i) {
      const rational p = v * f[i];
      if (reduce)
        d.reduction.push_back(
            {"mul", chain_lit(v) + "*" + chain_lit(f[i]), value_sstr(p)});
      v = p;
    }
    values.push_back(v);
    sum = sum + v;
  }
  // left-folded binary adds: one addition fact per row
  if (values.size() >= 2) {
    rational acc = values[0];
    for (std::size_t i = 1; i < values.size(); ++i) {
      const rational s = acc + values[i];
      d.reduction.push_back({"add",
                             acc.to_string() + " + " + chain_lit(values[i]),
                             value_sstr(s)});
      acc = s;
    }
  }
  // the solve row folds the sum when a reduction preceded it, so each
  // emission carries exactly one new arithmetic fact.
  const std::string s_lit = st.terms.empty()      ? "0"
                            : st.terms.size() == 1 ? term_str(st.terms[0])
                                                   : chain_lit(sum);
  d.solve_cur = "(" + st.q_n.to_string() + " - " + s_lit + ")/" +
                chain_lit(st.divisor);
  d.solve_nxt = value_sstr(st.a_n);
  return d;
}

}  // namespace ax::mathgen
