/** @file series_oracle.cpp Order-bounded ODE residual check (see
    series_oracle.hpp). */
#include <ax/sym/series_oracle.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace ax::sym {

namespace {

bool is_y_of_x(const expr& e, const expr& x) {
  return e.is_fn() && e.name() == "y" && e.args().size() == 1 &&
         e.args()[0].same(x);
}

/** Derivative(y(x), x, ..., x) carrier order; 0 when e is not one. */
std::size_t deriv_order(const expr& e, const expr& x) {
  if (!(e.is_fn() && e.name() == "Derivative" && e.args().size() >= 2 &&
        is_y_of_x(e.args()[0], x)))
    return 0;
  for (std::size_t i = 1; i < e.args().size(); ++i)
    if (!e.args()[i].same(x)) return 0;
  return e.args().size() - 1;
}

/** Expand e as a series in x with the unknown y bound to yd[0] and its
    carriers to yd[n]. Mirrors series::of_expr, plus the two carriers;
    throws std::domain_error out of scope (mapped to UNDECIDED). */
series expand_y(const expr& e, const expr& x, const std::vector<series>& yd,
                int order) {
  if (is_y_of_x(e, x)) return yd[0];
  if (const std::size_t n = deriv_order(e, x)) {
    if (n >= yd.size())
      throw std::domain_error("series: derivative beyond table");
    return yd[n];
  }
  switch (e.k()) {
    case kind::num:
      return series({e.value()}, order);
    case kind::sym:
      if (e.same(x)) return series({rational{}, rational{bigint(1)}}, order);
      throw std::domain_error("series: free symbol " + e.name());
    case kind::add: {
      series r(order);
      for (const expr& t : e.args()) r = r + expand_y(t, x, yd, order);
      return r;
    }
    case kind::mul: {
      series r({rational{bigint(1)}}, order);
      for (const expr& t : e.args()) r = r * expand_y(t, x, yd, order);
      return r;
    }
    case kind::pow: {
      const expr& ex = e.args()[1];
      if (ex.is_num() && ex.value().den() == bigint(1))
        return expand_y(e.args()[0], x, yd, order)
            .pow_int(std::stoll(ex.value().num().to_string()));
      if (ex.is_num() && ex.value() == rational(bigint(1), bigint(2)))
        return apply_fn("sqrt", expand_y(e.args()[0], x, yd, order));
      throw std::domain_error("series: non-integer exponent");
    }
    case kind::fn:
      if (e.args().size() != 1)
        throw std::domain_error("series: n-ary function " + e.name());
      return apply_fn(e.name(), expand_y(e.args()[0], x, yd, order));
  }
  throw std::domain_error("series: unreachable kind");
}

std::size_t max_deriv(const expr& e, const expr& x) {
  std::size_t best = deriv_order(e, x);
  for (const expr& a : e.args()) best = std::max(best, max_deriv(a, x));
  return best;
}

}  // namespace

series_check check_odesol_series(const expr& eq, const series& candidate,
                                 const expr& x) {
  expr lhs = eq;
  expr rhs = expr::num(0);
  if (eq.is_fn() && eq.name() == "Eq" && eq.args().size() == 2) {
    lhs = eq.args()[0];
    rhs = eq.args()[1];
  }
  const int order = candidate.order();
  try {
    const std::size_t k = std::max(max_deriv(lhs, x), max_deriv(rhs, x));
    std::vector<series> yd{candidate};
    for (std::size_t i = 0; i < k; ++i)
      yd.push_back(yd.back().derivative());
    const series residual =
        expand_y(lhs, x, yd, order) - expand_y(rhs, x, yd, order);
    for (int i = 0; i < residual.order(); ++i)
      if (!residual.coeff(static_cast<std::size_t>(i)).is_zero())
        return {series_verdict::not_equivalent, i};
    return {series_verdict::equivalent_to_order, residual.order()};
  } catch (const std::exception&) {
    return {series_verdict::undecided_beyond_order, 0};
  }
}

}  // namespace ax::sym
