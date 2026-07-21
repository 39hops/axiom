/** @file series_solve.cpp Coefficient-recurrence ODE solver (see
    series_solve.hpp). */
#include <ax/mathgen/series_solve.hpp>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>

namespace ax::mathgen {

namespace {

using sym::expr;
using sym::series;

bool is_y_of_x(const expr& e, const expr& x) {
  return e.is_fn() && e.name() == "y" && e.args().size() == 1 &&
         e.args()[0].same(x);
}

std::size_t deriv_order(const expr& e, const expr& x) {
  if (!(e.is_fn() && e.name() == "Derivative" && e.args().size() >= 2 &&
        is_y_of_x(e.args()[0], x)))
    return 0;
  for (std::size_t i = 1; i < e.args().size(); ++i)
    if (!e.args()[i].same(x)) return 0;
  return e.args().size() - 1;
}

bool is_carrier(const expr& e, const expr& x) {
  return is_y_of_x(e, x) || deriv_order(e, x) > 0;
}

/** falling factorial m (m-1) ... (m-k+1) as a rational. */
rational fall(int m, int k) {
  bigint p(1);
  for (int i = 0; i < k; ++i) p = p * bigint(m - i);
  return rational(std::move(p));
}

}  // namespace

series_solution series_solve(const ode_problem& p, int order) {
  if (p.x0 != 0)
    throw std::domain_error("series_solve: expansion point must be 0");
  const expr x = expr::symbol("x");
  expr lhs = p.eq;
  expr rhs = expr::num(0);
  if (p.eq.is_fn() && p.eq.name() == "Eq" && p.eq.args().size() == 2) {
    lhs = p.eq.args()[0];
    rhs = p.eq.args()[1];
  }
  const expr diffe = lhs - rhs;

  // collect c_i(x) (expr coefficient of the order-i carrier) and q(x)
  // from the Add terms of lhs - rhs == 0, i.e. sum c_i y^(i) == q
  std::vector<std::optional<expr>> coef;  // index i -> c_i
  std::optional<expr> q;
  const auto add_to = [](std::optional<expr>& slot, const expr& t) {
    slot = slot ? *slot + t : t;
  };
  const auto book_term = [&](const expr& term) {
    std::optional<std::size_t> ord;
    expr c = expr::num(1);
    const auto book_factor = [&](const expr& f) {
      if (is_carrier(f, x)) {
        if (ord)
          throw std::domain_error("series_solve: nonlinear in y");
        ord = deriv_order(f, x);
      } else {
        c = c * f;
      }
    };
    if (term.is_mul())
      for (const expr& f : term.args()) book_factor(f);
    else
      book_factor(term);
    if (!ord) {
      add_to(q, expr::num(0) - term);
      return;
    }
    if (coef.size() <= *ord) coef.resize(*ord + 1);
    add_to(coef[*ord], c);
  };
  if (diffe.is_add())
    for (const expr& t : diffe.args()) book_term(t);
  else
    book_term(diffe);

  const int k = static_cast<int>(coef.size()) - 1;
  if (k < 1 || !coef[static_cast<std::size_t>(k)])
    throw std::domain_error("series_solve: no derivative carrier");
  if (order < k)
    throw std::invalid_argument("series_solve: order below ODE order");

  std::vector<series> c;
  for (const auto& ci : coef)
    c.push_back(ci ? series::of_expr(*ci, x, order) : series(order));
  const series qs =
      q ? series::of_expr(*q, x, order) : series(order);
  const rational& top = c[static_cast<std::size_t>(k)].coeff(0);
  if (top.is_zero())
    throw std::domain_error("series_solve: zero top coefficient");

  // seed from the pinned ICs, then one exact recurrence step per a_{n+k}
  std::vector<rational> a(static_cast<std::size_t>(order));
  a[0] = p.y0;
  if (k == 2) {
    if (!p.yp0)
      throw std::domain_error("series_solve: second order needs yp0");
    a[1] = *p.yp0;
  }
  series_solution out;
  out.ode_order = k;
  for (int n = 0; n + k < order; ++n) {
    rational sum;
    for (int i = 0; i <= k; ++i)
      for (int j = 0; j <= n; ++j) {
        if (i == k && j == 0) continue;  // that's the unknown a_{n+k}
        const rational& cij = c[static_cast<std::size_t>(i)].coeff(
            static_cast<std::size_t>(j));
        if (cij.is_zero()) continue;
        sum = sum + cij * fall(n - j + i, i) *
                        a[static_cast<std::size_t>(n - j + i)];
      }
    const int m = n + k;
    const rational am =
        (qs.coeff(static_cast<std::size_t>(n)) - sum) / (top * fall(m, k));
    a[static_cast<std::size_t>(m)] = am;
    out.steps.push_back({m, am});
  }
  out.y = series(std::move(a), order);
  return out;
}

}  // namespace ax::mathgen
