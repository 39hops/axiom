#include <ax/sym/calc.hpp>

#include <stdexcept>
#include <vector>

namespace ax::sym {

namespace {

/** Derivative of fn(name, u) w.r.t. u (outer derivative, chain rule applies
    the du factor at the call site). */
expr fn_outer_derivative(const std::string& name, const expr& u) {
  if (name == "sin") return expr::fn("cos", u);
  if (name == "cos") return -expr::fn("sin", u);
  if (name == "tan")
    return expr::num(1) + expr::fn("tan", u).pow(expr::num(2));
  if (name == "exp") return expr::fn("exp", u);
  if (name == "log") return u.pow(expr::num(-1));
  if (name == "sqrt")
    return expr::num(rational(bigint(1), bigint(2))) *
           u.pow(expr::num(rational(bigint(-1), bigint(2))));
  if (name == "atan")  // 1/(1+u^2)
    return (expr::num(1) + u.pow(expr::num(2))).pow(expr::num(-1));
  if (name == "asin")  // 1/sqrt(1-u^2)
    return (expr::num(1) - u.pow(expr::num(2)))
        .pow(expr::num(rational(bigint(-1), bigint(2))));
  if (name == "acos")  // -1/sqrt(1-u^2)
    return -((expr::num(1) - u.pow(expr::num(2)))
                 .pow(expr::num(rational(bigint(-1), bigint(2)))));
  throw std::logic_error("diff: unknown function " + name);
}

}  // namespace

expr diff(const expr& e, const expr& s) {
  if (!s.is_sym()) throw std::invalid_argument("diff: s must be a symbol");
  if (e.same(s)) return expr::num(1);
  if (e.is_num() || e.is_sym()) return expr::num(0);
  if (e.is_add()) {
    expr sum = expr::num(0);
    for (const expr& t : e.args()) sum = sum + diff(t, s);
    return sum;
  }
  if (e.is_mul()) {
    // n-ary product rule: sum over i of (d f_i) * prod_{j != i} f_j
    const auto fs = e.args();
    expr sum = expr::num(0);
    for (std::size_t i = 0; i < fs.size(); ++i) {
      expr term = diff(fs[i], s);
      for (std::size_t j = 0; j < fs.size(); ++j)
        if (j != i) term = term * fs[j];
      sum = sum + term;
    }
    return sum;
  }
  if (e.is_pow()) {
    const expr& u = e.args()[0];
    const expr& v = e.args()[1];
    const expr du = diff(u, s);
    const expr dv = diff(v, s);
    if (dv.same(expr::num(0))) {
      // power rule: d(u^v) = v u^(v-1) du
      return v * u.pow(v - expr::num(1)) * du;
    }
    // general: u^v (dv log u + v du / u)
    return e * (dv * expr::fn("log", u) + v * du / u);
  }
  // Unevaluated carriers (Phase C search states):
  if (e.is_fn() && e.name() == "Integral") {
    // d/dx Integral(f, ..., x) = drop the outermost (last) limit; this is
    // the verify identity that never asks anyone to integrate.
    const auto a = e.args();
    if (a.back().same(s)) {
      if (a.size() == 2) return a[0];
      return expr::fn("Integral",
                      std::vector<expr>(a.begin(), a.end() - 1));
    }
    // independent variable: differentiate under the integral sign
    std::vector<expr> mapped(a.begin(), a.end());
    mapped[0] = diff(a[0], s);
    return expr::fn("Integral", std::move(mapped));
  }
  if (e.is_fn() && (e.name() == "Derivative" || e.name() == "Subs")) {
    // stays symbolic: append a derivative wrapper (resolved by doit
    // upstream, never differentiated through here)
    return expr::derivative(e, s);
  }
  // fn: chain rule
  const expr& u = e.args()[0];
  return fn_outer_derivative(e.name(), u) * diff(u, s);
}

}  // namespace ax::sym
