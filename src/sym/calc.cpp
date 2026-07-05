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
  // fn: chain rule
  const expr& u = e.args()[0];
  return fn_outer_derivative(e.name(), u) * diff(u, s);
}

}  // namespace ax::sym
