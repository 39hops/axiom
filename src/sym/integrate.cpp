#include <ax/sym/integrate.hpp>

#include <stdexcept>
#include <utility>
#include <vector>

namespace ax::sym {

namespace {

bool depends_on(const expr& e, const expr& x) {
  if (e.same(x)) return true;
  for (const expr& a : e.args())
    if (depends_on(a, x)) return true;
  return false;
}

/** e == a*x + b with rational a != 0? Returns {a_expr_inverse_factor, ok}.
    Detected structurally: e is x, or num*x, or a sum of such + x-free
    terms. */
std::optional<rational> linear_coefficient(const expr& e, const expr& x) {
  auto term_coeff = [&](const expr& t) -> std::optional<rational> {
    if (t.same(x)) return rational(bigint(1));
    if (t.is_mul()) {
      // canonical mul: at most one numeric factor first
      const auto fs = t.args();
      if (fs.size() == 2 && fs[0].is_num() && fs[1].same(x))
        return fs[0].value();
    }
    return std::nullopt;
  };
  if (auto c = term_coeff(e)) return c;
  if (e.is_add()) {
    std::optional<rational> coeff;
    for (const expr& t : e.args()) {
      if (!depends_on(t, x)) continue;
      if (coeff) return std::nullopt;  // two x-terms -> not simple linear
      coeff = term_coeff(t);
      if (!coeff) return std::nullopt;
    }
    return coeff;
  }
  return std::nullopt;
}

expr inv(const rational& a) {
  return expr::num(rational(bigint(1)) / a);
}

/** Table antiderivative of fn(name, u) for linear u (du = a dx). */
std::optional<expr> integrate_fn_linear(const std::string& name,
                                        const expr& u, const rational& a) {
  const expr ia = inv(a);
  if (name == "sin") return -(expr::fn("cos", u)) * ia;
  if (name == "cos") return expr::fn("sin", u) * ia;
  if (name == "tan") return -(expr::fn("log", expr::fn("cos", u))) * ia;
  if (name == "exp") return expr::fn("exp", u) * ia;
  if (name == "log") return (u * expr::fn("log", u) - u) * ia;
  if (name == "sqrt") {
    const expr three_halves = expr::num(rational(bigint(3), bigint(2)));
    return expr::num(rational(bigint(2), bigint(3))) * u.pow(three_halves) *
           ia;
  }
  return std::nullopt;
}

std::optional<expr> integrate_impl(const expr& e, const expr& x);

/** pow(base, exponent) cases. */
std::optional<expr> integrate_pow(const expr& base, const expr& ex,
                                  const expr& x) {
  if (depends_on(ex, x)) return std::nullopt;  // x in exponent: not here
  if (!ex.is_num()) return std::nullopt;
  const rational n = ex.value();
  const auto a = linear_coefficient(base, x);
  if (!a) return std::nullopt;
  if (n == rational(bigint(-1)))
    return expr::fn("log", base) * inv(*a);
  const rational n1 = n + rational(bigint(1));
  return base.pow(expr::num(n1)) * expr::num(rational(bigint(1)) / n1) *
         inv(*a);
}

std::optional<expr> integrate_impl(const expr& e, const expr& x) {
  // constants (in x)
  if (!depends_on(e, x)) return e * x;
  if (e.same(x))
    return expr::num(rational(bigint(1), bigint(2))) * x.pow(expr::num(2));
  // linearity over sums
  if (e.is_add()) {
    expr sum = expr::num(0);
    for (const expr& t : e.args()) {
      auto part = integrate_impl(t, x);
      if (!part) return std::nullopt;
      sum = sum + *part;
    }
    return sum;
  }
  // pull x-free factors out of products
  if (e.is_mul()) {
    expr c = expr::num(1), rest = expr::num(1);
    std::vector<expr> dep;
    for (const expr& f : e.args()) {
      if (depends_on(f, x))
        dep.push_back(f);
      else
        c = c * f;
    }
    if (dep.size() == 1) {
      auto part = integrate_impl(dep[0], x);
      if (part) return c * *part;
      return std::nullopt;
    }
    // multiple x-dependent factors: heuristics (Task 4)
    return std::nullopt;
  }
  if (e.is_pow()) return integrate_pow(e.args()[0], e.args()[1], x);
  if (e.is_fn()) {
    const expr& u = e.args()[0];
    if (const auto a = linear_coefficient(u, x))
      return integrate_fn_linear(e.name(), u, *a);
    return std::nullopt;
  }
  return std::nullopt;
}

}  // namespace

std::optional<expr> integrate(const expr& e, const expr& x) {
  if (!x.is_sym())
    throw std::invalid_argument("integrate: x must be a symbol");
  return integrate_impl(e, x);
}

}  // namespace ax::sym
