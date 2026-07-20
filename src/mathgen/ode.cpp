/** @file ode.cpp L9 ODE makers (see ode.hpp). Every rng call mirrors
    llmopt/mathgen/odes.py call-for-call; the eq/sol shapes reproduce
    the sympy construction so sstr output is byte-exact. */
#include <ax/mathgen/ode.hpp>

#include <ax/pyrand/pyrand.hpp>
#include <ax/sym/calc.hpp>
#include <ax/sym/expand.hpp>

#include <vector>

namespace ax::mathgen {

namespace {

using sym::expr;

const expr kX = expr::symbol("x");

expr y_of_x() { return expr::fn("y", kX); }

expr d1_y() { return expr::fn("Derivative", std::vector<expr>{y_of_x(), kX}); }

expr d2_y() {
  return expr::fn("Derivative", std::vector<expr>{y_of_x(), kX, kX});
}

expr make_eq(const expr& lhs, const expr& rhs) {
  return expr::fn("Eq", std::vector<expr>{lhs, rhs});
}

/** CPython random.sample(population, k) for n <= 21 (the pool copy
    algorithm: j = _randbelow(n-i); take pool[j]; backfill from the
    shrinking tail). Bit-exact with random.py. */
std::vector<long long> py_sample(pyrand::python_random& rng,
                                 const std::vector<long long>& population,
                                 std::size_t k) {
  std::vector<long long> pool = population;
  std::vector<long long> out;
  const std::size_t n = pool.size();
  for (std::size_t i = 0; i < k; ++i) {
    const std::size_t j = rng.choice_index(n - i);
    out.push_back(pool[j]);
    pool[j] = pool[n - i - 1];
  }
  return out;
}

}  // namespace

ode_problem make_linear_first_order(int level, long long seed) {
  pyrand::python_random rng("ode1-" + std::to_string(level) + "-" +
                            std::to_string(seed));
  const long long p = rng.randint(-3, 3);
  if (p == 0) return make_linear_first_order(level, seed + 1'000'003);
  expr sol = expr::num(0);
  // y0 = sol at x = 0 (exp(0) = 1, x = 0): computed from the draws, not
  // by folding the built expression (a c = 1 coefficient prints as a
  // bare exp term and a structural fold silently drops it)
  rational y0;
  if (level == 1) {
    const long long c = rng.randint(1, 5);
    sol = expr::num(c) * expr::fn("exp", expr::num(-p) * kX);
    y0 = rational(bigint(c));
  } else {
    const long long c1 = rng.randint(1, 4);
    const long long c2 = rng.randint(1, 5);
    const long long c3 = rng.randint(-3, 3);
    sol = expr::num(c1) * expr::fn("exp", expr::num(-p) * kX) +
          expr::num(c2) * kX + expr::num(c3);
    y0 = rational(bigint(c1 + c3));
  }
  const expr q = sym::expand(sym::diff(sol, kX) + expr::num(p) * sol);
  const expr eq = make_eq(d1_y() + expr::num(p) * y_of_x(), q);
  return {"ode_linear1", level, eq, sol, 0, y0, std::nullopt};
}

ode_problem make_second_order_cc(int level, long long seed) {
  pyrand::python_random rng("ode2-" + std::to_string(level) + "-" +
                            std::to_string(seed));
  const auto roots = py_sample(rng, {-3, -2, -1, 1, 2, 3}, 2);
  const long long r1 = roots[0];
  const long long r2 = roots[1];
  const long long c1 = rng.randint(1, 4);
  // explicit sequencing: operand evaluation order in `a * b` is
  // unspecified and MSVC runs right-to-left (the Phase B parity scar)
  const long long c2_mag = rng.randint(1, 4);
  const long long c2_sign = rng.choice_index(2) == 0 ? 1 : -1;
  const long long c2 = c2_mag * c2_sign;
  const expr sol = expr::num(c1) * expr::fn("exp", expr::num(r1) * kX) +
                   expr::num(c2) * expr::fn("exp", expr::num(r2) * kX);
  const expr eq =
      make_eq(d2_y() - expr::num(r1 + r2) * d1_y() +
                  expr::num(r1 * r2) * y_of_x(),
              expr::num(0));
  return {"ode_cc2",
          level,
          eq,
          sol,
          0,
          rational(bigint(c1 + c2)),
          rational(bigint(c1 * r1 + c2 * r2))};
}

ode_problem make_separable_growth(int level, long long seed) {
  pyrand::python_random rng("odes-" + std::to_string(level) + "-" +
                            std::to_string(seed));
  const std::vector<long long> ks = {-2, -1, 1, 2};
  const long long k = ks[rng.choice_index(4)];
  const long long n = level <= 2 ? 1 : rng.randint(1, 2);
  const long long c = rng.randint(1, 5);
  const expr sol =
      expr::num(c) *
      expr::fn("exp", expr::num(rational(bigint(k), bigint(n + 1))) *
                          kX.pow(expr::num(n + 1)));
  const expr eq = make_eq(
      d1_y(), expr::num(k) * kX.pow(expr::num(n)) * y_of_x());
  return {"ode_separable", level, eq, sol, 0, rational(bigint(c)),
          std::nullopt};
}

}  // namespace ax::mathgen
