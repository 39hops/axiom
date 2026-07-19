#include <ax/sym/expand.hpp>

#include <ax/sym/budget.hpp>

#include <vector>

#include <string>

namespace ax::sym {

namespace {

/** Product of two expanded operands, distributing over any add. */
expr mul_distribute(const expr& a, const expr& b) {
  if (a.is_add()) {
    expr sum = expr::num(0);
    for (const expr& t : a.args()) sum = sum + mul_distribute(t, b);
    return sum;
  }
  if (b.is_add()) {
    expr sum = expr::num(0);
    for (const expr& t : b.args()) {
      check_work_budget();
      sum = sum + mul_distribute(a, t);
    }
    return sum;
  }
  return a * b;
}

/** Exponent as small positive integer in [2, 64], or 0 when not. */
int small_int_exponent(const expr& e) {
  if (!e.is_num()) return 0;
  const rational& q = e.value();
  if (!(q.den() == bigint(1))) return 0;
  for (int k = 2; k <= 64; ++k)
    if (q.num() == bigint(k)) return k;
  return 0;
}

}  // namespace

expr expand(const expr& e) {
  switch (e.k()) {
    case kind::num:
    case kind::sym:
      return e;
    case kind::fn:
      {
        // n-ary fn nodes (search carriers): expand every argument
        std::vector<expr> mapped;
        mapped.reserve(e.args().size());
        for (const expr& a : e.args()) mapped.push_back(expand(a));
        return expr::fn(e.name(), std::move(mapped));
      }
    case kind::add: {
      expr sum = expr::num(0);
      for (const expr& t : e.args()) sum = sum + expand(t);
      return sum;
    }
    case kind::mul: {
      expr prod = expr::num(1);
      for (const expr& f : e.args()) prod = mul_distribute(prod, expand(f));
      return prod;
    }
    case kind::pow: {
      const expr base = expand(e.args()[0]);
      const expr ex = expand(e.args()[1]);
      const int k = small_int_exponent(ex);
      if (k != 0 && base.is_add()) {
        // multinomial blowup guard: (t-term sum)^k materializes
        // ~C(t+k-1,k) terms; a small count_ops input can explode into
        // millions (measured: an L8 sqrt-monster child held one expand
        // call for 13+ minutes at 1GB). Oversized stays unexpanded —
        // the oracle then answers UNDECIDED, which is conservative.
        const double t = static_cast<double>(base.args().size());
        double est = 1.0;
        for (int i = 1; i <= k; ++i) est *= (t + k - i) / i;
        if (est > 3000.0) return base.pow(ex);
        expr prod = base;
        for (int i = 1; i < k; ++i) prod = mul_distribute(prod, base);
        return prod;
      }
      return base.pow(ex);
    }
  }
  return e;  // unreachable
}

}  // namespace ax::sym
