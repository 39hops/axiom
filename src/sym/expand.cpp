#include <ax/sym/expand.hpp>

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
    for (const expr& t : b.args()) sum = sum + mul_distribute(a, t);
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
      return expr::fn(e.name(), expand(e.args()[0]));
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
