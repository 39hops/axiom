#include <ax/sym/count_ops.hpp>

#include <vector>

namespace ax::sym {

namespace {

const rational kZero{};
const rational kOne{bigint(1)};

bool is_int(const rational& q) { return q.den() == bigint(1); }

/** sympy _coeff_isneg: leading rational coefficient is negative. */
bool coeff_isneg(const expr& e) {
  if (e.is_num()) return e.value() < kZero;
  if (e.is_mul() && e.args()[0].is_num())
    return e.args()[0].value() < kZero;
  return false;
}

bool is_sqrt(const expr& e) { return e.is_fn() && e.name() == "sqrt"; }

long long count(const expr& e);

/** Rational contribution: (p<0) + (q!=1); One counts 0. */
long long count_rational(const rational& q) {
  if (q == kOne) return 0;
  long long ops = 0;
  if (q < kZero) ++ops;
  if (!is_int(q)) ++ops;
  return ops;
}

long long count_mul(const expr& e) {
  long long ops = 0;
  expr a = e;
  if (coeff_isneg(a)) {
    ++ops;  // NEG
    a = -a; // strip the sign (negating a Mul only flips its coefficient)
  }
  // fraction(a): split factors into numerator / denominator by negative
  // exponents; the rational coefficient contributes p to n, q to d.
  rational coeff = kOne;
  std::vector<expr> nf, df;  // df holds exponent-flipped factors
  const auto add_factor = [&](const expr& f) {
    if (f.is_num()) {
      coeff = coeff * f.value();
      return;
    }
    if (f.is_pow() && f.args()[1].is_num() &&
        f.args()[1].value() < kZero) {
      const rational flipped = -f.args()[1].value();
      df.push_back(flipped == kOne ? f.args()[0]
                                   : f.args()[0].pow(expr::num(flipped)));
      return;
    }
    if (is_sqrt(f) || !f.is_pow()) {
      nf.push_back(f);
      return;
    }
    nf.push_back(f);
  };
  if (a.is_mul())
    for (const expr& f : a.args()) add_factor(f);
  else
    add_factor(a);

  const bool den_nontrivial = !df.empty() || !is_int(coeff);
  const long long num_factor_ops = [&] {
    long long o = 0;
    for (const expr& f : nf) o += count(f);
    return o;
  }();
  const long long den_count = [&] {
    // count of d as a Mul: joins + factor counts (+ integer den = 0 ops)
    long long o = 0;
    std::size_t parts = df.size();
    if (!(coeff.den() == bigint(1))) ++parts;
    if (parts > 1) o += static_cast<long long>(parts) - 1;  // MULs
    for (const expr& f : df) o += count(f);
    return o;
  }();

  if (nf.empty()) {
    // n is an integer: DIV + (n<0 handled by the NEG strip) + count(d)
    if (den_nontrivial) return ops + 1 + den_count;
    return ops;  // pure number (already stripped)
  }
  if (den_nontrivial) {
    // DIV + count(n) + count(d)
    long long o = ops + 1 + den_count;
    // the coefficient's numerator joins n only when it is not 1
    // (fraction(x/2) has n = x, not (1/2's numerator)*x)
    const std::size_t nparts =
        nf.size() + (coeff.num() == bigint(1) ? 0 : 1);
    if (nparts > 1) o += static_cast<long long>(nparts) - 1;  // MULs in n
    return o + num_factor_ops;
  }
  // generic Mul: MUL*(len-1) over the original args (coeff included)
  std::size_t nargs = nf.size() + (coeff == kOne ? 0 : 1);
  long long o = ops;
  if (nargs > 1) o += static_cast<long long>(nargs) - 1;
  return o + num_factor_ops + count_rational(coeff);
}

long long count(const expr& e) {
  switch (e.k()) {
    case kind::num:
      return count_rational(e.value());
    case kind::sym:
      return e.name() == "E" ? 1 : 0;  // E counts as EXP
    case kind::add: {
      long long ops =
          static_cast<long long>(e.args().size()) - 1;  // ADD/SUB joins
      bool all_neg = true;
      for (const expr& t : e.args()) {
        if (coeff_isneg(t)) {
          ops += count(-t);
        } else {
          all_neg = false;
          ops += count(t);
        }
      }
      if (all_neg) ++ops;  // -x - y = NEG + SUB
      return ops;
    }
    case kind::mul:
      return count_mul(e);
    case kind::pow: {
      const expr& b = e.args()[0];
      const expr& ex = e.args()[1];
      if (ex.is_num() && ex.value() == rational(bigint(-1)))
        return 1 + count(b);  // DIV
      if (b.is_sym() && b.name() == "E") return 1 + count(ex);  // EXP
      return 1 + count(b) + count(ex);  // POW
    }
    case kind::fn: {
      const std::string& n = e.name();
      long long ops = 0;
      if (n == "Subs")
        ops = 0;  // Subs is not a counted class in sympy
      else if (is_sqrt(e))
        ops = 2;  // Pow(u, 1/2): POW + DIV
      else
        ops = 1;  // functions, exp (as EXP), Integral, Derivative
      for (const expr& a : e.args()) ops += count(a);
      return ops;
    }
  }
  return 0;
}

}  // namespace

long long count_ops(const expr& e) { return count(e); }

}  // namespace ax::sym
