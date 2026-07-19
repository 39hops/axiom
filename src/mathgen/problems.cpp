#include <ax/mathgen/problems.hpp>

#include <ax/sym/print_sstr.hpp>

#include <stdexcept>
#include <vector>

namespace ax::mathgen {

using pyrand::python_random;
using sym::expr;

namespace {

const expr kX = expr::symbol("x");

/** sympy Expr.could_extract_minus_sign (Add rule from core/add.py):
    majority of args negative, ties broken by sort_key(e) < sort_key(-e). */
bool could_extract_minus(const expr& e) {
  if (e.is_num()) return e.value() < ax::rational{};
  if (e.is_mul())
    return e.args()[0].is_num() && e.args()[0].value() < ax::rational{};
  if (e.is_add()) {
    int neg = 0;
    for (const expr& t : e.args())
      if (could_extract_minus(t)) ++neg;
    const int pos = static_cast<int>(e.args().size()) - neg;
    if (pos != neg) return neg > pos;
    return sym::sympy_sort_cmp(e, -e) < 0;
  }
  return false;
}

/** sympy Function auto-eval: odd fns (sin/tan/asin/atan) pull an
    extractable minus out of the argument; cos is even and drops it. */
expr make_fn(const std::string& name, const expr& arg) {
  const bool odd =
      name == "sin" || name == "tan" || name == "asin" || name == "atan";
  const bool even = name == "cos";
  if ((odd || even) && could_extract_minus(arg)) {
    const expr inner = expr::fn(name, -arg);
    return odd ? -inner : inner;
  }
  return expr::fn(name, arg);
}

expr fn_of(std::size_t idx, const expr& arg) {
  static const char* names[] = {"sin", "cos", "exp"};
  return make_fn(names[idx], arg);
}

/** Python: rng.randint(1,5)*X**rng.randint(1,3) + rng.randint(-4,4)*X
    + rng.randint(0,5) — argument evaluation strictly left-to-right. */
expr l3_inner_poly(python_random& rng) {
  const long long a = rng.randint(1, 5);
  const long long n = rng.randint(1, 3);
  const long long b = rng.randint(-4, 4);
  const long long c = rng.randint(0, 5);
  return expr::num(a) * kX.pow(expr::num(n)) + expr::num(b) * kX +
         expr::num(c);
}

}  // namespace

std::string seed_string(const std::string& kind, int level, long long seed) {
  return kind + "-" + std::to_string(level) + "-" + std::to_string(seed);
}

expr atom(python_random& rng, int level) {
  // c = rng.randint(1, 9) * rng.choice([1, 1, 1, -1])
  long long c = rng.randint(1, 9);
  if (rng.choice_index(4) == 3) c = -c;
  // n = rng.randint(1, 3 if level == 1 else 5)
  const long long n = rng.randint(1, level == 1 ? 3 : 5);
  // choices built fully (no rng), then one rng.choice
  std::vector<expr> choices = {
      expr::num(c) * kX.pow(expr::num(n)),
      expr::num(c) * kX.pow(expr::num(n)),
      expr::num(c) * kX,
      expr::num(c),
  };
  if (level >= 2) {
    choices.push_back(expr::num(c) * expr::fn("sin", kX));
    choices.push_back(expr::num(c) * expr::fn("cos", kX));
    choices.push_back(expr::num(c) * expr::fn("exp", kX));
    choices.push_back(expr::num(c) * expr::fn("log", kX));
  }
  return choices[rng.choice_index(choices.size())];
}

expr expression(python_random& rng, int level) {
  if (level <= 2) {
    const long long count = rng.randint(2, 4);
    expr out = expr::num(0);
    for (long long i = 0; i < count; ++i) out = out + atom(rng, level);
    return out;
  }
  if (level != 3)
    throw std::invalid_argument("mathgen: levels 4-8 not ported yet");
  // kind = rng.choice(["product", "compose", "mixed"])
  const std::size_t kind = rng.choice_index(3);
  // NB: C++ argument evaluation order is unspecified — every multi-draw
  // expression must be sequenced explicitly to match Python's
  // left-to-right semantics (this exact bug flipped 35/100 L3 rows).
  if (kind == 0) {
    const expr a = atom(rng, 2);
    const expr b = atom(rng, 2);
    return a * b;
  }
  if (kind == 1) {
    const expr inner = l3_inner_poly(rng);
    return fn_of(rng.choice_index(3), inner);
  }
  const expr a = atom(rng, 2);
  const expr b = atom(rng, 2);
  const expr c = atom(rng, 2);
  return a * b + c;
}

}  // namespace ax::mathgen
