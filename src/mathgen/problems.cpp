#include <ax/mathgen/problems.hpp>

#include <ax/sym/calc.hpp>
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

bool is_exp(const expr& e) { return e.is_fn() && e.name() == "exp"; }

/** sympy Mul auto-eval: exp is E**arg, so exp factors combine —
    exp(a)*exp(b) -> exp(a+b), exp(a)**k -> exp(k*a). Applied to builder
    products (axiom's canonical form keeps them separate). */
expr fix_exp(const expr& e) {
  if (e.is_pow() && is_exp(e.args()[0]) && e.args()[1].is_num())
    return expr::fn("exp", e.args()[1] * e.args()[0].args()[0]);
  if (!e.is_mul()) return e;
  // Probed rule: exp(a)*exp(b) merges only when a and b are rational
  // multiples of the same term (exp(x)*exp(7*x) -> exp(8*x), but
  // exp(x)*exp(4*x + 5) stays apart). Group exp args by their
  // non-numeric part and sum coefficients within a group.
  struct group {
    expr term;
    expr coeff_sum;
    int count;
  };
  std::vector<group> groups;
  expr others = expr::num(1);
  bool saw_pow_of_exp = false;
  const auto add_arg = [&groups](const expr& arg) {
    expr coeff = expr::num(1);
    expr term = arg;
    if (arg.is_num()) {
      coeff = arg;
      term = expr::num(1);
    } else if (arg.is_mul() && arg.args()[0].is_num()) {
      coeff = arg.args()[0];
      term = arg.args()[1];
      for (std::size_t i = 2; i < arg.args().size(); ++i)
        term = term * arg.args()[i];
    }
    for (auto& g : groups)
      if (g.term.same(term)) {
        g.coeff_sum = g.coeff_sum + coeff;
        ++g.count;
        return;
      }
    groups.push_back({term, coeff, 1});
  };
  for (const expr& f : e.args()) {
    if (is_exp(f)) {
      add_arg(f.args()[0]);
    } else if (f.is_pow() && is_exp(f.args()[0]) && f.args()[1].is_num()) {
      add_arg(f.args()[1] * f.args()[0].args()[0]);
      saw_pow_of_exp = true;  // exp(x)*exp(x) arrives as exp(x)**2
    } else {
      others = others * f;
    }
  }
  bool merged = saw_pow_of_exp;
  for (const auto& g : groups)
    if (g.count > 1) merged = true;
  if (!merged) return e;
  expr out = others;
  for (const auto& g : groups)
    out = out * expr::fn("exp", g.coeff_sum * g.term);
  return out;
}

/** Product with sympy Mul semantics (exp merging). */
expr pymul(const expr& a, const expr& b) { return fix_exp(a * b); }

/** sympy sqrt auto-eval: sqrt(c*R) with positive rational c splits into
    s*sqrt(r)*sqrt(R) where c = s^2*r (sqrt(8*x) -> 2*sqrt(2)*sqrt(x)). */
expr make_sqrt(const expr& u) {
  const auto square_split = [](long long c) {  // c = s*s*r
    long long s = 1;
    for (long long k = 2; k * k <= c; ++k)
      while (c % (k * k) == 0) {
        c /= k * k;
        s *= k;
      }
    return std::pair<long long, long long>{s, c};
  };
  const auto small_pos_int = [](const expr& e, long long& out) {
    if (!e.is_num()) return false;
    const ax::rational& q = e.value();
    if (!(q.den() == ax::bigint(1)) || q < ax::rational{}) return false;
    const std::string s = q.num().to_string();
    if (s.size() > 9) return false;
    out = std::stoll(s);
    return out > 0;
  };
  long long c = 0;
  if (small_pos_int(u, c)) {
    const auto [s, r] = square_split(c);
    if (r == 1) return expr::num(s);
    return expr::num(s) * expr::fn("sqrt", expr::num(r));
  }
  if (u.is_mul() && small_pos_int(u.args()[0], c)) {
    const auto fs = u.args();
    expr rest = fs[1];
    for (std::size_t i = 2; i < fs.size(); ++i) rest = rest * fs[i];
    const auto [s, r] = square_split(c);
    expr out = expr::fn("sqrt", rest);
    if (r != 1) out = expr::fn("sqrt", expr::num(r)) * out;
    if (s != 1) out = expr::num(s) * out;
    return out;
  }
  return expr::fn("sqrt", u);
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

/** L4+ helper poly(): redraw while the x-terms cancelled away. */
bool has_x(const expr& e) {
  if (e.is_sym()) return e.name() == "x";
  for (const expr& a : e.args())
    if (has_x(a)) return true;
  return false;
}

expr l4_poly(python_random& rng) {
  for (;;) {
    const expr p = l3_inner_poly(rng);  // same draw shape as Python poly()
    if (has_x(p)) return p;
  }
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

expr expression_l4(python_random& rng) {
  // rng call order mirrors problems._expression_l4 exactly; every
  // multi-draw expression sequenced explicitly (C++ eval-order trap).
  const auto deep = [&rng]() {
    const std::size_t outer = rng.choice_index(3);  // sin cos exp
    const std::size_t mid = rng.choice_index(3);    // sin cos sqrt
    const long long c = rng.randint(1, 9);
    const expr p = l4_poly(rng);
    const long long mc = rng.randint(1, 9);
    const expr marg = expr::num(mc) * kX;
    const expr mfn = mid == 0   ? make_fn("sin", marg)
                     : mid == 1 ? make_fn("cos", marg)
                                : make_sqrt(marg);
    return expr::num(c) * fn_of(outer, p + mfn);
  };
  const auto composed_product = [&rng]() {
    const std::size_t outer = rng.choice_index(3);
    const expr a = atom(rng, 2);
    const expr p = l4_poly(rng);
    const long long c = rng.randint(1, 9);
    return pymul(pymul(a, fn_of(outer, p)), expr::num(c));
  };
  const auto chained = [&rng]() {
    const expr g = l4_poly(rng);
    const std::size_t f = rng.choice_index(3);
    const long long c = rng.randint(1, 9);
    const long long k = rng.randint(1, 4);
    // sympy: exp(g) is E**g, so exp(g)**k auto-folds to exp(k*g).
    const expr fk = f == 2 ? make_fn("exp", expr::num(k) * g)
                           : fn_of(f, g).pow(expr::num(k));
    return expr::num(c) * sym::diff(g, kX) * fk;
  };
  const std::size_t kind = rng.choice_index(4);
  if (kind == 0) return deep();
  if (kind == 1) return composed_product();
  if (kind == 2) return chained();
  std::vector<expr> parts;
  parts.push_back(deep());
  parts.push_back(composed_product());
  parts.push_back(chained());
  rng.shuffle(parts);
  return parts[0] + parts[1];
}

expr expression(python_random& rng, int level) {
  if (level <= 2) {
    const long long count = rng.randint(2, 4);
    expr out = expr::num(0);
    for (long long i = 0; i < count; ++i) out = out + atom(rng, level);
    return out;
  }
  if (level >= 5)
    throw std::invalid_argument("mathgen: levels 5-8 not ported yet");
  if (level == 4) return expression_l4(rng);
  // kind = rng.choice(["product", "compose", "mixed"])
  const std::size_t kind = rng.choice_index(3);
  // NB: C++ argument evaluation order is unspecified — every multi-draw
  // expression must be sequenced explicitly to match Python's
  // left-to-right semantics (this exact bug flipped 35/100 L3 rows).
  if (kind == 0) {
    const expr a = atom(rng, 2);
    const expr b = atom(rng, 2);
    return pymul(a, b);
  }
  if (kind == 1) {
    const expr inner = l3_inner_poly(rng);
    return fn_of(rng.choice_index(3), inner);
  }
  const expr a = atom(rng, 2);
  const expr b = atom(rng, 2);
  const expr c = atom(rng, 2);
  return pymul(a, b) + c;
}

}  // namespace ax::mathgen
