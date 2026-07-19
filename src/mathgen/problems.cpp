#include <ax/mathgen/problems.hpp>

#include <ax/sym/calc.hpp>
#include <ax/sym/print_sstr.hpp>

#include <format>
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
  // sympy inverse composition: exp(log(u)) -> u (auto-eval).
  if (name == "exp" && arg.is_fn() && arg.name() == "log")
    return arg.args()[0];
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
  // |c| is extracted; the sign stays with the rest: sqrt(-2*x) ->
  // sqrt(2)*sqrt(-x) (probed sympy behavior).
  const auto small_int = [](const expr& e, long long& out) {
    if (!e.is_num()) return false;
    const ax::rational& q = e.value();
    if (!(q.den() == ax::bigint(1))) return false;
    const bool neg = q < ax::rational{};
    const std::string s = (neg ? -q : q).num().to_string();
    if (s.size() > 9) return false;
    out = std::stoll(s);
    if (neg) out = -out;
    return out != 0;
  };
  long long c = 0;
  if (small_int(u, c) && c > 0) {
    const auto [s, r] = square_split(c);
    if (r == 1) return expr::num(s);
    return expr::num(s) * expr::fn("sqrt", expr::num(r));
  }
  if (u.is_mul() && small_int(u.args()[0], c)) {
    const auto fs = u.args();
    expr rest = fs[1];
    for (std::size_t i = 2; i < fs.size(); ++i) rest = rest * fs[i];
    if (c < 0) rest = expr::num(-1) * rest;
    const auto [s, r] = square_split(c < 0 ? -c : c);
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


/** Generic Lk poly(): a*X**n + b*X + c with per-level ranges, redrawn
    while the x-terms cancel (Python `while not p.has(X)`). */
expr lk_poly(python_random& rng, long long amax, long long nmin,
             long long nmax, long long babs, long long cmax) {
  for (;;) {
    const long long a = rng.randint(1, amax);
    const long long n = rng.randint(nmin, nmax);
    const long long b = rng.randint(-babs, babs);
    const long long c = rng.randint(0, cmax);
    const expr p = expr::num(a) * kX.pow(expr::num(n)) + expr::num(b) * kX +
                   expr::num(c);
    if (has_x(p)) return p;
  }
}

/** sympy: sqrt(p)**k collapses to p**(k/2), and a positive numeric
    coefficient in a Mul base is extracted: (3*x)**(3/2) ->
    3**(3/2)*x**(3/2) = 3*sqrt(3)*x**(3/2). */
expr sqrt_pow(const expr& p, long long k) {
  if (k == 1) return make_sqrt(p);
  const expr half_k = expr::num(ax::rational(ax::bigint(k), ax::bigint(2)));
  if (p.is_mul() && p.args()[0].is_num() &&
      p.args()[0].value().den() == ax::bigint(1)) {
    // |coeff| extracted, sign stays with the rest:
    // (-2*x)**(3/2) -> 2**(3/2)*(-x)**(3/2) (probed sympy behavior).
    const bool neg = p.args()[0].value() < ax::rational{};
    const expr c =
        neg ? expr::num(-p.args()[0].value()) : p.args()[0];
    expr rest = p.args()[1];
    for (std::size_t i = 2; i < p.args().size(); ++i)
      rest = rest * p.args()[i];
    if (neg) rest = expr::num(-1) * rest;
    // c**(k/2) with odd k: c**((k-1)/2) * sqrt(c)
    expr cpart = make_sqrt(c);
    for (long long i = 0; i < (k - 1) / 2; ++i) cpart = cpart * c;
    return cpart * rest.pow(half_k);
  }
  return p.pow(half_k);
}

/** Python repr of a double (shortest round-trip) — verified identical to
    std::format on the pyrand random() stream. */
std::string py_repr(double d) { return std::format("{}", d); }

expr expression_l5(python_random& rng);
expr expression_l6(python_random& rng);
expr expression_l7(python_random& rng);

expr expression_l5(python_random& rng) {
  const auto poly = [&rng]() { return lk_poly(rng, 5, 1, 3, 4, 5); };
  const auto cross = [&rng]() {
    const long long a = rng.randint(1, 4);
    const long long b = rng.randint(1, 4);
    const long long c = rng.randint(1, 9);
    const std::size_t t = rng.choice_index(2);  // sin cos
    return pymul(expr::num(c) * make_fn("exp", expr::num(a) * kX),
                 make_fn(t == 0 ? "sin" : "cos", expr::num(b) * kX));
  };
  const auto inv_trig = [&rng]() {
    const long long c = rng.randint(1, 9);
    const std::size_t t = rng.choice_index(2);  // atan asin
    const long long a = rng.randint(1, 3);
    const expr f = make_fn(t == 0 ? "atan" : "asin", expr::num(a) * kX);
    const long long d = rng.randint(1, 5);
    return expr::num(c) * f + expr::num(d) * kX;
  };
  const auto log_power = [&rng]() {
    const long long c = rng.randint(1, 9);
    const long long n = rng.randint(1, 3);
    const long long a = rng.randint(1, 5);
    const long long k = rng.randint(1, 2);
    return expr::num(c) * kX.pow(expr::num(n)) *
           expr::fn("log", expr::num(a) * kX).pow(expr::num(k));
  };
  const auto root = [&rng, &poly]() {
    const long long c = rng.randint(1, 9);
    const expr p = poly();
    const expr q = poly();
    // sympy merges same-base powers: sqrt(p)*p -> p**(3/2).
    if (q.same(p)) return expr::num(c) * sqrt_pow(p, 3);
    return expr::num(c) * make_sqrt(p) * q;
  };
  const std::size_t kind = rng.choice_index(5);
  if (kind == 4) {  // sum2
    std::vector<expr> parts;
    parts.push_back(cross());
    parts.push_back(inv_trig());
    parts.push_back(log_power());
    rng.shuffle(parts);
    return parts[0] + parts[1];
  }
  if (kind == 0) return cross();
  if (kind == 1) return inv_trig();
  if (kind == 2) return log_power();
  return root();
}

expr expression_l6(python_random& rng) {
  const auto poly = [&rng](long long dmax) {
    return lk_poly(rng, 5, 2, dmax, 4, 5);
  };
  const auto l5 = [&rng]() {
    python_random sub("l6-sub-" + py_repr(rng.random()));
    return expression_l5(sub);
  };
  const auto cross_prod = [&rng]() {
    const long long c = rng.randint(1, 6);
    const long long n = rng.randint(1, 2);
    const long long la = rng.randint(1, 4);
    const expr a = expr::num(c) * kX.pow(expr::num(n)) *
                   expr::fn("log", expr::num(la) * kX);
    const std::size_t t = rng.choice_index(2);
    const long long b = rng.randint(1, 3);
    const expr bb = make_fn(t == 0 ? "sin" : "cos", expr::num(b) * kX);
    return pymul(a, bb);
  };
  const auto quotient = [&rng, &poly]() {
    const expr p = poly(3);
    const expr q = poly(2);
    const long long c = rng.randint(1, 5);
    return p / q + expr::num(c) * kX;
  };
  const auto deep_arg = [&rng, &poly]() {
    const long long c = rng.randint(1, 6);
    const std::size_t t = rng.choice_index(3);
    return expr::num(c) * fn_of(t, poly(2));
  };
  const std::size_t kind = rng.choice_index(5);
  if (kind == 0) {  // triple
    const expr a = l5();
    const expr b = l5();
    const expr c = l5();
    return a + b + c;
  }
  if (kind == 1) return cross_prod();
  if (kind == 2) return quotient();
  if (kind == 3) return deep_arg();
  const expr d = deep_arg();  // deep_sum
  return d + l5();
}

expr expression_l7(python_random& rng) {
  const auto poly = [&rng](long long dmax) {
    return lk_poly(rng, 4, 1, dmax, 3, 4);
  };
  const auto nest2 = [&rng, &poly]() {
    // Python builds ALL list elements before rng.choice: log(poly()),
    // trig(randint*X), poly() are all drawn, then one is selected.
    const expr cand0 = expr::fn("log", poly(2));
    const std::size_t t = rng.choice_index(2);
    const long long b = rng.randint(1, 3);
    const expr cand1 = make_fn(t == 0 ? "sin" : "cos", expr::num(b) * kX);
    const expr cand2 = poly(2);
    const std::size_t pick = rng.choice_index(3);
    const expr inner = pick == 0 ? cand0 : pick == 1 ? cand1 : cand2;
    const std::size_t outer = rng.choice_index(4);  // sin cos exp log
    const long long c = rng.randint(1, 6);
    const expr f =
        outer == 3 ? expr::fn("log", inner) : fn_of(outer, inner);
    return expr::num(c) * f;
  };
  const auto nest_prod = [&rng, &nest2]() {
    const expr n2 = nest2();
    const long long np = rng.randint(1, 2);
    const long long la = rng.randint(1, 4);
    const std::size_t pick = rng.choice_index(2);
    const expr other = pick == 0 ? kX.pow(expr::num(np))
                                 : expr::fn("log", expr::num(la) * kX);
    return pymul(n2, other);
  };
  const auto atan_log = [&rng, &poly]() {
    const long long c = rng.randint(1, 6);
    const expr lg = expr::fn("log", poly(2));
    const long long a = rng.randint(1, 3);
    return expr::num(c) * lg * make_fn("atan", expr::num(a) * kX);
  };
  const auto root_wrap = [&rng, &poly]() {
    const long long c = rng.randint(1, 6);
    const expr p = poly(2);
    const long long k = rng.choice_index(2) == 0 ? 1 : 3;
    return expr::num(c) * sqrt_pow(p, k);
  };
  const std::size_t kind = rng.choice_index(5);
  if (kind == 0) return nest2();
  if (kind == 1) return nest_prod();
  if (kind == 2) return atan_log();
  if (kind == 3) return root_wrap();
  const expr n2 = nest2();  // nest_sum
  python_random sub("l7-sub-" + py_repr(rng.random()));
  return n2 + expression_l6(sub);
}

expr expression_l8(python_random& rng) {
  const auto poly = [&rng](long long dmax, long long dmin) {
    return lk_poly(rng, 4, dmin, dmax, 3, 4);
  };
  const auto trig_log = [&rng, &poly]() {
    const std::size_t t = rng.choice_index(2);
    const expr lg = expr::fn("log", poly(2, 1));
    const expr tf = make_fn(t == 0 ? "sin" : "cos", lg);
    const long long c = rng.randint(1, 6);
    const long long n = rng.randint(0, 2);
    return expr::num(c) * tf * kX.pow(expr::num(n));
  };
  const auto sqrt_log = [&rng, &poly]() {
    const long long c = rng.randint(1, 6);
    const expr p = poly(2, 1);
    const expr q = poly(2, 1);
    return expr::num(c) * make_sqrt(p) * expr::fn("log", q);
  };
  const auto sqrt_monster = [&rng, &poly]() {
    const long long c = rng.randint(1, 6);
    const expr p = poly(3, 2);
    const long long k = rng.choice_index(2) == 0 ? 1 : 3;
    const long long n = rng.randint(0, 2);
    return expr::num(c) * sqrt_pow(p, k) * kX.pow(expr::num(n));
  };
  const auto nest3 = [&rng, &poly]() {
    const expr inner = expr::fn("log", poly(2, 1));
    const std::size_t t = rng.choice_index(2);
    const expr mid = make_fn(t == 0 ? "sin" : "cos", inner);
    const std::size_t outer = rng.choice_index(3);
    if (outer == 0) return make_fn("exp", mid);
    if (outer == 1) return mid.pow(expr::num(2));
    const long long c = rng.randint(1, 4);
    return expr::num(c) * mid;
  };
  const std::size_t kind = rng.choice_index(5);
  if (kind == 0) return trig_log();
  if (kind == 1) return sqrt_log();
  if (kind == 2) return sqrt_monster();
  if (kind == 3) return nest3();
  const expr t = trig_log();  // combo_sum
  python_random sub("l8-sub-" + py_repr(rng.random()));
  return t + expression_l7(sub);
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
  if (level >= 8) return expression_l8(rng);
  if (level == 7) return expression_l7(rng);
  if (level == 6) return expression_l6(rng);
  if (level == 5) return expression_l5(rng);
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
