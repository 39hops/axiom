#include <ax/sym/print_sstr.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <numbers>
#include <string>
#include <vector>

namespace ax::sym {

namespace {

const rational kZero{};
const rational kOne{bigint(1)};
const rational kMinusOne{bigint(-1)};
const rational kHalf{bigint(1), bigint(2)};

bool is_int(const rational& q) { return q.den() == bigint(1); }

// ---------------------------------------------------------------- views
// sqrt(u) is a Pow(u, 1/2) in sympy; axiom stores it as an fn node. All
// structural logic below goes through this view.

bool is_sqrt(const expr& e) { return e.is_fn() && e.name() == "sqrt"; }

struct pow_view {
  expr base;
  bool exp_is_num;
  rational exp_num;   // valid when exp_is_num
  expr exp_expr;      // valid otherwise
};

/** Pow decomposition per sympy as_base_exp; nullopt = not a power. */
std::optional<pow_view> as_pow(const expr& e) {
  if (is_sqrt(e))
    return pow_view{e.args()[0], true, kHalf, expr::num(0)};
  if (e.is_pow()) {
    const expr& ex = e.args()[1];
    if (ex.is_num()) return pow_view{e.args()[0], true, ex.value(), ex};
    return pow_view{e.args()[0], false, kOne, ex};
  }
  return std::nullopt;
}

/** as_coeff_Mul: leading rational coefficient and the rest. */
std::pair<rational, expr> as_coeff_mul(const expr& e) {
  if (e.is_num()) return {e.value(), expr::num(1)};
  if (e.is_mul() && e.args()[0].is_num()) {
    const auto fs = e.args();
    expr rest = fs[1];
    for (std::size_t i = 2; i < fs.size(); ++i) rest = rest * fs[i];
    return {fs[0].value(), rest};
  }
  return {kOne, e};
}

/** Is e numeric (no symbols beyond pi/E)? Value for coefficient folding. */
bool numeric_value(const expr& e, double& out) {
  try {
    out = e.eval({{"pi", std::numbers::pi}, {"E", std::numbers::e}});
  } catch (const std::exception&) {
    return false;
  }
  return true;
}

bool is_numberlike(const expr& e) {
  double v;
  if (e.is_num()) return true;
  if (e.is_sym()) return e.name() == "pi" || e.name() == "E";
  // sqrt(2), 2**(1/2), pi**2 ... : numeric subtrees
  switch (e.k()) {
    case kind::pow:
    case kind::mul:
    case kind::add:
      return numeric_value(e, v);
    case kind::fn:
      return numeric_value(e, v);
    default:
      return false;
  }
}

// ------------------------------------------------------------- sort_key
// Faithful transcription of sympy Expr.sort_key() (probed on 1.14) as a
// three-way comparison; no key objects are materialized.

struct class_key {
  int a;
  long long b;
  std::string name;
};

int cmp(const class_key& x, const class_key& y) {
  if (x.a != y.a) return x.a < y.a ? -1 : 1;
  if (x.b != y.b) return x.b < y.b ? -1 : 1;
  return x.name.compare(y.name);
}

class_key class_key_of(const expr& rest) {
  switch (rest.k()) {
    case kind::num: return {1, 0, "Number"};
    case kind::sym:
      if (rest.name() == "E") return {2, 0, "Exp1"};
      if (rest.name() == "pi") return {2, 0, "Pi"};
      return {2, 0, "Symbol"};
    case kind::mul: return {3, 0, "Mul"};
    case kind::add: return {3, 1, "Add"};
    case kind::pow: return {3, 2, "Pow"};
    case kind::fn: {
      const std::string& n = rest.name();
      if (n == "exp") return {4, 10, "exp"};
      if (n == "log") return {4, 11, "log"};
      if (n == "sin") return {4, 20, "sin"};
      if (n == "cos") return {4, 21, "cos"};
      if (n == "tan") return {4, 22, "tan"};
      return {4, 10000, n};
    }
  }
  return {9, 0, "?"};
}

int cmp_sort_key(const expr& a, const expr& b);
std::vector<expr> ordered_terms(const expr& add);
std::vector<expr> ordered_factors(const expr& e);

int cmp_rat(const rational& x, const rational& y) {
  if (x == y) return 0;
  return x < y ? -1 : 1;
}

/** Decomposed sort key: coeff * base^expo, plus the pure-number flag
    (Numbers key as ((1,0,Number),(0,()),(),value)). */
struct key_parts {
  bool pure_number = false;
  rational coeff = kOne;
  expr base;
  bool exp_is_num = true;
  rational exp_num = kOne;
  expr exp_expr;
  key_parts() : base(expr::num(0)), exp_expr(expr::num(0)) {}
};

key_parts decompose_key(const expr& e) {
  key_parts k;
  if (e.is_num()) {
    k.pure_number = true;
    k.coeff = e.value();
    return k;
  }
  auto [c, rest] = as_coeff_mul(e);
  k.coeff = c;
  if (const auto pv = as_pow(rest)) {
    k.base = pv->base;
    k.exp_is_num = pv->exp_is_num;
    k.exp_num = pv->exp_num;
    k.exp_expr = pv->exp_expr;
  } else {
    k.base = rest;
  }
  return k;
}

/** Compare the `args` component for two bases of equal class. */
int cmp_args(const expr& x, const expr& y) {
  const auto list_of = [](const expr& e) -> std::vector<expr> {
    if (e.is_add()) return ordered_terms(e);
    if (e.is_mul()) return ordered_factors(e);
    std::vector<expr> out;
    for (const expr& a : e.args()) out.push_back(a);
    return out;
  };
  // Atoms: args = (1, (str,)).
  const bool xa = x.is_sym() || x.is_num();
  const bool ya = y.is_sym() || y.is_num();
  if (xa || ya) {
    if (xa != ya) {
      // atom args len 1 vs composite len n
      const std::size_t xl = xa ? 1 : list_of(x).size();
      const std::size_t yl = ya ? 1 : list_of(y).size();
      if (xl != yl) return xl < yl ? -1 : 1;
      return 0;  // same length, incomparable shapes: treat equal
    }
    const std::string xs = x.is_sym() ? x.name() : x.value().to_string();
    const std::string ys = y.is_sym() ? y.name() : y.value().to_string();
    return xs.compare(ys);
  }
  const auto xs = list_of(x);
  const auto ys = list_of(y);
  if (xs.size() != ys.size()) return xs.size() < ys.size() ? -1 : 1;
  for (std::size_t i = 0; i < xs.size(); ++i)
    if (const int c = cmp_sort_key(xs[i], ys[i])) return c;
  return 0;
}

int cmp_sort_key(const expr& a, const expr& b) {
  const key_parts ka = decompose_key(a);
  const key_parts kb = decompose_key(b);

  // class_key
  const class_key ca =
      ka.pure_number ? class_key{1, 0, "Number"} : class_key_of(ka.base);
  const class_key cb =
      kb.pure_number ? class_key{1, 0, "Number"} : class_key_of(kb.base);
  if (const int c = cmp(ca, cb)) return c;

  // args: pure numbers have (0, ()); everything else per cmp_args
  if (ka.pure_number != kb.pure_number)
    return ka.pure_number ? -1 : 1;  // (0,()) < (1,(...))
  if (!ka.pure_number)
    if (const int c = cmp_args(ka.base, kb.base)) return c;

  // exp: pure numbers carry the empty tuple (equal for both); otherwise
  // compare exponents as sort keys (numeric exponents reduce to Number
  // keys, i.e. their rational values).
  if (!ka.pure_number) {
    if (ka.exp_is_num && kb.exp_is_num) {
      if (const int c = cmp_rat(ka.exp_num, kb.exp_num)) return c;
    } else if (ka.exp_is_num != kb.exp_is_num) {
      return ka.exp_is_num ? -1 : 1;  // Number class < symbolic classes
    } else {
      if (const int c = cmp_sort_key(ka.exp_expr, kb.exp_expr)) return c;
    }
  }

  return cmp_rat(ka.coeff, kb.coeff);
}

// ---------------------------------------------------- as_ordered_factors

/** sympy distributes integer exponents over Mul at construction
    ((2*x)**-1 -> 2**-1 * x**-1); axiom keeps the Pow. Splice such factors
    into per-base powers locally — a view, never a tree rebuild (rebuilding
    through expr operators re-canonicalizes and diverges from sympy). */
void push_factor(const expr& f, std::vector<expr>& out) {
  const auto pv = as_pow(f);
  if (pv && pv->exp_is_num && is_int(pv->exp_num) &&
      !(pv->exp_num == kOne) && pv->base.is_mul()) {
    const expr k = expr::num(pv->exp_num);
    for (const expr& g : pv->base.args()) out.push_back(g.pow(k));
    return;
  }
  out.push_back(f);
}

std::vector<expr> factor_view(const expr& e) {
  std::vector<expr> fs;
  if (e.is_mul())
    for (const expr& f : e.args()) push_factor(f, fs);
  else
    push_factor(e, fs);
  return fs;
}

std::vector<expr> ordered_factors(const expr& e) {
  std::vector<expr> fs = factor_view(e);
  std::stable_sort(fs.begin(), fs.end(), [](const expr& x, const expr& y) {
    return cmp_sort_key(x, y) < 0;
  });
  return fs;
}

// ------------------------------------------------------ as_ordered_terms

/** decompose_power per sympy exprtools: integer exponent split; rational
    p/q -> (base^(1/q), p); exp(c*u) behaves as E**(c*u). */
std::pair<expr, rational> decompose_power(const expr& f) {
  // exp fn: E**arg
  if (f.is_fn() && f.name() == "exp") {
    const expr& u = f.args()[0];
    auto [c, tail] = as_coeff_mul(u);
    if (c == kOne || tail.is_num())  // exp(x) or exp(const)
      return {f, kOne};
    if (is_int(c)) {
      if (c == kMinusOne) return {expr::fn("exp", tail), kMinusOne};
      // base = exp(tail * 1/den(c)) with den = 1: exp(tail), e = num(c)
      return {expr::fn("exp", tail), c};
    }
    const rational inv_den(kOne.num(), c.den());
    return {expr::fn("exp", expr::num(inv_den) * tail),
            rational(c.num(), bigint(1))};
  }
  const auto pv = as_pow(f);
  if (!pv) return {f, kOne};
  if (pv->exp_is_num) {
    const rational& q = pv->exp_num;
    if (is_int(q)) return {pv->base, q};
    const rational root(bigint(1), q.den());
    return {pv->base.pow(expr::num(root)), rational(q.num(), bigint(1))};
  }
  // symbolic exponent: (c, tail) = exp.as_coeff_Mul
  auto [c, tail] = as_coeff_mul(pv->exp_expr);
  if (c == kMinusOne) return {pv->base.pow(tail), kMinusOne};
  if (c == kOne) return {f, kOne};
  const rational inv_den(bigint(1), c.den());
  return {pv->base.pow(expr::num(inv_den) * tail),
          rational(c.num(), bigint(1))};
}

struct term_info {
  expr term;
  double coeff;
  std::vector<rational> monom;  // over gens
  term_info() : term(expr::num(0)) {}
};

struct expr_cmp_less {
  bool operator()(const expr& a, const expr& b) const {
    if (const int c = cmp_sort_key(a, b)) return c < 0;
    // sort_key ties: fall back to structural order for map identity
    return expr::compare(a, b) < 0;
  }
};

std::vector<expr> ordered_terms(const expr& add) {
  std::vector<expr> terms;
  if (add.is_add())
    for (const expr& t : add.args()) terms.push_back(t);
  else
    terms.push_back(add);

  // Special case: Add(positive Number, Mul with negative Number coeff).
  if (terms.size() == 2) {
    const expr* numarg = nullptr;
    const expr* other = nullptr;
    for (const expr& t : terms) {
      if (t.is_num()) numarg = &t;
      else other = &t;
    }
    if (numarg && other && kZero < numarg->value() && other->is_mul() &&
        other->args()[0].is_num() && other->args()[0].value() < kZero)
      return {*numarg, *other};
  }

  // as_terms: per-term coeff + cpart monomial
  std::vector<term_info> infos;
  std::map<expr, std::size_t, expr_cmp_less> gen_index;
  std::vector<std::map<expr, rational, expr_cmp_less>> cparts;
  for (const expr& t : terms) {
    term_info ti;
    ti.term = t;
    auto [c, rest] = as_coeff_mul(t);
    double coeff = 0.0;
    (void)numeric_value(expr::num(c), coeff);
    std::map<expr, rational, expr_cmp_less> cpart;
    if (!(rest.is_num() && rest.value() == kOne)) {
      const std::vector<expr> factors = factor_view(rest);
      for (const expr& f : factors) {
        double v;
        if (is_numberlike(f) && numeric_value(f, v)) {
          coeff *= v;
          continue;
        }
        auto [base, ex] = decompose_power(f);
        cpart[base] = ex;
        gen_index.emplace(base, 0);
      }
    }
    ti.coeff = coeff;
    infos.push_back(ti);
    cparts.push_back(std::move(cpart));
  }
  std::size_t gi = 0;
  for (auto& [g, idx] : gen_index) idx = gi++;
  for (std::size_t i = 0; i < infos.size(); ++i) {
    infos[i].monom.assign(gen_index.size(), kZero);
    for (const auto& [base, ex] : cparts[i])
      infos[i].monom[gen_index.at(base)] = ex;
  }

  std::stable_sort(infos.begin(), infos.end(),
                   [](const term_info& x, const term_info& y) {
                     // descending lex on monom
                     for (std::size_t i = 0; i < x.monom.size(); ++i) {
                       if (!(x.monom[i] == y.monom[i]))
                         return y.monom[i] < x.monom[i];
                     }
                     return x.coeff < y.coeff;  // ascending coefficient
                   });

  std::vector<expr> out;
  for (const auto& ti : infos) out.push_back(ti.term);
  return out;
}

// ------------------------------------------------------------- printing
// Precedence per sympy: Add 40, Mul 50, Pow 60, Atom 1000.

constexpr int kPrecAdd = 40;
constexpr int kPrecMul = 50;
constexpr int kPrecPow = 60;
constexpr int kPrecAtom = 1000;

std::string print(const expr& e);

int precedence_of(const expr& e) {
  switch (e.k()) {
    case kind::num: {
      const rational& q = e.value();
      if (q < kZero) return kPrecAdd;          // leading minus
      if (!is_int(q)) return kPrecMul;         // p/q renders as division
      return kPrecAtom;
    }
    case kind::sym:
    case kind::fn:
      return is_sqrt(e) ? kPrecAtom : kPrecAtom;
    case kind::add: return kPrecAdd;
    case kind::mul: return kPrecMul;
    case kind::pow: return kPrecPow;
  }
  return kPrecAtom;
}

std::string parenthesize(const expr& e, int level) {
  const std::string s = print(e);
  if (precedence_of(e) < level) return "(" + s + ")";
  return s;
}

std::string print_pow(const expr& base, const expr& ex) {
  if (ex.is_num()) {
    const rational& q = ex.value();
    if (q == kHalf) return "sqrt(" + print(base) + ")";
    if (q == -kHalf) return "1/sqrt(" + print(base) + ")";
    if (q == kMinusOne) {
      // sympy parenthesizes strictly here: 1/(2*sqrt(2 - x)).
      const std::string bs = print(base);
      if (precedence_of(base) <= kPrecMul) return "1/(" + bs + ")";
      return "1/" + bs;
    }
  }
  std::string bs = parenthesize(base, kPrecPow);
  if (base.is_pow()) bs = "(" + print(base) + ")";
  const std::string es = parenthesize(ex, kPrecPow);
  return bs + "**" + es;
}

std::string print_mul(const expr& e) {
  auto [c, rest] = as_coeff_mul(e);
  std::string sign;
  if (c < kZero) {
    sign = "-";
    c = -c;
  }
  // rebuild factor list: coefficient (if != 1) + rest factors, sorted
  std::vector<expr> factors;
  if (!(c == kOne)) factors.push_back(expr::num(c));
  if (!(rest.is_num() && rest.value() == kOne))
    for (const expr& f : factor_view(rest)) factors.push_back(f);
  std::stable_sort(factors.begin(), factors.end(),
                   [](const expr& x, const expr& y) {
                     return cmp_sort_key(x, y) < 0;
                   });

  std::vector<std::string> num_parts, den_parts;
  for (const expr& f : factors) {
    const auto pv = as_pow(f);
    const bool neg_exp =
        pv && ((pv->exp_is_num && pv->exp_num < kZero) ||
               (!pv->exp_is_num &&
                as_coeff_mul(pv->exp_expr).first < kZero));
    if (neg_exp) {
      // move to denominator with negated exponent
      expr inv = expr::num(1);
      if (pv->exp_is_num) {
        const rational ne = -pv->exp_num;
        inv = (ne == kOne) ? pv->base : pv->base.pow(expr::num(ne));
      } else {
        inv = pv->base.pow(-pv->exp_expr);
      }
      // sympy #14160: parenthesize Mul/Pow bases landing bare in the
      // denominator
      std::string ds = parenthesize(inv, kPrecMul);
      if ((inv.is_mul() || inv.is_pow()) && pv->exp_is_num &&
          pv->exp_num == kMinusOne && inv.is_mul())
        ds = "(" + print(inv) + ")";
      den_parts.push_back(ds);
      continue;
    }
    if (f.is_num()) {
      const rational& q = f.value();
      if (!(q.num() == bigint(1)))
        num_parts.push_back(rational(q.num(), bigint(1)).to_string());
      if (!(q.den() == bigint(1)))
        den_parts.push_back(q.den().to_string());
      continue;
    }
    num_parts.push_back(parenthesize(f, kPrecMul));
  }
  if (num_parts.empty()) num_parts.push_back("1");

  std::string n;
  for (std::size_t i = 0; i < num_parts.size(); ++i) {
    if (i) n += "*";
    n += num_parts[i];
  }
  if (den_parts.empty()) return sign + n;
  std::string d;
  for (std::size_t i = 0; i < den_parts.size(); ++i) {
    if (i) d += "*";
    d += den_parts[i];
  }
  if (den_parts.size() > 1) return sign + n + "/(" + d + ")";
  return sign + n + "/" + d;
}

std::string print_add(const expr& e) {
  const std::vector<expr> terms = ordered_terms(e);
  std::string out;
  bool first = true;
  for (const expr& t : terms) {
    std::string ts = print(t);
    std::string sgn = "+";
    if (!ts.empty() && ts[0] == '-') {
      sgn = "-";
      ts = ts.substr(1);
    }
    if (first) {
      out = (sgn == "-" ? "-" : "") + ts;
      first = false;
    } else {
      out += " " + sgn + " " + ts;
    }
  }
  return out;
}

std::string print(const expr& e) {
  switch (e.k()) {
    case kind::num:
      return e.value().to_string();
    case kind::sym:
      return e.name();
    case kind::fn:
      if (is_sqrt(e)) return "sqrt(" + print(e.args()[0]) + ")";
      return e.name() + "(" + print(e.args()[0]) + ")";
    case kind::add:
      return print_add(e);
    case kind::mul:
      return print_mul(e);
    case kind::pow:
      return print_pow(e.args()[0], e.args()[1]);
  }
  return "?";
}

}  // namespace

std::string to_sstr(const expr& e) { return print(e); }

}  // namespace ax::sym
