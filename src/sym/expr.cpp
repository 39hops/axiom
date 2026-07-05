#include <ax/sym/expr.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace ax::sym {

struct node {
  kind k;
  rational val;           // num
  std::string nm;         // sym / fn
  std::vector<expr> a;    // operands
  std::size_t h = 0;      // cached structural hash
};

struct node_access {
  static const node* get(const expr& e) { return e.p_.get(); }
  static expr wrap(std::shared_ptr<const node> p) { return expr(std::move(p)); }
};

namespace {

std::size_t mix(std::size_t seed, std::size_t v) {
  return seed ^ (v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

std::size_t node_hash(const node& n) {
  std::size_t h = static_cast<std::size_t>(n.k) * 0x9e3779b9U;
  if (n.k == kind::num)
    h = mix(h, std::hash<std::string>{}(n.val.num().to_string() + "/" +
                                        n.val.den().to_string()));
  if (!n.nm.empty()) h = mix(h, std::hash<std::string>{}(n.nm));
  for (const expr& e : n.a) h = mix(h, e.hash());
  return h;
}

bool node_equal(const node& x, const node& y) {
  if (x.k != y.k || x.nm != y.nm || x.a.size() != y.a.size()) return false;
  if (x.k == kind::num && !(x.val == y.val)) return false;
  for (std::size_t i = 0; i < x.a.size(); ++i)
    if (!x.a[i].same(y.a[i])) return false;
  return true;
}

/** Global hash-consing pool. Nodes live as long as any expr references
    them; expired entries are lazily swept on collision. */
class pool {
 public:
  static expr intern(node&& n) {
    static pool p;
    n.h = node_hash(n);
    std::lock_guard<std::mutex> lock(p.mu_);
    auto [lo, hi] = p.map_.equal_range(n.h);
    for (auto it = lo; it != hi;) {
      if (auto sp = it->second.lock()) {
        if (node_equal(*sp, n)) return node_access::wrap(std::move(sp));
        ++it;
      } else {
        it = p.map_.erase(it);
      }
    }
    auto sp = std::make_shared<const node>(std::move(n));
    p.map_.emplace(sp->h, sp);
    return node_access::wrap(std::move(sp));
  }

 private:
  std::mutex mu_;
  std::unordered_multimap<std::size_t, std::weak_ptr<const node>> map_;
};

const node& deref(const expr& e) { return *node_access::get(e); }

expr make_num(const rational& q) {
  return pool::intern(node{kind::num, q, {}, {}, 0});
}
expr num_i(std::int64_t v) { return make_num(rational(bigint(v))); }

/** Exponent as small integer if exact; nullopt otherwise. */
std::optional<long long> as_small_int(const rational& q) {
  if (!(q.den() == bigint(1))) return std::nullopt;
  try {
    return std::stoll(q.num().to_string());
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<rational> rational_pow(const rational& base, long long e) {
  if (e == 0) return rational(bigint(1));
  const long long mag = e < 0 ? -e : e;
  if (mag > 1024) return std::nullopt;  // keep symbolic, avoid blowup
  rational r(bigint(1));
  for (long long i = 0; i < mag; ++i) r = r * base;
  if (e < 0) {
    if (r.is_zero()) return std::nullopt;
    return rational(bigint(1)) / r;
  }
  return r;
}

expr make_add(std::vector<expr> terms);
expr make_mul(std::vector<expr> factors);

expr make_pow(const expr& b, const expr& e) {
  const node& bn = deref(b);
  const node& en = deref(e);
  if (en.k == kind::num) {
    if (en.val.is_zero()) return num_i(1);
    if (en.val == rational(bigint(1))) return b;
    if (bn.k == kind::num) {
      if (const auto ie = as_small_int(en.val)) {
        if (const auto folded = rational_pow(bn.val, *ie))
          return make_num(*folded);
      }
    }
    // (x^a)^b with numeric a, b -> x^(a*b)
    if (bn.k == kind::pow) {
      const node& inner_exp = deref(bn.a[1]);
      if (inner_exp.k == kind::num)
        return make_pow(bn.a[0], make_num(inner_exp.val * en.val));
    }
  }
  return pool::intern(node{kind::pow, {}, {}, {b, e}, 0});
}

struct expr_less {
  bool operator()(const expr& a, const expr& b) const {
    return expr::compare(a, b) < 0;
  }
};

expr make_mul(std::vector<expr> factors) {
  // flatten
  std::vector<expr> flat;
  for (const expr& f : factors) {
    const node& n = deref(f);
    if (n.k == kind::mul)
      flat.insert(flat.end(), n.a.begin(), n.a.end());
    else
      flat.push_back(f);
  }
  rational coeff(bigint(1));
  std::map<expr, expr, expr_less> pows;  // base -> exponent sum
  for (const expr& f : flat) {
    const node& n = deref(f);
    if (n.k == kind::num) {
      coeff = coeff * n.val;
      continue;
    }
    expr base = f, ex = num_i(1);
    if (n.k == kind::pow) {
      base = n.a[0];
      ex = n.a[1];
    }
    const auto it = pows.find(base);
    if (it == pows.end())
      pows.emplace(base, ex);
    else
      it->second = it->second + ex;  // exponent arithmetic re-canonicalizes
  }
  if (coeff.is_zero()) return num_i(0);
  std::vector<expr> out;
  for (const auto& [base, ex] : pows) {
    const expr p = make_pow(base, ex);
    const node& pn = deref(p);
    if (pn.k == kind::num)
      coeff = coeff * pn.val;
    else
      out.push_back(p);
  }
  std::sort(out.begin(), out.end(), expr_less{});
  if (out.empty()) return make_num(coeff);
  const bool unit = coeff == rational(bigint(1));
  // distribute a numeric coefficient over a lone sum: -(x+y) -> -x - y
  if (!unit && out.size() == 1 && deref(out[0]).k == kind::add) {
    std::vector<expr> terms;
    for (const expr& t : deref(out[0]).a)
      terms.push_back(make_mul({make_num(coeff), t}));
    return make_add(std::move(terms));
  }
  if (unit && out.size() == 1) return out[0];
  std::vector<expr> ops;
  if (!unit) ops.push_back(make_num(coeff));
  ops.insert(ops.end(), out.begin(), out.end());
  if (ops.size() == 1) return ops[0];
  return pool::intern(node{kind::mul, {}, {}, std::move(ops), 0});
}

expr make_add(std::vector<expr> terms) {
  std::vector<expr> flat;
  for (const expr& t : terms) {
    const node& n = deref(t);
    if (n.k == kind::add)
      flat.insert(flat.end(), n.a.begin(), n.a.end());
    else
      flat.push_back(t);
  }
  rational constant;
  std::map<expr, rational, expr_less> coeffs;  // term -> coefficient
  for (const expr& t : flat) {
    const node& n = deref(t);
    if (n.k == kind::num) {
      constant = constant + n.val;
      continue;
    }
    rational c(bigint(1));
    expr rest = t;
    if (n.k == kind::mul && deref(n.a[0]).k == kind::num) {
      c = deref(n.a[0]).val;
      std::vector<expr> remaining(n.a.begin() + 1, n.a.end());
      rest = remaining.size() == 1 ? remaining[0]
                                   : pool::intern(node{kind::mul, {}, {},
                                                       std::move(remaining), 0});
    }
    const auto it = coeffs.find(rest);
    if (it == coeffs.end())
      coeffs.emplace(rest, c);
    else
      it->second = it->second + c;
  }
  std::vector<expr> out;
  for (const auto& [rest, c] : coeffs) {
    if (c.is_zero()) continue;
    if (c == rational(bigint(1)))
      out.push_back(rest);
    else
      out.push_back(make_mul({make_num(c), rest}));
  }
  std::sort(out.begin(), out.end(), expr_less{});
  if (!constant.is_zero()) out.push_back(make_num(constant));
  if (out.empty()) return num_i(0);
  if (out.size() == 1) return out[0];
  return pool::intern(node{kind::add, {}, {}, std::move(out), 0});
}

double to_double(const rational& q) {
  return std::stod(q.num().to_string()) / std::stod(q.den().to_string());
}

}  // namespace

// ---------------------------------------------------------------- expr

expr expr::num(std::int64_t n) { return num_i(n); }
expr expr::num(const rational& q) { return make_num(q); }
expr expr::symbol(std::string name) {
  return pool::intern(node{kind::sym, {}, std::move(name), {}, 0});
}
expr expr::fn(std::string name, expr arg) {
  return pool::intern(node{kind::fn, {}, std::move(name), {std::move(arg)}, 0});
}

kind expr::k() const { return p_->k; }
const rational& expr::value() const {
  if (!is_num()) throw std::logic_error("expr::value: not a num");
  return p_->val;
}
const std::string& expr::name() const {
  if (!is_sym() && !is_fn()) throw std::logic_error("expr::name: no name");
  return p_->nm;
}
std::span<const expr> expr::args() const { return p_->a; }
std::size_t expr::hash() const { return p_->h; }

int expr::compare(const expr& a, const expr& b) {
  if (a.same(b)) return 0;
  const node& x = deref(a);
  const node& y = deref(b);
  if (x.k != y.k) return x.k < y.k ? -1 : 1;
  switch (x.k) {
    case kind::num: {
      const auto c = x.val <=> y.val;
      return c < 0 ? -1 : 1;  // equal impossible: same() was false
    }
    case kind::sym:
      return x.nm.compare(y.nm);
    case kind::fn: {
      if (const int c = x.nm.compare(y.nm)) return c;
      return compare(x.a[0], y.a[0]);
    }
    default: {  // add, mul, pow: lexicographic on operands
      const std::size_t n = std::min(x.a.size(), y.a.size());
      for (std::size_t i = 0; i < n; ++i)
        if (const int c = compare(x.a[i], y.a[i])) return c;
      if (x.a.size() != y.a.size()) return x.a.size() < y.a.size() ? -1 : 1;
      return 0;
    }
  }
}

expr operator+(const expr& a, const expr& b) { return make_add({a, b}); }
expr operator-(const expr& a, const expr& b) {
  return make_add({a, make_mul({num_i(-1), b})});
}
expr operator*(const expr& a, const expr& b) { return make_mul({a, b}); }
expr operator/(const expr& a, const expr& b) {
  return make_mul({a, make_pow(b, num_i(-1))});
}
expr operator-(const expr& a) { return make_mul({num_i(-1), a}); }
expr expr::pow(const expr& exponent) const {
  return make_pow(*this, exponent);
}

expr expr::subs(const expr& sym, const expr& replacement) const {
  if (same(sym)) return replacement;
  const node& n = *p_;
  switch (n.k) {
    case kind::num:
    case kind::sym:
      return *this;
    case kind::add: {
      std::vector<expr> ts;
      for (const expr& t : n.a) ts.push_back(t.subs(sym, replacement));
      return make_add(std::move(ts));
    }
    case kind::mul: {
      std::vector<expr> fs;
      for (const expr& f : n.a) fs.push_back(f.subs(sym, replacement));
      return make_mul(std::move(fs));
    }
    case kind::pow:
      return make_pow(n.a[0].subs(sym, replacement),
                      n.a[1].subs(sym, replacement));
    case kind::fn:
      return fn(n.nm, n.a[0].subs(sym, replacement));
  }
  throw std::logic_error("expr::subs: unreachable");
}

double expr::eval(const std::map<std::string, double>& env) const {
  const node& n = *p_;
  switch (n.k) {
    case kind::num:
      return to_double(n.val);
    case kind::sym: {
      const auto it = env.find(n.nm);
      if (it == env.end())
        throw std::logic_error("expr::eval: unbound symbol " + n.nm);
      return it->second;
    }
    case kind::add: {
      double s = 0.0;
      for (const expr& t : n.a) s += t.eval(env);
      return s;
    }
    case kind::mul: {
      double p = 1.0;
      for (const expr& f : n.a) p *= f.eval(env);
      return p;
    }
    case kind::pow:
      return std::pow(n.a[0].eval(env), n.a[1].eval(env));
    case kind::fn: {
      const double x = n.a[0].eval(env);
      if (n.nm == "sin") return std::sin(x);
      if (n.nm == "cos") return std::cos(x);
      if (n.nm == "tan") return std::tan(x);
      if (n.nm == "exp") return std::exp(x);
      if (n.nm == "log") return std::log(x);
      if (n.nm == "sqrt") return std::sqrt(x);
      if (n.nm == "atan") return std::atan(x);
      if (n.nm == "asin") return std::asin(x);
      if (n.nm == "acos") return std::acos(x);
      throw std::logic_error("expr::eval: unknown function " + n.nm);
    }
  }
  throw std::logic_error("expr::eval: unreachable");
}

expr simplify(const expr& e) {
  const node& n = deref(e);
  switch (n.k) {
    case kind::num:
    case kind::sym:
      return e;
    case kind::add: {
      std::vector<expr> ts;
      for (const expr& t : n.a) ts.push_back(simplify(t));
      return make_add(std::move(ts));
    }
    case kind::mul: {
      std::vector<expr> fs;
      for (const expr& f : n.a) fs.push_back(simplify(f));
      return make_mul(std::move(fs));
    }
    case kind::pow:
      return make_pow(simplify(n.a[0]), simplify(n.a[1]));
    case kind::fn:
      return expr::fn(n.nm, simplify(n.a[0]));
  }
  throw std::logic_error("simplify: unreachable");
}

}  // namespace ax::sym
