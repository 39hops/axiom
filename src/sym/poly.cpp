#include <ax/sym/poly.hpp>

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ax::sym {

namespace {

const rational kZero{};
const rational kOne{bigint(1)};

long long to_ll(const bigint& b) {
  try {
    return std::stoll(b.to_string());
  } catch (...) {
    throw std::overflow_error("poly: coefficient exceeds 64 bits");
  }
}

/** Positive divisors of |n|, n != 0. */
std::vector<long long> divisors(long long n) {
  if (n < 0) n = -n;
  std::vector<long long> out;
  for (long long d = 1; d * d <= n; ++d) {
    if (n % d == 0) {
      out.push_back(d);
      if (d != n / d) out.push_back(n / d);
    }
  }
  return out;
}

}  // namespace

void poly::trim() {
  while (!c_.empty() && c_.back().is_zero()) c_.pop_back();
}

poly::poly(std::vector<rational> coeffs) : c_(std::move(coeffs)) { trim(); }

const rational& poly::coeff(std::size_t k) const {
  return k < c_.size() ? c_[k] : kZero;
}

poly operator+(const poly& a, const poly& b) {
  std::vector<rational> c(std::max(a.c_.size(), b.c_.size()));
  for (std::size_t i = 0; i < c.size(); ++i) c[i] = a.coeff(i) + b.coeff(i);
  return poly(std::move(c));
}

poly operator-(const poly& a, const poly& b) {
  std::vector<rational> c(std::max(a.c_.size(), b.c_.size()));
  for (std::size_t i = 0; i < c.size(); ++i) c[i] = a.coeff(i) - b.coeff(i);
  return poly(std::move(c));
}

poly operator*(const poly& a, const poly& b) {
  if (a.c_.empty() || b.c_.empty()) return poly();
  std::vector<rational> c(a.c_.size() + b.c_.size() - 1);
  for (std::size_t i = 0; i < a.c_.size(); ++i)
    for (std::size_t j = 0; j < b.c_.size(); ++j)
      c[i + j] = c[i + j] + a.c_[i] * b.c_[j];
  return poly(std::move(c));
}

std::pair<poly, poly> poly::divmod(const poly& d) const {
  if (d.c_.empty()) throw std::domain_error("poly::divmod: division by zero");
  poly rem = *this;
  std::vector<rational> quo;
  const int dd = d.degree();
  if (degree() >= dd) quo.resize(static_cast<std::size_t>(degree() - dd) + 1);
  while (rem.degree() >= dd) {
    const int k = rem.degree() - dd;
    const rational f = rem.c_.back() / d.c_.back();
    quo[static_cast<std::size_t>(k)] = f;
    for (std::size_t i = 0; i < d.c_.size(); ++i)
      rem.c_[static_cast<std::size_t>(k) + i] =
          rem.c_[static_cast<std::size_t>(k) + i] - f * d.c_[i];
    rem.trim();
  }
  return {poly(std::move(quo)), rem};
}

poly gcd(poly a, poly b) {
  while (b.degree() >= 0) {
    poly r = a.divmod(b).second;
    a = std::move(b);
    b = std::move(r);
  }
  // monic normalization
  if (a.degree() >= 0 && !(a.c_.back() == kOne)) {
    const rational lead = a.c_.back();
    for (rational& c : a.c_) c = c / lead;
  }
  return a;
}

poly poly::derivative() const {
  if (c_.size() <= 1) return poly();
  std::vector<rational> d(c_.size() - 1);
  for (std::size_t i = 1; i < c_.size(); ++i)
    d[i - 1] = c_[i] * rational(bigint(static_cast<long long>(i)));
  return poly(std::move(d));
}

poly poly::square_free() const {
  if (degree() <= 0) return *this;
  const poly g = gcd(*this, derivative());
  poly sf = divmod(g).first;
  if (sf.degree() >= 0 && !(sf.c_.back() == kOne)) {
    const rational lead = sf.c_.back();
    for (rational& c : sf.c_) c = c / lead;
  }
  return sf;
}

std::vector<rational> poly::rational_roots() const {
  std::vector<rational> roots;
  if (degree() < 0) return roots;
  // strip x^k factor: 0 is a root if constant term vanishes
  std::size_t low = 0;
  while (low < c_.size() && c_[low].is_zero()) ++low;
  if (low > 0) roots.push_back(kZero);
  if (low >= c_.size() - 1 && degree() >= 0 && low == c_.size()) return roots;
  // clear denominators over the remaining coefficients
  bigint lcm_den(1);
  for (std::size_t i = low; i < c_.size(); ++i) {
    const bigint& d = c_[i].den();
    lcm_den = lcm_den / gcd(lcm_den, d) * d;
  }
  const long long a0 = to_ll((c_[low] * rational(lcm_den)).num());
  const long long an = to_ll((c_.back() * rational(lcm_den)).num());
  if (a0 == 0 || an == 0) return roots;  // degenerate after strip: no more
  for (const long long p : divisors(a0)) {
    for (const long long q : divisors(an)) {
      for (const int sign : {1, -1}) {
        const rational cand(bigint(sign * p), bigint(q));
        if (!eval(cand).is_zero()) continue;
        bool seen = false;
        for (const rational& r : roots) seen = seen || r == cand;
        if (!seen) roots.push_back(cand);
      }
    }
  }
  return roots;
}

rational poly::eval(const rational& x) const {
  rational acc;
  for (std::size_t i = c_.size(); i-- > 0;) acc = acc * x + c_[i];
  return acc;
}

poly poly::from_expr(const expr& e, const expr& s) {
  if (e.same(s)) return poly({kZero, kOne});
  if (e.is_num()) return poly({e.value()});
  if (e.is_sym())
    throw std::invalid_argument("poly::from_expr: foreign symbol " + e.name());
  if (e.is_add()) {
    poly acc;
    for (const expr& t : e.args()) acc = acc + from_expr(t, s);
    return acc;
  }
  if (e.is_mul()) {
    poly acc({kOne});
    for (const expr& f : e.args()) acc = acc * from_expr(f, s);
    return acc;
  }
  if (e.is_pow()) {
    const expr& ex = e.args()[1];
    if (!ex.is_num() || !(ex.value().den() == bigint(1)))
      throw std::invalid_argument("poly::from_expr: non-integer power");
    const long long n = to_ll(ex.value().num());
    if (n < 0)
      throw std::invalid_argument("poly::from_expr: negative power");
    const poly base = from_expr(e.args()[0], s);
    poly acc({kOne});
    for (long long i = 0; i < n; ++i) acc = acc * base;
    return acc;
  }
  throw std::invalid_argument("poly::from_expr: not polynomial (function)");
}

expr poly::to_expr(const expr& s) const {
  expr acc = expr::num(0);
  for (std::size_t i = 0; i < c_.size(); ++i) {
    if (c_[i].is_zero()) continue;
    acc = acc + expr::num(c_[i]) * s.pow(expr::num(static_cast<std::int64_t>(i)));
  }
  return acc;
}

}  // namespace ax::sym
