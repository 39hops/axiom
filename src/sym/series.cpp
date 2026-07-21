/** @file series.cpp Truncated power-series arithmetic (see series.hpp). */
#include <ax/sym/series.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ax::sym {

namespace {

const rational kZero{};
const rational kOne{bigint(1)};

rational q_ll(long long n, long long d = 1) {
  return rational(bigint(n), bigint(d));
}

}  // namespace

series::series(int order) {
  if (order < 0) throw std::invalid_argument("series: negative order");
  c_.assign(static_cast<std::size_t>(order), kZero);
}

series::series(std::vector<rational> coeffs, int order)
    : c_(std::move(coeffs)) {
  if (order < 0) throw std::invalid_argument("series: negative order");
  c_.resize(static_cast<std::size_t>(order), kZero);
}

const rational& series::coeff(std::size_t k) const {
  if (k >= c_.size()) throw std::out_of_range("series::coeff beyond order");
  return c_[k];
}

bool series::is_zero() const {
  return std::all_of(c_.begin(), c_.end(),
                     [](const rational& q) { return q.is_zero(); });
}

series operator+(const series& a, const series& b) {
  const int n = std::min(a.order(), b.order());
  series r(n);
  for (int k = 0; k < n; ++k) r.c_[k] = a.c_[k] + b.c_[k];
  return r;
}

series operator-(const series& a, const series& b) {
  const int n = std::min(a.order(), b.order());
  series r(n);
  for (int k = 0; k < n; ++k) r.c_[k] = a.c_[k] - b.c_[k];
  return r;
}

series series::operator-() const {
  series r(order());
  for (int k = 0; k < order(); ++k) r.c_[k] = -c_[k];
  return r;
}

series operator*(const series& a, const series& b) {
  const int n = std::min(a.order(), b.order());
  series r(n);
  for (int i = 0; i < n; ++i) {
    if (a.c_[i].is_zero()) continue;
    for (int j = 0; i + j < n; ++j)
      r.c_[i + j] = r.c_[i + j] + a.c_[i] * b.c_[j];
  }
  return r;
}

series series::derivative() const {
  if (order() == 0) return series(0);
  series r(order() - 1);
  for (int k = 1; k < order(); ++k) r.c_[k - 1] = q_ll(k) * c_[k];
  return r;
}

series series::integrate() const {
  series r(order() + 1);
  for (int k = 0; k < order(); ++k) r.c_[k + 1] = c_[k] / q_ll(k + 1);
  return r;
}

series series::inverse() const {
  if (order() == 0 || c_[0].is_zero())
    throw std::domain_error("series::inverse: zero constant term");
  series r(order());
  r.c_[0] = kOne / c_[0];
  for (int n = 1; n < order(); ++n) {
    rational s;
    for (int k = 1; k <= n; ++k) s = s + c_[k] * r.c_[n - k];
    r.c_[n] = -s / c_[0];
  }
  return r;
}

series series::pow_int(long long e) const {
  if (e < 0) return inverse().pow_int(-e);
  series r({kOne}, order());
  series base = *this;
  for (; e > 0; e >>= 1) {
    if (e & 1) r = r * base;
    base = base * base;
  }
  return r;
}

series series::compose(const series& inner) const {
  if (inner.order() > 0 && !inner.c_[0].is_zero())
    throw std::domain_error("series::compose: inner constant term nonzero");
  const int n = std::min(order(), inner.order());
  series r(n);
  for (int k = std::min(order(), n) - 1; k >= 0; --k) {
    r = r * inner;
    if (r.order() > 0) r.c_[0] = r.c_[0] + c_[k];
  }
  // degenerate n == 0 keeps the empty series
  return r;
}

expr series::to_expr(const expr& x) const {
  expr sum = expr::num(0);
  for (int k = 0; k < order(); ++k) {
    if (c_[k].is_zero()) continue;
    sum = sum + expr::num(c_[k]) * x.pow(expr::num(k));
  }
  return sum;
}

series apply_fn(const std::string& name, const series& u) {
  const int n = u.order();
  if (n == 0) return series(0);
  if (name == "exp" || name == "sin" || name == "cos") {
    if (!u.coeff(0).is_zero())
      throw std::domain_error("series: " + name + " with nonzero u(0)");
    std::vector<rational> m(static_cast<std::size_t>(n));
    rational fact = kOne;  // k!
    for (int k = 0; k < n; ++k) {
      if (k > 0) fact = fact * q_ll(k);
      const rational inv = kOne / fact;
      if (name == "exp") m[k] = inv;
      else if (name == "sin" && k % 2 == 1)
        m[k] = k % 4 == 1 ? inv : -inv;
      else if (name == "cos" && k % 2 == 0)
        m[k] = k % 4 == 0 ? inv : -inv;
    }
    return series(std::move(m), n).compose(u);
  }
  if (name == "log") {
    if (u.coeff(0) != kOne)
      throw std::domain_error("series: log with u(0) != 1");
    std::vector<rational> m(static_cast<std::size_t>(n));
    for (int k = 1; k < n; ++k)
      m[k] = k % 2 == 1 ? q_ll(1, k) : q_ll(-1, k);
    // log(1 + v) with v = u - 1
    return series(std::move(m), n).compose(u - series({kOne}, n));
  }
  if (name == "sqrt") {
    if (u.coeff(0) != kOne)
      throw std::domain_error("series: sqrt with u(0) != 1");
    // binomial series (1 + v)^(1/2): b_0 = 1, b_k = b_{k-1}(3/2 - k)/k
    std::vector<rational> m(static_cast<std::size_t>(n));
    m[0] = kOne;
    for (int k = 1; k < n; ++k)
      m[k] = m[k - 1] * (q_ll(3, 2) - q_ll(k)) / q_ll(k);
    return series(std::move(m), n).compose(u - series({kOne}, n));
  }
  throw std::domain_error("series: unsupported function " + name);
}

series series::of_expr(const expr& e, const expr& x, int order) {
  switch (e.k()) {
    case kind::num:
      return series({e.value()}, order);
    case kind::sym:
      if (e.same(x)) return series({kZero, kOne}, order);
      throw std::domain_error("series: free symbol " + e.name());
    case kind::add: {
      series r(order);
      for (const expr& t : e.args()) r = r + of_expr(t, x, order);
      return r;
    }
    case kind::mul: {
      series r({kOne}, order);
      for (const expr& t : e.args()) r = r * of_expr(t, x, order);
      return r;
    }
    case kind::pow: {
      const expr& b = e.args()[0];
      const expr& ex = e.args()[1];
      if (ex.is_num() && ex.value().den() == bigint(1)) {
        const std::string s = ex.value().num().to_string();
        return of_expr(b, x, order).pow_int(std::stoll(s));
      }
      if (ex.is_num() && ex.value() == rational(bigint(1), bigint(2)))
        return apply_fn("sqrt", of_expr(b, x, order));
      throw std::domain_error("series: non-integer exponent");
    }
    case kind::fn: {
      if (e.args().size() != 1)
        throw std::domain_error("series: n-ary function " + e.name());
      return apply_fn(e.name(), of_expr(e.args()[0], x, order));
    }
  }
  throw std::domain_error("series: unreachable kind");
}

}  // namespace ax::sym
