#include <ax/core/rational.hpp>

#include <stdexcept>

namespace ax {

void rational::normalize() {
  if (den_.is_zero()) throw std::domain_error("rational: zero denominator");
  if (den_.is_negative()) {
    num_ = -num_;
    den_ = -den_;
  }
  if (num_.is_zero()) {
    den_ = bigint{1};
    return;
  }
  const bigint g = gcd(num_, den_);
  num_ = num_ / g;
  den_ = den_ / g;
}

rational::rational(bigint n, bigint d)
    : num_{std::move(n)}, den_{std::move(d)} {
  normalize();
}

std::string rational::to_string() const {
  if (den_ == bigint{1}) return num_.to_string();
  return num_.to_string() + "/" + den_.to_string();
}

rational rational::operator-() const {
  rational r = *this;
  r.num_ = -r.num_;
  return r;
}

rational operator+(const rational& a, const rational& b) {
  return {a.num_ * b.den_ + b.num_ * a.den_, a.den_ * b.den_};
}

rational operator-(const rational& a, const rational& b) { return a + (-b); }

rational operator*(const rational& a, const rational& b) {
  return {a.num_ * b.num_, a.den_ * b.den_};
}

rational operator/(const rational& a, const rational& b) {
  if (b.is_zero()) throw std::domain_error("rational: division by zero");
  return {a.num_ * b.den_, a.den_ * b.num_};
}

std::strong_ordering operator<=>(const rational& a, const rational& b) {
  return (a.num_ * b.den_) <=> (b.num_ * a.den_);
}

}  // namespace ax
