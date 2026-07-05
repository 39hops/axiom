#include <ax/core/bigint.hpp>

#include <algorithm>
#include <stdexcept>

namespace ax {

namespace {

/// 128-bit product of two 64-bit values via 32-bit split (portable).
struct u128 {
  std::uint64_t lo, hi;
};

u128 mul64(std::uint64_t a, std::uint64_t b) {
  const std::uint64_t a0 = a & 0xffffffffu, a1 = a >> 32;
  const std::uint64_t b0 = b & 0xffffffffu, b1 = b >> 32;
  const std::uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
  const std::uint64_t mid = p01 + (p00 >> 32) + (p10 & 0xffffffffu);
  return {(p00 & 0xffffffffu) | (mid << 32), p11 + (p10 >> 32) + (mid >> 32)};
}

/// magnitude compare: -1, 0, +1
int cmp_mag(const std::vector<std::uint64_t>& a,
            const std::vector<std::uint64_t>& b) {
  if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
  for (std::size_t i = a.size(); i-- > 0;)
    if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
  return 0;
}

/// magnitude = magnitude * m + add
void mul_add_small(std::vector<std::uint64_t>& v, std::uint64_t m,
                   std::uint64_t add) {
  std::uint64_t carry = add;
  for (auto& limb : v) {
    const u128 p = mul64(limb, m);
    const std::uint64_t nlo = p.lo + carry;
    limb = nlo;
    carry = p.hi + (nlo < p.lo ? 1 : 0);
  }
  if (carry) v.push_back(carry);
}

/// magnitude divmod by small d (nonzero); returns remainder.
std::uint64_t divmod_small(std::vector<std::uint64_t>& v, std::uint64_t d) {
  std::uint64_t rem = 0;
  for (std::size_t i = v.size(); i-- > 0;) {
    // 128-bit dividend (rem:v[i]) / d, bit-by-bit; rem < d holds throughout.
    const std::uint64_t cur_lo = v[i];
    std::uint64_t q = 0, r = rem;
    for (int bit = 63; bit >= 0; --bit) {
      const std::uint64_t top = r >> 63;
      r = (r << 1) | ((cur_lo >> bit) & 1u);
      if (top || r >= d) {
        r -= d;
        q |= (1ull << bit);
      }
    }
    v[i] = q;
    rem = r;
  }
  while (!v.empty() && v.back() == 0) v.pop_back();
  return rem;
}

/// schoolbook magnitude product
std::vector<std::uint64_t> mul_school(const std::vector<std::uint64_t>& a,
                                      const std::vector<std::uint64_t>& b) {
  if (a.empty() || b.empty()) return {};
  std::vector<std::uint64_t> r(a.size() + b.size(), 0);
  for (std::size_t i = 0; i < a.size(); ++i) {
    std::uint64_t carry = 0;
    for (std::size_t j = 0; j < b.size(); ++j) {
      const u128 p = mul64(a[i], b[j]);
      std::uint64_t lo = p.lo + carry;
      std::uint64_t hi = p.hi + (lo < carry ? 1 : 0);
      lo += r[i + j];
      hi += (lo < r[i + j]) ? 1 : 0;
      r[i + j] = lo;
      carry = hi;
    }
    std::size_t k = i + b.size();
    while (carry) {
      r[k] += carry;
      carry = (r[k] < carry) ? 1 : 0;
      ++k;
    }
  }
  while (!r.empty() && r.back() == 0) r.pop_back();
  return r;
}

/// r = a + b (magnitudes)
std::vector<std::uint64_t> add_mag(const std::vector<std::uint64_t>& a,
                                   const std::vector<std::uint64_t>& b) {
  const auto& big = a.size() >= b.size() ? a : b;
  const auto& small = a.size() >= b.size() ? b : a;
  std::vector<std::uint64_t> r(big.size());
  std::uint64_t carry = 0;
  for (std::size_t i = 0; i < big.size(); ++i) {
    std::uint64_t s = big[i] + carry;
    std::uint64_t c1 = (s < carry) ? 1u : 0u;
    if (i < small.size()) {
      s += small[i];
      c1 += (s < small[i]) ? 1u : 0u;
    }
    r[i] = s;
    carry = c1;
  }
  if (carry) r.push_back(carry);
  return r;
}

/// r = a - b (magnitudes), requires a >= b
std::vector<std::uint64_t> sub_mag(const std::vector<std::uint64_t>& a,
                                   const std::vector<std::uint64_t>& b) {
  std::vector<std::uint64_t> r(a.size());
  std::uint64_t borrow = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::uint64_t bi = i < b.size() ? b[i] : 0;
    const std::uint64_t d = a[i] - bi;
    const std::uint64_t d2 = d - borrow;
    borrow = ((a[i] < bi) || (d < borrow)) ? 1u : 0u;
    r[i] = d2;
  }
  while (!r.empty() && r.back() == 0) r.pop_back();
  return r;
}

/// split u64 limbs into u32 halves (little-endian)
std::vector<std::uint32_t> to_h(const std::vector<std::uint64_t>& v) {
  std::vector<std::uint32_t> h;
  h.reserve(v.size() * 2);
  for (auto x : v) {
    h.push_back(static_cast<std::uint32_t>(x));
    h.push_back(static_cast<std::uint32_t>(x >> 32));
  }
  while (!h.empty() && h.back() == 0) h.pop_back();
  return h;
}

std::vector<std::uint64_t> from_h(const std::vector<std::uint32_t>& h) {
  std::vector<std::uint64_t> v((h.size() + 1) / 2, 0);
  for (std::size_t i = 0; i < h.size(); ++i)
    v[i / 2] |= static_cast<std::uint64_t>(h[i]) << (32 * (i % 2));
  while (!v.empty() && v.back() == 0) v.pop_back();
  return v;
}

/// Knuth TAOCP vol.2 4.3.1 Algorithm D, base 2^32, magnitudes only.
/// Requires v nonempty (nonzero divisor).
void divmod_mag(std::vector<std::uint32_t> u, std::vector<std::uint32_t> v,
                std::vector<std::uint32_t>& q, std::vector<std::uint32_t>& r) {
  constexpr std::uint64_t base = 1ull << 32;
  if (v.size() == 1) {
    q.assign(u.size(), 0);
    std::uint64_t rem = 0;
    for (std::size_t i = u.size(); i-- > 0;) {
      const std::uint64_t cur = (rem << 32) | u[i];
      q[i] = static_cast<std::uint32_t>(cur / v[0]);
      rem = cur % v[0];
    }
    while (!q.empty() && q.back() == 0) q.pop_back();
    r.clear();
    if (rem) r.push_back(static_cast<std::uint32_t>(rem));
    return;
  }
  const std::size_t n = v.size();
  if (u.size() < n) {
    q.clear();
    r = std::move(u);
    return;
  }
  const std::size_t m = u.size() - n;

  // D1 normalize: shift so v.back() >= base/2
  int s = 0;
  for (std::uint32_t top = v.back(); top < (1u << 31); top <<= 1) ++s;
  auto shl = [&](std::vector<std::uint32_t>& a) {
    if (!s) return;
    std::uint32_t carry = 0;
    for (auto& x : a) {
      const std::uint32_t nc = x >> (32 - s);
      x = (x << s) | carry;
      carry = nc;
    }
    if (carry) a.push_back(carry);
  };
  shl(v);
  u.push_back(0);
  shl(u);
  if (u.size() < m + n + 1) u.resize(m + n + 1, 0);

  q.assign(m + 1, 0);
  for (std::size_t j = m + 1; j-- > 0;) {
    // D3 trial digit
    const std::uint64_t num =
        (static_cast<std::uint64_t>(u[j + n]) << 32) | u[j + n - 1];
    std::uint64_t qh = num / v[n - 1];
    std::uint64_t rh = num % v[n - 1];
    while (qh >= base ||
           qh * v[n - 2] > ((rh << 32) | u[j + n - 2])) {
      --qh;
      rh += v[n - 1];
      if (rh >= base) break;
    }
    // D4 multiply-subtract
    std::int64_t borrow = 0;
    std::uint64_t carry = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint64_t p = qh * v[i] + carry;
      carry = p >> 32;
      const std::int64_t t = static_cast<std::int64_t>(u[i + j]) -
                             static_cast<std::int64_t>(p & 0xffffffffu) -
                             borrow;
      u[i + j] = static_cast<std::uint32_t>(t);
      borrow = t < 0 ? 1 : 0;
    }
    const std::int64_t t = static_cast<std::int64_t>(u[j + n]) -
                           static_cast<std::int64_t>(carry) - borrow;
    u[j + n] = static_cast<std::uint32_t>(t);
    // D5/D6 add back on overshoot
    if (t < 0) {
      --qh;
      std::uint64_t c = 0;
      for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t sum =
            static_cast<std::uint64_t>(u[i + j]) + v[i] + c;
        u[i + j] = static_cast<std::uint32_t>(sum);
        c = sum >> 32;
      }
      u[j + n] = static_cast<std::uint32_t>(u[j + n] + c);
    }
    q[j] = static_cast<std::uint32_t>(qh);
  }
  while (!q.empty() && q.back() == 0) q.pop_back();
  // D8 denormalize remainder
  r.assign(u.begin(), u.begin() + n);
  if (s) {
    for (std::size_t i = 0; i < r.size(); ++i) {
      r[i] >>= s;
      if (i + 1 < n) r[i] |= u[i + 1] << (32 - s);
    }
  }
  while (!r.empty() && r.back() == 0) r.pop_back();
}

}  // namespace

void bigint::trim() {
  while (!limbs_.empty() && limbs_.back() == 0) limbs_.pop_back();
  if (limbs_.empty()) neg_ = false;
}

bigint::bigint(long long v) {
  if (v == 0) return;
  neg_ = v < 0;
  // two's complement negate as unsigned; safe for LLONG_MIN
  const std::uint64_t mag = neg_ ? (~static_cast<std::uint64_t>(v) + 1u)
                                 : static_cast<std::uint64_t>(v);
  limbs_.push_back(mag);
}

bigint::bigint(std::string_view dec) {
  std::size_t i = 0;
  bool neg = false;
  if (i < dec.size() && (dec[i] == '-' || dec[i] == '+')) {
    neg = dec[i] == '-';
    ++i;
  }
  if (i >= dec.size()) throw std::invalid_argument("bigint: empty numeral");
  for (; i < dec.size(); ++i) {
    const char c = dec[i];
    if (c < '0' || c > '9') throw std::invalid_argument("bigint: bad digit");
    mul_add_small(limbs_, 10, static_cast<std::uint64_t>(c - '0'));
  }
  neg_ = neg;
  trim();
}

std::string bigint::to_string() const {
  if (is_zero()) return "0";
  std::vector<std::uint64_t> tmp = limbs_;
  std::string digits;
  while (!tmp.empty()) {
    const std::uint64_t r = divmod_small(tmp, 10);
    digits.push_back(static_cast<char>('0' + r));
  }
  if (neg_) digits.push_back('-');
  std::reverse(digits.begin(), digits.end());
  return digits;
}

bigint bigint::operator-() const {
  bigint r = *this;
  if (!r.is_zero()) r.neg_ = !r.neg_;
  return r;
}

bigint operator+(const bigint& a, const bigint& b) {
  bigint r;
  if (a.neg_ == b.neg_) {
    r.limbs_ = add_mag(a.limbs_, b.limbs_);
    r.neg_ = a.neg_;
  } else {
    const int c = cmp_mag(a.limbs_, b.limbs_);
    if (c == 0) return r;  // magnitudes equal, opposite signs: zero
    const bigint& big = c > 0 ? a : b;
    const bigint& small = c > 0 ? b : a;
    r.limbs_ = sub_mag(big.limbs_, small.limbs_);
    r.neg_ = big.neg_;
  }
  r.trim();
  return r;
}

bigint operator-(const bigint& a, const bigint& b) { return a + (-b); }

bigint operator*(const bigint& a, const bigint& b) {
  bigint r;
  r.limbs_ = mul_school(a.limbs_, b.limbs_);
  r.neg_ = !r.limbs_.empty() && (a.neg_ != b.neg_);
  return r;
}

bigint operator<<(const bigint& a, unsigned bits) {
  if (a.is_zero() || bits == 0) return a;
  const unsigned limb_shift = bits / 64, bit_shift = bits % 64;
  bigint r;
  r.neg_ = a.neg_;
  r.limbs_.assign(limb_shift, 0);
  std::uint64_t carry = 0;
  for (std::uint64_t limb : a.limbs_) {
    if (bit_shift == 0) {
      r.limbs_.push_back(limb);
    } else {
      r.limbs_.push_back((limb << bit_shift) | carry);
      carry = limb >> (64 - bit_shift);
    }
  }
  if (carry) r.limbs_.push_back(carry);
  return r;
}

bigint operator>>(const bigint& a, unsigned bits) {
  const unsigned limb_shift = bits / 64, bit_shift = bits % 64;
  bigint r;
  if (limb_shift >= a.limbs_.size()) return r;
  r.neg_ = a.neg_;
  r.limbs_.assign(a.limbs_.begin() + limb_shift, a.limbs_.end());
  if (bit_shift) {
    for (std::size_t i = 0; i < r.limbs_.size(); ++i) {
      r.limbs_[i] >>= bit_shift;
      if (i + 1 < r.limbs_.size())
        r.limbs_[i] |= r.limbs_[i + 1] << (64 - bit_shift);
    }
  }
  r.trim();
  return r;
}

std::pair<bigint, bigint> bigint::divmod(const bigint& a, const bigint& b) {
  if (b.is_zero()) throw std::domain_error("bigint: division by zero");
  std::vector<std::uint32_t> qh, rh;
  divmod_mag(to_h(a.limbs_), to_h(b.limbs_), qh, rh);
  bigint q, r;
  q.limbs_ = from_h(qh);
  r.limbs_ = from_h(rh);
  q.neg_ = !q.limbs_.empty() && (a.neg_ != b.neg_);
  r.neg_ = !r.limbs_.empty() && a.neg_;
  return {q, r};
}

bigint operator/(const bigint& a, const bigint& b) {
  return bigint::divmod(a, b).first;
}

bigint operator%(const bigint& a, const bigint& b) {
  return bigint::divmod(a, b).second;
}

std::strong_ordering operator<=>(const bigint& a, const bigint& b) {
  if (a.neg_ != b.neg_)
    return a.neg_ ? std::strong_ordering::less : std::strong_ordering::greater;
  const int c = cmp_mag(a.limbs_, b.limbs_);
  const int s = a.neg_ ? -c : c;
  return s < 0   ? std::strong_ordering::less
         : s > 0 ? std::strong_ordering::greater
                 : std::strong_ordering::equal;
}

}  // namespace ax
