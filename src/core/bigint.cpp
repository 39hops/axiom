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
