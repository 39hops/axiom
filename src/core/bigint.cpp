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
