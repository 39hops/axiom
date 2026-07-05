#include <ax/st/rng.hpp>

#include <stdexcept>

namespace ax::st {

namespace {

/** splitmix64 — seed expander. */
std::uint64_t splitmix64(std::uint64_t& s) {
  s += 0x9e3779b97f4a7c15ULL;
  std::uint64_t z = s;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

/** 64x64 -> 128 multiply via 32-bit halves (no intrinsics/__int128). */
void mul_64x64(std::uint64_t a, std::uint64_t b, std::uint64_t& hi,
               std::uint64_t& lo) {
  const std::uint64_t a_lo = a & 0xffffffffULL, a_hi = a >> 32;
  const std::uint64_t b_lo = b & 0xffffffffULL, b_hi = b >> 32;
  const std::uint64_t p0 = a_lo * b_lo;
  const std::uint64_t p1 = a_lo * b_hi;
  const std::uint64_t p2 = a_hi * b_lo;
  const std::uint64_t p3 = a_hi * b_hi;
  const std::uint64_t mid = (p0 >> 32) + (p1 & 0xffffffffULL) +
                            (p2 & 0xffffffffULL);
  lo = (mid << 32) | (p0 & 0xffffffffULL);
  hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
}

std::uint64_t rotr64(std::uint64_t v, unsigned r) {
  return (v >> r) | (v << ((64u - r) & 63u));
}

// PCG64 multiplier (Melissa O'Neill's 128-bit constant) and odd increment.
constexpr std::uint64_t kMulHi = 0x2360ed051fc65da4ULL;
constexpr std::uint64_t kMulLo = 0x4385df649fccf645ULL;
constexpr std::uint64_t kIncHi = 0x5851f42d4c957f2dULL;
constexpr std::uint64_t kIncLo = 0x14057b7ef767814fULL;  // odd

}  // namespace

rng::rng(std::uint64_t seed) {
  std::uint64_t s = seed;
  hi_ = splitmix64(s);
  lo_ = splitmix64(s);
  next_u64();  // decorrelate first output from raw seed bits
}

std::uint64_t rng::next_u64() {
  // state = state * MUL + INC (mod 2^128)
  std::uint64_t p_hi, p_lo;
  mul_64x64(lo_, kMulLo, p_hi, p_lo);
  p_hi += lo_ * kMulHi + hi_ * kMulLo;
  const std::uint64_t old_hi = p_hi, old_lo = p_lo;
  std::uint64_t new_lo = old_lo + kIncLo;
  std::uint64_t new_hi = old_hi + kIncHi + (new_lo < old_lo ? 1 : 0);
  hi_ = new_hi;
  lo_ = new_lo;
  // XSL-RR output on the *new* state
  return rotr64(hi_ ^ lo_, static_cast<unsigned>(hi_ >> 58));
}

double rng::next_double() {
  return static_cast<double>(next_u64() >> 11) * 0x1.0p-53;
}

double rng::uniform(double a, double b) { return a + (b - a) * next_double(); }

std::uint64_t rng::below(std::uint64_t n) {
  if (n == 0) throw std::invalid_argument("rng::below: n must be > 0");
  const std::uint64_t threshold = (~n + 1) % n;  // 2^64 mod n
  for (;;) {
    const std::uint64_t v = next_u64();
    if (v >= threshold) return v % n;
  }
}

}  // namespace ax::st
