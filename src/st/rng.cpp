#include <ax/st/rng.hpp>

#include <cmath>
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

namespace {

/** Ziggurat tables for the standard normal (128 layers).
    x_[0] = v/f(r) (virtual base-strip edge), x_[1] = r, x_[128] = 0. */
struct zig_tables {
  static constexpr double kR = 3.442619855899;
  static constexpr double kV = 9.91256303526217e-3;
  double x[129];
  zig_tables() {
    x[1] = kR;
    x[0] = kV * std::exp(0.5 * kR * kR);
    for (int i = 2; i <= 127; ++i) {
      const double prev = x[i - 1];
      x[i] = std::sqrt(
          -2.0 * std::log(kV / prev + std::exp(-0.5 * prev * prev)));
    }
    x[128] = 0.0;
  }
};

const zig_tables& zig() {
  static const zig_tables t;
  return t;
}

}  // namespace

double rng::normal() {
  const zig_tables& t = zig();
  for (;;) {
    const std::uint64_t bits = next_u64();
    const int i = static_cast<int>(bits & 127u);
    const double u = 2.0 * next_double() - 1.0;  // (-1,1)
    const double x = u * t.x[i];
    if (std::abs(x) < t.x[i + 1]) return x;  // inside rectangle
    if (i == 0) {
      // Marsaglia tail beyond r
      double xt, yt;
      do {
        xt = -std::log(1.0 - next_double()) / zig_tables::kR;
        yt = -std::log(1.0 - next_double());
      } while (2.0 * yt < xt * xt);
      return u > 0.0 ? zig_tables::kR + xt : -(zig_tables::kR + xt);
    }
    // wedge: accept under the density
    const double f0 = std::exp(-0.5 * t.x[i] * t.x[i]);
    const double f1 = std::exp(-0.5 * t.x[i + 1] * t.x[i + 1]);
    if (f1 + next_double() * (f0 - f1) < std::exp(-0.5 * x * x)) return x;
  }
}

double rng::normal(double mu, double sigma) { return mu + sigma * normal(); }

double rng::exponential(double lambda) {
  if (!(lambda > 0.0))
    throw std::invalid_argument("rng::exponential: lambda must be > 0");
  return -std::log(1.0 - next_double()) / lambda;
}

double rng::gamma(double shape, double scale) {
  if (!(shape > 0.0) || !(scale > 0.0))
    throw std::invalid_argument("rng::gamma: shape and scale must be > 0");
  if (shape < 1.0) {
    // boost: G(a) = G(a+1) * U^{1/a}
    const double u = 1.0 - next_double();  // (0,1]
    return gamma(shape + 1.0, scale) * std::pow(u, 1.0 / shape);
  }
  const double d = shape - 1.0 / 3.0;
  const double c = 1.0 / std::sqrt(9.0 * d);
  for (;;) {
    double x, v;
    do {
      x = normal();
      v = 1.0 + c * x;
    } while (v <= 0.0);
    v = v * v * v;
    const double u = 1.0 - next_double();  // (0,1], safe for log
    if (u < 1.0 - 0.0331 * x * x * x * x) return d * v * scale;
    if (std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v)))
      return d * v * scale;
  }
}

}  // namespace ax::st
