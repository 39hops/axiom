#pragma once
/** @file rng.hpp PCG64 random number generator and sampling primitives. */
#include <cstdint>

namespace ax::st {

/** PCG64 (XSL-RR 128/64 output, 128-bit LCG state as two u64 words).
    Seeded via splitmix64 expansion of a single u64 seed. Deterministic:
    identical seeds produce identical streams. Not cryptographic. */
class rng {
 public:
  using result_type = std::uint64_t;

  explicit rng(std::uint64_t seed = 0x853c49e6748fea9bULL);

  /** Next raw 64-bit output. */
  std::uint64_t next_u64();
  /** Uniform double in [0,1) with 53 random bits. */
  double next_double();
  /** Uniform double in [a,b). */
  double uniform(double a, double b);
  /** Uniform integer in [0,n), unbiased via rejection. n must be > 0. */
  std::uint64_t below(std::uint64_t n);

  /** Standard normal N(0,1) via 128-layer ziggurat. */
  double normal();
  /** Normal N(mu, sigma^2). */
  double normal(double mu, double sigma);
  /** Exponential with rate lambda > 0. Throws std::invalid_argument. */
  double exponential(double lambda = 1.0);
  /** Gamma(shape, scale), shape > 0, scale > 0 (Marsaglia-Tsang).
      Throws std::invalid_argument. */
  double gamma(double shape, double scale = 1.0);

  // std UniformRandomBitGenerator surface
  std::uint64_t operator()() { return next_u64(); }
  static constexpr std::uint64_t min() { return 0; }
  static constexpr std::uint64_t max() { return ~0ULL; }

 private:
  std::uint64_t hi_ = 0, lo_ = 0;  ///< 128-bit LCG state
};

}  // namespace ax::st
