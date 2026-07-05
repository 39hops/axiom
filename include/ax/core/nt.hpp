#pragma once
/** @file nt.hpp Number theory: primality, factorization, modular arithmetic. */
#include <ax/core/bigint.hpp>

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace ax {

/** Deterministic Miller-Rabin for 64-bit n. */
bool is_prime(std::uint64_t n);
/** Probabilistic Miller-Rabin (default 25 rounds, error < 4^-25). */
bool is_prime(const bigint& n, int rounds = 25);
/** Prime factorization of n, ascending with multiplicity.
    Pollard rho + trial division. factor(0) == factor(1) == {}. */
std::vector<std::uint64_t> factor(std::uint64_t n);
/** Extended gcd: returns (g, x, y) with a*x + b*y == g == gcd(a,b). */
std::tuple<bigint, bigint, bigint> ext_gcd(const bigint& a, const bigint& b);
/** Modular inverse of a mod m (m > 1). Throws std::domain_error if
    gcd(a, m) != 1. Result in [0, m). */
bigint modinv(const bigint& a, const bigint& m);
/** CRT over pairwise-coprime moduli. Input pairs (residue, modulus).
    Returns (x, M) with x == r_i (mod m_i), M = prod m_i, x in [0, M).
    Throws std::domain_error if moduli not coprime or list empty. */
std::pair<bigint, bigint> crt(
    const std::vector<std::pair<bigint, bigint>>& congruences);

}  // namespace ax
