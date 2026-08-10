#pragma once
/** @file rns.hpp anchor-v2 ring core: residue number system over
    pinned 61-bit primes + CRT + rational reconstruction. This is
    where the gcd disappears: ring ops act residue-wise (u64 mod-p
    arithmetic), and gcd runs only inside the rare reconstruction
    fallback. Reconstruction VERIFIES against held-out primes — an
    exhausted modulus is a loud ok=false, never a wrong value. */
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <ax/core/bigint.hpp>
#include <ax/core/nt.hpp>
#include <ax/core/rational.hpp>

namespace ax::rns {

using u64 = std::uint64_t;
using u128 = unsigned __int128;

inline u64 addm(u64 a, u64 b, u64 p) {
  const u64 s = a + b;  // p < 2^61: no overflow
  return s >= p ? s - p : s;
}
inline u64 subm(u64 a, u64 b, u64 p) { return a >= b ? a - b : a + p - b; }
inline u64 mulm(u64 a, u64 b, u64 p) { return u64(u128(a) * b % p); }
inline u64 powm(u64 a, u64 e, u64 p) {
  u64 r = 1 % p;
  for (; e; e >>= 1, a = mulm(a, a, p))
    if (e & 1) r = mulm(r, a, p);
  return r;
}
inline u64 invm(u64 a, u64 p) { return powm(a % p, p - 2, p); }

/** residue of a signed bigint mod p (result in [0,p)) */
inline u64 res_of(const bigint& x, u64 p) {
  const bigint r = x % bigint((long long)p);  // sign of x
  long long v = std::stoll(r.to_string());
  if (v < 0) v += (long long)p;
  return u64(v);
}

/** the pinned prime basis: the k largest primes below 2^61,
    deterministic (same list on every build/platform) */
struct ctx {
  std::vector<u64> P;
  static ctx make(int k) {
    ctx c;
    for (u64 n = (u64(1) << 61) - 1; int(c.P.size()) < k; n -= 2)
      if (ax::is_prime(n)) c.P.push_back(n);
    return c;
  }
};

/** rational -> residues; ok[i]=0 marks a pole (p | denominator) */
inline std::pair<std::vector<u64>, std::vector<std::uint8_t>> to_res(
    const rational& v, const ctx& c) {
  std::vector<u64> r(c.P.size());
  std::vector<std::uint8_t> ok(c.P.size(), 1);
  for (std::size_t i = 0; i < c.P.size(); ++i) {
    const u64 p = c.P[i];
    const u64 d = res_of(v.den(), p);
    if (d == 0) { ok[i] = 0; r[i] = 0; continue; }
    r[i] = mulm(res_of(v.num(), p), invm(d, p), p);
  }
  return {r, ok};
}

struct recon {
  rational v;
  bool ok = false;
};

/** CRT over clean primes (minus held-out verifiers) + extended-
    Euclid rational reconstruction + verification. HOLDOUT = 2. */
inline recon reconstruct(const std::vector<u64>& res,
                         const std::vector<std::uint8_t>& okm,
                         const ctx& c) {
  std::vector<std::size_t> clean;
  for (std::size_t i = 0; i < c.P.size(); ++i)
    if (okm[i]) clean.push_back(i);
  const std::size_t HOLD = clean.size() > 4 ? 2 : 1;
  if (clean.size() <= HOLD) return {};
  const std::size_t used = clean.size() - HOLD;

  // CRT: x mod M over the used primes
  bigint M(1);
  for (std::size_t j = 0; j < used; ++j)
    M = M * bigint((long long)c.P[clean[j]]);
  bigint x(0);
  for (std::size_t j = 0; j < used; ++j) {
    const u64 p = c.P[clean[j]];
    const bigint Mi = M / bigint((long long)p);
    const u64 mi = res_of(Mi, p);
    const u64 ci = mulm(res_of(bigint((long long)res[clean[j]]), p),
                        invm(mi, p), p);
    x = x + Mi * bigint((long long)ci);
  }
  x = x % M;
  if (x.is_negative()) x = x + M;

  // rational reconstruction: ext-Euclid, stop at 2*r^2 <= M
  bigint r0 = M, r1 = x, t0(0), t1(1);
  while (!r1.is_zero() && (r1 * r1 + r1 * r1) > M) {
    const bigint q = r0 / r1;
    bigint r2 = r0 - q * r1, t2 = t0 - q * t1;
    r0 = std::move(r1); r1 = std::move(r2);
    t0 = std::move(t1); t1 = std::move(t2);
  }
  if (t1.is_zero()) return {};
  bigint num = r1, den = t1;
  if (den.is_negative()) { num = -num; den = -den; }
  if ((den * den + den * den) > M) return {};  // den bound
  if (!(ax::gcd(num, den) == bigint(1))) return {};

  // verify against every held-out prime the candidate can reach
  int usable = 0;
  for (std::size_t j = used; j < clean.size(); ++j) {
    const u64 p = c.P[clean[j]];
    const u64 dv = res_of(den, p);
    if (dv == 0) continue;
    usable++;
    if (mulm(res_of(num, p), invm(dv, p), p) != res[clean[j]])
      return {};
  }
  if (usable == 0) return {};
  recon out;
  out.v = rational(std::move(num), std::move(den));
  out.ok = true;
  return out;
}

}  // namespace ax::rns
