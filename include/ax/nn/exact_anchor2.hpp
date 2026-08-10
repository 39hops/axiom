#pragma once
/** @file exact_anchor2.hpp ANCHOR-V2: the gcd-free exact anchor
    (PRE-REG ANCHOR-V2, relays 2026-08-09-5/-6). Same templated
    birth loop, third scalar family: `rx` carries the value twice —
    residues mod pinned 61-bit primes for ring arithmetic (no gcd
    anywhere on the hot path) and a certified growing dyadic
    interval for the branch sites Z_p cannot see (compares, floors).

    The four frozen pins:
      1. zero-tests route to residues, never to the interval;
      2. exact-boundary floor fallbacks are cached per residue-value
         (structural sites reconstruct once);
      3. shadow precision grows with the prefix (driver sets
         dyi::prec ~ 50 + 20*N) — fixed precision collapses at d64;
      4. per-class fallback counters are registered observables.

    Fallback = CRT + rational reconstruction (the only gcd left),
    verified against held-out primes; an exhausted modulus throws
    loudly, never returns a plausible wrong value. */
#include <compare>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ax/core/bigint.hpp>
#include <ax/core/dyadic.hpp>
#include <ax/core/rational.hpp>
#include <ax/core/rns.hpp>
#include <ax/nn/intbirth_core.hpp>

namespace ax::nn::ib::anchor2 {

/** pin 4: the registered observable */
struct fb_counters {
  long eq_zero = 0;      // zero-tests that needed reconstruction
  long floor_exact = 0;  // floor fallbacks landing ON a grain integer
  long floor_near = 0;   // floor fallbacks near a boundary
  long cmp = 0;          // ordered-compare fallbacks
  long recon = 0;        // total CRT reconstructions
  long cache_hit = 0;    // pin-2 site-cache hits
};

struct rx {
  // empty res == canonical exact zero (gemm fills vectors of Op
  // before assignment; the lazy form keeps that cheap)
  std::vector<std::uint64_t> res;
  std::vector<std::uint8_t> okm;
  ax::dyi sh;  // certified: true value always inside

  static inline ax::rns::ctx ctx;
  static inline fb_counters fb;
  struct cache_ent {
    std::vector<std::uint64_t> res;
    ax::bigint fl;
  };
  static inline std::unordered_map<std::uint64_t, cache_ent> site_cache;

  /** freeze the run configuration (primes + shadow precision) */
  static void init(int nprimes, int prec_bits) {
    ctx = ax::rns::ctx::make(nprimes);
    ax::dyi::prec = prec_bits;
    fb = {};
    site_cache.clear();
  }

  rx() = default;
  rx(int x) : rx((long long)x) {}    // NOLINT
  rx(long x) : rx((long long)x) {}   // NOLINT
  rx(long long x) : sh(x) {          // NOLINT
    if (x) {
      res.resize(ctx.P.size());
      okm.assign(ctx.P.size(), 1);
      for (std::size_t i = 0; i < ctx.P.size(); ++i) {
        long long m = x % (long long)ctx.P[i];
        if (m < 0) m += (long long)ctx.P[i];
        res[i] = std::uint64_t(m);
      }
    }
  }

  std::uint64_t rat(std::size_t i) const { return res.empty() ? 0 : res[i]; }
  std::uint8_t okat(std::size_t i) const { return okm.empty() ? 1 : okm[i]; }

  /** exact integer -> rx (floor results, reconstructed points) */
  static rx from_int(const ax::bigint& f) {
    rx r;
    r.res.resize(ctx.P.size());
    r.okm.assign(ctx.P.size(), 1);
    for (std::size_t i = 0; i < ctx.P.size(); ++i)
      r.res[i] = ax::rns::res_of(f, ctx.P[i]);
    r.sh = ax::dyi(f, 0);
    return r;
  }

  /** the fallback court: CRT + rational reconstruction, loud */
  ax::rational reconstruct_rat() const {
    if (res.empty()) return ax::rational(ax::bigint(0));
    fb.recon++;
    const std::vector<std::uint8_t> ok =
        okm.empty() ? std::vector<std::uint8_t>(ctx.P.size(), 1) : okm;
    const auto rec = ax::rns::reconstruct(res, ok, ctx);
    if (!rec.ok)
      throw std::runtime_error("anchor2: modulus exhausted");
    return rec.v;
  }

  // ---- ring ops (residue-wise + interval) ---------------------
  friend rx operator+(const rx& a, const rx& b) {
    rx r;
    r.sh = a.sh + b.sh;
    r.res.resize(ctx.P.size());
    r.okm.resize(ctx.P.size());
    for (std::size_t i = 0; i < ctx.P.size(); ++i) {
      r.res[i] = ax::rns::addm(a.rat(i), b.rat(i), ctx.P[i]);
      r.okm[i] = a.okat(i) & b.okat(i);
    }
    return r;
  }
  friend rx operator-(const rx& a, const rx& b) {
    rx r;
    r.sh = a.sh - b.sh;
    r.res.resize(ctx.P.size());
    r.okm.resize(ctx.P.size());
    for (std::size_t i = 0; i < ctx.P.size(); ++i) {
      r.res[i] = ax::rns::subm(a.rat(i), b.rat(i), ctx.P[i]);
      r.okm[i] = a.okat(i) & b.okat(i);
    }
    return r;
  }
  friend rx operator*(const rx& a, const rx& b) {
    rx r;
    r.sh = a.sh * b.sh;
    r.res.resize(ctx.P.size());
    r.okm.resize(ctx.P.size());
    for (std::size_t i = 0; i < ctx.P.size(); ++i) {
      r.res[i] = ax::rns::mulm(a.rat(i), b.rat(i), ctx.P[i]);
      r.okm[i] = a.okat(i) & b.okat(i);
    }
    return r;
  }
  rx operator-() const {
    rx r = *this;
    r.sh = -r.sh;
    for (std::size_t i = 0; i < r.res.size(); ++i)
      r.res[i] = ax::rns::subm(0, r.res[i], ctx.P[i]);
    return r;
  }
  rx& operator+=(const rx& b) { return *this = *this + b; }
  rx& operator-=(const rx& b) { return *this = *this - b; }
  rx& operator*=(const rx& b) { return *this = *this * b; }
  friend rx operator<<(const rx& a, int n) {  // exact * 2^n
    rx r = a;
    r.sh = r.sh << n;
    for (std::size_t i = 0; i < r.res.size(); ++i)
      r.res[i] = ax::rns::mulm(
          r.res[i], ax::rns::powm(2, std::uint64_t(n), ctx.P[i]),
          ctx.P[i]);
    return r;
  }
  friend rx operator>>(const rx& a, int n) {  // exact / 2^n
    rx r = a;
    r.sh = r.sh >> n;
    for (std::size_t i = 0; i < r.res.size(); ++i)
      r.res[i] = ax::rns::mulm(
          r.res[i],
          ax::rns::invm(ax::rns::powm(2, std::uint64_t(n), ctx.P[i]),
                        ctx.P[i]),
          ctx.P[i]);
    return r;
  }

  // ---- branch sites -------------------------------------------
  /** pin 1: residue-native zero test */
  bool is_zero() const {
    if (res.empty()) return true;
    if (!sh.contains_zero()) return false;
    int clean = 0;
    for (std::size_t i = 0; i < res.size(); ++i) {
      if (!okat(i)) continue;
      clean++;
      if (res[i] != 0) return false;
    }
    if (clean >= 3) return true;
    fb.eq_zero++;  // pole-starved: reconstruct to be sure
    return reconstruct_rat().num().is_zero();
  }
  friend bool operator==(const rx& a, const rx& b) {
    return (a - b).is_zero();
  }
  friend std::strong_ordering operator<=>(const rx& a, const rx& b) {
    if (ax::dyi::lt_certain(a.sh, b.sh)) return std::strong_ordering::less;
    if (ax::dyi::lt_certain(b.sh, a.sh)) return std::strong_ordering::greater;
    const rx d = a - b;
    if (d.is_zero()) return std::strong_ordering::equal;
    fb.cmp++;
    return d.reconstruct_rat().num().is_negative()
               ? std::strong_ordering::less
               : std::strong_ordering::greater;
  }

  /** DECLARED floor conversion (seams + de-grain); the shadow is a
      point integer at every conversion site by construction */
  explicit operator long long() const {
    return std::stoll(sh.point_int().to_string());
  }
  explicit operator long() const {
    return (long)static_cast<long long>(*this);
  }
};

/** Zero value-rounding policy over rx: div exact (modular inverse +
    interval division), to_grain the declared exact floor. */
struct Exact2 {
  static std::uint64_t res_hash(const rx& s) {  // FNV-1a over words
    std::uint64_t h = 1469598103934665603ull;
    for (const std::uint64_t w : s.res)
      for (int b = 0; b < 8; ++b) {
        h ^= (w >> (8 * b)) & 0xff;
        h *= 1099511628211ull;
      }
    return h;
  }

  static rx div(const rx& x, const rx& d) {
    ax::dyi dsh = d.sh;
    if (dsh.contains_zero()) {
      // shadow can't certify the divisor's sign: exact-zero check
      // first (loud), then reconstruct a tight point interval
      if (d.is_zero())
        throw std::runtime_error("anchor2: division by exact zero");
      const ax::rational dr = d.reconstruct_rat();
      dsh = ax::dyi::div(ax::dyi(dr.num(), 0), ax::dyi(dr.den(), 0));
    }
    rx r;
    r.sh = ax::dyi::div(x.sh, dsh);
    r.res.resize(rx::ctx.P.size());
    r.okm.resize(rx::ctx.P.size());
    for (std::size_t i = 0; i < rx::ctx.P.size(); ++i) {
      const std::uint64_t dv = d.rat(i);
      if (dv == 0) {  // pole: this prime can't see the quotient
        r.res[i] = 0;
        r.okm[i] = 0;
        continue;
      }
      r.res[i] = ax::rns::mulm(x.rat(i),
                               ax::rns::invm(dv, rx::ctx.P[i]),
                               rx::ctx.P[i]);
      r.okm[i] = x.okat(i) & d.okat(i);
    }
    return r;
  }
  static rx div_trunc(const rx& x, const rx& d) { return div(x, d); }

  static rx to_grain(const rx& x, int gshift) {
    const rx s = gshift ? (x >> gshift) : x;
    const auto [fl, fh] = s.sh.floor_pair();
    if (fl == fh) return rx::from_int(fl);  // shadow decides
    // straddle: pin-2 site cache, then reconstruction
    const std::uint64_t key = res_hash(s);
    const auto it = rx::site_cache.find(key);
    if (it != rx::site_cache.end() && it->second.res == s.res) {
      rx::fb.cache_hit++;
      return rx::from_int(it->second.fl);
    }
    const ax::rational v = s.reconstruct_rat();
    const auto [q, r] = ax::bigint::divmod(v.num(), v.den());
    ax::bigint f = q;
    if (v.num().is_negative() && !r.is_zero()) f = f - ax::bigint(1);
    if (r.is_zero())
      rx::fb.floor_exact++;  // exactly ON the grain integer
    else
      rx::fb.floor_near++;
    rx::site_cache[key] = {s.res, f};
    return rx::from_int(f);
  }
  static rx from_grain(const rx& x, int gshift) {
    return gshift ? (x << gshift) : x;
  }
};

/** the anchor-v2 loop: same placement sites, same frozen-grain
    conventions, shipped grain — gcd-free */
using anchor2_birth = core::birth_impl<rx, rx, Exact2>;

}  // namespace ax::nn::ib::anchor2
