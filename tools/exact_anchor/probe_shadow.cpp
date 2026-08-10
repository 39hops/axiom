/** @file probe_shadow.cpp anchor-v2 probe 1 (relay 2026-08-09-5,
    GO 2026-08-09): would the certified interval shadow decide the
    anchor's branches, and how often would it fall back to exact
    reconstruction?

    Part A: instantiate the templated engine core with an interval
    scalar (double endpoints, outward 1-ulp widening) — the shadow
    dry run llmopt proposed. Runs the full 12-step d8 prefix at
    machine speed; counts straddles per branch class per step.
    Straddled branches proceed on the midpoint decision, so counts
    are approximate after a first wrong branch — probe semantics.

    Part B (compile with AX_ANCHOR_PROBE): the true margin
    distribution from the EXACT anchor for the steps it can reach
    (step 2+ is minutes-class at d8 — the growth law), validating
    the shadow's counts where they overlap.

    Build (from repo root; probe tool, not part of the suite):
      c++ -std=c++20 -O2 -DAX_ANCHOR_PROBE -Iinclude \
        tools/exact_anchor/probe_shadow.cpp build/libaxiom.a \
        -o /tmp/probe_shadow */
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <ax/nn/exact_anchor.hpp>
#include <ax/nn/intbirth_core.hpp>

namespace core = ax::nn::ib::core;
using i64 = std::int64_t;

// ------------------------------------------------------ counters
struct shadow_stats {
  long cmp_total = 0, cmp_straddle = 0;
  long eq_total = 0, eq_straddle = 0;
  long floor_total = 0, floor_straddle = 0;
  double min_lg_cmp = 1e9;    // min log2 rel margin of CERTAIN cmps
  double min_lg_floor = 1e9;  // min log2 frac margin of CERTAIN floors
  void reset() { *this = shadow_stats{}; }
};
static shadow_stats st;

// ---------------------------------------------- interval scalar
static inline double dn(double x) { return std::nextafter(x, -DBL_MAX); }
static inline double up(double x) { return std::nextafter(x, DBL_MAX); }

struct ivl {
  double lo = 0, hi = 0;
  ivl() = default;
  ivl(int v) : lo(v), hi(v) {}                       // NOLINT
  ivl(long v) : ivl(static_cast<long long>(v)) {}    // NOLINT
  ivl(long long v) {                                 // NOLINT
    lo = hi = double(v);
    if (i64(lo) != v) { lo = dn(lo); hi = up(hi); }
  }
  double mid() const { return (lo + hi) / 2; }

  friend ivl operator+(const ivl& a, const ivl& b) {
    ivl r; r.lo = dn(a.lo + b.lo); r.hi = up(a.hi + b.hi); return r;
  }
  ivl operator-() const { ivl r; r.lo = -hi; r.hi = -lo; return r; }
  friend ivl operator-(const ivl& a, const ivl& b) { return a + (-b); }
  friend ivl operator*(const ivl& a, const ivl& b) {
    const double p[4] = {a.lo * b.lo, a.lo * b.hi, a.hi * b.lo,
                         a.hi * b.hi};
    ivl r; r.lo = p[0]; r.hi = p[0];
    for (double x : p) { if (x < r.lo) r.lo = x; if (x > r.hi) r.hi = x; }
    r.lo = dn(r.lo); r.hi = up(r.hi); return r;
  }
  ivl& operator+=(const ivl& b) { return *this = *this + b; }
  ivl& operator-=(const ivl& b) { return *this = *this - b; }
  ivl& operator*=(const ivl& b) { return *this = *this * b; }
  friend ivl operator<<(const ivl& a, int n) {  // exact 2^n scale
    ivl r; r.lo = std::ldexp(a.lo, n); r.hi = std::ldexp(a.hi, n);
    return r;
  }
  friend ivl operator>>(const ivl& a, int n) {
    ivl r; r.lo = std::ldexp(a.lo, -n); r.hi = std::ldexp(a.hi, -n);
    return r;
  }

  // branch decisions: certain when the intervals are disjoint,
  // midpoint-decided (and counted) when they overlap.
  static double rel_gap(const ivl& a, const ivl& b) {
    const double gap = (a.hi < b.lo) ? b.lo - a.hi : a.lo - b.hi;
    const double mag = std::max({std::fabs(a.lo), std::fabs(a.hi),
                                 std::fabs(b.lo), std::fabs(b.hi),
                                 1e-300});
    return std::log2(gap / mag);
  }
  friend bool operator<(const ivl& a, const ivl& b) {
    st.cmp_total++;
    if (a.hi < b.lo || b.hi < a.lo) {
      const double m = rel_gap(a, b);
      if (m < st.min_lg_cmp) st.min_lg_cmp = m;
      return a.hi < b.lo;
    }
    if (!(a.lo == b.lo && a.hi == b.hi)) st.cmp_straddle++;
    return a.mid() < b.mid();
  }
  friend bool operator>(const ivl& a, const ivl& b) { return b < a; }
  friend bool operator<=(const ivl& a, const ivl& b) { return !(b < a); }
  friend bool operator>=(const ivl& a, const ivl& b) { return !(a < b); }
  friend bool operator==(const ivl& a, const ivl& b) {
    st.eq_total++;
    if (a.lo == a.hi && b.lo == b.hi) return a.lo == b.lo;  // exact
    if (a.hi < b.lo || b.hi < a.lo) return false;           // disjoint
    st.eq_straddle++;
    return a.mid() == b.mid();
  }
  friend bool operator!=(const ivl& a, const ivl& b) { return !(a == b); }

  explicit operator long long() const {
    return (long long)(std::floor(mid()));
  }
  explicit operator long() const {
    return (long)(std::floor(mid()));
  }
};

/** Exact-anchor policy semantics over intervals: div exact,
    to_grain = declared floor (the branch site), from_grain exact. */
struct Shadow {
  static ivl div(const ivl& x, const ivl& d) {
    // d never straddles 0 in the anchor (divisors are Q powers,
    // lr denominators, isqrt outputs); assert stays honest.
    const double c[4] = {x.lo / d.lo, x.lo / d.hi, x.hi / d.lo,
                         x.hi / d.hi};
    ivl r; r.lo = c[0]; r.hi = c[0];
    for (double v : c) { if (v < r.lo) r.lo = v; if (v > r.hi) r.hi = v; }
    r.lo = dn(r.lo); r.hi = up(r.hi); return r;
  }
  static ivl div_trunc(const ivl& x, const ivl& d) { return div(x, d); }
  static ivl to_grain(const ivl& x, int gshift) {
    const ivl s = gshift ? (x >> gshift) : x;
    st.floor_total++;
    const double fl = std::floor(s.lo), fh = std::floor(s.hi);
    if (fl != fh) {
      st.floor_straddle++;
      ivl r; r.lo = r.hi = std::floor(s.mid()); return r;
    }
    const double fr = std::min(s.mid() - fl, fl + 1 - s.mid());
    const double m = std::log2(std::max(fr, 1e-300));
    if (m < st.min_lg_floor) st.min_lg_floor = m;
    ivl r; r.lo = r.hi = fl; return r;
  }
  static ivl from_grain(const ivl& x, int gshift) {
    return gshift ? (x << gshift) : x;
  }
};

// ------------------------------------------------- tiny fixture
// (exact_anchor_test.cpp fixture, raw-engine draw, seed 613)
static void put_u16(std::string& b, std::uint16_t v) {
  b.append(reinterpret_cast<const char*>(&v), 2);
}
static void put_u32(std::string& b, std::uint32_t v) {
  b.append(reinterpret_cast<const char*>(&v), 4);
}
static void put_u64(std::string& b, std::uint64_t v) {
  b.append(reinterpret_cast<const char*>(&v), 8);
}
static void put_tensor(std::string& b, const std::string& name,
                       const std::vector<i64>& v) {
  put_u16(b, std::uint16_t(name.size()));
  b.append(name);
  b.push_back(char(1));
  put_u64(b, v.size());
  b.append(reinterpret_cast<const char*>(v.data()), v.size() * 8);
}
static core::birth_cfg_t tiny_cfg() {
  return {4, 8, 4, 16, 8, 12, 9, 256, 8192, 16384, 42950, 1, 1000};
}
static std::string tiny_tables(int T, int DH) {
  const i64 ts = 100, tse = 50, RS = i64{1} << 14;
  std::vector<i64> sil(2 * ts + 1), dsl(2 * ts + 1), ex(tse + 1);
  for (i64 i = 0; i < i64(sil.size()); i++) sil[i] = (i - ts) / 2;
  for (i64 i = 0; i < i64(dsl.size()); i++) dsl[i] = 256;
  for (i64 i = 0; i < i64(ex.size()); i++) ex[i] = 1 + i * 7;
  std::vector<i64> rc(std::size_t(T) * (DH / 2), RS);
  std::vector<i64> rs(std::size_t(T) * (DH / 2), 0);
  std::string b("AXP3", 4);
  put_u32(b, 5);
  put_tensor(b, "silu.tab", sil);
  put_tensor(b, "dsilu.tab", dsl);
  put_tensor(b, "exp.tab", ex);
  put_tensor(b, "rope.cos", rc);
  put_tensor(b, "rope.sin", rs);
  return b;
}
static std::string tiny_init(const core::birth_cfg_t& c) {
  std::mt19937_64 rng(613);
  const auto d = [](std::mt19937_64& r) { return i64(r() % 513) - 256; };
  const std::size_t sizes[11] = {
      std::size_t(c.DH) * c.D, std::size_t(c.DH) * c.D,
      std::size_t(c.DH) * c.D, std::size_t(c.D) * c.DH,
      std::size_t(c.F) * c.D,  std::size_t(c.F) * c.D,
      std::size_t(c.D) * c.F,  std::size_t(c.V) * c.D,
      std::size_t(c.D),        std::size_t(c.D),
      std::size_t(c.D)};
  std::string b;
  for (std::size_t n : sizes)
    for (std::size_t i = 0; i < n; i++)
      put_u64(b, std::uint64_t(d(rng)));
  for (int i = 0; i < c.T * c.D; i++)
    put_u64(b, std::uint64_t(d(rng)));
  for (int t = 0; t < c.T; t++) put_u64(b, std::uint64_t(t % c.V));
  return b;
}

// -------------------------------------------------------- main
static std::string slurp(const char* p) {
  FILE* f = std::fopen(p, "rb");
  if (!f) { std::perror(p); std::exit(1); }
  std::string b;
  char buf[65536];
  for (std::size_t n; (n = std::fread(buf, 1, sizeof buf, f));)
    b.append(buf, n);
  std::fclose(f);
  return b;
}

int main(int argc, char** argv) {
  const int shadow_steps = argc > 1 ? std::atoi(argv[1]) : 12;
  const int exact_steps = argc > 2 ? std::atoi(argv[2]) : 2;
  core::birth_cfg_t c = tiny_cfg();
  std::string tb, in;
  if (argc > 4) {  // real inputs: probe_shadow N M tables.bin init.bin
    // d64-class r2b contract defaults (ib::contract)
    c = {32, 64, 16, 128, 64, 12, 9, 256, 8192, 16384, 42950, 1, 1000};
    tb = slurp(argv[3]);
    in = slurp(argv[4]);
  } else {
    tb = tiny_tables(c.T, c.DH);
    in = tiny_init(c);
  }

  std::printf("== Part A: interval-shadow dry run, %d steps ==\n",
              shadow_steps);
  core::birth_impl<ivl, ivl, Shadow> sb(tb, in, c);
  for (int s = 1; s <= shadow_steps; s++) {
    st.reset();
    sb.run(1);
    std::printf(
        "step %2d: floors %ld/%ld straddle, cmps %ld/%ld, "
        "eq %ld/%ld | min certain margin lg2: floor %.1f cmp %.1f | "
        "loss %lld\n",
        s, st.floor_straddle, st.floor_total, st.cmp_straddle,
        st.cmp_total, st.eq_straddle, st.eq_total,
        st.min_lg_floor > 1e8 ? 0.0 : st.min_lg_floor,
        st.min_lg_cmp > 1e8 ? 0.0 : st.min_lg_cmp,
        (long long)sb.last_loss());
  }

  namespace an = ax::nn::ib::anchor;
  std::printf("== Part B: exact-anchor margins, %d steps ==\n",
              exact_steps);
  an::anchor_birth ab(tb, in, c);
  for (int s = 1; s <= exact_steps; s++) {
    an::probe::floors.clear();
    an::probe::cmps.clear();
    an::probe::eq_total = an::probe::eq_true = 0;
    ab.run(1);
    double mf = 1e9, mc = 1e9;  // min log2 margins (abs / rel)
    for (const auto& r : an::probe::floors)
      if (r.lg_margin < mf) mf = r.lg_margin;
    for (const auto& r : an::probe::cmps) {
      const double rel = r.lg_margin - r.lg_mag;
      if (rel < mc) mc = rel;
    }
    std::printf(
        "step %2d: floors %zu (min lg2 frac-margin %.1f), cmps %zu "
        "(min lg2 rel-margin %.1f), eq %ld (%ld true) | loss %lld\n",
        s, an::probe::floors.size(), mf, an::probe::cmps.size(), mc,
        an::probe::eq_total, an::probe::eq_true,
        (long long)ab.last_loss());
  }
  return 0;
}
