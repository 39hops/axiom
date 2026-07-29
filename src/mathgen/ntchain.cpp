/** @file ntchain.cpp Number-theory chain makers (see ntchain.hpp). */
#include <ax/mathgen/ntchain.hpp>

#include <ax/core/nt.hpp>
#include <ax/pyrand/pyrand.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ax::mathgen {

namespace {

constexpr long long kReseed = 1'000'003;

// ---------------------------------------------------------------- nt_eval

struct nt_parser {
  const std::string& s;
  std::size_t i = 0;

  void skip() {
    while (i < s.size() && s[i] == ' ') ++i;
  }
  bool eat(char c) {
    skip();
    if (i < s.size() && s[i] == c) {
      ++i;
      return true;
    }
    return false;
  }
  void expect(char c) {
    if (!eat(c))
      throw std::invalid_argument("nt_eval: expected '" +
                                  std::string(1, c) + "' in " + s);
  }

  bigint expr() {
    bigint v = term();
    for (;;) {
      skip();
      // '+' always binds; '-' only when not the '**' spelling (no such
      // ambiguity — '*' is consumed in term) and not end of input
      if (eat('+'))
        v = v + term();
      else if (i < s.size() && s[i] == '-') {
        ++i;
        v = v - term();
      } else
        return v;
    }
  }

  bigint term() {
    bigint v = factor();
    for (;;) {
      skip();
      if (i + 1 < s.size() && s[i] == '*' && s[i + 1] == '*')
        throw std::invalid_argument("nt_eval: stray '**' in " + s);
      if (i < s.size() && s[i] == '*') {
        ++i;
        v = v * factor();
      } else
        return v;
    }
  }

  bigint factor() {
    skip();
    if (eat('-')) return -factor();
    bigint base = primary();
    skip();
    if (i + 1 < s.size() && s[i] == '*' && s[i + 1] == '*') {
      i += 2;
      const bigint e = factor();
      if (e.is_negative() || e > bigint(1'000'000))
        throw std::domain_error("nt_eval: exponent out of range in " + s);
      unsigned long long ev = 0;
      for (const char c : e.to_string()) ev = ev * 10 + (c - '0');
      return pow(base, ev);
    }
    return base;
  }

  bigint primary() {
    skip();
    if (eat('(')) {
      bigint v = expr();
      expect(')');
      return v;
    }
    if (i < s.size() && (s[i] >= '0' && s[i] <= '9')) {
      const std::size_t j = i;
      while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
      return bigint(std::string_view(s).substr(j, i - j));
    }
    const std::size_t j = i;
    while (i < s.size() &&
           ((s[i] >= 'a' && s[i] <= 'z') || (s[i] >= 'A' && s[i] <= 'Z')))
      ++i;
    const std::string name = s.substr(j, i - j);
    if (name != "gcd" && name != "Mod")
      throw std::invalid_argument("nt_eval: unknown token in " + s);
    expect('(');
    bigint a = expr();
    expect(',');
    bigint b = expr();
    expect(')');
    if (name == "gcd") return gcd(std::move(a), std::move(b));
    if (!(bigint(0) < b))
      throw std::domain_error("nt_eval: Mod with m <= 0 in " + s);
    bigint r = a % b;
    if (r.is_negative()) r = r + b;
    return r;
  }
};

}  // namespace

bigint nt_eval(const std::string& s) {
  nt_parser p{s};
  const bigint v = p.expr();
  p.skip();
  if (p.i != s.size())
    throw std::invalid_argument("nt_eval: trailing input in " + s);
  return v;
}

namespace {

/** Literal spelling; negatives parenthesized so every binary spelling
    stays inside the grammar. */
std::string lit(const bigint& n) {
  const std::string s = n.to_string();
  return n.is_negative() ? "(" + s + ")" : s;
}

/** Resolve every call site in s innermost-first: emit one span per
    site and return s with each site replaced by its value's
    spelling (lit, same parenthesization as the emitters). */
std::string resolve_calls(const std::string& s,
                          std::vector<std::string>& spans) {
  std::string out;
  std::size_t i = 0;
  while (i < s.size()) {
    // find the next call-name token at a non-identifier boundary
    if ((s[i] >= 'a' && s[i] <= 'z') || (s[i] >= 'A' && s[i] <= 'Z')) {
      std::size_t j = i;
      while (j < s.size() && ((s[j] >= 'a' && s[j] <= 'z') ||
                              (s[j] >= 'A' && s[j] <= 'Z')))
        ++j;
      const std::string name = s.substr(i, j - i);
      if ((name == "gcd" || name == "Mod") && j < s.size() && s[j] == '(') {
        // match the site's closing paren
        int depth = 0;
        std::size_t k = j;
        for (; k < s.size(); ++k) {
          if (s[k] == '(') ++depth;
          if (s[k] == ')' && --depth == 0) break;
        }
        if (depth != 0)
          throw std::invalid_argument("nt_call_spans: unbalanced site in " +
                                      s);
        // inner calls first (their spans land before this site's)
        const std::string args =
            resolve_calls(s.substr(j + 1, k - j - 1), spans);
        const std::string site = name + "(" + args + ")";
        const bigint v = nt_eval(site);
        spans.push_back("call: " + site + " -> " + v.to_string());
        out += lit(v);
        i = k + 1;
        continue;
      }
      out += name;
      i = j;
      continue;
    }
    out += s[i++];
  }
  return out;
}

}  // namespace

std::vector<std::string> nt_call_spans(const std::string& s) {
  std::vector<std::string> spans;
  const std::string resolved = resolve_calls(s, spans);
  // the resolved string must still evaluate — and to the same value —
  // or the extraction itself is broken
  if (!spans.empty() && !(nt_eval(resolved) == nt_eval(s)))
    throw std::logic_error("nt_call_spans: substitution changed value of " +
                           s);
  return spans;
}

namespace {

/** Row builder: every row is certified by exact evaluation of BOTH
    emitted strings — the oracle runs on the bytes, not the
    construction. */
struct nt_builder {
  pchain_problem& out;

  void put(const std::string& kind, const std::string& cur,
           const std::string& nxt) {
    bool ok = false;
    try {
      ok = nt_eval(cur) == nt_eval(nxt);
    } catch (const std::exception&) {
    }
    if (!ok) {
      out.certified = false;
      if (out.error.empty()) out.error = kind + " row failed nt_eval check";
    }
    out.rows.push_back({kind, cur, nxt});
  }

  void fail(const std::string& why) {
    out.certified = false;
    if (out.error.empty()) out.error = why;
  }
};

/** Euclid division chain a -> b -> ... -> g, emitting mul/sub/gcdstep
    rows; returns the quotient sequence and sets g. */
std::vector<bigint> emit_euclid(nt_builder& b, bigint a, bigint bb,
                                bigint& g) {
  std::vector<bigint> quotients;
  while (!bb.is_zero()) {
    const auto [q, r] = bigint::divmod(a, bb);
    const bigint qb = q * bb;
    b.put("mul", lit(q) + "*" + lit(bb), lit(qb));
    b.put("sub", lit(a) + " - " + lit(qb), lit(r));
    b.put("gcdstep", "gcd(" + lit(a) + ", " + lit(bb) + ")",
          "gcd(" + lit(bb) + ", " + lit(r) + ")");
    quotients.push_back(q);
    a = bb;
    bb = r;
  }
  b.put("gcdend", "gcd(" + lit(a) + ", 0)", lit(a));
  g = a;
  return quotients;
}

/** Bezout coefficient recurrence c_{k+1} = c_{k-1} - q_k * c_k emitted
    as mul/sub rows; returns the final coefficient. */
bigint emit_bezout_coeff(nt_builder& b, const std::vector<bigint>& q,
                         bigint c0, bigint c1) {
  for (const bigint& qk : q) {
    const bigint p = qk * c1;
    b.put("mul", lit(qk) + "*" + lit(c1), lit(p));
    const bigint c2 = c0 - p;
    b.put("sub", lit(c0) + " - " + lit(p), lit(c2));
    c0 = c1;
    c1 = c2;
  }
  return c0;  // coefficient aligned with the last nonzero remainder
}

struct level_range {
  long long lo, hi;
};

/** Operand magnitude per level: chains lengthen with the digits. */
level_range operand_range(int level) {
  switch (level) {
    case 1:
      return {12, 99};
    case 2:
      return {100, 9999};
    default:
      return {1'000'000, 99'999'999};
  }
}

bigint draw(pyrand::python_random& rng, const level_range& r) {
  return bigint(rng.randint(r.lo, r.hi));
}

}  // namespace

pchain_problem make_nt_gcd_chain(int level, long long seed) {
  pyrand::python_random rng("nt_gcd-" + std::to_string(level) + "-" +
                            std::to_string(seed));
  const level_range r = operand_range(level);
  bigint a = draw(rng, r), b = draw(rng, r);
  if (b > a) std::swap(a, b);
  if (a == b || b.is_zero()) return make_nt_gcd_chain(level, seed + kReseed);

  pchain_problem out{"nt_gcd", level, seed, {}, true, ""};
  nt_builder bld{out};
  bigint g;
  const auto q = emit_euclid(bld, a, b, g);
  if (q.size() < 2) return make_nt_gcd_chain(level, seed + kReseed);
  if (!(gcd(a, b) == g)) bld.fail("gcd cross-check failed");
  return out;
}

namespace {

/** Shared bezout emission on (a, b): euclid rows, s/t recurrences,
    closing bezout row. Returns (g, s, t) with a*s + b*t == g. */
std::tuple<bigint, bigint, bigint> emit_bezout(nt_builder& bld,
                                               const bigint& a,
                                               const bigint& b) {
  bigint g;
  const auto q = emit_euclid(bld, a, b, g);
  const bigint s = emit_bezout_coeff(bld, q, bigint(1), bigint(0));
  const bigint t = emit_bezout_coeff(bld, q, bigint(0), bigint(1));
  bld.put("bezout", lit(a) + "*" + lit(s) + " + " + lit(b) + "*" + lit(t),
          lit(g));
  return {g, s, t};
}

}  // namespace

pchain_problem make_nt_bezout_chain(int level, long long seed) {
  pyrand::python_random rng("nt_bezout-" + std::to_string(level) + "-" +
                            std::to_string(seed));
  const level_range r = operand_range(level);
  bigint a = draw(rng, r), b = draw(rng, r);
  if (b > a) std::swap(a, b);
  if (a == b || b.is_zero())
    return make_nt_bezout_chain(level, seed + kReseed);

  pchain_problem out{"nt_bezout", level, seed, {}, true, ""};
  nt_builder bld{out};
  const auto [g, s, t] = emit_bezout(bld, a, b);
  const auto [xg, xs, xt] = ext_gcd(a, b);
  if (!(xg == g) || !(a * s + b * t == g))
    bld.fail("ext_gcd cross-check failed");
  (void)xs;
  (void)xt;
  return out;
}

pchain_problem make_nt_modinv_chain(int level, long long seed) {
  pyrand::python_random rng("nt_modinv-" + std::to_string(level) + "-" +
                            std::to_string(seed));
  const level_range r = operand_range(level);
  const bigint m = draw(rng, r);
  const bigint a = bigint(rng.randint(2, r.hi)) % m;
  if (a < bigint(2) || !(gcd(a, m) == bigint(1)))
    return make_nt_modinv_chain(level, seed + kReseed);

  pchain_problem out{"nt_modinv", level, seed, {}, true, ""};
  nt_builder bld{out};
  const auto [g, s, t] = emit_bezout(bld, a, m);
  (void)t;
  bigint inv = s % m;
  if (inv.is_negative()) inv = inv + m;
  bld.put("inv", "Mod(" + lit(s) + ", " + lit(m) + ")", lit(inv));
  bld.put("invcheck",
          "Mod(" + lit(a) + "*" + lit(inv) + ", " + lit(m) + ")", "1");
  if (!(g == bigint(1)) || !(modinv(a, m) == inv))
    bld.fail("modinv cross-check failed");
  return out;
}

pchain_problem make_nt_crt_chain(int level, long long seed) {
  pyrand::python_random rng("nt_crt-" + std::to_string(level) + "-" +
                            std::to_string(seed));
  const level_range r =
      level == 1 ? level_range{3, 40} : operand_range(level - 1);
  const int k = level == 1 ? 2 : 3;
  std::vector<std::pair<bigint, bigint>> cong;  // (residue, modulus)
  bigint prod(1);
  for (int tries = 0; static_cast<int>(cong.size()) < k; ++tries) {
    if (tries > 64) return make_nt_crt_chain(level, seed + kReseed);
    const bigint m = draw(rng, r);
    if (m < bigint(2) || !(gcd(m, prod) == bigint(1))) continue;
    const bigint res = bigint(rng.randint(0, r.hi)) % m;
    cong.push_back({res, m});
    prod = prod * m;
  }

  pchain_problem out{"nt_crt", level, seed, {}, true, ""};
  nt_builder bld{out};
  bigint x = cong[0].first, big_m = cong[0].second;
  for (int idx = 1; idx < k; ++idx) {
    const bigint& res = cong[static_cast<std::size_t>(idx)].first;
    const bigint& m = cong[static_cast<std::size_t>(idx)].second;
    // inverse of M mod m via the bezout sub-chain on (M mod m, m)
    bigint c = big_m % m;
    bld.put("mod", "Mod(" + lit(big_m) + ", " + lit(m) + ")", lit(c));
    const auto [g, s, t] = emit_bezout(bld, c, m);
    (void)t;
    if (!(g == bigint(1))) bld.fail("crt moduli not coprime");
    bigint inv = s % m;
    if (inv.is_negative()) inv = inv + m;
    bld.put("inv", "Mod(" + lit(s) + ", " + lit(m) + ")", lit(inv));
    bld.put("invcheck",
            "Mod(" + lit(c) + "*" + lit(inv) + ", " + lit(m) + ")", "1");
    const bigint d = res - x;
    bld.put("sub", lit(res) + " - " + lit(x), lit(d));
    bigint dm = d % m;
    if (dm.is_negative()) dm = dm + m;
    bld.put("mod", "Mod(" + lit(d) + ", " + lit(m) + ")", lit(dm));
    const bigint t0 = inv * dm;
    bld.put("mul", lit(inv) + "*" + lit(dm), lit(t0));
    bigint tt = t0 % m;
    bld.put("mod", "Mod(" + lit(t0) + ", " + lit(m) + ")", lit(tt));
    const bigint lift = big_m * tt;
    bld.put("mul", lit(big_m) + "*" + lit(tt), lit(lift));
    const bigint x2 = x + lift;
    bld.put("add", lit(x) + " + " + lit(lift), lit(x2));
    const bigint m2 = big_m * m;
    bld.put("mul", lit(big_m) + "*" + lit(m), lit(m2));
    x = x2;
    big_m = m2;
  }
  for (const auto& [res, m] : cong)
    bld.put("crtcheck", "Mod(" + lit(x) + ", " + lit(m) + ")", lit(res));
  const auto [cx, cm] = crt(cong);
  if (!(cm == big_m) || !(x % big_m == cx))
    bld.fail("crt cross-check failed");
  return out;
}

pchain_problem make_nt_modexp_chain(int level, long long seed) {
  pyrand::python_random rng("nt_modexp-" + std::to_string(level) + "-" +
                            std::to_string(seed));
  const level_range r = operand_range(level);
  const bigint m = draw(rng, r);
  const bigint b = draw(rng, r);
  const long long e =
      rng.randint(level == 1 ? 8 : level == 2 ? 40 : 300,
                  level == 1 ? 40 : level == 2 ? 300 : 2000);
  if (m < bigint(2)) return make_nt_modexp_chain(level, seed + kReseed);

  pchain_problem out{"nt_modexp", level, seed, {}, true, ""};
  nt_builder bld{out};
  bigint acc = b % m;
  bld.put("mod", "Mod(" + lit(b) + ", " + lit(m) + ")", lit(acc));
  std::vector<int> bits;
  for (long long v = e; v > 0; v >>= 1) bits.push_back(static_cast<int>(v & 1));
  for (std::size_t i = bits.size() - 1; i-- > 0;) {
    const bigint sq = acc * acc % m;
    bld.put("sq", "Mod(" + lit(acc) + "*" + lit(acc) + ", " + lit(m) + ")",
            lit(sq));
    acc = sq;
    if (bits[i]) {
      const bigint mu = acc * b % m;
      bld.put("mulstep",
              "Mod(" + lit(acc) + "*" + lit(b) + ", " + lit(m) + ")",
              lit(mu));
      acc = mu;
    }
  }
  bld.put("modexp",
          "Mod(" + lit(b) + "**" + std::to_string(e) + ", " + lit(m) + ")",
          lit(acc));
  if (!(modpow(b, bigint(e), m) == acc)) bld.fail("modpow cross-check failed");
  return out;
}

pchain_problem make_nt_cf_chain(int level, long long seed) {
  pyrand::python_random rng("nt_cf-" + std::to_string(level) + "-" +
                            std::to_string(seed));
  const level_range r = operand_range(level);
  bigint p = draw(rng, r), q = draw(rng, r);
  if (q > p) std::swap(p, q);
  if (p == q || q.is_zero()) return make_nt_cf_chain(level, seed + kReseed);

  pchain_problem out{"nt_cf", level, seed, {}, true, ""};
  nt_builder bld{out};
  bigint g;
  const auto quot = emit_euclid(bld, p, q, g);
  if (quot.size() < 2) return make_nt_cf_chain(level, seed + kReseed);
  // convergent recurrences h_k = a_k*h_{k-1} + h_{k-2} (k likewise)
  bigint h0(0), h1(1), k0(1), k1(0);
  for (const bigint& a : quot) {
    const bigint hp = a * h1;
    bld.put("mul", lit(a) + "*" + lit(h1), lit(hp));
    const bigint h2 = hp + h0;
    bld.put("add", lit(hp) + " + " + lit(h0), lit(h2));
    const bigint kp = a * k1;
    bld.put("mul", lit(a) + "*" + lit(k1), lit(kp));
    const bigint k2 = kp + k0;
    bld.put("add", lit(kp) + " + " + lit(k0), lit(k2));
    h0 = h1;
    h1 = h2;
    k0 = k1;
    k1 = k2;
  }
  const bigint det = h1 * k0 - h0 * k1;
  bld.put("det",
          lit(h1) + "*" + lit(k0) + " - " + lit(h0) + "*" + lit(k1),
          lit(det));
  if (!(abs(det) == bigint(1)) || !(h1 * g == p) || !(k1 * g == q))
    bld.fail("convergent cross-check failed");
  return out;
}

}  // namespace ax::mathgen
