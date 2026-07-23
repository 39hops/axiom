/** @file polychain.cpp Poly-algebra chain makers (see polychain.hpp).
    Rung 2 of the pilot (llmopt relay 2026-07-23): every multi-fact row
    is decomposed into one-primitive rows — the 15->88 tree recipe,
    third use. Constant primitives: "mul", "add", "sub", "div" (fold-
    certified via parse -> canonical). Polynomial primitives: "pmul"
    (one monomial-or-accumulator times one factor), "psub"/"padd" (one
    binary poly step), certified parse -> expand -> exact poly. The
    closing "monic"/"assemble" rows are now fully determined by their
    predecessor rows. */
#include <ax/mathgen/polychain.hpp>

#include <ax/mathgen/series_chain.hpp>
#include <ax/pyrand/pyrand.hpp>
#include <ax/sym/expand.hpp>
#include <ax/sym/parse.hpp>
#include <ax/sym/poly.hpp>
#include <ax/sym/print_sstr.hpp>

#include <stdexcept>
#include <utility>

namespace ax::mathgen {

namespace {

using sym::expr;
using sym::poly;

const expr kX = expr::symbol("x");
constexpr long long kReseed = 1'000'003;
const rational kOne(bigint(1));

std::string psstr(const poly& p) { return sym::to_sstr(p.to_expr(kX)); }

std::string value_sstr(const rational& r) {
  return sym::to_sstr(expr::num(r));
}

/** Constant-arithmetic row: parse -> canonical must fold byte-exactly. */
void arith_row(pchain_problem& out, const std::string& kind,
               const std::string& cur, const rational& val) {
  const std::string nxt = value_sstr(val);
  bool ok = false;
  try {
    ok = sym::to_sstr(sym::parse(cur)) == nxt;
  } catch (const std::exception&) {
  }
  if (!ok) {
    out.certified = false;
    out.error = kind + " row failed engine certification";
  }
  out.rows.push_back({kind, cur, nxt});
}

/** Polynomial-identity row: parse -> expand -> exact poly equality;
    nxt is printed from the engine polynomial. */
void poly_row(pchain_problem& out, const std::string& kind,
              const std::string& cur, const poly& want) {
  bool ok = false;
  try {
    ok = poly::from_expr(sym::expand(sym::parse(cur)), kX) == want;
  } catch (const std::exception&) {
  }
  if (!ok) {
    out.certified = false;
    out.error = kind + " row failed engine certification";
  }
  out.rows.push_back({kind, cur, psstr(want)});
}

/** Left-folded binary products over factors: one "mul" row each. */
rational mul_fold(pchain_problem& out, const std::vector<rational>& f) {
  rational acc = f.at(0);
  for (std::size_t i = 1; i < f.size(); ++i) {
    const rational p = acc * f[i];
    arith_row(out, "mul", chain_lit(acc) + "*" + chain_lit(f[i]), p);
    acc = p;
  }
  return acc;
}

/** Left-folded binary additions over values: one "add" row each. */
rational add_fold(pchain_problem& out, const std::vector<rational>& v) {
  rational acc = v.at(0);
  for (std::size_t i = 1; i < v.size(); ++i) {
    const rational s = acc + v[i];
    arith_row(out, "add", acc.to_string() + " + " + chain_lit(v[i]), s);
    acc = s;
  }
  return acc;
}

poly draw_poly(pyrand::python_random& rng, int degree) {
  std::vector<rational> c(static_cast<std::size_t>(degree) + 1);
  for (int i = 0; i < degree; ++i)
    c[static_cast<std::size_t>(i)] = rational(bigint(rng.randint(-4, 4)));
  c[static_cast<std::size_t>(degree)] =
      rational(bigint(rng.randint(1, 3)));  // nonzero lead
  return poly(std::move(c));
}

}  // namespace

pchain_problem make_gcd_chain(int level, long long seed) {
  pyrand::python_random rng("pgcd-" + std::to_string(level) + "-" +
                            std::to_string(seed));
  const int dg = level >= 2 ? 2 : 1;   // shared factor degree
  const int du = level >= 3 ? 2 : 1;   // cofactor degree
  const poly g = draw_poly(rng, dg);
  const poly u = draw_poly(rng, du);
  const poly v = draw_poly(rng, du);
  if (gcd(u, v).degree() != 0) return make_gcd_chain(level, seed + kReseed);
  const poly p1 = g * u;
  const poly p2 = g * v;

  pchain_problem out{"poly_gcd", level, seed, {}, true, ""};
  poly r0 = p1, r1 = p2;
  while (r1.degree() >= 0) {  // degree() == -1 iff zero
    const auto [q, r2] = r0.divmod(r1);
    // one q-term times r1 per "pmul" row, one subtraction per "psub"
    poly acc = r0;
    if (q.degree() < 0) {
      poly_row(out, "psub",
               "(" + psstr(r0) + ") - (0)*(" + psstr(r1) + ")", r2);
      acc = r2;
    } else {
      for (int k = q.degree(); k >= 0; --k) {
        const rational& qk = q.coeff(static_cast<std::size_t>(k));
        if (qk.is_zero()) continue;
        std::vector<rational> mc(static_cast<std::size_t>(k) + 1);
        mc[static_cast<std::size_t>(k)] = qk;
        const poly mono(std::move(mc));
        const poly prod = mono * r1;
        poly_row(out, "pmul",
                 "(" + psstr(mono) + ")*(" + psstr(r1) + ")", prod);
        const poly next = acc - prod;
        poly_row(out, "psub",
                 "(" + psstr(acc) + ") - (" + psstr(prod) + ")", next);
        acc = next;
      }
    }
    if (!(acc == r2)) {
      out.certified = false;
      out.error = "divstep tree does not reach the remainder";
    }
    r0 = r1;
    r1 = r2;
  }
  const rational lc = r0.coeff(static_cast<std::size_t>(r0.degree()));
  poly monic = r0;
  if (!(lc == kOne)) {
    std::vector<rational> mc;
    // one divide row per coefficient, then the assembly row they
    // fully determine
    for (int i = r0.degree(); i >= 0; --i) {
      const rational& ci = r0.coeff(static_cast<std::size_t>(i));
      arith_row(out, "div", "(" + ci.to_string() + ")/" + chain_lit(lc),
                ci / lc);
    }
    for (int i = 0; i <= r0.degree(); ++i)
      mc.push_back(r0.coeff(static_cast<std::size_t>(i)) / lc);
    monic = poly(std::move(mc));
    poly_row(out, "monic", "(" + psstr(r0) + ")/" + chain_lit(lc), monic);
  }
  if (!(gcd(p1, p2) == monic)) {
    out.certified = false;
    out.error = "chain result disagrees with engine gcd";
  }
  return out;
}

pchain_problem make_pf_chain(int level, long long seed) {
  pyrand::python_random rng("ppf-" + std::to_string(level) + "-" +
                            std::to_string(seed));
  const std::size_t nroots = level >= 2 ? 3 : 2;
  // distinct roots without replacement from {-3..3}
  std::vector<long long> pool = {-3, -2, -1, 0, 1, 2, 3};
  std::vector<rational> roots;
  for (std::size_t i = 0; i < nroots; ++i) {
    const std::size_t j = rng.choice_index(pool.size() - i);
    roots.push_back(rational(bigint(pool[j])));
    pool[j] = pool[pool.size() - i - 1];
  }
  // numerator of degree < nroots, nonzero at every root
  std::vector<rational> nc(nroots);
  for (std::size_t i = 0; i + 1 < nroots; ++i)
    nc[i] = rational(bigint(rng.randint(-4, 4)));
  nc[nroots - 1] = rational(bigint(rng.randint(1, 4)));
  const poly num(std::move(nc));
  for (const rational& a : roots)
    if (num.eval(a).is_zero()) return make_pf_chain(level, seed + kReseed);

  pchain_problem out{"poly_pf", level, seed, {}, true, ""};
  std::vector<rational> residues;
  for (std::size_t i = 0; i < roots.size(); ++i) {
    // N(a) as a mul/add tree: per term, coefficient times root powers
    // as binary "mul" rows, then term values folded with "add" rows
    std::vector<rational> term_vals;
    for (int k = num.degree(); k >= 0; --k) {
      const rational& ck = num.coeff(static_cast<std::size_t>(k));
      if (ck.is_zero()) continue;
      std::vector<rational> factors;
      if (!(ck == kOne) || k == 0) factors.push_back(ck);
      for (int p = 0; p < k; ++p) factors.push_back(roots[i]);
      term_vals.push_back(mul_fold(out, factors));
    }
    const rational nval =
        term_vals.empty() ? rational() : add_fold(out, term_vals);
    // D'(a): one "sub" row per root gap, then "mul" rows fold them
    std::vector<rational> gaps;
    for (std::size_t j = 0; j < roots.size(); ++j) {
      if (j == i) continue;
      const rational gap = roots[i] - roots[j];
      arith_row(out, "sub",
                roots[i].to_string() + " - " + chain_lit(roots[j]), gap);
      gaps.push_back(gap);
    }
    const rational dval = mul_fold(out, gaps);
    const rational res = nval / dval;
    arith_row(out, "res",
              "(" + nval.to_string() + ")/(" + dval.to_string() + ")",
              res);
    residues.push_back(res);
  }
  // recombination: per root res_i * prod_{j != i}(x - a_j) as binary
  // "pmul" rows, folded with "padd" rows back to N — the assemble row's
  // certificate, emitted as its determining context
  std::vector<poly> parts;
  for (std::size_t i = 0; i < roots.size(); ++i) {
    poly part({residues[i]});
    for (std::size_t j = 0; j < roots.size(); ++j) {
      if (j == i) continue;
      const poly fac({rational() - roots[j], kOne});
      const poly grown = part * fac;
      poly_row(out, "pmul",
               "(" + psstr(part) + ")*(" + psstr(fac) + ")", grown);
      part = grown;
    }
    parts.push_back(part);
  }
  poly recon = parts.at(0);
  for (std::size_t i = 1; i < parts.size(); ++i) {
    const poly s = recon + parts[i];
    poly_row(out, "padd",
             "(" + psstr(recon) + ") + (" + psstr(parts[i]) + ")", s);
    recon = s;
  }
  // assemble row: N/prod(x - a_i) -> sum res_i/(x - a_i)
  std::string den_str, sum_str;
  for (std::size_t i = 0; i < roots.size(); ++i) {
    const std::string fac = sym::to_sstr(kX - expr::num(roots[i]));
    if (!den_str.empty()) den_str += "*";
    den_str += "(" + fac + ")";
    if (!sum_str.empty()) sum_str += " + ";
    sum_str += chain_lit(residues[i]) + "/(" + fac + ")";
  }
  const std::string cur = "(" + psstr(num) + ")/(" + den_str + ")";
  bool parses = true;
  try {
    sym::parse(cur);
    sym::parse(sum_str);
  } catch (const std::exception&) {
    parses = false;
  }
  if (!(recon == num) || !parses) {
    out.certified = false;
    out.error = "assemble row failed engine certification";
  }
  out.rows.push_back({"assemble", cur, sum_str});
  return out;
}

}  // namespace ax::mathgen
