/** @file polychain.cpp Poly-algebra chain makers (see polychain.hpp). */
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

std::string psstr(const poly& p) { return sym::to_sstr(p.to_expr(kX)); }

/** Engine certification of a polynomial-identity row: cur must parse,
    expand, and read back as exactly the engine polynomial want. */
bool poly_folds_to(const std::string& cur, const poly& want) {
  try {
    return poly::from_expr(sym::expand(sym::parse(cur)), kX) == want;
  } catch (const std::exception&) {
    return false;
  }
}

/** Engine certification of a constant-arithmetic row: parse ->
    canonical must fold byte-exactly to nxt. */
bool folds_to(const std::string& cur, const std::string& nxt) {
  try {
    return sym::to_sstr(sym::parse(cur)) == nxt;
  } catch (const std::exception&) {
    return false;
  }
}

std::string value_sstr(const rational& r) {
  return sym::to_sstr(expr::num(r));
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
    const std::string cur =
        "(" + psstr(r0) + ") - (" + psstr(q) + ")*(" + psstr(r1) + ")";
    if (!poly_folds_to(cur, r2)) {
      out.certified = false;
      out.error = "divstep row failed engine certification";
    }
    out.rows.push_back({"divstep", cur, psstr(r2)});
    r0 = r1;
    r1 = r2;
  }
  const rational lc = r0.coeff(static_cast<std::size_t>(r0.degree()));
  poly monic = r0;
  if (!(lc == rational(bigint(1)))) {
    std::vector<rational> mc;
    for (int i = 0; i <= r0.degree(); ++i)
      mc.push_back(r0.coeff(static_cast<std::size_t>(i)) / lc);
    monic = poly(std::move(mc));
    const std::string cur = "(" + psstr(r0) + ")/" + chain_lit(lc);
    if (!poly_folds_to(cur, monic)) {
      out.certified = false;
      out.error = "monic row failed engine certification";
    }
    out.rows.push_back({"monic", cur, psstr(monic)});
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
  const auto arith = [&](const std::string& kind, const std::string& cur,
                         const rational& val) {
    const std::string nxt = value_sstr(val);
    if (!folds_to(cur, nxt)) {
      out.certified = false;
      out.error = kind + " row failed engine certification";
    }
    out.rows.push_back({kind, cur, nxt});
  };
  std::vector<rational> residues;
  for (std::size_t i = 0; i < roots.size(); ++i) {
    // N(a) spelled by substitution, highest power first
    std::string ns;
    for (int k = num.degree(); k >= 0; --k) {
      const rational& ck = num.coeff(static_cast<std::size_t>(k));
      if (ck.is_zero()) continue;
      if (!ns.empty()) ns += " + ";
      std::string term =
          ck == rational(bigint(1)) && k > 0 ? std::string() : chain_lit(ck);
      if (k > 0) {
        if (!term.empty()) term += "*";
        term += chain_lit(roots[i]);
        if (k > 1) term += "**" + std::to_string(k);
      }
      ns += term;
    }
    if (ns.empty()) ns = "0";
    const rational nval = num.eval(roots[i]);
    arith("num", ns, nval);
    // D'(a) = prod of root gaps, spelled factor by factor
    std::string ds;
    rational dval(bigint(1));
    for (std::size_t j = 0; j < roots.size(); ++j) {
      if (j == i) continue;
      if (!ds.empty()) ds += "*";
      ds += "(" + roots[i].to_string() + " - " + chain_lit(roots[j]) + ")";
      dval = dval * (roots[i] - roots[j]);
    }
    arith("den", ds, dval);
    const rational res = nval / dval;
    arith("res", "(" + nval.to_string() + ")/(" + dval.to_string() + ")",
          res);
    residues.push_back(res);
  }
  // assemble row: N/prod(x - a_i) -> sum res_i/(x - a_i); certified by
  // the exact polynomial identity N == sum res_i * prod_{j != i}(x - a_j)
  std::string den_str, sum_str;
  for (std::size_t i = 0; i < roots.size(); ++i) {
    const std::string fac =
        sym::to_sstr(kX - expr::num(roots[i]));
    if (!den_str.empty()) den_str += "*";
    den_str += "(" + fac + ")";
    if (!sum_str.empty()) sum_str += " + ";
    sum_str += chain_lit(residues[i]) + "/(" + fac + ")";
  }
  const std::string cur = "(" + psstr(num) + ")/(" + den_str + ")";
  poly recon;
  for (std::size_t i = 0; i < roots.size(); ++i) {
    poly part({residues[i]});
    for (std::size_t j = 0; j < roots.size(); ++j)
      if (j != i) part = part * poly({rational() - roots[j],
                                      rational(bigint(1))});
    recon = recon + part;
  }
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
