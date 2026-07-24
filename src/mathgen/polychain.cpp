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
#include <ax/search/search.hpp>
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

/** Termwise precursors (llmopt 2026-07-23 postscript: operand
    complexity is a second axis — p-kind rows get their coefficient
    arithmetic emitted first as constant primitives, so the poly row
    is an assembly of already-derived facts). */
void termwise_mul(pchain_problem& out, const rational& c, const poly& p) {
  if (c == kOne) return;  // nothing to derive
  for (int j = 0; j <= p.degree(); ++j) {
    const rational& pj = p.coeff(static_cast<std::size_t>(j));
    if (pj.is_zero() || pj == kOne) continue;
    arith_row(out, "mul", chain_lit(c) + "*" + chain_lit(pj), c * pj);
  }
}

void termwise_sub(pchain_problem& out, const poly& a, const poly& b) {
  const int deg = a.degree() > b.degree() ? a.degree() : b.degree();
  for (int j = 0; j <= deg; ++j) {
    const rational& aj = a.coeff(static_cast<std::size_t>(j));
    const rational& bj = b.coeff(static_cast<std::size_t>(j));
    if (aj.is_zero() || bj.is_zero()) continue;  // copies, not facts
    arith_row(out, "sub", aj.to_string() + " - " + chain_lit(bj), aj - bj);
  }
}

void termwise_add(pchain_problem& out, const poly& a, const poly& b) {
  const int deg = a.degree() > b.degree() ? a.degree() : b.degree();
  for (int j = 0; j <= deg; ++j) {
    const rational& aj = a.coeff(static_cast<std::size_t>(j));
    const rational& bj = b.coeff(static_cast<std::size_t>(j));
    if (aj.is_zero() || bj.is_zero()) continue;
    arith_row(out, "add", aj.to_string() + " + " + chain_lit(bj), aj + bj);
  }
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
        termwise_mul(out, qk, r1);
        poly_row(out, "pmul",
                 "(" + psstr(mono) + ")*(" + psstr(r1) + ")", prod);
        const poly next = acc - prod;
        termwise_sub(out, acc, prod);
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

namespace {

/** Shared partial-fraction chain builder (poly_pf and the ibridge
    family draw identically but under different rng tags, so their
    problem populations are independent). */
pchain_problem pf_build(const std::string& family, const std::string& tag,
                        int level, long long seed,
                        std::vector<rational>& roots_out,
                        std::vector<rational>& residues_out,
                        poly& num_out) {
  pyrand::python_random rng(tag + "-" + std::to_string(level) + "-" +
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
    if (num.eval(a).is_zero())
      return pf_build(family, tag, level, seed + kReseed, roots_out,
                      residues_out, num_out);

  pchain_problem out{family, level, seed, {}, true, ""};
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
      termwise_mul(out, rational() - roots[j], part);
      poly_row(out, "pmul",
               "(" + psstr(part) + ")*(" + psstr(fac) + ")", grown);
      part = grown;
    }
    parts.push_back(part);
  }
  poly recon = parts.at(0);
  for (std::size_t i = 1; i < parts.size(); ++i) {
    // one COEFFICIENT per row (llmopt: operand size is a second axis):
    // fold each nonzero term of the incoming part into the accumulator,
    // constant add facts first, then the single-term poly fold
    termwise_add(out, recon, parts[i]);
    for (int j = 0; j <= parts[i].degree(); ++j) {
      const rational& cj = parts[i].coeff(static_cast<std::size_t>(j));
      if (cj.is_zero()) continue;
      std::vector<rational> tc(static_cast<std::size_t>(j) + 1);
      tc[static_cast<std::size_t>(j)] = cj;
      const poly term(std::move(tc));
      const poly s = recon + term;
      poly_row(out, "padd",
               "(" + psstr(term) + ") + (" + psstr(recon) + ")", s);
      recon = s;
    }
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
  roots_out = roots;
  residues_out = residues;
  num_out = num;
  return out;
}

}  // namespace

pchain_problem make_pf_chain(int level, long long seed) {
  std::vector<rational> roots, residues;
  poly num;
  return pf_build("poly_pf", "ppf", level, seed, roots, residues, num);
}

pchain_problem make_bridge_chain(int level, long long seed) {
  std::vector<rational> roots, residues;
  poly num;
  pchain_problem out =
      pf_build("poly_ibridge", "pbrg", level, seed, roots, residues, num);

  // the bridge: Integral(N/D, x) -> sum Integral(r_i/(x - a_i), x),
  // then each piece closes to r_i*log(x - a_i). Every edge certified by
  // the farm-grade three-valued oracle (verify_edge, empty slots).
  const search::external_slots ext{};
  const auto edge_row = [&](const std::string& kind, const expr& parent,
                            const expr& child) {
    if (!search::verify_edge(parent, child, ext)) {
      out.certified = false;
      out.error = kind + " row failed verify_edge";
    }
    out.rows.push_back({kind, sym::to_sstr(parent), sym::to_sstr(child)});
  };
  expr den = kX - expr::num(roots[0]);
  for (std::size_t i = 1; i < roots.size(); ++i)
    den = den * (kX - expr::num(roots[i]));
  const expr whole = expr::fn(
      "Integral", std::vector<expr>{num.to_expr(kX) / den, kX});
  std::vector<expr> pieces, antis;
  for (std::size_t i = 0; i < roots.size(); ++i) {
    const expr fac = kX - expr::num(roots[i]);
    pieces.push_back(expr::fn(
        "Integral",
        std::vector<expr>{expr::num(residues[i]) / fac, kX}));
    antis.push_back(expr::num(residues[i]) * expr::fn("log", fac));
  }
  // per-piece peel (llmopt 2026-07-23 night: the one-fact append
  // pattern that closed series at 98): each "ibridge" row splits off
  // ONE residue integral, leaving a smaller certified rational
  // integral. M_{k+1} = (M_k - r_k * prod_{j>k}(x - a_j)) / (x - a_k),
  // exact by the residue construction (division checked remainder-0).
  expr state = whole;
  poly rem = num;
  for (std::size_t k = 0; k + 1 < roots.size(); ++k) {
    poly tail_prod({kOne});
    expr tail_den = kX - expr::num(roots[k + 1]);
    for (std::size_t j = k + 1; j < roots.size(); ++j) {
      tail_prod = tail_prod * poly({rational() - roots[j], kOne});
      if (j > k + 1) tail_den = tail_den * (kX - expr::num(roots[j]));
    }
    const poly scaled = poly({residues[k]}) * tail_prod;
    const poly sub_num = rem - scaled;
    const auto [m_next, m_rem] =
        sub_num.divmod(poly({rational() - roots[k], kOne}));
    if (!(m_rem == poly())) {
      out.certified = false;
      out.error = "peel division left a remainder";
    }
    // determining context for the subtracted numerator: r_k * T_k as
    // constant "mul" facts + the pmul row, then per-coefficient "sub"
    // facts + the psub row (same doctrine as the pf recombination)
    termwise_mul(out, residues[k], tail_prod);
    poly_row(out, "pmul",
             chain_lit(residues[k]) + "*(" + psstr(tail_prod) + ")", scaled);
    termwise_sub(out, rem, scaled);
    poly_row(out, "psub",
             "(" + psstr(rem) + ") - (" + psstr(scaled) + ")", sub_num);
    // recipe application #6 (llmopt 2026-07-24 morning: peel was still
    // a two-integral row at 4/15): the peel is now TWO one-fact rows —
    // "ibridge" splits off the residue integral, leaving the literal
    // subtraction (M - r*T)/D un-divided (copyable arithmetic from
    // cur); "icancel" then cancels the (x - a_k) factor, one integral
    // rewritten to one integral.
    const expr cur_den = (kX - expr::num(roots[k])) * tail_den;
    expr prefix = pieces[0];
    for (std::size_t j = 1; j <= k; ++j) prefix = prefix + pieces[j];
    const expr sub_integral = expr::fn(
        "Integral", std::vector<expr>{sub_num.to_expr(kX) / cur_den, kX});
    const expr mid = k == 0 ? pieces[0] + sub_integral
                            : prefix + sub_integral;
    edge_row("ibridge", state, mid);
    rem = m_next;
    const expr remaining = expr::fn(
        "Integral", std::vector<expr>{rem.to_expr(kX) / tail_den, kX});
    const expr next = k == 0 ? pieces[0] + remaining : prefix + remaining;
    edge_row("icancel", mid, next);
    state = next;
  }
  // the last remaining integral IS the last residue piece; assert so
  if (!(rem == poly({residues[roots.size() - 1]}))) {
    out.certified = false;
    out.error = "peel chain did not terminate on the last residue";
  }
  expr split = pieces[0];
  for (std::size_t i = 1; i < pieces.size(); ++i) split = split + pieces[i];
  for (std::size_t i = 0; i < pieces.size(); ++i)
    edge_row("iclose", pieces[i], antis[i]);
  // per-piece close fold: replace one integral piece with its log per
  // row, walking the full state from the split to the closed form
  state = split;
  for (std::size_t i = 0; i < pieces.size(); ++i) {
    expr next = antis[0];
    for (std::size_t j = 1; j < pieces.size(); ++j)
      next = next + (j <= i ? antis[j] : pieces[j]);
    edge_row("close", state, next);
    state = next;
  }
  return out;
}

}  // namespace ax::mathgen
