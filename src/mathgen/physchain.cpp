/** @file physchain.cpp Physics chain makers (see physchain.hpp). */
#include <ax/mathgen/physchain.hpp>

#include <ax/mathgen/ode.hpp>
#include <ax/mathgen/series_chain.hpp>
#include <ax/mathgen/series_solve.hpp>
#include <ax/pyrand/pyrand.hpp>
#include <ax/sym/calc.hpp>
#include <ax/sym/expand.hpp>
#include <ax/sym/parse.hpp>
#include <ax/sym/poly.hpp>
#include <ax/sym/print_sstr.hpp>
#include <ax/sym/series.hpp>
#include <ax/sym/series_oracle.hpp>

#include <stdexcept>
#include <vector>

namespace ax::mathgen {

namespace {

using sym::expr;
using sym::poly;

const expr kT = expr::symbol("t");
constexpr long long kReseed = 1'000'003;

/** Certify one antiderivative row: parse(nxt), differentiate in t,
    expand, and compare as exact polynomials against parse(cur). */
bool int_row_certifies(const std::string& cur, const std::string& nxt) {
  try {
    return poly::from_expr(sym::expand(sym::diff(sym::parse(nxt), kT)),
                           kT) ==
           poly::from_expr(sym::expand(sym::parse(cur)), kT);
  } catch (const std::exception&) {
    return false;
  }
}

/** Emit "int" + "append" rows integrating src (a polynomial in t) with
    integration constant ic; returns the integrated poly (v or x). */
poly integrate_stage(pchain_problem& out, const poly& src,
                     const rational& ic) {
  std::vector<rational> anti(static_cast<std::size_t>(src.degree()) + 2);
  anti[0] = ic;
  poly partial({ic});
  for (int k = 0; k <= src.degree(); ++k) {
    const rational& ck = src.coeff(static_cast<std::size_t>(k));
    if (ck.is_zero()) continue;
    const rational ik = ck / rational(bigint(k + 1));
    anti[static_cast<std::size_t>(k) + 1] = ik;
    std::vector<rational> tc(static_cast<std::size_t>(k) + 1);
    tc[static_cast<std::size_t>(k)] = ck;
    std::vector<rational> ti(static_cast<std::size_t>(k) + 2);
    ti[static_cast<std::size_t>(k) + 1] = ik;
    const std::string cur = sym::to_sstr(poly(std::move(tc)).to_expr(kT));
    const std::string nxt = sym::to_sstr(poly(std::move(ti)).to_expr(kT));
    if (!int_row_certifies(cur, nxt)) {
      out.certified = false;
      out.error = "int row failed engine certification";
    }
    out.rows.push_back({"int", cur, nxt});
    // fold spelling (llmopt 2026-07-23: cur must determine nxt) — the
    // integrated term and the running partial are both in the string
    const poly grown = partial + poly::from_expr(sym::parse(nxt), kT);
    const std::string acur = "(" + nxt + ") + (" +
                             sym::to_sstr(partial.to_expr(kT)) + ")";
    bool ok = false;
    try {
      ok = poly::from_expr(sym::expand(sym::parse(acur)), kT) == grown;
    } catch (const std::exception&) {
    }
    if (!ok) {
      out.certified = false;
      out.error = "append fold failed engine certification";
    }
    out.rows.push_back(
        {"append", acur, sym::to_sstr(grown.to_expr(kT))});
    partial = grown;
  }
  return poly(std::move(anti));
}

}  // namespace

pchain_problem make_kin_chain(int level, long long seed) {
  pyrand::python_random rng("pkin-" + std::to_string(level) + "-" +
                            std::to_string(seed));
  const int deg = level >= 3 ? 2 : level;  // a(t) degree: 1, 2 at L3
  std::vector<rational> ac(static_cast<std::size_t>(deg) + 1);
  bool all_zero = true;
  for (int k = 0; k <= deg; ++k) {
    const long long c = rng.randint(-4, 4);
    ac[static_cast<std::size_t>(k)] = rational(bigint(c));
    if (c != 0) all_zero = false;
  }
  const rational v0(bigint(rng.randint(-4, 4)));
  const rational x0(bigint(rng.randint(-4, 4)));
  if (all_zero) return make_kin_chain(level, seed + kReseed);
  const poly a(std::move(ac));

  pchain_problem out{"phys_kin", level, seed, {}, true, ""};
  const poly v = integrate_stage(out, a, v0);
  const poly x = integrate_stage(out, v, x0);
  // problem-level certification on the emitted final strings
  const auto ends_with = [&](const poly& want) {
    for (auto it = out.rows.rbegin(); it != out.rows.rend(); ++it)
      if (it->kind == "append")
        return poly::from_expr(sym::parse(it->nxt), kT) == want;
    return false;
  };
  if (!(v.derivative() == a) || !(x.derivative() == v) ||
      !(v.eval(rational()) == v0) || !(x.eval(rational()) == x0) ||
      !ends_with(x)) {
    out.certified = false;
    if (out.error.empty()) out.error = "kinematics cross-check failed";
  }
  return out;
}

pchain_problem make_shm_chain(int level, long long seed, int order) {
  pyrand::python_random rng("pshm-" + std::to_string(level) + "-" +
                            std::to_string(seed));
  const long long w = level;  // angular frequency scales with level
  const long long y0i = rng.randint(-4, 4);
  const long long v0i = rng.randint(-4, 4);
  if (y0i == 0 && v0i == 0) return make_shm_chain(level, seed + kReseed, order);
  const expr x = expr::symbol("x");  // series_solve's internal symbol
  const expr y = expr::fn("y", x);
  const expr d2 =
      expr::fn("Derivative", std::vector<expr>{y, x, x});
  const expr eq = expr::fn(
      "Eq", std::vector<expr>{d2 + expr::num(w * w) * y, expr::num(0)});
  const expr sol =
      expr::num(y0i) * expr::fn("cos", expr::num(w) * x) +
      expr::num(rational(bigint(v0i), bigint(w))) *
          expr::fn("sin", expr::num(w) * x);
  const ode_problem p{"phys_shm", level,
                      eq,         sol,
                      0,          rational(bigint(y0i)),
                      rational(bigint(v0i))};

  pchain_problem out{"phys_shm", level, seed, {}, true, ""};
  try {
    const auto s = series_solve(p, order);
    const auto chk = sym::check_odesol_series(p.eq, s.y, x);
    if (chk.v != sym::series_verdict::equivalent_to_order ||
        !(s.y == sym::series::of_expr(p.sol, x, order))) {
      out.certified = false;
      out.error = "shm residual/coefficient cross-check failed";
    }
    const auto partial = [&](int upto) {
      const sym::series ps(
          std::vector<rational>(s.y.coeffs().begin(),
                                s.y.coeffs().begin() + upto),
          upto);
      return sym::to_sstr(ps.to_expr(kT)) + " + O(t**" +
             std::to_string(upto) + ")";
    };
    for (const auto& st : s.steps) {
      const auto d = derivation_rows(st);
      for (const auto& r : d.reduction) {
        if (sym::to_sstr(sym::parse(r.cur)) != r.nxt) {
          out.certified = false;
          out.error = "reduction row failed engine certification";
        }
        out.rows.push_back({r.kind, r.cur, r.nxt});
      }
      if (sym::to_sstr(sym::parse(d.solve_cur)) != d.solve_nxt) {
        out.certified = false;
        out.error = "solve row failed engine certification";
      }
      out.rows.push_back({"solve", d.solve_cur, d.solve_nxt});
      out.rows.push_back({"append", partial(st.n), partial(st.n + 1)});
    }
  } catch (const std::exception& ex) {
    out.certified = false;
    out.error = ex.what();
  }
  return out;
}

}  // namespace ax::mathgen
