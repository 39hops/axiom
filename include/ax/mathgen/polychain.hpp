#pragma once
/** @file polychain.hpp Poly-algebra pilot (llmopt relay 2026-07-23 GO):
    gcd chains (Euclidean division steps) and partial fractions as
    certified cur/nxt rewrite rows in the existing vocab — digits, x,
    + - * / ( ) space, ** for powers; no new characters.

    Certification per row kind:
    - constant-arithmetic rows ("num", "den", "res") fold byte-exactly
      via sym::parse -> canonical, the series-chain standard;
    - polynomial-identity rows ("divstep", "monic", "assemble") are
      certified by parse -> expand -> poly::from_expr equality against
      the exact polynomial the engine computed independently, and their
      nxt is printed from that engine polynomial (byte-exact spelling).
    A problem whose rows do not all certify is reported with
    certified == false and an honest error, never dropped. */
#include <ax/core/rational.hpp>

#include <string>
#include <vector>

namespace ax::mathgen {

struct pchain_row {
  std::string kind;
  std::string cur;
  std::string nxt;
};

struct pchain_problem {
  std::string family;  ///< poly_gcd | poly_pf
  int level = 1;
  long long seed = 0;
  std::vector<pchain_row> rows;
  bool certified = false;
  std::string error;  ///< empty when certified
};

/** Euclidean gcd chain on p1 = g*u, p2 = g*v (gcd(u, v) == 1 by
    construction): one "divstep" row per division r0 - q*r1 -> r2, a
    "monic" row when the last remainder needs normalizing, and the
    engine's own gcd(p1, p2) as the cross-check. */
pchain_problem make_gcd_chain(int level, long long seed);

/** Partial fractions over distinct integer linear factors: per root a
    "num" row (N(a) substituted), a "den" row (prod of root gaps), a
    "res" row (their quotient), then one "assemble" row rewriting
    N/prod(x - a_i) as the residue sum. Residues are nonzero by
    construction (degenerate draws are reseeded). */
pchain_problem make_pf_chain(int level, long long seed);

/** Rational-integral bridge chains (llmopt transfer experiment,
    2026-07-23): the full pf derivation rows (shared steps in context
    are the transfer mechanism), then Integral(N/D, x) -> the residue
    split ("ibridge"), each piece closed to r*log(x - a) ("iclose"),
    and the whole integral closed ("close"). Integral-grammar rows are
    certified by search::verify_edge — the same three-valued oracle
    that guards farm chain emission. Draws under an independent rng
    tag, so populations do not overlap poly_pf. */
pchain_problem make_bridge_chain(int level, long long seed);

}  // namespace ax::mathgen
