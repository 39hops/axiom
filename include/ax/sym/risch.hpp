#pragma once
/** @file risch.hpp Magic boards r2: Risch-certified dead states
    (relay 2026-07-27-0 ask 3).

    Per-atom certificates: Liouville's theorem for the exponential /
    trigonometric monomial cases decides elementarity of
      Integral(q(x) * exp(p(x)), x)   and   Integral(q(x) * trig(p(x)), x)
    by solving the Risch ODE y' + p'*y = q for rational y over exact
    (complex-)rational arithmetic — the solve is triangular top-down, so
    "no solution" is a PROOF of non-elementarity, never a heuristic. The
    log family Integral(c*x^n * log(x)^m, x) reduces to the exponential
    case under u = log(x).

    Covered certificate families (everything else -> undecided, never
    masked):
      risch-exp-poly    q polynomial, deg p >= 2         (exp(x**2), ...)
      risch-trig-poly   q polynomial, deg p >= 2         (Fresnel, ...)
      risch-exp-laurent q Laurent w/ pole only at 0, p linear (Ei, ...)
      risch-trig-laurent  same, trig                     (Si, Ci, ...)
      risch-log         c*x^n*log(x)^m under u = log x   (li, ...)

    State-level lift (the S5 mask consumer): a state is certified dead
    when some certified atom's transcendental core (the exp/sin/cos/log
    node) occurs NOWHERE else in the state and the atom occurs once —
    value-preserving moves can then never cancel the non-elementary
    component, so no solved (carrier-free elementary) state is
    reachable. The core check is syntactic (hash-consed): a masked
    state costs the scorer a candidate it could never close; an
    unmasked dead state costs only scoring time. */
#include <ax/sym/expr.hpp>

#include <string>
#include <vector>

namespace ax::sym {

struct board_cert {
  bool dead = false;   // certified non-elementary / unreachable-solved
  std::string reason;  // certificate tag ("" when not dead)
};

/** Certify one Integral(f, x) atom non-elementary. Undecided and
    provably-elementary shapes both return {false, ""}. */
board_cert risch_certify(const expr& integral_node);

/** State-level certificate (see the lift condition above). */
board_cert dead_state(const expr& state);

/** Batch mask for S5: mask[i] == true iff states[i] is certified dead.
    Pure syntactic + exact arithmetic; no oracle calls, no cache. */
std::vector<bool> dead_state_mask(const std::vector<expr>& states);

}  // namespace ax::sym
