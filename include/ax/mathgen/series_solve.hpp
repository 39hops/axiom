#pragma once
/** @file series_solve.hpp L10 rung 3: the series ODE method as a chain
    producer. Substitute y = sum a_n x^n into a certified-family linear
    ODE, derive the coefficient recurrence structurally from the Eq
    carrier, and produce a_n one exact rational step at a time — each
    step is one farm chain row (a_{n+k} from its predecessors). */
#include <ax/core/rational.hpp>
#include <ax/mathgen/ode.hpp>
#include <ax/sym/series.hpp>

#include <vector>

namespace ax::mathgen {

/** One contributing product c * fall * a of the recurrence sum for a
    step: c is the series coefficient of c_i(x), fall the falling
    factorial from differentiating x^m i times, a the earlier series
    coefficient it multiplies. All exact rationals so the arithmetic can
    be re-emitted as certifiable rewrite rows. */
struct series_term {
  rational c;
  rational fall;
  rational a;
};

struct series_step {
  int n = 0;      ///< coefficient index produced (a_n)
  rational a_n;   ///< its exact value
  rational q_n;      ///< forcing coefficient q_n on the rhs
  rational divisor;  ///< c_k[0] * fall(n, k): a_n == (q - sum)/divisor
  std::vector<series_term> terms;  ///< the recurrence sum, term by term
};

struct series_solution {
  int ode_order = 0;             ///< k: highest Derivative order in eq
  sym::series y;                 ///< a_0..a_{N-1} + O(x^N)
  std::vector<series_step> steps;  ///< one per recurrence application
                                   ///< (IC-seeded coefficients excluded)
};

/** Solve p.eq around x0 == 0 to the given truncation order via the
    coefficient recurrence. Requirements (all hold for the certified L9b
    families): p.x0 == 0; eq linear in y — every Add term carries at most
    one y(x)/Derivative carrier as a Mul factor; the top-derivative
    coefficient is a nonzero constant; non-carrier factors expand through
    series::of_expr. Throws std::domain_error / std::invalid_argument
    when a requirement fails (callers report honest UNDECIDED). */
series_solution series_solve(const ode_problem& p, int order);

}  // namespace ax::mathgen
