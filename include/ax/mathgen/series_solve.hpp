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

struct series_step {
  int n = 0;      ///< coefficient index produced (a_n)
  rational a_n;   ///< its exact value
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
