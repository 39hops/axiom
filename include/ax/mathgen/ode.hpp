#pragma once
/** @file ode.hpp L9 rung 3: native ODE problem makers, ported
    call-for-call from llmopt/mathgen/odes.py (solution drawn first,
    equation built from it — rows verify via check_odesol, never by
    comparing to the stored solution). Parity gate: byte-exact sstr
    equality per (family, level, seed) against fixtures generated from
    llmopt's actual module (tests/mathgen/fixtures/ode_fixture.jsonl). */
#include <ax/core/rational.hpp>
#include <ax/sym/expr.hpp>

#include <optional>
#include <string>

namespace ax::mathgen {

struct ode_problem {
  std::string family;  // ode_linear1 | ode_cc2 | ode_separable
  int level = 1;
  sym::expr eq;   // Eq(lhs, rhs) carrier over y(x)
  sym::expr sol;  // the drawn solution (an expression in x)
  long long x0 = 0;
  rational y0;
  std::optional<rational> yp0;  // second-order rows pin y'(x0) too
};

ode_problem make_linear_first_order(int level, long long seed);
ode_problem make_second_order_cc(int level, long long seed);
ode_problem make_separable_growth(int level, long long seed);

}  // namespace ax::mathgen
