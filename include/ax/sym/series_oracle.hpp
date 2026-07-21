#pragma once
/** @file series_oracle.hpp Order-bounded ODE residual check over exact
    truncated series — the series-method analogue of check_odesol.

    Soundness contract (same shape as oracle.hpp, order replaces the IC
    check as the keystone):
    - equivalent_to_order only when the exact residual series is
      identically zero through its natural truncation (a bounded
      structural proof: residual is O(x^order));
    - not_equivalent only on a nonzero residual coefficient — exact
      rational arithmetic makes that a confirmed witness, not a sample;
    - undecided_beyond_order when the equation cannot be expanded
      (unsupported function, shifted transcendental, free symbol). The
      oracle never guesses "valid". */
#include <ax/sym/expr.hpp>
#include <ax/sym/series.hpp>

namespace ax::sym {

enum class series_verdict {
  equivalent_to_order,
  not_equivalent,
  undecided_beyond_order,
};

struct series_check {
  series_verdict v = series_verdict::undecided_beyond_order;
  /** equivalent_to_order: residual proven O(x^order).
      not_equivalent: index of the first nonzero residual coefficient.
      undecided_beyond_order: 0. */
  int order = 0;
};

/** Substitute candidate for y(x) (Derivative(y, x..n) carriers become the
    candidate's termwise nth derivative), expand everything else with
    series::of_expr semantics, and judge the residual lhs - rhs of the
    Eq carrier (bare expressions read as eq == 0). Diff-only doctrine
    holds: only termwise series derivatives, never integration. */
series_check check_odesol_series(const expr& eq, const series& candidate,
                                 const expr& x);

}  // namespace ax::sym
