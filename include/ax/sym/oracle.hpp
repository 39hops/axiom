#pragma once
/** @file oracle.hpp Verification-oracle primitives for llmopt.

    Soundness contract (docs/specs/2026-07-18-llmopt-oracle.md):
    - verdict::equivalent only on structural proof (canonical difference
      is literally zero);
    - verdict::not_equivalent only on a confirmed numeric witness;
    - verdict::undecided otherwise — the oracle never guesses "valid". */
#include <ax/sym/expr.hpp>

namespace ax::sym {

/** Canonical form of e as a function of x: fn arguments and exponents are
    canonicalized recursively, the whole expression is combined over a
    common denominator, numerator and denominator expanded, and — when both
    are polynomials in x — reduced by polynomial gcd with a monic
    denominator. No trig identities are applied (sin^2+cos^2 stays as-is);
    that incompleteness surfaces as verdict::undecided downstream. */
expr canonical(const expr& e, const expr& x);

enum class verdict { equivalent, not_equivalent, undecided };

/** Sound three-valued equivalence of a and b as functions of x. Symbols
    other than x (and pi/E, bound to their values) are treated as fixed
    deterministic parameters. */
verdict equivalent(const expr& a, const expr& b, const expr& x);

/** Integral-step check: is candidate an antiderivative of integrand?
    Computed as equivalent(diff(candidate, x), integrand) — differentiation
    only, never integration. */
verdict equivalent_mod_const(const expr& candidate, const expr& integrand,
                             const expr& x);

}  // namespace ax::sym
