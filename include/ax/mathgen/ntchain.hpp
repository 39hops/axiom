#pragma once
/** @file ntchain.hpp Number-theory chains (llmopt relay 2026-07-27 -9,
    priority 1): Euclid / extended-Euclid, modular inverse, CRT
    reconstruction, modular exponentiation, continued-fraction
    convergents — 3+ step exact-integer derivations as certified
    cur/nxt rewrite rows. Oracle is free: every row is certified by
    parsing BOTH emitted strings and evaluating them in exact bigint
    arithmetic (nt_eval), then cross-checked problem-level against the
    independent core/nt implementations (ext_gcd, crt, modpow).

    Serialization spec (VOCAB_EXTRA contract — rows ship their atoms):
      characters: digits 0-9, '+', '-', '*', '(', ')', ',', ' '
      names:      gcd(a, b)   — nonnegative gcd
                  Mod(a, m)   — least nonnegative residue, m > 0
      operator:   '**'        — power, exponent a nonnegative literal
    Grammar (nt_eval): expr := term (('+'|'-') term)*;
    term := factor ('*' factor)*; factor := '-' factor
    | primary ('**' factor)?; primary := integer | name '(' expr ','
    expr ')' | '(' expr ')'. Negative literals inside binary spellings
    are always parenthesized by the emitters.

    A problem whose rows do not all certify is reported with
    certified == false and an honest error, never dropped. */
#include <ax/core/bigint.hpp>
#include <ax/mathgen/polychain.hpp>

#include <string>

namespace ax::mathgen {

/** Exact evaluator for the NT row grammar above. Throws
    std::invalid_argument on any spelling outside the grammar,
    std::domain_error on Mod with m <= 0 or negative/huge exponent. */
bigint nt_eval(const std::string& s);

/** Call-span resolution trace (Leg B pilot, llmopt relay 2026-07-29-2):
    for every call site (gcd/Mod) in s, a span
    "call: <site> -> <value>" with the site text verbatim and the value
    from nt_eval. Nested calls flatten innermost-first in evaluation
    order — an outer site's span shows its arguments with inner calls
    already substituted by their values (the resolution trace, not a
    tree). Throws like nt_eval on anything outside the grammar. */
std::vector<std::string> nt_call_spans(const std::string& s);

/** Euclid gcd chain: per division a = q*b + r the rows
    "mul" (q*b), "sub" (a - qb -> r), "gcdstep" (gcd(a, b) ->
    gcd(b, r)), closing "gcdend" (gcd(g, 0) -> g). */
pchain_problem make_nt_gcd_chain(int level, long long seed);

/** Extended Euclid: the gcd chain plus the Bezout coefficient
    recurrences s/t (one "mul" + "sub" pair per quotient per
    coefficient), closing "bezout" (a*s + b*t -> g). */
pchain_problem make_nt_bezout_chain(int level, long long seed);

/** Modular inverse of a mod m (coprime by construction): the bezout
    rows on (a, m), then "inv" (Mod(s, m) -> a^-1) and "invcheck"
    (Mod(a*inv, m) -> 1). */
pchain_problem make_nt_modinv_chain(int level, long long seed);

/** CRT reconstruction over 2 (L1) or 3 (L2+) pairwise-coprime moduli:
    iterative combine — inverse sub-chain, then sub/mod/mul/add rows to
    the lifted x, one "crtcheck" row (Mod(x, m_i) -> r_i) per input
    congruence. */
pchain_problem make_nt_crt_chain(int level, long long seed);

/** Modular exponentiation by square-and-multiply: "mod" (Mod(b, m)),
    per bit "sq" (Mod(acc*acc, m)) and "mulstep" (Mod(s*b, m)),
    closing "modexp" (Mod(b**e, m) -> result). */
pchain_problem make_nt_modexp_chain(int level, long long seed);

/** Continued fraction of p/q: the Euclid quotient rows, then the
    convergent recurrences h/k ("mul" + "add" per coefficient),
    closing "det" (h*k' - h'*k -> +-1, the best-rational certificate). */
pchain_problem make_nt_cf_chain(int level, long long seed);

}  // namespace ax::mathgen
