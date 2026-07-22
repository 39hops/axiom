#pragma once
/** @file physchain.hpp Physics rung 1 (llmopt relay 2026-07-23 GO):
    kinematics and SHM chains over the time symbol t. Units live
    engine-side per the determinability doctrine — every family carries
    a fixed dimension signature (kin: a in L/T^2 -> v in L/T -> x in L;
    shm: [w] = 1/T balances y'' against w^2 y), checked by construction;
    the emitted strings stay pure arithmetic/calculus, vocab-41 (t is
    the only new character; no y, no '=').

    Row kinds: kin emits "int" rows (one antiderivative term, certified
    by parse -> diff -> expand -> poly equality against the source term)
    and "append" rows building v(t) then x(t) from their ICs; shm reuses
    the certified series machinery (mul/add/solve/append) with partial
    sums printed in t. */
#include <ax/mathgen/polychain.hpp>

namespace ax::mathgen {

/** Kinematics: draw polynomial a(t) plus v0, x0; derive v(t) then x(t)
    term by term. Chain-certified per row, problem-certified by
    d/dt v == a, d/dt x == v, v(0) == v0, x(0) == x0 (all engine ops on
    the emitted strings, not the construction). */
pchain_problem make_kin_chain(int level, long long seed);

/** Simple harmonic motion: y'' + w^2 y = 0 with y(0), y'(0) pinned —
    the certified cc2 recurrence relabeled. Emits the series chain to
    the given truncation order with partial sums in t. */
pchain_problem make_shm_chain(int level, long long seed, int order);

}  // namespace ax::mathgen
