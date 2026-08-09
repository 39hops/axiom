# ENGINE-EXACT-1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Precision-ladder rungs (Q9 shipped / Q32 / Q64) plus a dyadic exact-prefix anchor in the intbirth engine, per `docs/specs/2026-08-08-engine-exact-ladder.md`.

**Architecture:** Extract intbirth's function bodies into a header-only core templated on `<Op, Acc, Round>` (operand scalar, accumulator, rounding policy); the existing i64 symbols in `src/nn/intbirth.cpp` become the Q9 instantiation, gated bit-identical by every existing digest test. Rungs differ only in `contract.precision` and widths; the anchor is the same core instantiated with a bigint-backed dyadic scalar and an `Exact` rounding policy.

**Tech Stack:** C++23, CMake+Ninja, GoogleTest (test-only), `__int128` (both dev compilers support it), `ax::core::bigint` for the anchor scalar, pybind11 (bindings-only) for drivers.

## Global Constraints

- Q9 (`precision = 9`, the default) must remain BIT-IDENTICAL to the shipped engine: the whole existing suite (495 tests) plus `tools/int_adamw/verify_*.py` digest drivers are the no-op gate after every task.
- Rounding *placement* never moves; only grain. Frozen-grain convention (spec §Convention pin): transcendental/isqrt sites truncate rung→shipped scale, apply the shipped convention, shift back.
- Tables stay opaque shipped bytes; never regenerated.
- No changes to ENGINE-SCALE-1 cells/contracts; no GPU work.
- STL-only rule holds (no new dependencies).
- Commit after every green task; do not push mid-plan (push at plan completion).

---

### Task 1: Baseline gate snapshot

**Files:**
- None created (evidence-only task).

**Interfaces:**
- Produces: a known-green baseline; every later task re-runs these exact commands.

- [ ] **Step 1: Build and run the full suite**

Run: `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug && cmake --build build && ctest --test-dir build --output-on-failure`
Expected: all tests pass (495 at last booking).

- [ ] **Step 2: Run the digest drivers**

Run: `cmake -S . -B build -G Ninja -DAXIOM_BUILD_PYTHON=ON -Dpybind11_DIR=$(python3.12 -m pybind11 --cmakedir) && cmake --build build && cd tools/int_adamw && python3.12 verify_intbirth.py ../../build && python3.12 verify_multiblock.py ../../build && python3.12 verify_gravmoe.py ../../build && python3.12 verify_primitives.py ../../build`
Expected: every driver reports digest match against its pinned reference. If any fails, STOP — the tree is not at a green baseline and the plan must not proceed.

---

### Task 2: `contract.precision` field + scale plumbing (no behavior change)

**Files:**
- Modify: `include/ax/nn/intbirth.hpp` (struct `contract`, line ~61)
- Modify: `src/nn/intbirth.cpp` (replace file-local `constexpr i64 Q = 512` reads with a contract-derived value)
- Test: `tests/nn/test_intbirth_precision.cpp` (new)

**Interfaces:**
- Produces: `contract::precision` (int, default 9); `i64 contract_Q(const contract&)` returning `i64{1} << c.precision`; constructor validation throwing `std::runtime_error("intbirth: bad precision")` for `precision < 9 || precision > 64 || precision != 9` (only 9 accepted until Task 5 widens it).

- [ ] **Step 1: Write the failing test**

```cpp
// tests/nn/test_intbirth_precision.cpp
#include <gtest/gtest.h>
#include <ax/nn/intbirth.hpp>

TEST(IntbirthPrecision, DefaultIsShippedGrain) {
  ax::nn::ib::contract c;
  EXPECT_EQ(c.precision, 9);
  EXPECT_EQ(ax::nn::ib::contract_Q(c), 512);
}

TEST(IntbirthPrecision, UnsupportedPrecisionThrows) {
  ax::nn::ib::contract c;
  c.precision = 32;  // not yet wired; must refuse loudly, not misbehave
  EXPECT_THROW(ax::nn::ib::block("", c), std::runtime_error);
}
```

Register it in `tests/nn/CMakeLists.txt` next to the existing intbirth test target (same pattern as its neighbors).

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build && ctest --test-dir build -R IntbirthPrecision --output-on-failure`
Expected: FAIL — `precision`/`contract_Q` undeclared.

- [ ] **Step 3: Implement**

In `include/ax/nn/intbirth.hpp`, add to `struct contract` (after `lrn/lrd`):

```cpp
  int precision = 9;          // operand grain Q_p = 2^precision
                              // (ENGINE-EXACT-1 ladder; 9 = shipped)
```

and a free function next to the struct:

```cpp
inline i64 contract_Q(const contract& c) { return i64{1} << c.precision; }
```

In `src/nn/intbirth.cpp`: keep `constexpr i64 Q = 512;` as the *shipped-grain* constant (it is now the frozen-grain reference, still needed by the convention pin), but in `block::block(...)` and `full_birth`/`multi_birth`/`moe_birth` constructors add:

```cpp
  if (c.precision != 9)
    throw std::runtime_error("intbirth: bad precision");
```

(Refuse-if-disagree doctrine: an unwired precision aborts before a step runs.)

- [ ] **Step 4: Run the full gate**

Run: `ctest --test-dir build --output-on-failure` then the Task-1 digest drivers.
Expected: all green — this task must be digest-invisible.

- [ ] **Step 5: Commit**

```bash
git add include/ax/nn/intbirth.hpp src/nn/intbirth.cpp tests/nn/test_intbirth_precision.cpp tests/nn/CMakeLists.txt
git commit -m "feat: contract.precision field, Q9-only with loud refusal (ENGINE-EXACT-1 task 2)"
```

---

### Task 3: Core extraction — gemm/rdiv become `<Op, Acc, Round>` templates

**Files:**
- Create: `include/ax/nn/intbirth_core.hpp`
- Modify: `src/nn/intbirth.cpp:233-268` (int_gemm / int_gemm_nt / int_gemm_xty / rdiv_inplace become thin wrappers)
- Test: `tests/nn/test_intbirth_core.cpp` (new)

**Interfaces:**
- Produces (namespace `ax::nn::ib::core`):

```cpp
template <class Op> struct RoundHalfAway {
  // round-half-away division at grain d; identical to shipped rdiv for Op=i64
  static Op div(Op x, Op d);
};
struct Exact;  // tag; specializations divide exactly (anchor only, Task 8)

template <class Op, class Acc, class Round = RoundHalfAway<Op>>
std::vector<Op> gemm(const std::vector<Op>& a, int rows, int K,
                     const std::vector<Op>& w, int N);        // a @ w^T, Acc sum
// gemm_nt, gemm_xty: same shape conventions as the shipped free functions
template <class Op, class Round>
void rdiv_inplace(std::vector<Op>& m, Op d);
```

- Consumes: nothing new; bodies are verbatim moves of `src/nn/intbirth.cpp:233-268` with `i64`→`Op`, sum variable →`Acc`.

- [ ] **Step 1: Write the failing test**

```cpp
// tests/nn/test_intbirth_core.cpp
#include <gtest/gtest.h>
#include <ax/nn/intbirth.hpp>
#include <ax/nn/intbirth_core.hpp>
#include <random>

using namespace ax::nn::ib;

TEST(IntbirthCore, I64InstantiationMatchesShippedGemm) {
  std::mt19937_64 rng(613);
  std::uniform_int_distribution<i64> d(-1000, 1000);
  Mat a(6 * 8), w(4 * 8);
  for (auto& v : a) v = d(rng);
  for (auto& v : w) v = d(rng);
  EXPECT_EQ((core::gemm<i64, i64>(a, 6, 8, w, 4)), int_gemm(a, 6, 8, w, 4));
}

TEST(IntbirthCore, RoundHalfAwayMatchesShippedRdiv) {
  Mat m{7, -7, 5, -5, 512, -513}, m2 = m;
  rdiv_inplace(m, 2);                          // shipped
  core::rdiv_inplace<i64, core::RoundHalfAway<i64>>(m2, 2);
  EXPECT_EQ(m, m2);
}

TEST(IntbirthCore, WideAccDoesNotOverflow) {
  // K=2 dot with operands near 2^40: i64 acc would overflow, i128 must not.
  std::vector<i64> a{i64{1} << 40, i64{1} << 40}, w{i64{1} << 22, -(i64{1} << 22)};
  auto r = core::gemm<i64, __int128>(a, 1, 2, w, 1);
  EXPECT_EQ(r[0], 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build && ctest --test-dir build -R IntbirthCore --output-on-failure`
Expected: FAIL — header missing.

- [ ] **Step 3: Implement `intbirth_core.hpp`**

Move the four function bodies verbatim, templated. `RoundHalfAway<Op>::div` is the shipped `rdiv` (src/nn/intbirth.cpp:24) generalized:

```cpp
template <class Op> struct RoundHalfAway {
  static Op div(Op x, Op d) {   // d > 0; round half away from zero
    const Op q = x / d, r = x % d;
    if (r * 2 >= d) return q + 1;
    if (-r * 2 >= d) return q - 1;
    return q;
  }
};
```

(Copy the SHIPPED body exactly — if it differs from the sketch above, the shipped body wins; the test in Step 1 is the arbiter.) In `gemm` forms the inner sum accumulates in `Acc` and narrows back to `Op` at store — with a `static_assert(sizeof(Acc) >= sizeof(Op))` and, for `Acc != Op`, a range check that throws on narrowing overflow (loud, not wrapped).

Rewrite the four shipped free functions as one-line wrappers calling the core instantiation `<i64, i64>`.

- [ ] **Step 4: Run the full gate**

Run: `ctest --test-dir build --output-on-failure` + Task-1 digest drivers.
Expected: all green (wrappers must be digest-invisible).

- [ ] **Step 5: Commit**

```bash
git add include/ax/nn/intbirth_core.hpp src/nn/intbirth.cpp tests/nn/test_intbirth_core.cpp tests/nn/CMakeLists.txt
git commit -m "refactor: gemm/rdiv extracted to <Op,Acc,Round> core, i64 wrappers digest-identical (ENGINE-EXACT-1 task 3)"
```

---

### Task 4: Core extraction — block forward/backward + adamw

**Files:**
- Modify: `include/ax/nn/intbirth_core.hpp` (grow)
- Modify: `src/nn/intbirth.cpp` (block::*, adamw::step bodies delegate to core)
- Test: existing suite is the test (verbatim-move task; Task 3's pattern)

**Interfaces:**
- Produces: `core::block_impl<Op, Acc, Round>` carrying the moved bodies of `softmax_rows`, `rms_fwd`, `rms_bwd`, `attn_fwd/bwd`, `ffn_fwd/bwd`, `moe_body_fwd/bwd`, `body_fwd/bwd`, `fwd/bwd`, and `core::adamw_impl<Op, Acc, Round>` (moved `adamw::step`, src/nn/intbirth.cpp:866+; BigV bias-correction path stays i64/bigint as shipped — it is scale-free). The public `block`/`adamw` classes keep their exact signatures and internally hold the `<i64, i64>` instantiation.
- Frozen-grain seam: everywhere a moved body consults a table or isqrt (`softmax_rows` exp table, `rms_fwd`/`rms_bwd` `isqrt_newton`+R16, RoPE tables+RS, attn `scale_`, adamw `isqrt_newton(vh * Q)`), route the *input* through `Round::to_grain(x, gshift)` and the *result* through `Round::from_grain(x, gshift)` where `gshift = c.precision - 9`. Add both as static members of `RoundHalfAway` (`x >> gshift` floor-truncate resp. `x << gshift`; no-ops when `gshift == 0`).

Steps (same cycle as Task 3, one module at a time — commit after EACH green module, in this order):

- [ ] **Step 1: Move softmax_rows + rms_fwd/rms_bwd; gate; commit** (`refactor: softmax/rms into core (task 4a)`)
- [ ] **Step 2: Move attn_fwd/attn_bwd (incl. RoPE site); gate; commit** (`task 4b`)
- [ ] **Step 3: Move ffn + moe bodies; gate; commit** (`task 4c`)
- [ ] **Step 4: Move body_/fwd/bwd composition + adamw::step; gate; commit** (`task 4d`)

Gate after every step = full ctest + all four digest drivers (Task 1 commands). Any digest mismatch: stop, bisect the move — a verbatim move CANNOT change digests; a mismatch means a transcription error, not an acceptable drift.

---

### Task 5: Q32 rung (i64 operands, `__int128` accumulation)

**Files:**
- Modify: `src/nn/intbirth.cpp` (constructor accepts `precision == 9 || precision == 32`; dispatch to `<i64, __int128>` instantiation with grain plumbed)
- Modify: `bindings/` intbirth pybind module (contract dict gains optional `"precision"` key, default 9 — follow the existing contract-parsing pattern in the module source)
- Test: `tests/nn/test_intbirth_ladder.cpp` (new)

**Interfaces:**
- Produces: runnable Q32 rung: `full_birth`/`multi_birth`/`moe_birth` with `c.precision = 32`.
- Scale derivation (spec §Contract changes) implemented in ONE function used by all loops:

```cpp
struct scales {                 // all derived from contract, nowhere else
  i64 Q;                        // 1 << precision
  int gshift;                   // precision - 9 (frozen-grain shift)
  i64 act_clamp;                // c.act_clamp << gshift
  i64 eps32;                    // c.eps32 << (2 * gshift)
};
inline scales derive(const contract& c) {
  return {i64{1} << c.precision, c.precision - 9,
          c.act_clamp << (c.precision - 9),
          c.eps32 << (2 * (c.precision - 9))};
}
```

`pq`, `gboost`, R16, RS stay put (frozen carries, spec pin).

- [ ] **Step 1: Write the failing tests**

```cpp
// tests/nn/test_intbirth_ladder.cpp
#include <gtest/gtest.h>
#include <ax/nn/intbirth.hpp>
// build_tiny_tables()/tiny_weights(): construct the minimal shipped-format
// tables/weights blob the same way tests/nn's existing intbirth test
// fixture does — reuse that fixture (include or lift it into a shared
// header tests/nn/intbirth_fixture.hpp as part of this step).

TEST(IntbirthLadder, Q32AcceptedAndRuns) {
  ax::nn::ib::contract c;  c.precision = 32;
  // tiny dims: T=4, D=8, DH=4, F=16, V=8 via the fixture
  auto fb = make_tiny_full_birth(c);   // fixture helper
  EXPECT_NO_THROW(fb.run(3));
}

TEST(IntbirthLadder, Q32GrainZeroInputsMatchQ9) {
  // With all-zero weights/inputs, every rounding site sees 0 at every
  // grain: the Q32 and Q9 logits must be bit-equal after de-shifting.
  auto l9  = tiny_logits_after_fwd(/*precision=*/9,  /*zero=*/true);
  auto l32 = tiny_logits_after_fwd(/*precision=*/32, /*zero=*/true);
  for (size_t i = 0; i < l9.size(); i++)
    EXPECT_EQ(l9[i], l32[i] >> 23);   // 23 = 32 - 9
}
```

- [ ] **Step 2: Run to verify it fails** — Q32 currently throws (Task 2's refusal).

- [ ] **Step 3: Implement** — widen the constructor acceptance, plumb `derive(c)` through the core instantiations, dispatch `<i64, __int128>` when `precision == 32`. pybind: read `"precision"` from the contract dict with default 9.

- [ ] **Step 4: Full gate + new tests green.** Q9 digests unchanged (drivers), ladder tests pass.

- [ ] **Step 5: Emit and book the Q32 reference digest**

Run the tiny fixture 40 steps at Q32, print the running weight sha256. Record it IN THE TEST as `EXPECT_EQ(sha, "<value>")` (self-reference until llmopt counter-books; note in commit message that the counterparty booking is pending).

- [ ] **Step 6: Commit** (`feat: Q32 rung - i128 accumulation, frozen-grain sites, pinned reference digest (ENGINE-EXACT-1 task 5)`)

---

### Task 6: `i256` fixed-width helper

**Files:**
- Create: `include/ax/core/int256.hpp`
- Test: `tests/core/test_int256.cpp` (new)

**Interfaces:**
- Produces: `ax::core::i256` — two's-complement 256-bit: construct from `i64`/`__int128`; `+ - *` (i128×i128→i256 full multiply), `/ %` by `__int128` divisor, `<< >>`, comparisons, `to_i128()` (throws on overflow). No division by i256 needed (rdiv divisors are ≤ 2^127).

- [ ] **Step 1: Write the failing test** — property tests against `ax::core::bigint` as reference oracle (the codebase's cross-check doctrine):

```cpp
#include <gtest/gtest.h>
#include <ax/core/int256.hpp>
#include <ax/core/bigint.hpp>   // adjust include to the actual bigint header
#include <random>

TEST(Int256, MulDivRoundtripVsBigint) {
  std::mt19937_64 rng(613);
  for (int it = 0; it < 2000; it++) {
    __int128 a = (__int128(rng()) << 64) | rng();
    __int128 b = (__int128(rng()) << 40) | rng();
    if (rng() & 1) a = -a;
    if (!b) continue;
    ax::core::i256 p = ax::core::i256::mul(a, b);
    // reference via bigint string round-trip
    EXPECT_EQ(p.to_string(), (ax::core::bigint(i128_to_string(a)) *
                              ax::core::bigint(i128_to_string(b))).to_string());
    EXPECT_EQ((p / b).to_i128(), a);
  }
}
```

(Adapt bigint's actual construction/printing API on contact — the test's *shape* is the contract: 2000 random cross-checks vs bigint.)

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement** (four u64 limbs, schoolbook mul, shift-subtract division by i128).
- [ ] **Step 4: Green + full gate.**
- [ ] **Step 5: Commit** (`feat: i256 fixed-width int, bigint-cross-checked (ENGINE-EXACT-1 task 6)`)

---

### Task 7: Q64 rung (`__int128` operands, i256 accumulation)

**Files:**
- Modify: `include/ax/nn/intbirth_core.hpp` (Op=`__int128` instantiation needs `RoundHalfAway<__int128>` using i256 intermediates at product-then-divide sites)
- Modify: `src/nn/intbirth.cpp` (accept `precision == 64`, dispatch `<__int128, ax::core::i256>`)
- Modify: pybind module (Q64 weights cross the boundary as int64 pairs (hi, lo) — document in the module docstring; drivers only need digests + milestone i64 readouts, so the wide type mostly stays engine-side)
- Test: extend `tests/nn/test_intbirth_ladder.cpp`

- [ ] **Step 1: Failing tests** — same pattern as Task 5: `Q64AcceptedAndRuns`, `Q64GrainZeroInputsMatchQ9` (de-shift by 55), plus `Q64ProductSiteNoOverflow`: drive one rmsnorm row with operands near max Q64 magnitude (`(i64 value) << 55`) and assert no throw and sign-correct output.
- [ ] **Step 2: Verify fail.** — Q64 throws (refusal).
- [ ] **Step 3: Implement.** Every `a * b` on the Op path where both factors carry operand scale routes through `i256::mul` then `Round::div`.
- [ ] **Step 4: Full gate + ladder tests green; emit + pin the Q64 tiny-fixture reference digest (as Task 5 Step 5).**
- [ ] **Step 5: Commit** (`feat: Q64 rung on i128/i256 (ENGINE-EXACT-1 task 7)`)

---

### Task 8: Dyadic anchor scalar + `Exact` rounding policy

**Files:**
- Create: `include/ax/nn/dyadic.hpp`
- Modify: `include/ax/nn/intbirth_core.hpp` (`Exact` policy: `div` = exact dyadic division — divisor is always a power of two times a small integer at these sites; where the divisor is NOT dyadic (softmax z, adamw den, rms isq), `Exact::div` performs true rational division via the dyadic type's `div_exact`, which requires the Dyadic scalar and is `static_assert`-blocked for integer Ops)
- Test: `tests/nn/test_dyadic_anchor.cpp`

**Interfaces:**
- Produces:

```cpp
// ax::nn::dyadic — exact scalar for the anchor. value = num * 2^exp
struct Dyadic {
  ax::core::bigint num;   // arbitrary precision
  int64_t exp;            // power-of-two exponent (negative = fractional)
  static inline int64_t bit_ceiling = int64_t{1} << 22;  // loud abort guard
  Dyadic operator+(const Dyadic&) const;   // align exps, add; throws
  Dyadic operator-(const Dyadic&) const;   //   std::runtime_error("anchor: bit ceiling")
  Dyadic operator*(const Dyadic&) const;   //   if num bits exceed bit_ceiling
  // NOTE: no operator/ — division is not closed over dyadics.
  // Exact division sites go through rational pairs:
  static Dyadic div_exact_by_pow2(const Dyadic& a, int64_t k);  // exp -= k
};
```

Non-dyadic exact division (softmax z etc.): the anchor carries those
values as `DyadicRatio {Dyadic n, d;}` locally within each site and
clears the ratio at the next frozen-grain seam (where truncation to the
shipped grain is DECLARED by the convention pin — so the ratio never
propagates and dyadic closure is preserved between sites). This is the
load-bearing design point: the frozen-grain seams are exactly where the
exact arm is allowed to quantize, because the convention pin makes that
quantization part of the *function definition*, not rounding.

- [ ] **Step 1: Failing unit tests for Dyadic** (add/mul exactness vs Fraction-style hand values; bit-ceiling throw; `div_exact_by_pow2`).
- [ ] **Step 2: Verify fail. Implement dyadic.hpp. Green. Commit** (`feat: dyadic exact scalar (task 8a)`).
- [ ] **Step 3: Failing anchor test:** tiny fixture, `<Dyadic, Dyadic, Exact>` instantiation, 2 steps: (a) runs without throw; (b) with `bit_ceiling = 64` it throws the declared error (guard is observable); (c) SANITY: at gshift=0 with the Exact policy replaced by RoundHalfAway-at-grain… must equal the Q9 engine bit-for-bit on 1 step (policy seam verified).
- [ ] **Step 4: Implement `Exact` + the core's DyadicRatio site-local handling. Green + full gate. Commit** (`feat: exact-prefix anchor instantiation (task 8b)`).

---

### Task 9: Anchor driver + divergence readout

**Files:**
- Create: `tools/exact_anchor/run_anchor.py` (pybind driver; anchor exposed via a `run_anchor(contract_dict, tables_bytes, init_bytes, steps)` binding added to the intbirth module returning per-step weight sha256 list + final weights at rung-9 truncation)
- Create: `tools/exact_anchor/divergence.py`
- Test: `tools/exact_anchor/test_anchor_driver.py` (pytest, runs under python3.12 like the other drivers)

**Interfaces:**
- Produces: `divergence.py rungs.jsonl` → per-step table: for each pair in {anchor, Q9, Q32, Q64}, mean/max abs weight delta at the common (shipped) grain + first-divergence step. Output one JSONL row per step: `{"step": n, "pairs": {"anchor-q9": {"mean": ..., "max": ...}, ...}}`.

- [ ] **Step 1: Failing pytest** — run tiny fixture 4 steps at Q9 + anchor, assert the driver emits 4 rows, monotone step field, anchor-q9 delta 0 at step 0 (shared init).
- [ ] **Step 2: Implement binding + drivers. Green.**
- [ ] **Step 3: d64-class anchor probe:** contract T=32 D=64 (shipped anchor dims), run the anchor until bit-ceiling abort or 12 steps, book actual horizon + wall-clock in the commit message (this number goes to llmopt).
- [ ] **Step 4: Full gate. Commit** (`feat: anchor driver + divergence readout, d64 horizon measured (ENGINE-EXACT-1 task 9)`).

---

### Task 10: Docs, relay, push

- [ ] **Step 1:** README `ax::nn` module row: add one clause "precision-ladder rungs (Q9/Q32/Q64) and a dyadic exact-prefix anchor (ENGINE-EXACT-1)". Status section: one sentence with pointer to the spec.
- [ ] **Step 2:** Write `docs/relay/2026-08-0X-engine-exact-build.md`: rung digests (Q32/Q64 pinned values), anchor horizon + wall-clock, ask llmopt to counter-book the two rung digests, restate that comparison runs wait for their ladder-limit pre-reg.
- [ ] **Step 3:** Full gate one last time; `git push origin main`.

## Self-review notes

- Spec coverage: convention pin → Task 4 seam + Task 8 DyadicRatio paragraph; contract changes → Tasks 2/5; widths → Tasks 3/5/6/7; anchor → Tasks 8/9; deliverable 1 (law-leg booking) is llmopt-side per the GO and not a code task; deliverable 4's capability race is llmopt-side (non-goal here).
- The Q32/Q64 "GrainZeroInputsMatchQ9" tests are deliberately weak (zero case); the real gates are the pinned rung digests (Tasks 5/7) counter-booked by llmopt.
- Verbatim-move discipline (Tasks 3/4): shipped body always wins over any sketch in this plan; digests are the arbiter.
