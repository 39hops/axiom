# FP32LIMB R1 CPU Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** C++ CPU oracle for fp32-limb exact GEMM (relay 2026-08-10-10, PRE-REG FP32LIMB-METAL rung R1): fp32 slice arithmetic verified bit-exact against an `ax::bigint` integer reference, with a runtime envelope fence that survives release builds.

**Architecture:** Port the house's `two_sum`/`slices`/`aligned_partials` (llmopt `scratch/ozaki_rung2bc.py`) to C++ as library code in `ax::la::fp32limb`. Per 32-wide K-block: align by max exponent, slice mantissas into s=7-bit signed slices (cap `MAX_SLICES=8`, loud reject if residual survives), do slice-pair dot products **in fp32** (each product and each of the ≤32 adds provably exact), recombine into an exact `(bigint, exp)` dyadic accumulator. Reference = pure bigint GEMM from frexp-decoded mantissas. DO NOT port `dd_chain` (its carrier term is `* 0`, an unmeasured placeholder).

**Tech Stack:** C++23, ax::bigint, GoogleTest via existing `ax_tests` binary, CMake/Ninja.

## Global Constraints

- Registered constants: slice width `s = 7` bits (house file's signature default is 8 — REGISTERED value is 7), block `b = 32`, constraint `2s + log2(b) <= 24` (19 ≤ 24) checked with `static_assert`.
- Envelope/exactness guards must be `throw std::runtime_error` (or equivalent), never `assert` — must survive `NDEBUG`/Release.
- Reference comparisons are integer/bigint equality only — never fp-vs-fp.
- No fast-math anywhere (repo build already has none; do not add flags).
- Library code: `include/ax/la/fp32limb.hpp` + `src/la/fp32limb.cpp` (add to `axiom` source list in root CMakeLists). Tests: `tests/la/fp32limb_test.cpp` (add to `ax_tests` list in tests/CMakeLists.txt). Driver: `tools/fp32limb/r1_oracle.cpp` (NOT in CMake; self-documented build line in header comment, per `tools/exact_anchor/run_anchor2.cpp` convention).
- Build/test: `cmake --build build && ctest --test-dir build --output-on-failure`. Also verify guards in Release: `build-rel`.
- No Metal, no GPU, one CPU worker. Commit messages: no session links, `Co-Authored-By: Claude <noreply@anthropic.com>`.

---

### Task 1: Exact fp32 decode + bigint reference GEMM

**Files:**
- Create: `include/ax/la/fp32limb.hpp`, `src/la/fp32limb.cpp`
- Modify: root `CMakeLists.txt` (add `src/la/fp32limb.cpp`), `tests/CMakeLists.txt` (add test file)
- Test: `tests/la/fp32limb_test.cpp`

**Interfaces (Produces):**

```cpp
namespace ax::la::fp32limb {
struct dyadic { bigint m; int e; };            // value = m * 2^e
dyadic decode(float x);                         // exact; throws on non-finite
// accumulate: r += m2 * 2^e2  (aligns exponents exactly)
void acc(dyadic& r, const bigint& m2, int e2);
bool dyadic_eq(const dyadic& a, const dyadic& b);   // exact value equality
struct matf { int rows, cols; std::vector<float> v;
              float& at(int i,int j); float at(int i,int j) const; };
std::vector<dyadic> gemm_ref(const matf& A, const matf& B);  // bigint GEMM, row-major
}
```

`decode`: `std::frexp(x,&e)` → mantissa `f in [0.5,1)`; `m = llround(ldexp(f,24))`, result `{bigint(m), e-24}` (24 not 53: fp32 mantissa fits; use double frexp on the float value — exact). `dyadic_eq`: align to min exponent via `<<`, compare bigint `==` (handles m=0/e differences). `gemm_ref`: per entry, `acc` over k of `m_a*m_b` at `e_a+e_b`.

- [ ] Step 1: write failing tests: `decode(1.5f)=={3,-1}`-valued (via dyadic_eq against hand-built), decode of a denormal (`std::ldexp(1.0f,-140)`), decode(-0.75f), gemm_ref on a 2x2 integer matrix equals hand-computed bigints, gemm_ref throws on inf/nan input.
- [ ] Step 2: build, run `ctest --test-dir build -R Fp32Limb --output-on-failure`, verify FAIL (link error → stub first, then assert-fail).
- [ ] Step 3: implement; run same command, verify PASS.
- [ ] Step 4: full suite `ctest --test-dir build --output-on-failure` green; commit `feat(fp32limb): exact fp32 decode + bigint reference GEMM (R1 task 1)`.

### Task 2: EFT primitives + expansion recombination (rider: triple-double exit)

**Files:** same header/src/test.

**Interfaces (Produces):**

```cpp
struct f2 { float s, r; };
f2 two_sum(float a, float b);                       // Knuth, branch-free
// exact expansion add, verbatim port of house exp_add (drops zeros)
std::vector<float> exp_add(std::vector<float> e, float x);
// rider: cap expansion length; throws std::runtime_error on overflow past cap
std::vector<float> exp_add_capped(std::vector<float> e, float x, std::size_t cap);
bigint exp_to_bigint_scaled(const std::vector<float>& e, int shift); // Σ round(c*2^shift), each term exact or throw
```

two_sum verbatim from house: `s=a+b; bb=s-a; r=(a-(s-bb))+(b-bb)`. Mark inputs/outputs `volatile`-free but compile-time-fence not needed (no fast-math in build).

- [ ] Step 1: failing tests: `two_sum` exactness checked via bigint (`decode(a)+decode(b) == decode(s)+decode(r)` under dyadic acc) over 10k seeded random float pairs (use `ax::st::rng`); `exp_add` of 1000 random floats sums to exact bigint total; `exp_add_capped(..., 3)` succeeds on values within triple-double range and **throws** on an adversarial sequence needing length 4 (e.g. {2^80, 2^40, 1.0, 2^-40}).
- [ ] Step 2: verify fail → implement → verify pass (same ctest -R).
- [ ] Step 3: commit `feat(fp32limb): two_sum + expansion recombine, len-3 capped exit (R1 task 2)`.

### Task 3: Alignment + slicing + envelope fence

**Interfaces (Produces):**

```cpp
inline constexpr int SLICE_W = 7;      // registered s=7 (house default 8 — do not use)
inline constexpr int BLOCK   = 32;
inline constexpr int MAX_SLICES = 8;   // fixed cap; residual past this = envelope reject
static_assert(2*SLICE_W + 5 /*log2 BLOCK*/ <= 24);
// scale block column-window by 2^-e_align (max exponent), slice into signed
// 7-bit integer slices stored as float. Throws std::runtime_error("fp32limb: envelope")
// if residual nonzero after MAX_SLICES.
struct sliced { std::vector<std::vector<float>> sl; int e_align; };
sliced slice_row(const float* x, int n);   // one row (or col) segment, n<=BLOCK
```

Port of house `slices()`: `R = x*2^-e_align`; loop `Q = nearbyint(R*2^s); R = R*2^s - Q` in **double** (exact: fp32 payload ≤ 24+spread bits, double holds it), slices stored as float (each |Q| ≤ 2^(s-1), exact). e_align from `frexp(max|x|)`; all-zero segment → one zero slice.

- [ ] Step 1: failing tests: (a) reconstruction — Σ sl[i]·2^(−s(i+1)+e_align) as dyadic equals Σ decode of inputs elementwise (test per element with n=1 windows and with full windows via a reconstruct helper in the test); (b) uniform-exponent inputs need ≤ 4 slices; (c) engineered spread (element pair {1.0f, ldexp(1.0f, -40)} in one window) **throws** the envelope error; (d) the throw fires in a Release build too — assert via building `build-rel` and running `ctest --test-dir build-rel -R Fp32Limb`.
- [ ] Step 2: fail → implement → pass; commit `feat(fp32limb): alignment + s=7 slicing with release-safe envelope fence (R1 task 3)`.

### Task 4: fp32-limb GEMM oracle — P-ENVELOPE-EXACT

**Interfaces (Produces):**

```cpp
// fp32-limb GEMM: slice-pair dots computed IN fp32 (simulating the kernel),
// recombined exactly. Every fp32 add is guarded: |partial| < 2^24 or throw.
std::vector<dyadic> gemm_fp32limb(const matf& A, const matf& B);
```

Per (i,j,block): `pa = slice_row(A row segment)`, `pb = slice_row(B col segment)`; for each slice pair (p,q): `float acc=0; for k: acc += pa.sl[p][k]*pb.sl[q][k];` with a guard `if (std::fabs(acc) >= 16777216.0f) throw` after each add (release-safe fence, per the sloppiest-link contract); recombine `acc` (an exact integer ≤ 2^19 by construction) into the dyadic accumulator at exponent `-s(p+1)-s(q+1)+e_align_a+e_align_b`.

- [ ] Step 1: failing test P-ENVELOPE-EXACT: for sizes {8, 33 (straddles one block), 64} and 5 seeds, `gemm_fp32limb(A,B)` entry-wise `dyadic_eq` `gemm_ref(A,B)`, inputs `uniform(-1,1)` floats and `normal()*0.05` floats (house's class).
- [ ] Step 2: fail → implement → pass.
- [ ] Step 3: commit `feat(fp32limb): fp32-limb GEMM oracle, P-ENVELOPE-EXACT vs bigint (R1 task 4)`.

### Task 5: Registered input classes — exponent-spread + K-permutation

- [ ] Step 1: failing tests:
  - **ExponentSpreadInsideBlock:** matrices where each 32-window mixes magnitudes `2^0 … 2^-25` (inside envelope: 24+25 ≤ 56 bits, 8 slices × 7 = 56 ✓) → exact equality with `gemm_ref`; and a variant with spread 40 → envelope throw (loud reject, both build types).
  - **KPermutation:** permute the K axis of A's columns and B's rows with the same seeded permutation; `gemm_fp32limb` output must be **bit-identical**: serialize each dyadic (normalize by stripping trailing zero bits of m, then compare `m ==` and `e ==`) AND compare a byte-digest of the recombined values. Run 3 permutations × 3 seeds, N=64.
  - **DenormalClass:** inputs scaled near `FLT_MIN` (`ldexp(uniform, -120)`) → still exact (CPU has no FTZ by default; document in the per-link table that the Metal port must re-verify FTZ).
- [ ] Step 2: fail → implement (likely zero impl changes; fixes if not) → pass; full suite green in Debug AND Release.
- [ ] Step 3: commit `test(fp32limb): exponent-spread + K-permutation bit-identity classes (R1 task 5)`.

### Task 6: Depth-L chain rider + receipt driver + per-link table

**Files:** Create `tools/fp32limb/r1_oracle.cpp`, `docs/specs/2026-08-10-fp32limb-r1-links.md`.

- [ ] Step 1: driver (header documents build line `c++ -std=c++23 -O2 -Iinclude tools/fp32limb/r1_oracle.cpp build-rel/libaxiom.a -o /tmp/r1_oracle`): runs every registered class at N=128, plus a depth-L chain harness (L=6 linear chain, `gemm_fp32limb` per layer carrying exact dyadic → re-encode is NOT exact, so the chain carries the bigint result forward and verifies each layer against `gemm_ref` of that layer — no dd_chain port). Emits JSONL receipts: `{"class":..., "n":..., "seed":..., "pass":true, "max_int_dev":"0", "slices_max":..., "digest":"<sha256 of serialized output>"}` one line per run.
- [ ] Step 2: run driver, all-pass JSONL captured into `tools/fp32limb/receipts/` (commit the JSONL).
- [ ] Step 3: write per-link table doc (align → split → product → local sum → block carry → recombine; per link: widest value, why it cannot round, which guard fences it, incl. FTZ note and the 2^24 guard from Task 4).
- [ ] Step 4: commit `feat(fp32limb): R1 receipt driver + depth-chain rider + per-link table (R1 task 6)`.

### Task 7: Relay + close

- [ ] Step 1: write `docs/relay/2026-08-10-11-fp32limb-r1-receipt.md` (`# axiom → llmopt: ...`): R1 built and green, P-ENVELOPE-EXACT fired, constants used (s=7 registered, not the file default 8), dd_chain not ported, envelope fence release-verified, K-permutation bit-identity receipts, RNS promotion NOT triggered (restating relay), R2/R3 remain HOLD pending crown window + Artin GO. Include commit SHAs and the receipts digest.
- [ ] Step 2: full suite green both build types; commit `docs: relay - FP32LIMB R1 receipt`; push.

## Self-Review

Spec coverage: envelope-exact ✓(T4), loud reject + release-safe ✓(T3/T4/T5), exponent-spread class ✓(T5), K-permutation ✓(T5), bigint ground truth ✓(T1), s=7 not 8 ✓(T3), no dd_chain ✓(T6 chain design), triple-double rider ✓(T2), depth-L chain rider ✓(T6), per-link table ✓(T6), no RNS promotion ✓(T7), fences (CPU only, no GPU) ✓, receipts for counter-book ✓(T6/T7). Types consistent: `dyadic`/`matf`/`sliced` defined T1/T3, consumed T4-6.
