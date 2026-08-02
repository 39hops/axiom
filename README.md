# axiom

A from-scratch, STL-only C++23 mathematics and statistics library: exact
arithmetic, linear algebra, statistics, numerical methods, a symbolic (CAS)
engine, exact fixed-point neural-network inference and integer training, and
(planned) a small proof kernel. No Eigen, no GMP, no Boost, no BLAS, no torch —
every algorithm is implemented in-house on top of the standard library.
GoogleTest is the only dependency, and it is test-only (fetched by CMake).

Two things follow from that rule and shape everything below. Results are
**exact or explicitly undecided**, never silently approximate — see [Soundness
contract](#soundness-contract). And where axiom computes in integers, its
results are **bit-identical across platforms by construction** rather than by
tuning — see [Exact NN and deterministic birth](#exact-nn-and-deterministic-birth).

## Modules

| Namespace  | Contents |
|------------|----------|
| `ax::core` | Arbitrary-precision `bigint` (Karatsuba multiplication), exact `rational`, number theory (Miller-Rabin, Pollard rho, modular arithmetic, CRT), FFT/NTT |
| `ax::par`  | Thread pool, `parallel_for` / `parallel_reduce` |
| `ax::la`   | Dense `mat`/`vec`, cache-blocked multithreaded matmul, LU, Cholesky, Householder QR, least squares |
| `ax::st`   | PCG64 RNG (ziggurat normal), special functions (lgamma, erf, incomplete gamma/beta), 15 probability distributions with pdf/cdf/quantile/sample, descriptive statistics, hypothesis tests (t, chi-square, ANOVA, KS), OLS with inference, GLMs (logistic/Poisson via IRLS), Metropolis-Hastings MCMC, time series (ACF/PACF, AR/ARMA, periodogram) |
| `ax::num`  | Adaptive Gauss-Kronrod and tanh-sinh quadrature, RK45 (Dormand-Prince) ODE solver, Brent root finding, Newton, optimization (Brent 1-d, BFGS, Nelder-Mead) |
| `ax::sym`  | CAS: immutable hash-consed expression DAG, canonicalizing simplifier, symbolic differentiation, integration (table, u-substitution, parts, partial fractions), equation solving (exact polynomial roots through quartic, Durand-Kerner numeric fallback, symbolic linear systems), univariate polynomial algebra over exact rationals, expansion, text/LaTeX printers, sympy-sstr parser, verification oracle (canonical forms, three-valued equivalence) |
| `ax::nn`   | Exact NN: AXNN container format (fp32 and fixed-point), reference fp32 inference (`model`), integer-only fixed-point inference (`exact_model`, Q.16 operands / int64 accumulation / container-shipped transcendental tables), and the `intbirth` integer-training engine (`block`, `adamw`, `full_birth`, `multi_birth`, `moe_birth`) with a pybind11 bridge |

See [`docs/specs/2026-07-05-axiom-design.md`](docs/specs/2026-07-05-axiom-design.md)
for the full design and roadmap (hypothesis tests, GLM, MCMC, time series,
symbolic integration/solvers, proof kernel, optional CUDA backend).

## Exact NN and deterministic birth

`ax::nn` exists because floating-point neural networks are not reproducible:
change the reduction order, the device, or the BLAS kernel and the bits move.
axiom's answer is to leave floating point entirely. Integer addition is
associative and exact, so a reduction's *order* cannot change its *value* —
bit-identity across platforms becomes a property of the arithmetic rather than
a testing goal.

**AXNN container.** A self-describing little-endian model format (`"AXNN"`
magic, flat-JSON config, then named tensors). Every architectural convention is
*declared*, never assumed — norm, activation, positional scheme, tied vs.
separate head, dense/SwiGLU/MoE FFN. Three minor versions ship: v1 (dense
fp32), v1.1 (SwiGLU, fused QKV), v1.2 (top-1 switch-MoE scalar gate). Spec:
[`docs/specs/2026-07-27-axnn-exact.md`](docs/specs/2026-07-27-axnn-exact.md).

**Exact inference** (`ax::nn::exact_model`). Integer-only forward over a
fixed-point model: Q.16 operands, int64 accumulation, container-shipped tables
for every transcendental (activation, exp, rsqrt, RoPE), declared floor/RNE
rounding at every site. No float touches the forward path after load. Logits
read out as Q.32 int64 and are never rescaled; greedy decode uses a KV cache
that is bit-exact with the uncached forward *by construction* (a position's
integer ops never depend on later positions, so caching changes cost, never
values).

**`intbirth` — integer training.** Not just inference: a complete training loop
in integers, so a training *trajectory* is a reproducible object. Layers, each
certified against the one below it:

| Type | Role |
|------|------|
| `int_gemm` / `int_gemm_nt` / `int_gemm_xty` | exact int64 sum-reduce GEMM forms; the caller rounds |
| `rdiv_inplace` | round-half-away division — the program-wide rounding primitive |
| `block` | forward/backward for one transformer block, integer softmax |
| `adamw` | IntAdamW with exact big-integer bias correction |
| `full_birth` | the composed single-block training loop |
| `multi_birth` | multi-block: embedding → *n* bodies → RMSNorm → tied head, with window cycling |
| `moe_birth` | mixture-of-experts bodies, top-1 routing, periodic gravity events |

Rounding *placement* is part of the contract, not an implementation detail:
every multi-GEMM sum rounds exactly once, after the sum. Scale constants are
carried explicitly in a `contract` rather than derived from `libm` (the
attention scale, for example, is computed with exact integer `isqrt`).
Trajectories are attested by a running SHA-256 over the weights at declared
milestones.

**Acceptance tooling** (`tools/int_adamw/`). Python drivers that exercise the
engine through its pybind11 module and compare digests against pinned
references — `verify_intbirth.py`, `verify_multiblock.py`,
`verify_primitives.py`, `verify_gravmoe.py`. Two rules are deliberate. *Digests
engine-side, comparison house-side*: the engine never grades its own homework.
*Refuse-if-disagree*: a mismatch in parameter order or an init hash aborts
before a single training step runs, rather than producing a plausible wrong
number. `verify_primitives.py` rebuilds the whole training loop in Python from
the exported primitives and checks it reproduces the composed loop's digests —
so the primitive layer is certified by the same references as the engine.

**Cross-lab role.** axiom is an *independent implementation*, in a different
language and runtime, of trajectories and decodes first produced elsewhere
(the llmopt program). Its value is precisely that it shares no code with what
it checks: agreement between the two is evidence about the arithmetic, not
about a shared bug. Determinism claims that rest on a single implementation are
worth much less than they look.

## Soundness contract

axiom's verification surface is **three-valued**, and the third value is the
point:

| Verdict | Emitted only when |
|---------|-------------------|
| `EQUIVALENT` | a structural proof establishes it |
| `NOT_EQUIVALENT` | a confirmed numeric witness disproves it |
| `UNDECIDED` | otherwise |

`UNDECIDED` is never a soft `EQUIVALENT`. It is the honest report that axiom
could not settle the question, and consumers must not read it as validation —
treating it as a pass is the failure mode this contract exists to prevent. The
oracle never guesses in either direction: a verdict is a claim axiom is
prepared to defend, and silence is preferable to a confident wrong answer.

The same doctrine governs the NN side, where it appears as *refuse-if-disagree*
— tooling aborts on a contract mismatch instead of reporting a number it cannot
stand behind.

## Example

```cpp
#include <ax/sym/expr.hpp>
#include <ax/sym/calc.hpp>
#include <ax/sym/print.hpp>

using namespace ax::sym;

auto x = expr::symbol("x");
auto f = x.pow(expr::num(3)) + expr::num(2) * x;   // x^3 + 2x
auto df = diff(f, x);                              // 3x^2 + 2
std::string s = to_string(df);                     // "3*x^2 + 2"
```

```cpp
#include <ax/st/dist.hpp>

ax::st::rng g{42};
ax::st::normal_dist n{0.0, 1.0};
double p = n.cdf(1.96);        // ≈ 0.975
double z = n.quantile(0.975);  // ≈ 1.96
double x = n.sample(g);
```

## Building

Requirements: a C++23 compiler, CMake ≥ 3.28, Ninja. Developed against MSVC
(VS 2026 Build Tools) on Windows; the code itself is standard C++23.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

On Windows with VS Build Tools, `scripts\build.cmd [test|rel]` wraps the above
(edit the tool paths at the top if your installation differs).

The Python bridge is optional and off by default. The core stays
dependency-free; pybind11 is bindings-only, the way GoogleTest is test-only:

```sh
cmake -S . -B build -G Ninja -DAXIOM_BUILD_PYTHON=ON \
  -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
cmake --build build            # -> axiom_sym, intbirth
```

With `build/` on `sys.path`, the acceptance drivers run against a reference
directory:

```sh
cd tools/int_adamw && python verify_intbirth.py ../../build
```

## Testing

TDD throughout — every feature was written test-first. Oracles favor
cross-checks over copied constants: symbolic derivatives are verified against
numeric central differences, quantiles against `quantile ∘ cdf ≈ id`,
decompositions by reconstruction, bigint arithmetic by ring axioms on random
inputs.

## Status

Phases 0–7 of the [design spec](docs/specs/2026-07-05-axiom-design.md) are
complete, plus Phase 8 — the [llmopt verification
oracle](docs/specs/2026-07-18-llmopt-oracle.md) (330 tests passing): a
sympy-`sstr` parser, `canonical()`/`equivalent()` primitives under the
[soundness contract](#soundness-contract) above, and the `axiom-oracle` JSONL
harness for the sympy parity audit (~11 ms/row Release on farm-shaped
`equiv_mod_const` rows). axiom becomes llmopt's oracle of record only after
that parity run passes.

`ax::nn` landed after that: AXNN fp32 inference, then FX-V1 exact fixed-point
inference, then the `intbirth` integer-training engine through its MoE stage.
Next up: the proof kernel (Phase 9) and the optional CUDA backend (Phase 10).
Implementation plans for every phase live in [`docs/plans/`](docs/plans/).

## Citing

axiom is cited by external verdicts that depend on reproducing its numbers, so
citations must name an exact commit — both this ledger and the ones citing it
are living. Machine-readable metadata is in [`CITATION.cff`](CITATION.cff);
replace the `commit:` field with the sha you actually ran.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
