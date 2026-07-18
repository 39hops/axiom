# axiom

A from-scratch, STL-only C++23 mathematics and statistics library: exact
arithmetic, linear algebra, statistics, numerical methods, a symbolic (CAS)
engine, and (planned) a small proof kernel. No Eigen, no GMP, no Boost — every
algorithm is implemented in-house on top of the standard library. GoogleTest is
the only dependency, and it is test-only (fetched by CMake).

## Modules

| Namespace  | Contents |
|------------|----------|
| `ax::core` | Arbitrary-precision `bigint` (Karatsuba multiplication), exact `rational`, number theory (Miller-Rabin, Pollard rho, modular arithmetic, CRT), FFT/NTT |
| `ax::par`  | Thread pool, `parallel_for` / `parallel_reduce` |
| `ax::la`   | Dense `mat`/`vec`, cache-blocked multithreaded matmul, LU, Cholesky, Householder QR, least squares |
| `ax::st`   | PCG64 RNG (ziggurat normal), special functions (lgamma, erf, incomplete gamma/beta), 15 probability distributions with pdf/cdf/quantile/sample, descriptive statistics, hypothesis tests (t, chi-square, ANOVA, KS), OLS with inference, GLMs (logistic/Poisson via IRLS), Metropolis-Hastings MCMC, time series (ACF/PACF, AR/ARMA, periodogram) |
| `ax::num`  | Adaptive Gauss-Kronrod and tanh-sinh quadrature, RK45 (Dormand-Prince) ODE solver, Brent root finding, Newton, optimization (Brent 1-d, BFGS, Nelder-Mead) |
| `ax::sym`  | CAS: immutable hash-consed expression DAG, canonicalizing simplifier, symbolic differentiation, integration (table, u-substitution, parts, partial fractions), equation solving (exact polynomial roots through quartic, Durand-Kerner numeric fallback, symbolic linear systems), univariate polynomial algebra over exact rationals, expansion, text/LaTeX printers, sympy-sstr parser, verification oracle (canonical forms, three-valued equivalence) |

See [`docs/specs/2026-07-05-axiom-design.md`](docs/specs/2026-07-05-axiom-design.md)
for the full design and roadmap (hypothesis tests, GLM, MCMC, time series,
symbolic integration/solvers, proof kernel, optional CUDA backend).

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
sympy-`sstr` parser, `canonical()`/`equivalent()` primitives with a strict
soundness contract (EQUIVALENT only on structural proof, NOT_EQUIVALENT only
on a confirmed numeric witness, UNDECIDED otherwise — never guessed), and the
`axiom-oracle` JSONL harness for the sympy parity audit (~11 ms/row Release
on farm-shaped `equiv_mod_const` rows). axiom becomes llmopt's oracle of
record only after that parity run passes; the pybind11 bridge lands after
parity. Next up: the proof kernel (Phase 9) and the optional CUDA backend
(Phase 10). Implementation plans for every phase live in
[`docs/plans/`](docs/plans/).

## License

MIT — see [LICENSE](LICENSE).
