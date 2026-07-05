# axiom — C++ mathematical/statistical library: design spec

Date: 2026-07-05
Status: approved pending user review

## Purpose

A from-scratch, STL-only C++23 mathematics library covering exact arithmetic,
linear algebra, heavy statistics, a symbolic (CAS) engine, and a small proof
kernel — engineered for performance (thread-parallel CPU kernels, opt-in CUDA
matmul). A math-specialized fine-tuned LLM is a later phase and is out of scope
here, but the CAS and proof kernel keep hooks (machine-checkable proof terms,
verifiable step traces) so a model can bolt on later.

## Constraints (user-set)

- **STL-only** for the default build. No Eigen, no GMP, no Boost. Own bigint,
  own RNG, hand-rolled decompositions.
- **CUDA** is the single allowed exception: an *opt-in* compile flag
  (`AXIOM_CUDA`) enabling a GPU matmul/linalg backend. Default build never
  requires it.
- **Naming**: std-style. `snake_case` types/functions/files, short lowercase
  namespaces and dirs. Root namespace `ax::`.
- **Doxygen** comments on all public headers.
- **Distributed = single-machine parallel** (thread pool + GPU). Multi-node
  (MPI-style) is explicitly out of scope.
- Model training deferred entirely.

## Toolchain

- MSVC 14.50 (VS 2026 Build Tools v18), C++23.
- CMake 4.2 + Ninja (bundled with Build Tools).
- CUDA 13.3 for the optional GPU backend. Risk: nvcc host-compiler whitelist
  may not accept MSVC 14.50 yet — verify at GPU phase; does not block CPU work.
- GoogleTest via CMake `FetchContent` (test-only dep; library itself stays
  STL-only).

## Layout

```
math/
├── include/ax/        public headers (header-heavy, impl in src/ where hot)
│   ├── core/          bigint, rational, fixed-precision float utils, constants
│   ├── la/            dense matrix/vector, decompositions, backend dispatch
│   ├── st/            distributions, tests, regression, mcmc, ts
│   ├── num/           numeric methods: quadrature, ODE solvers, optimization
│   ├── sym/           CAS: expr DAG, simplify, calculus, solve, print
│   ├── prf/           proof kernel: terms, typechecker, tactics
│   └── par/           thread pool, parallel_for, blocking helpers
├── src/               non-header implementation
├── cuda/              .cu kernels (compiled only with AXIOM_CUDA)
├── tests/             GoogleTest, mirrors module layout
├── bench/             micro-benchmarks (own tiny harness, STL-only)
└── docs/specs/        design docs
```

## Modules

### ax::core — exact arithmetic
- `bigint`: sign + magnitude `std::vector<uint64_t>` limbs. Schoolbook mul with
  Karatsuba above threshold; later Toom-3 if profiling justifies. Division via
  Knuth algorithm D. Comparison, gcd (binary), pow, modpow.
- `rational`: normalized `bigint` pair. Exact CAS substrate.
- `real`: `double` wrappers + kahan summation, ulp helpers, constants.
- Number theory: primality (Miller-Rabin deterministic < 2^64, probabilistic on
  bigint), Pollard rho factorization, modular arithmetic (modpow, modinv, CRT).
- `fft`: iterative radix-2 complex FFT + NTT (used by bigint mul above
  Karatsuba range, convolutions, st spectral analysis).
- Everything `constexpr`-friendly where practical.

### ax::par — parallelism substrate
- `thread_pool` (jthread, work-stealing deque later; simple queue v1).
- `parallel_for(range, grain, fn)`, `parallel_reduce`.
- All heavy modules take an optional `executor&`; default = global pool.

### ax::la — linear algebra
- `mat<T>` / `vec<T>` dense, row-major, own aligned allocator.
- CPU matmul: cache-blocked, panel-packed, multithreaded via `ax::par`;
  autovectorizable inner kernels (no intrinsics, pragma-friendly loops).
- Decompositions: LU (partial pivot), QR (Householder), Cholesky, SVD
  (Golub-Kahan bidiagonalization + implicit QR), eigen (symmetric via QR).
- Backend dispatch: `enum class backend { cpu, cuda }`; `cuda` present only
  under `AXIOM_CUDA`, falls back to cpu at runtime if no device.

### ax::st — statistics (heavy scope)
- RNG: own PCG64 + splitmix seeding, `uniform`, `normal` (ziggurat),
  `gamma`, etc.
- Distributions (~15: normal, t, chi2, f, gamma, beta, binomial, poisson,
  exponential, uniform, lognormal, weibull, cauchy, laplace, negbinom):
  pdf/pmf, cdf, quantile, sample. Special functions in-house: lgamma,
  incomplete gamma/beta (continued fractions), erf.
- Descriptive: mean/var/skew/kurtosis (one-pass, numerically stable),
  quantiles, covariance/correlation matrices.
- Tests: t (1/2-sample, Welch), chi², ANOVA, KS, Shapiro-Wilk (later).
- Regression: OLS via QR; GLMs (logistic, Poisson) via IRLS on ax::la.
- MCMC: Metropolis-Hastings v1; NUTS later.
- Time series: ACF/PACF, AR/MA/ARIMA fit (CSS then MLE via Kalman later),
  periodogram/spectral density via ax::core fft.

### ax::num — numeric methods
- Quadrature: adaptive Gauss-Kronrod, tanh-sinh for endpoint singularities.
- ODE: RK45 (Dormand-Prince) adaptive; stiff (implicit BDF) v1.5.
- Root finding: bisection/Brent, Newton with numeric or symbolic (ax::sym)
  derivative.
- Optimization: golden-section/Brent 1-d, Newton, BFGS, Nelder-Mead;
  linear programming (simplex) if st/GLM work surfaces need grows.

### ax::sym — CAS
- Immutable expression DAG: `num, sym, add, mul, pow, fn` nodes,
  hash-consing pool for structural sharing; small-object arena.
- Canonicalization + rewrite-rule simplifier.
- Calculus: differentiation (complete); integration = table + heuristics
  (u-sub, parts, partial fractions) — "Risch-lite", not full Risch.
- Solvers: polynomial exact to quartic (rational-coef), numeric fallback
  (Durand-Kerner); symbolic linear systems; basic trig/exp equations.
- Polynomial algebra (first-class, also internal substrate for solve/simplify):
  univariate/multivariate poly over rationals, arithmetic, GCD (Euclidean +
  modular), square-free + rational-root factorization, resultants. Gröbner
  bases deferred to v1.5.
- Printers: plain text + LaTeX.
- Hook: solver/simplifier steps can emit trace records for future
  proof-obligation generation.

### ax::prf — proof kernel
- Small trusted kernel (~2-3k lines), calculus-of-constructions-style
  dependent type theory (mini-Lean).
- Terms, de Bruijn indices, typechecker, minimal inductive types, universe
  hierarchy (simple, non-cumulative v1).
- Proof terms serialized to files; kernel re-checks independently.
- Thin tactic layer: `intro, apply, exact, rewrite` — enough surface for a
  future model to target.
- CAS bridge: selected `ax::sym` results emit kernel-checkable obligations
  (e.g. polynomial identities via ring normalization).

## Performance strategy

1. Correct scalar first (TDD), then benchmark, then optimize — never blind.
2. `bench/` micro-harness gates regressions on matmul, bigint mul,
   distribution sampling, simplifier throughput.
3. Threads via ax::par everywhere data-parallel (matmul panels, MCMC chains,
   bootstrap, batch sampling).
4. GPU phase: CUDA matmul (tiled shared-memory kernel), then batched
   decompositions if profiling justifies. Opt-in flag; runtime fallback.

## Testing

- GoogleTest per module; TDD workflow.
- Oracles: known closed forms, cross-checks between modules (e.g. quantile∘cdf
  ≈ id, symbolic derivative vs numeric diff, OLS vs hand-computed).
- Property tests via randomized inputs + invariants (bigint ring axioms,
  simplifier idempotence, typechecker subject-reduction spot checks).
- Coverage target 80%+ on non-GPU code.

## Phases

0. Repo skeleton, CMake, CI-less local build, ax::par pool, test harness.
1. core: bigint + rational (heavy TDD; Karatsuba benched), number theory,
   FFT/NTT (wire into bigint mul).
2. la: mat/vec, blocked parallel matmul, LU/QR/Cholesky; bench baseline.
3. st part 1: RNG, special functions, distributions, descriptive.
4. num: quadrature, RK45, root finding, optimization core (needed by st
   quantiles/GLM/MLE).
5. sym part 1: expr DAG, hash-consing, simplifier, differentiation,
   polynomial algebra, printers.
6. st part 2: tests, OLS/GLM, MCMC, time series; SVD/eigen in la as needed.
7. sym part 2: integration heuristics, solvers.
8. prf: kernel, tactics, CAS bridge for polynomial identities.
9. cuda: matmul backend, dispatch, bench vs CPU.
10. (future, separate spec) model fine-tune + verifier loop.

## Out of scope

- Multi-node distribution, sparse matrices (v2 candidate), full Risch,
  arbitrary-precision floats (bigfloat is v2 candidate), Gröbner bases (v1.5),
  stiff ODE solvers (v1.5), Windows-only build assumed (no cross-platform CI
  yet), the LLM itself.
