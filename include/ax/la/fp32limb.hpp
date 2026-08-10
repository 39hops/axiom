#pragma once
/** @file fp32limb.hpp
 *  fp32-limb exact GEMM CPU oracle (PRE-REG FP32LIMB-METAL, rung R1).
 *
 *  Carries fp64-class exactness in fp32 limbs: inputs are aligned per
 *  32-wide K-block, sliced into signed SLICE_W-bit integer slices stored
 *  as float, slice-pair dot products run in fp32 (each op provably exact:
 *  2*SLICE_W + log2(BLOCK) <= 24), and partials recombine into an exact
 *  dyadic (bigint mantissa, exponent) accumulator.
 *
 *  Ground truth is always ax::bigint integer arithmetic — never fp-vs-fp.
 *  All exactness fences throw std::runtime_error so they survive NDEBUG.
 */
#include <ax/core/bigint.hpp>

#include <cstddef>
#include <vector>

namespace ax::la::fp32limb {

// Registered constants (relay 2026-08-10-10). The house source's signature
// default is s=8; the REGISTERED slice width is 7. Do not change without a
// new pre-registration.
inline constexpr int SLICE_W = 7;
inline constexpr int BLOCK = 32;
// Fixed slice cap: covers 24 mantissa bits + up to 32 bits of in-block
// exponent spread. Residual past the cap is an envelope reject.
inline constexpr int MAX_SLICES = 8;
static_assert(2 * SLICE_W + 5 /* log2(BLOCK) */ <= 24,
              "slice-pair fp32 accumulation would round");

/// Exact dyadic rational: value = m * 2^e.
struct dyadic {
  bigint m;
  int e = 0;
};

/// Exact decode of a finite fp32. Throws std::runtime_error on inf/nan.
dyadic decode(float x);

/// r += m2 * 2^e2, exactly (aligns exponents by shifting mantissas).
void acc(dyadic& r, const bigint& m2, int e2);

/// Exact value equality (representation-independent).
bool dyadic_eq(const dyadic& a, const dyadic& b);

/// Minimal row-major float matrix for the oracle.
struct matf {
  int rows = 0, cols = 0;
  std::vector<float> v;
  matf() = default;
  matf(int r, int c) : rows(r), cols(c), v(static_cast<std::size_t>(r) * c) {}
  float& at(int i, int j) { return v[static_cast<std::size_t>(i) * cols + j]; }
  float at(int i, int j) const {
    return v[static_cast<std::size_t>(i) * cols + j];
  }
};

/// Ground-truth GEMM: pure bigint arithmetic on decoded mantissas.
/// Returns row-major rows(A) x cols(B) dyadics.
std::vector<dyadic> gemm_ref(const matf& A, const matf& B);

/// Error-free transform: a + b = s + r exactly (Knuth two_sum).
struct f2 {
  float s, r;
};
f2 two_sum(float a, float b);

/// Exact expansion add (Shewchuk-style grow; drops zero components).
/// Verbatim port of the house exp_add.
std::vector<float> exp_add(std::vector<float> e, float x);

/// Rider (triple-double exit): expansion add with a hard length cap.
/// Throws std::runtime_error if the exact result needs more components.
std::vector<float> exp_add_capped(std::vector<float> e, float x,
                                  std::size_t cap);

/// One aligned, sliced segment (a row of A or column of B restricted to a
/// K-block). sl[p][k] holds the p-th signed SLICE_W-bit slice of element k,
/// stored exactly as float. Element k reconstructs as
///   sum_p sl[p][k] * 2^(-SLICE_W*(p+1)) * 2^e_align.
struct sliced {
  std::vector<std::vector<float>> sl;
  int e_align = 0;
};

/// Align by max exponent and slice. n <= BLOCK. Throws
/// std::runtime_error("fp32limb: envelope") when the residual is still
/// nonzero after MAX_SLICES slices (in-block exponent spread too wide).
sliced slice_row(const float* x, int n);

/// fp32-limb GEMM oracle: slice-pair dots computed in fp32, recombined
/// exactly. Every fp32 accumulation is fenced (|acc| < 2^24 or throw).
std::vector<dyadic> gemm_fp32limb(const matf& A, const matf& B);

}  // namespace ax::la::fp32limb
