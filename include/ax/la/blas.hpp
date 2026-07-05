#pragma once
/** @file blas.hpp Matrix multiplication kernels. */
#include <ax/la/mat.hpp>
#include <ax/par/pool.hpp>

namespace ax::la {

/** c = a * b, serial reference (i-k-j loop order).
    Throws std::invalid_argument on inner-dimension mismatch. */
mat matmul(const mat& a, const mat& b);

/** c = a * b, row-blocked and parallelized across pool. */
mat matmul(const mat& a, const mat& b, thread_pool& pool);

/// Serial kernel.
inline mat operator*(const mat& a, const mat& b) { return matmul(a, b); }

}  // namespace ax::la
