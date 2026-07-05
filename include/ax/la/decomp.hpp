#pragma once
/** @file decomp.hpp Dense decompositions: LU, Cholesky, QR. */
#include <ax/la/mat.hpp>

#include <cstddef>
#include <vector>

namespace ax::la {

/** PA = LU factorization with partial pivoting. */
struct lu_result {
  mat lu;                        ///< L (unit diag, strict lower) + U packed.
  std::vector<std::size_t> piv;  ///< row i of PA is row piv[i] of A.
  int sign = 1;                  ///< permutation parity (+1/-1).
  bool singular = false;         ///< true if a zero pivot was hit.
};

/** Factor square a. Throws std::invalid_argument if not square. */
lu_result lu_decompose(const mat& a);
/** Solve a x = b via factorization. Throws std::domain_error if singular,
    std::invalid_argument on size mismatch. */
vec lu_solve(const lu_result& f, const vec& b);
/** Determinant via LU (0 for singular). */
double det(const mat& a);
/** Inverse via LU solve per unit vector. Throws std::domain_error if
    singular. */
mat inverse(const mat& a);

/** Cholesky A = L L^T for symmetric positive definite A.
    Returns lower-triangular L. Throws std::domain_error if not SPD
    (non-positive pivot), std::invalid_argument if not square.
    Only lower triangle of input is read. */
mat cholesky(const mat& a);
/** Solve a x = b given L from cholesky(). Throws std::invalid_argument
    on size mismatch. */
vec cholesky_solve(const mat& l, const vec& b);

/** Householder QR: A (m x n, m >= n) = Q R with Q m x m orthogonal,
    R m x n upper trapezoidal. */
struct qr_result {
  mat q;  ///< orthogonal, m x m
  mat r;  ///< upper trapezoidal, m x n
};

/** Factor a. Throws std::invalid_argument if rows < cols. */
qr_result qr_decompose(const mat& a);
/** Least squares min ||a x - b|| via QR (full column rank assumed).
    Throws std::invalid_argument if rows < cols or size mismatch,
    std::domain_error on rank deficiency (zero diagonal in R). */
vec lstsq(const mat& a, const vec& b);

}  // namespace ax::la
