#include <ax/la/blas.hpp>

#include <stdexcept>

namespace ax::la {

namespace {

/// i-k-j inner kernel over row range [i0, i1): streams b rows, vectorizes.
void mm_rows(const mat& a, const mat& b, mat& c, std::size_t i0,
             std::size_t i1) {
  const std::size_t k_n = a.cols(), n = b.cols();
  const double* bd = b.data();
  for (std::size_t i = i0; i < i1; ++i) {
    double* ci = c.data() + i * n;
    for (std::size_t k = 0; k < k_n; ++k) {
      const double aik = a(i, k);
      const double* bk = bd + k * n;
      for (std::size_t j = 0; j < n; ++j) ci[j] += aik * bk[j];
    }
  }
}

void check_shapes(const mat& a, const mat& b) {
  if (a.cols() != b.rows())
    throw std::invalid_argument("matmul: inner dimension mismatch");
}

}  // namespace

mat matmul(const mat& a, const mat& b) {
  check_shapes(a, b);
  mat c(a.rows(), b.cols());
  mm_rows(a, b, c, 0, a.rows());
  return c;
}

mat matmul(const mat& a, const mat& b, thread_pool& pool) {
  check_shapes(a, b);
  mat c(a.rows(), b.cols());
  constexpr std::size_t rows_per_task = 64;
  const std::size_t nblocks = (a.rows() + rows_per_task - 1) / rows_per_task;
  parallel_for(pool, std::size_t{0}, nblocks, 1, [&](std::size_t blk) {
    const std::size_t i0 = blk * rows_per_task;
    const std::size_t i1 = std::min(i0 + rows_per_task, a.rows());
    mm_rows(a, b, c, i0, i1);
  });
  return c;
}

}  // namespace ax::la
