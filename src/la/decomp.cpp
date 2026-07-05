#include <ax/la/decomp.hpp>

#include <cmath>
#include <stdexcept>

namespace ax::la {

lu_result lu_decompose(const mat& a) {
  if (a.rows() != a.cols())
    throw std::invalid_argument("lu_decompose: matrix not square");
  const std::size_t n = a.rows();
  lu_result f;
  f.lu = a;
  f.piv.resize(n);
  for (std::size_t i = 0; i < n; ++i) f.piv[i] = i;

  for (std::size_t k = 0; k < n; ++k) {
    // pivot: largest |value| in column k at or below diagonal
    std::size_t p = k;
    double best = std::abs(f.lu(k, k));
    for (std::size_t i = k + 1; i < n; ++i) {
      const double v = std::abs(f.lu(i, k));
      if (v > best) {
        best = v;
        p = i;
      }
    }
    if (best == 0.0) {
      f.singular = true;
      continue;
    }
    if (p != k) {
      for (std::size_t j = 0; j < n; ++j)
        std::swap(f.lu(k, j), f.lu(p, j));
      std::swap(f.piv[k], f.piv[p]);
      f.sign = -f.sign;
    }
    const double pivot = f.lu(k, k);
    for (std::size_t i = k + 1; i < n; ++i) {
      const double m = f.lu(i, k) / pivot;
      f.lu(i, k) = m;
      for (std::size_t j = k + 1; j < n; ++j) f.lu(i, j) -= m * f.lu(k, j);
    }
  }
  return f;
}

vec lu_solve(const lu_result& f, const vec& b) {
  if (f.singular) throw std::domain_error("lu_solve: singular matrix");
  const std::size_t n = f.lu.rows();
  if (b.size() != n) throw std::invalid_argument("lu_solve: size mismatch");
  // forward substitution with permuted b (L has unit diagonal)
  vec y(n);
  for (std::size_t i = 0; i < n; ++i) {
    double s = b[f.piv[i]];
    for (std::size_t j = 0; j < i; ++j) s -= f.lu(i, j) * y[j];
    y[i] = s;
  }
  // back substitution
  vec x(n);
  for (std::size_t i = n; i-- > 0;) {
    double s = y[i];
    for (std::size_t j = i + 1; j < n; ++j) s -= f.lu(i, j) * x[j];
    x[i] = s / f.lu(i, i);
  }
  return x;
}

double det(const mat& a) {
  const lu_result f = lu_decompose(a);
  if (f.singular) return 0.0;
  double d = static_cast<double>(f.sign);
  for (std::size_t i = 0; i < f.lu.rows(); ++i) d *= f.lu(i, i);
  return d;
}

mat inverse(const mat& a) {
  const lu_result f = lu_decompose(a);
  if (f.singular) throw std::domain_error("inverse: singular matrix");
  const std::size_t n = a.rows();
  mat inv(n, n);
  vec e(n);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < n; ++i) e[i] = (i == j) ? 1.0 : 0.0;
    const vec col = lu_solve(f, e);
    for (std::size_t i = 0; i < n; ++i) inv(i, j) = col[i];
  }
  return inv;
}

}  // namespace ax::la
