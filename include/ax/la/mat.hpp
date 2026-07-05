#pragma once
/** @file mat.hpp Dense row-major matrix and vector over double. */
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace ax::la {

/** Dense vector of double. Thin wrapper over std::vector with math ops. */
class vec {
 public:
  vec() = default;
  explicit vec(std::size_t n, double fill = 0.0) : d_(n, fill) {}
  vec(std::initializer_list<double> xs) : d_(xs) {}

  std::size_t size() const noexcept { return d_.size(); }
  double& operator[](std::size_t i) noexcept { return d_[i]; }
  double operator[](std::size_t i) const noexcept { return d_[i]; }
  double* data() noexcept { return d_.data(); }
  const double* data() const noexcept { return d_.data(); }

  /// Euclidean norm.
  double norm() const {
    double s = 0.0;
    for (double x : d_) s += x * x;
    return std::sqrt(s);
  }

  friend vec operator+(const vec& a, const vec& b) {
    if (a.size() != b.size())
      throw std::invalid_argument("vec: size mismatch");
    vec r(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) r[i] = a[i] + b[i];
    return r;
  }

  friend vec operator-(const vec& a, const vec& b) {
    if (a.size() != b.size())
      throw std::invalid_argument("vec: size mismatch");
    vec r(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) r[i] = a[i] - b[i];
    return r;
  }

  friend vec operator*(double s, const vec& a) {
    vec r(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) r[i] = s * a[i];
    return r;
  }

  friend double dot(const vec& a, const vec& b) {
    if (a.size() != b.size())
      throw std::invalid_argument("vec: size mismatch");
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
  }

 private:
  std::vector<double> d_;
};

/** Dense row-major matrix of double. */
class mat {
 public:
  mat() = default;
  mat(std::size_t rows, std::size_t cols, double fill = 0.0)
      : r_(rows), c_(cols), d_(rows * cols, fill) {}

  /// From rows; all rows must be equal length (throws std::invalid_argument).
  mat(std::initializer_list<std::initializer_list<double>> rows) {
    r_ = rows.size();
    c_ = r_ ? rows.begin()->size() : 0;
    d_.reserve(r_ * c_);
    for (const auto& row : rows) {
      if (row.size() != c_) throw std::invalid_argument("mat: ragged rows");
      d_.insert(d_.end(), row.begin(), row.end());
    }
  }

  /// n x n identity.
  static mat identity(std::size_t n) {
    mat i(n, n);
    for (std::size_t k = 0; k < n; ++k) i(k, k) = 1.0;
    return i;
  }

  std::size_t rows() const noexcept { return r_; }
  std::size_t cols() const noexcept { return c_; }

  /// Unchecked element access.
  double& operator()(std::size_t i, std::size_t j) noexcept {
    return d_[i * c_ + j];
  }
  double operator()(std::size_t i, std::size_t j) const noexcept {
    return d_[i * c_ + j];
  }

  /// Checked element access; throws std::out_of_range.
  double& at(std::size_t i, std::size_t j) {
    if (i >= r_ || j >= c_) throw std::out_of_range("mat: index");
    return d_[i * c_ + j];
  }
  double at(std::size_t i, std::size_t j) const {
    if (i >= r_ || j >= c_) throw std::out_of_range("mat: index");
    return d_[i * c_ + j];
  }

  double* data() noexcept { return d_.data(); }
  const double* data() const noexcept { return d_.data(); }

  mat transposed() const {
    mat t(c_, r_);
    for (std::size_t i = 0; i < r_; ++i)
      for (std::size_t j = 0; j < c_; ++j) t(j, i) = (*this)(i, j);
    return t;
  }

  friend mat operator+(const mat& a, const mat& b) {
    check_same_shape(a, b);
    mat r(a.r_, a.c_);
    for (std::size_t k = 0; k < a.d_.size(); ++k) r.d_[k] = a.d_[k] + b.d_[k];
    return r;
  }

  friend mat operator-(const mat& a, const mat& b) {
    check_same_shape(a, b);
    mat r(a.r_, a.c_);
    for (std::size_t k = 0; k < a.d_.size(); ++k) r.d_[k] = a.d_[k] - b.d_[k];
    return r;
  }

  friend mat operator*(double s, const mat& a) {
    mat r(a.r_, a.c_);
    for (std::size_t k = 0; k < a.d_.size(); ++k) r.d_[k] = s * a.d_[k];
    return r;
  }

  friend vec operator*(const mat& a, const vec& x) {
    if (a.c_ != x.size()) throw std::invalid_argument("mat*vec: shape");
    vec y(a.r_);
    for (std::size_t i = 0; i < a.r_; ++i) {
      double s = 0.0;
      for (std::size_t j = 0; j < a.c_; ++j) s += a(i, j) * x[j];
      y[i] = s;
    }
    return y;
  }

 private:
  static void check_same_shape(const mat& a, const mat& b) {
    if (a.r_ != b.r_ || a.c_ != b.c_)
      throw std::invalid_argument("mat: shape mismatch");
  }

  std::size_t r_ = 0, c_ = 0;
  std::vector<double> d_;
};

/// True iff same shape and max element diff <= tol.
inline bool approx_equal(const mat& a, const mat& b, double tol) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
  for (std::size_t i = 0; i < a.rows(); ++i)
    for (std::size_t j = 0; j < a.cols(); ++j)
      if (std::abs(a(i, j) - b(i, j)) > tol) return false;
  return true;
}

/// True iff same size and max element diff <= tol.
inline bool approx_equal(const vec& a, const vec& b, double tol) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (std::abs(a[i] - b[i]) > tol) return false;
  return true;
}

}  // namespace ax::la
