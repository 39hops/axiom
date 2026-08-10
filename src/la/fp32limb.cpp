#include <ax/la/fp32limb.hpp>

#include <cmath>
#include <stdexcept>

namespace ax::la::fp32limb {

dyadic decode(float x) {
  if (!std::isfinite(x)) throw std::runtime_error("fp32limb: non-finite input");
  if (x == 0.0f) return {bigint(0), 0};
  int e = 0;
  // frexp on double is exact for any fp32 value; mantissa in [0.5, 1).
  const double f = std::frexp(static_cast<double>(x), &e);
  // fp32 has a 24-bit significand; f * 2^24 is an exact integer.
  const long long m = std::llround(std::ldexp(f, 24));
  return {bigint(m), e - 24};
}

void acc(dyadic& r, const bigint& m2, int e2) {
  if (r.m == bigint(0)) {
    r.m = m2;
    r.e = e2;
    return;
  }
  if (m2 == bigint(0)) return;
  if (r.e <= e2) {
    r.m = r.m + (m2 << static_cast<unsigned>(e2 - r.e));
  } else {
    r.m = (r.m << static_cast<unsigned>(r.e - e2)) + m2;
    r.e = e2;
  }
}

bool dyadic_eq(const dyadic& a, const dyadic& b) {
  if (a.e == b.e) return a.m == b.m;
  if (a.e < b.e) return a.m == (b.m << static_cast<unsigned>(b.e - a.e));
  return (a.m << static_cast<unsigned>(a.e - b.e)) == b.m;
}

std::vector<dyadic> gemm_ref(const matf& A, const matf& B) {
  if (A.cols != B.rows) throw std::runtime_error("fp32limb: shape mismatch");
  std::vector<dyadic> c(static_cast<std::size_t>(A.rows) * B.cols);
  for (int i = 0; i < A.rows; ++i) {
    for (int j = 0; j < B.cols; ++j) {
      dyadic r{bigint(0), 0};
      for (int k = 0; k < A.cols; ++k) {
        const dyadic da = decode(A.at(i, k));
        const dyadic db = decode(B.at(k, j));
        acc(r, da.m * db.m, da.e + db.e);
      }
      c[static_cast<std::size_t>(i) * B.cols + j] = r;
    }
  }
  return c;
}

f2 two_sum(float a, float b) { throw std::runtime_error("unimplemented"); }
std::vector<float> exp_add(std::vector<float> e, float x) {
  throw std::runtime_error("unimplemented");
}
std::vector<float> exp_add_capped(std::vector<float> e, float x,
                                  std::size_t cap) {
  throw std::runtime_error("unimplemented");
}
sliced slice_row(const float* x, int n) {
  throw std::runtime_error("unimplemented");
}
std::vector<dyadic> gemm_fp32limb(const matf& A, const matf& B) {
  throw std::runtime_error("unimplemented");
}

}  // namespace ax::la::fp32limb
