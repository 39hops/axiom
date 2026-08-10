#include <ax/la/fp32limb.hpp>

#include <algorithm>
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

// Knuth two-sum; correct only under strict IEEE semantics (this build has
// no fast-math; the Metal port must pin fast-math OFF — pre-reg risk item).
f2 two_sum(float a, float b) {
  const float s = a + b;
  const float bb = s - a;
  const float r = (a - (s - bb)) + (b - bb);
  return {s, r};
}

std::vector<float> exp_add(std::vector<float> e, float x) {
  std::vector<float> out;
  for (float c : e) {
    const f2 sr = two_sum(c, x);
    x = sr.s;
    if (sr.r != 0.0f) out.push_back(sr.r);
  }
  out.push_back(x);
  return out;
}

std::vector<float> exp_add_capped(std::vector<float> e, float x,
                                  std::size_t cap) {
  auto out = exp_add(std::move(e), x);
  if (out.size() > cap)
    throw std::runtime_error("fp32limb: expansion exceeds cap");
  return out;
}
sliced slice_row(const float* x, int n) {
  if (n < 1 || n > BLOCK) throw std::runtime_error("fp32limb: bad window");
  sliced out;
  // align to the window's max exponent: frexp(max|x|) so scaled values
  // land in (-1, 1] before slicing (house alignment, per-window)
  float mx = 0.0f;
  for (int k = 0; k < n; ++k) {
    if (!std::isfinite(x[k]))
      throw std::runtime_error("fp32limb: non-finite input");
    mx = std::max(mx, std::fabs(x[k]));
  }
  out.e_align = 0;
  if (mx != 0.0f) std::frexp(mx, &out.e_align);
  // residuals in double: exact. Each element carries only 24 significant
  // bits (spread shifts the exponent, never widens the significand), and
  // every step is a power-of-two scale, an integer round, or a suffix-
  // extracting subtract — all exact in a 53-bit significand.
  std::vector<double> r(static_cast<std::size_t>(n));
  for (int k = 0; k < n; ++k)
    r[static_cast<std::size_t>(k)] =
        std::ldexp(static_cast<double>(x[k]), -out.e_align);
  const double sc = std::ldexp(1.0, SLICE_W);
  bool nonzero = true;
  for (int p = 0; p < MAX_SLICES && nonzero; ++p) {
    std::vector<float> q(static_cast<std::size_t>(n));
    nonzero = false;
    for (int k = 0; k < n; ++k) {
      auto& rk = r[static_cast<std::size_t>(k)];
      const double scaled = rk * sc;
      const double Q = std::nearbyint(scaled);
      rk = scaled - Q;
      q[static_cast<std::size_t>(k)] = static_cast<float>(Q);
      if (rk != 0.0) nonzero = true;
    }
    out.sl.push_back(std::move(q));
  }
  if (nonzero) throw std::runtime_error("fp32limb: envelope");
  return out;
}
std::vector<dyadic> gemm_fp32limb(const matf& A, const matf& B) {
  throw std::runtime_error("unimplemented");
}

}  // namespace ax::la::fp32limb
