#include <ax/st/sf.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace ax::st::sf {

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kTiny = 1e-300;
constexpr int kMaxIter = 300;

/** Series for P(a,x), converges fast for x < a+1. */
double gamma_p_series(double a, double x) {
  double term = 1.0 / a;
  double sum = term;
  double ap = a;
  for (int i = 0; i < kMaxIter; ++i) {
    ap += 1.0;
    term *= x / ap;
    sum += term;
    if (std::abs(term) < std::abs(sum) * kEps) break;
  }
  return sum * std::exp(-x + a * std::log(x) - lgamma(a));
}

/** Continued fraction for Q(a,x) (Lentz), converges for x > a+1. */
double gamma_q_cf(double a, double x) {
  double b = x + 1.0 - a;
  double c = 1.0 / kTiny;
  double d = 1.0 / b;
  double h = d;
  for (int i = 1; i <= kMaxIter; ++i) {
    const double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
    b += 2.0;
    d = an * d + b;
    if (std::abs(d) < kTiny) d = kTiny;
    c = b + an / c;
    if (std::abs(c) < kTiny) c = kTiny;
    d = 1.0 / d;
    const double delta = d * c;
    h *= delta;
    if (std::abs(delta - 1.0) < kEps) break;
  }
  return h * std::exp(-x + a * std::log(x) - lgamma(a));
}

/** Continued fraction for incomplete beta (Lentz). */
double beta_cf(double a, double b, double x) {
  const double qab = a + b, qap = a + 1.0, qam = a - 1.0;
  double c = 1.0;
  double d = 1.0 - qab * x / qap;
  if (std::abs(d) < kTiny) d = kTiny;
  d = 1.0 / d;
  double h = d;
  for (int m = 1; m <= kMaxIter; ++m) {
    const double dm = static_cast<double>(m);
    const double m2 = 2.0 * dm;
    double aa = dm * (b - dm) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (std::abs(d) < kTiny) d = kTiny;
    c = 1.0 + aa / c;
    if (std::abs(c) < kTiny) c = kTiny;
    d = 1.0 / d;
    h *= d * c;
    aa = -(a + dm) * (qab + dm) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (std::abs(d) < kTiny) d = kTiny;
    c = 1.0 + aa / c;
    if (std::abs(c) < kTiny) c = kTiny;
    d = 1.0 / d;
    const double delta = d * c;
    h *= delta;
    if (std::abs(delta - 1.0) < kEps) break;
  }
  return h;
}

}  // namespace

double lgamma(double x) {
  if (!(x > 0.0)) throw std::domain_error("lgamma: requires x > 0");
  // Lanczos, g = 7, n = 9
  static const double kCoef[9] = {
      0.99999999999980993,      676.5203681218851,   -1259.1392167224028,
      771.32342877765313,       -176.61502916214059, 12.507343278686905,
      -0.13857109526572012,     9.9843695780195716e-6,
      1.5056327351493116e-7};
  if (x < 0.5) {
    // reflection: Γ(x)Γ(1-x) = π / sin(πx)
    return std::log(kPi / std::sin(kPi * x)) - lgamma(1.0 - x);
  }
  const double z = x - 1.0;
  double sum = kCoef[0];
  for (int i = 1; i < 9; ++i) sum += kCoef[i] / (z + static_cast<double>(i));
  const double t = z + 7.5;
  return 0.5 * std::log(2.0 * kPi) + (z + 0.5) * std::log(t) - t +
         std::log(sum);
}

double gamma_p(double a, double x) {
  if (!(a > 0.0) || x < 0.0)
    throw std::domain_error("gamma_p: requires a > 0, x >= 0");
  if (x == 0.0) return 0.0;
  if (x < a + 1.0) return gamma_p_series(a, x);
  return 1.0 - gamma_q_cf(a, x);
}

double gamma_q(double a, double x) {
  if (!(a > 0.0) || x < 0.0)
    throw std::domain_error("gamma_q: requires a > 0, x >= 0");
  if (x == 0.0) return 1.0;
  if (x < a + 1.0) return 1.0 - gamma_p_series(a, x);
  return gamma_q_cf(a, x);
}

double erf(double x) {
  if (x == 0.0) return 0.0;
  const double p = gamma_p(0.5, x * x);
  return x > 0.0 ? p : -p;
}

double erfc(double x) {
  if (x < 0.0) return 2.0 - erfc(-x);
  if (x == 0.0) return 1.0;
  return gamma_q(0.5, x * x);
}

double erf_inv(double p) {
  if (!(p > -1.0 && p < 1.0))
    throw std::domain_error("erf_inv: requires |p| < 1");
  if (p == 0.0) return 0.0;
  // initial guess (Winitzki-style), refined by Newton
  const double sign = p > 0.0 ? 1.0 : -1.0;
  const double ap = std::abs(p);
  const double ln1mp2 = std::log(1.0 - ap * ap);
  const double aa = 8.0 * (kPi - 3.0) / (3.0 * kPi * (4.0 - kPi));
  const double t1 = 2.0 / (kPi * aa) + 0.5 * ln1mp2;
  double y = sign * std::sqrt(std::sqrt(t1 * t1 - ln1mp2 / aa) - t1);
  for (int i = 0; i < 4; ++i) {
    const double err = erf(y) - p;
    y -= err * 0.5 * std::sqrt(kPi) * std::exp(y * y);
  }
  return y;
}

double log_beta(double a, double b) {
  return lgamma(a) + lgamma(b) - lgamma(a + b);
}

double beta_inc(double a, double b, double x) {
  if (!(a > 0.0) || !(b > 0.0))
    throw std::domain_error("beta_inc: requires a, b > 0");
  if (x < 0.0 || x > 1.0)
    throw std::domain_error("beta_inc: requires x in [0,1]");
  if (x == 0.0) return 0.0;
  if (x == 1.0) return 1.0;
  const double front =
      std::exp(a * std::log(x) + b * std::log(1.0 - x) - log_beta(a, b));
  if (x < (a + 1.0) / (a + b + 2.0)) return front * beta_cf(a, b, x) / a;
  return 1.0 - front * beta_cf(b, a, 1.0 - x) / b;
}

}  // namespace ax::st::sf
