#include <ax/num/quad.hpp>

#include <cmath>

namespace ax::num {

namespace {

// G7-K15 abscissae/weights on [-1,1] (QUADPACK dqk15).
constexpr double kXgk[8] = {
    0.991455371120813, 0.949107912342759, 0.864864423359769,
    0.741531185599394, 0.586087235467691, 0.405845151377397,
    0.207784955007898, 0.0};
constexpr double kWgk[8] = {
    0.022935322010529, 0.063092092629979, 0.104790010322250,
    0.140653259715525, 0.169004726639267, 0.190350578064785,
    0.204432940075298, 0.209482141084728};
constexpr double kWg[4] = {0.129484966168870, 0.279705391489277,
                           0.381830050505119, 0.417959183673469};

struct panel {
  double kronrod, err;
};

/** One G7-K15 evaluation on [a,b]. */
panel gk15(const std::function<double(double)>& f, double a, double b,
           int& evals) {
  const double mid = 0.5 * (a + b), half = 0.5 * (b - a);
  const double fc = f(mid);
  double gauss = fc * kWg[3];
  double kron = fc * kWgk[7];
  for (int j = 0; j < 7; ++j) {
    const double dx = half * kXgk[j];
    const double fsum = f(mid - dx) + f(mid + dx);
    kron += fsum * kWgk[j];
    if (j % 2 == 1) gauss += fsum * kWg[j / 2];
  }
  evals += 15;
  kron *= half;
  gauss *= half;
  const double diff = std::abs(kron - gauss);
  // QUADPACK-style sharpened estimate
  const double err = diff == 0.0 ? 0.0 : std::min(diff, std::pow(200.0 * diff, 1.5));
  return {kron, err};
}

void adapt(const std::function<double(double)>& f, double a, double b,
           double abstol, double reltol, int depth, int max_depth,
           double& value, double& err, int& evals) {
  const panel p = gk15(f, a, b, evals);
  const double tol = std::max(abstol, reltol * std::abs(p.kronrod));
  if (p.err <= tol || depth >= max_depth) {
    value += p.kronrod;
    err += p.err;
    return;
  }
  const double mid = 0.5 * (a + b);
  adapt(f, a, mid, 0.5 * abstol, reltol, depth + 1, max_depth, value, err,
        evals);
  adapt(f, mid, b, 0.5 * abstol, reltol, depth + 1, max_depth, value, err,
        evals);
}

}  // namespace

quad_result integrate(const std::function<double(double)>& f, double a,
                      double b, double abstol, double reltol, int max_depth) {
  if (a == b) return {0.0, 0.0, 0};
  const double sign = a < b ? 1.0 : -1.0;
  if (a > b) std::swap(a, b);
  double value = 0.0, err = 0.0;
  int evals = 0;
  adapt(f, a, b, abstol, reltol, 0, max_depth, value, err, evals);
  return {sign * value, err, evals};
}

quad_result integrate_ts(const std::function<double(double)>& f, double a,
                         double b, double tol, int max_level) {
  const double half = 0.5 * (b - a);
  constexpr double kPiHalf = 1.570796326794896619231321691639751442;
  constexpr double kTMax = 4.0;
  int evals = 0;

  // weighted integrand over transformed variable t
  const auto term = [&](double t) -> double {
    const double sh = kPiHalf * std::sinh(t);
    // distance to the nearer endpoint, computed without cancellation:
    // 1 - |tanh(s)| = 2 / (1 + e^{2|s|})
    const double delta = half * 2.0 / (1.0 + std::exp(2.0 * std::abs(sh)));
    const double x = t <= 0.0 ? a + delta : b - delta;
    if (x <= a || x >= b) return 0.0;  // denormal-level saturation only
    const double ch = std::cosh(sh);
    const double w = kPiHalf * std::cosh(t) / (ch * ch);
    const double v = f(x) * w;
    ++evals;
    return std::isfinite(v) ? v : 0.0;
  };

  double h = 1.0;
  double sum = term(0.0);
  for (int k = 1; k * h <= kTMax; ++k)
    sum += term(k * h) + term(-k * h);
  double prev = half * h * sum;

  for (int level = 1; level <= max_level; ++level) {
    h *= 0.5;
    // add new nodes at odd multiples of h
    for (int k = 1; k * h <= kTMax; k += 2)
      sum += term(k * h) + term(-k * h);
    const double cur = half * h * sum;
    const double change = std::abs(cur - prev);
    if (change < tol * std::max(1.0, std::abs(cur)) && level >= 3)
      return {cur, change, evals};
    prev = cur;
  }
  return {prev, std::abs(prev) * 1e-6, evals};
}

}  // namespace ax::num
