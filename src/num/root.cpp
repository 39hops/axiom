#include <ax/num/root.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace ax::num {

root_result bisect(const std::function<double(double)>& f, double a, double b,
                   double xtol, int max_iter) {
  double fa = f(a), fb = f(b);
  if (fa == 0.0) return {a, 0.0, 0, true};
  if (fb == 0.0) return {b, 0.0, 0, true};
  if (fa * fb > 0.0)
    throw std::invalid_argument("bisect: f(a) and f(b) must differ in sign");
  int i = 0;
  for (; i < max_iter; ++i) {
    const double mid = 0.5 * (a + b);
    const double fm = f(mid);
    if (fm == 0.0 || 0.5 * (b - a) < xtol)
      return {mid, fm, i + 1, true};
    if (fa * fm < 0.0) {
      b = mid;
      fb = fm;
    } else {
      a = mid;
      fa = fm;
    }
  }
  const double mid = 0.5 * (a + b);
  return {mid, f(mid), i, false};
}

root_result brent(const std::function<double(double)>& f, double a, double b,
                  double xtol, int max_iter) {
  double fa = f(a), fb = f(b);
  if (fa == 0.0) return {a, 0.0, 0, true};
  if (fb == 0.0) return {b, 0.0, 0, true};
  if (fa * fb > 0.0)
    throw std::invalid_argument("brent: f(a) and f(b) must differ in sign");
  double c = a, fc = fa;
  double d = b - a, e = d;
  for (int i = 0; i < max_iter; ++i) {
    if (std::abs(fc) < std::abs(fb)) {
      a = b;
      b = c;
      c = a;
      fa = fb;
      fb = fc;
      fc = fa;
    }
    const double tol1 =
        2.0 * std::numeric_limits<double>::epsilon() * std::abs(b) +
        0.5 * xtol;
    const double xm = 0.5 * (c - b);
    if (std::abs(xm) <= tol1 || fb == 0.0) return {b, fb, i + 1, true};
    if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
      // attempt inverse quadratic / secant
      const double s = fb / fa;
      double p, q;
      if (a == c) {
        p = 2.0 * xm * s;
        q = 1.0 - s;
      } else {
        const double qq = fa / fc, r = fb / fc;
        p = s * (2.0 * xm * qq * (qq - r) - (b - a) * (r - 1.0));
        q = (qq - 1.0) * (r - 1.0) * (s - 1.0);
      }
      if (p > 0.0) q = -q;
      p = std::abs(p);
      const double min1 = 3.0 * xm * q - std::abs(tol1 * q);
      const double min2 = std::abs(e * q);
      if (2.0 * p < (min1 < min2 ? min1 : min2)) {
        e = d;
        d = p / q;
      } else {
        d = xm;
        e = d;
      }
    } else {
      d = xm;
      e = d;
    }
    a = b;
    fa = fb;
    b += std::abs(d) > tol1 ? d : (xm > 0.0 ? tol1 : -tol1);
    fb = f(b);
    if ((fb > 0.0) == (fc > 0.0)) {
      c = a;
      fc = fa;
      d = b - a;
      e = d;
    }
  }
  return {b, fb, max_iter, false};
}

root_result newton(const std::function<double(double)>& f,
                   const std::function<double(double)>& df, double x0,
                   double xtol, int max_iter) {
  double x = x0;
  for (int i = 0; i < max_iter; ++i) {
    const double fx = f(x);
    const double d = df(x);
    if (d == 0.0)
      throw std::runtime_error("newton: zero derivative encountered");
    const double step = fx / d;
    x -= step;
    if (std::abs(step) < xtol) return {x, f(x), i + 1, true};
  }
  return {x, f(x), max_iter, false};
}

}  // namespace ax::num
