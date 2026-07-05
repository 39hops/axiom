#include <ax/num/opt.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace ax::num {

using la::vec;

min_result_1d minimize(const std::function<double(double)>& f, double a,
                       double b, double xtol, int max_iter) {
  // Brent: golden section with parabolic interpolation (NR-style).
  constexpr double kGold = 0.3819660112501051;
  double x = a + kGold * (b - a), w = x, v = x;
  double fx = f(x), fw = fx, fv = fx;
  double d = 0.0, e = 0.0;
  for (int i = 0; i < max_iter; ++i) {
    const double xm = 0.5 * (a + b);
    const double tol1 = xtol * std::abs(x) + 1e-15;
    const double tol2 = 2.0 * tol1;
    if (std::abs(x - xm) <= tol2 - 0.5 * (b - a))
      return {x, fx, i + 1, true};
    bool parabolic = false;
    if (std::abs(e) > tol1) {
      const double r = (x - w) * (fx - fv);
      double q = (x - v) * (fx - fw);
      double p = (x - v) * q - (x - w) * r;
      q = 2.0 * (q - r);
      if (q > 0.0) p = -p;
      q = std::abs(q);
      const double e_old = e;
      e = d;
      if (std::abs(p) < std::abs(0.5 * q * e_old) && p > q * (a - x) &&
          p < q * (b - x)) {
        d = p / q;
        const double u = x + d;
        if (u - a < tol2 || b - u < tol2) d = xm > x ? tol1 : -tol1;
        parabolic = true;
      }
    }
    if (!parabolic) {
      e = (x >= xm ? a : b) - x;
      d = kGold * e;
    }
    const double u = std::abs(d) >= tol1 ? x + d : x + (d > 0.0 ? tol1 : -tol1);
    const double fu = f(u);
    if (fu <= fx) {
      if (u >= x)
        a = x;
      else
        b = x;
      v = w;
      fv = fw;
      w = x;
      fw = fx;
      x = u;
      fx = fu;
    } else {
      if (u < x)
        a = u;
      else
        b = u;
      if (fu <= fw || w == x) {
        v = w;
        fv = fw;
        w = u;
        fw = fu;
      } else if (fu <= fv || v == x || v == w) {
        v = u;
        fv = fu;
      }
    }
  }
  return {x, fx, max_iter, false};
}

min_result nelder_mead(const std::function<double(const vec&)>& f, vec x0,
                       double ftol, int max_iter) {
  const std::size_t n = x0.size();
  std::vector<vec> s(n + 1, x0);
  std::vector<double> fs(n + 1);
  for (std::size_t i = 0; i < n; ++i) {
    const double step = x0[i] != 0.0 ? 0.05 * std::abs(x0[i]) : 0.00025;
    s[i + 1][i] += step;
  }
  for (std::size_t i = 0; i <= n; ++i) fs[i] = f(s[i]);

  const auto order = [&] {
    for (std::size_t i = 1; i <= n; ++i)
      for (std::size_t j = i; j > 0 && fs[j] < fs[j - 1]; --j) {
        std::swap(fs[j], fs[j - 1]);
        std::swap(s[j], s[j - 1]);
      }
  };
  order();

  for (int it = 0; it < max_iter; ++it) {
    const double spread =
        2.0 * std::abs(fs[n] - fs[0]) /
        (std::abs(fs[n]) + std::abs(fs[0]) + 1e-10);  // floor: minima at 0
    if (spread < ftol) return {s[0], fs[0], it, true};

    vec centroid(n);
    for (std::size_t i = 0; i < n; ++i) {
      double c = 0.0;
      for (std::size_t j = 0; j < n; ++j) c += s[j][i];
      centroid[i] = c / static_cast<double>(n);
    }
    const auto point = [&](double coef) {
      vec p(n);
      for (std::size_t i = 0; i < n; ++i)
        p[i] = centroid[i] + coef * (s[n][i] - centroid[i]);
      return p;
    };
    const vec xr = point(-1.0);  // reflection
    const double fr = f(xr);
    if (fr < fs[0]) {
      const vec xe = point(-2.0);  // expansion
      const double fe = f(xe);
      if (fe < fr) {
        s[n] = xe;
        fs[n] = fe;
      } else {
        s[n] = xr;
        fs[n] = fr;
      }
    } else if (fr < fs[n - 1]) {
      s[n] = xr;
      fs[n] = fr;
    } else {
      const vec xc = point(fr < fs[n] ? -0.5 : 0.5);  // contraction
      const double fc = f(xc);
      if (fc < std::min(fr, fs[n])) {
        s[n] = xc;
        fs[n] = fc;
      } else {
        // shrink toward best
        for (std::size_t i = 1; i <= n; ++i) {
          for (std::size_t j = 0; j < n; ++j)
            s[i][j] = s[0][j] + 0.5 * (s[i][j] - s[0][j]);
          fs[i] = f(s[i]);
        }
      }
    }
    order();
  }
  return {s[0], fs[0], max_iter, false};
}

namespace {

double max_abs(const vec& g) {
  double m = 0.0;
  for (std::size_t i = 0; i < g.size(); ++i)
    m = std::max(m, std::abs(g[i]));
  return m;
}

}  // namespace

min_result bfgs(const std::function<double(const vec&)>& f,
                const std::function<vec(const vec&)>& grad, vec x0,
                double gtol, int max_iter) {
  const std::size_t n = x0.size();
  vec x = std::move(x0);
  double fx = f(x);
  vec g = grad(x);
  // inverse Hessian approximation, starts as identity
  la::mat h_inv(n, n);
  for (std::size_t i = 0; i < n; ++i) h_inv(i, i) = 1.0;

  for (int it = 0; it < max_iter; ++it) {
    if (max_abs(g) < gtol) return {x, fx, it, true};
    // direction p = -H g
    vec p(n);
    for (std::size_t i = 0; i < n; ++i) {
      double s = 0.0;
      for (std::size_t j = 0; j < n; ++j) s += h_inv(i, j) * g[j];
      p[i] = -s;
    }
    // Armijo backtracking
    const double slope = dot(g, p);
    double alpha = 1.0;
    double fn = 0.0;
    vec xn(n);
    for (int ls = 0; ls < 60; ++ls) {
      xn = x + alpha * p;
      fn = f(xn);
      if (fn <= fx + 1e-4 * alpha * slope) break;
      alpha *= 0.5;
    }
    const vec gn = grad(xn);
    const vec s_vec = xn - x;
    const vec y_vec = gn - g;
    const double sy = dot(s_vec, y_vec);
    if (sy > 1e-12) {
      // BFGS inverse update: H = (I - r s yᵀ) H (I - r y sᵀ) + r s sᵀ
      const double rho = 1.0 / sy;
      // t = H y
      vec t(n);
      for (std::size_t i = 0; i < n; ++i) {
        double acc = 0.0;
        for (std::size_t j = 0; j < n; ++j) acc += h_inv(i, j) * y_vec[j];
        t[i] = acc;
      }
      const double yty_h = dot(y_vec, t);
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
          h_inv(i, j) += rho * rho * yty_h * s_vec[i] * s_vec[j] +
                         rho * s_vec[i] * s_vec[j] -
                         rho * (s_vec[i] * t[j] + t[i] * s_vec[j]);
    }
    x = xn;
    fx = fn;
    g = gn;
  }
  return {x, fx, max_iter, max_abs(g) < gtol};
}

min_result bfgs(const std::function<double(const vec&)>& f, vec x0,
                double gtol, int max_iter) {
  const auto grad = [&f](const vec& x) {
    const double heps = std::cbrt(std::numeric_limits<double>::epsilon());
    vec g(x.size());
    vec xp = x;
    for (std::size_t i = 0; i < x.size(); ++i) {
      const double h = heps * std::max(1.0, std::abs(x[i]));
      const double orig = xp[i];
      xp[i] = orig + h;
      const double fp = f(xp);
      xp[i] = orig - h;
      const double fm = f(xp);
      xp[i] = orig;
      g[i] = (fp - fm) / (2.0 * h);
    }
    return g;
  };
  return bfgs(f, grad, std::move(x0), gtol, max_iter);
}

}  // namespace ax::num
