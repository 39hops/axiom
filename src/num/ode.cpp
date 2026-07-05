#include <ax/num/ode.hpp>

#include <algorithm>
#include <cmath>

namespace ax::num {

using la::vec;

namespace {

// Dormand-Prince RK5(4) coefficients.
constexpr double kA21 = 1.0 / 5.0;
constexpr double kA31 = 3.0 / 40.0, kA32 = 9.0 / 40.0;
constexpr double kA41 = 44.0 / 45.0, kA42 = -56.0 / 15.0, kA43 = 32.0 / 9.0;
constexpr double kA51 = 19372.0 / 6561.0, kA52 = -25360.0 / 2187.0,
                 kA53 = 64448.0 / 6561.0, kA54 = -212.0 / 729.0;
constexpr double kA61 = 9017.0 / 3168.0, kA62 = -355.0 / 33.0,
                 kA63 = 46732.0 / 5247.0, kA64 = 49.0 / 176.0,
                 kA65 = -5103.0 / 18656.0;
// 5th-order solution weights (also row 7 — FSAL)
constexpr double kB1 = 35.0 / 384.0, kB3 = 500.0 / 1113.0,
                 kB4 = 125.0 / 192.0, kB5 = -2187.0 / 6784.0,
                 kB6 = 11.0 / 84.0;
// error weights: b5 - b4
constexpr double kE1 = kB1 - 5179.0 / 57600.0;
constexpr double kE3 = kB3 - 7571.0 / 16695.0;
constexpr double kE4 = kB4 - 393.0 / 640.0;
constexpr double kE5 = kB5 + 92097.0 / 339200.0;
constexpr double kE6 = kB6 - 187.0 / 2100.0;
constexpr double kE7 = -1.0 / 40.0;

constexpr double kC2 = 1.0 / 5.0, kC3 = 3.0 / 10.0, kC4 = 4.0 / 5.0,
                 kC5 = 8.0 / 9.0;

vec axpy(const vec& y, double h, std::initializer_list<double> cs,
         std::initializer_list<const vec*> ks) {
  vec r = y;
  auto c = cs.begin();
  for (const vec* k : ks) {
    const double hc = h * *c++;
    for (std::size_t i = 0; i < r.size(); ++i) r[i] += hc * (*k)[i];
  }
  return r;
}

}  // namespace

ode_result solve_ivp(const std::function<vec(double, const vec&)>& f,
                     double t0, double t1, vec y0, double reltol,
                     double abstol, double h0) {
  constexpr int kMaxSteps = 1000000;
  ode_result out;
  double t = t0;
  vec y = std::move(y0);
  out.t.push_back(t);
  out.y.push_back(y);
  double h = h0 > 0.0 ? h0 : (t1 - t0) / 100.0;
  vec k1 = f(t, y);  // FSAL: reused across accepted steps

  while (t < t1) {
    if (out.steps + out.rejected >= kMaxSteps) {
      out.converged = false;
      return out;
    }
    h = std::min(h, t1 - t);
    const vec k2 = f(t + kC2 * h, axpy(y, h, {kA21}, {&k1}));
    const vec k3 = f(t + kC3 * h, axpy(y, h, {kA31, kA32}, {&k1, &k2}));
    const vec k4 =
        f(t + kC4 * h, axpy(y, h, {kA41, kA42, kA43}, {&k1, &k2, &k3}));
    const vec k5 = f(t + kC5 * h, axpy(y, h, {kA51, kA52, kA53, kA54},
                                       {&k1, &k2, &k3, &k4}));
    const vec k6 = f(t + h, axpy(y, h, {kA61, kA62, kA63, kA64, kA65},
                                 {&k1, &k2, &k3, &k4, &k5}));
    const vec y1 = axpy(y, h, {kB1, kB3, kB4, kB5, kB6},
                        {&k1, &k3, &k4, &k5, &k6});
    const vec k7 = f(t + h, y1);
    // scaled RMS error norm
    double err2 = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
      const double e = h * (kE1 * k1[i] + kE3 * k3[i] + kE4 * k4[i] +
                            kE5 * k5[i] + kE6 * k6[i] + kE7 * k7[i]);
      const double sc =
          abstol + reltol * std::max(std::abs(y[i]), std::abs(y1[i]));
      err2 += (e / sc) * (e / sc);
    }
    const double err = std::sqrt(err2 / static_cast<double>(y.size()));
    if (err <= 1.0) {
      t += h;
      y = y1;
      k1 = k7;  // FSAL
      out.t.push_back(t);
      out.y.push_back(y);
      ++out.steps;
    } else {
      ++out.rejected;
    }
    const double factor =
        err == 0.0 ? 5.0
                   : std::clamp(0.9 * std::pow(err, -0.2), 0.2, 5.0);
    h *= factor;
  }
  return out;
}

}  // namespace ax::num
