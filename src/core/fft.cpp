#include <ax/core/fft.hpp>

#include <cmath>
#include <numbers>
#include <utility>

namespace ax {

void fft(std::vector<std::complex<double>>& a, bool invert) {
  const std::size_t n = a.size();
  for (std::size_t i = 1, j = 0; i < n; ++i) {
    std::size_t bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }
  for (std::size_t len = 2; len <= n; len <<= 1) {
    const double ang =
        2 * std::numbers::pi / static_cast<double>(len) * (invert ? -1 : 1);
    const std::complex<double> wl{std::cos(ang), std::sin(ang)};
    for (std::size_t i = 0; i < n; i += len) {
      std::complex<double> w{1, 0};
      for (std::size_t j = 0; j < len / 2; ++j) {
        const auto u = a[i + j], v = a[i + j + len / 2] * w;
        a[i + j] = u + v;
        a[i + j + len / 2] = u - v;
        w *= wl;
      }
    }
  }
  if (invert)
    for (auto& x : a) x /= static_cast<double>(n);
}

std::vector<double> convolve(const std::vector<double>& a,
                             const std::vector<double>& b) {
  if (a.empty() || b.empty()) return {};
  const std::size_t rs = a.size() + b.size() - 1;
  std::size_t n = 1;
  while (n < rs) n <<= 1;
  std::vector<std::complex<double>> fa(n), fb(n);
  for (std::size_t i = 0; i < a.size(); ++i) fa[i] = a[i];
  for (std::size_t i = 0; i < b.size(); ++i) fb[i] = b[i];
  fft(fa, false);
  fft(fb, false);
  for (std::size_t i = 0; i < n; ++i) fa[i] *= fb[i];
  fft(fa, true);
  std::vector<double> r(rs);
  for (std::size_t i = 0; i < rs; ++i) r[i] = fa[i].real();
  return r;
}

namespace {

std::uint64_t pm(std::uint64_t a, std::uint64_t e, std::uint64_t m) {
  std::uint64_t r = 1;
  a %= m;
  while (e) {
    if (e & 1) r = r * a % m;
    a = a * a % m;
    e >>= 1;
  }
  return r;
}

/// In-place NTT mod prime (root 3), size power of two dividing mod-1.
void ntt(std::vector<std::uint64_t>& a, bool invert, std::uint64_t mod) {
  const std::size_t n = a.size();
  for (std::size_t i = 1, j = 0; i < n; ++i) {
    std::size_t bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }
  for (std::size_t len = 2; len <= n; len <<= 1) {
    std::uint64_t wl = pm(3, (mod - 1) / len, mod);
    if (invert) wl = pm(wl, mod - 2, mod);
    for (std::size_t i = 0; i < n; i += len) {
      std::uint64_t w = 1;
      for (std::size_t j = 0; j < len / 2; ++j) {
        const std::uint64_t u = a[i + j];
        const std::uint64_t v = a[i + j + len / 2] * w % mod;
        a[i + j] = (u + v) % mod;
        a[i + j + len / 2] = (u + mod - v) % mod;
        w = w * wl % mod;
      }
    }
  }
  if (invert) {
    const std::uint64_t ninv = pm(n % mod, mod - 2, mod);
    for (auto& x : a) x = x * ninv % mod;
  }
}

}  // namespace

std::vector<std::uint64_t> ntt_convolve(const std::vector<std::uint64_t>& a,
                                        const std::vector<std::uint64_t>& b,
                                        std::uint64_t mod) {
  if (a.empty() || b.empty()) return {};
  const std::size_t rs = a.size() + b.size() - 1;
  std::size_t n = 1;
  while (n < rs) n <<= 1;
  std::vector<std::uint64_t> fa(n, 0), fb(n, 0);
  for (std::size_t i = 0; i < a.size(); ++i) fa[i] = a[i] % mod;
  for (std::size_t i = 0; i < b.size(); ++i) fb[i] = b[i] % mod;
  ntt(fa, false, mod);
  ntt(fb, false, mod);
  for (std::size_t i = 0; i < n; ++i) fa[i] = fa[i] * fb[i] % mod;
  ntt(fa, true, mod);
  fa.resize(rs);
  return fa;
}

}  // namespace ax
