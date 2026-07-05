#include <ax/core/nt.hpp>

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>

namespace ax {

namespace {

/// (a * b) % m without overflow (double-and-add).
std::uint64_t mulmod_u64(std::uint64_t a, std::uint64_t b, std::uint64_t m) {
  std::uint64_t result = 0;
  a %= m;
  while (b) {
    if (b & 1) {
      result += a;
      if (result >= m || result < a) result -= m;
    }
    const std::uint64_t a2 = a + a;
    a = (a2 >= m || a2 < a) ? a2 - m : a2;
    b >>= 1;
  }
  return result;
}

std::uint64_t powmod_u64(std::uint64_t a, std::uint64_t e, std::uint64_t m) {
  std::uint64_t r = 1 % m;
  a %= m;
  while (e) {
    if (e & 1) r = mulmod_u64(r, a, m);
    a = mulmod_u64(a, a, m);
    e >>= 1;
  }
  return r;
}

/// One Miller-Rabin round; true = "probably prime for base a".
bool miller_rabin_u64(std::uint64_t n, std::uint64_t a) {
  if (a % n == 0) return true;
  std::uint64_t d = n - 1;
  int s = 0;
  while ((d & 1) == 0) {
    d >>= 1;
    ++s;
  }
  std::uint64_t x = powmod_u64(a, d, n);
  if (x == 1 || x == n - 1) return true;
  for (int i = 1; i < s; ++i) {
    x = mulmod_u64(x, x, n);
    if (x == n - 1) return true;
  }
  return false;
}

std::uint64_t pollard_rho(std::uint64_t n, std::mt19937_64& rng) {
  if ((n & 1) == 0) return 2;
  for (;;) {
    const std::uint64_t c = rng() % (n - 1) + 1;
    std::uint64_t x = rng() % n, y = x, d = 1;
    auto f = [&](std::uint64_t v) {
      const std::uint64_t s = mulmod_u64(v, v, n) + c;
      return s >= n ? s - n : s;
    };
    while (d == 1) {
      x = f(x);
      y = f(f(y));
      d = std::gcd(x > y ? x - y : y - x, n);
    }
    if (d != n) return d;
  }
}

}  // namespace

bool is_prime(std::uint64_t n) {
  if (n < 2) return false;
  for (std::uint64_t p : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull,
                          23ull, 29ull, 31ull, 37ull}) {
    if (n == p) return true;
    if (n % p == 0) return false;
  }
  // deterministic base set for all n < 2^64
  for (std::uint64_t a : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull,
                          23ull, 29ull, 31ull, 37ull})
    if (!miller_rabin_u64(n, a)) return false;
  return true;
}

bool is_prime(const bigint& n, int rounds) {
  if (n < bigint{2}) return false;
  const bigint one{1}, two{2};
  if (n == two) return true;
  if ((n % two).is_zero()) return false;
  const bigint n1 = n - one;
  bigint d = n1;
  int s = 0;
  while ((d % two).is_zero()) {
    d = d / two;
    ++s;
  }
  std::mt19937_64 rng{0x9e3779b97f4a7c15ull};
  for (int i = 0; i < rounds; ++i) {
    // random base in [2, n-2]
    const bigint a =
        (bigint{static_cast<long long>(rng() >> 1)} % (n - bigint{3})) + two;
    bigint x = modpow(a, d, n);
    if (x == one || x == n1) continue;
    bool witness = true;
    for (int r = 1; r < s; ++r) {
      x = (x * x) % n;
      if (x == n1) {
        witness = false;
        break;
      }
    }
    if (witness) return false;
  }
  return true;
}

std::vector<std::uint64_t> factor(std::uint64_t n) {
  std::vector<std::uint64_t> out;
  if (n < 2) return out;
  for (std::uint64_t p : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull}) {
    while (n % p == 0) {
      out.push_back(p);
      n /= p;
    }
  }
  std::mt19937_64 rng{42};
  std::vector<std::uint64_t> stack{n};
  while (!stack.empty()) {
    const std::uint64_t m = stack.back();
    stack.pop_back();
    if (m == 1) continue;
    if (is_prime(m)) {
      out.push_back(m);
      continue;
    }
    const std::uint64_t d = pollard_rho(m, rng);
    stack.push_back(d);
    stack.push_back(m / d);
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::tuple<bigint, bigint, bigint> ext_gcd(const bigint& a, const bigint& b) {
  if (b.is_zero()) return {a, bigint{1}, bigint{0}};
  auto [q, r] = bigint::divmod(a, b);
  auto [g, x, y] = ext_gcd(b, r);
  return {g, y, x - q * y};
}

bigint modinv(const bigint& a, const bigint& m) {
  auto [g, x, y] = ext_gcd(a, m);
  (void)y;
  if (g != bigint{1} && g != bigint{-1})
    throw std::domain_error("modinv: not coprime");
  bigint r = x % m;
  if (r.is_negative()) r = r + m;
  return r;
}

std::pair<bigint, bigint> crt(
    const std::vector<std::pair<bigint, bigint>>& congruences) {
  if (congruences.empty()) throw std::domain_error("crt: empty input");
  bigint x = congruences[0].first % congruences[0].second;
  if (x.is_negative()) x = x + congruences[0].second;
  bigint m = congruences[0].second;
  for (std::size_t i = 1; i < congruences.size(); ++i) {
    const auto& [ri, mi] = congruences[i];
    if (gcd(m, mi) != bigint{1})
      throw std::domain_error("crt: moduli not coprime");
    // x' = x + m * ((ri - x) * modinv(m, mi) mod mi)
    bigint diff = (ri - x) % mi;
    if (diff.is_negative()) diff = diff + mi;
    const bigint t = (diff * modinv(m, mi)) % mi;
    x = x + m * t;
    m = m * mi;
  }
  return {x % m, m};
}

}  // namespace ax
