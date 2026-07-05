#include <ax/core/bigint.hpp>
#include <ax/par/pool.hpp>

#include <random>
#include <string>

#include "bench.hpp"

namespace {

ax::bigint random_bigint(std::size_t digits, std::mt19937_64& rng) {
  std::string s;
  s.push_back(static_cast<char>('1' + rng() % 9));
  for (std::size_t i = 1; i < digits; ++i)
    s.push_back(static_cast<char>('0' + rng() % 10));
  return ax::bigint{s};
}

}  // namespace

int main() {
  std::mt19937_64 rng{2026};
  const auto a1k = random_bigint(1000, rng), b1k = random_bigint(1000, rng);
  const auto a10k = random_bigint(10000, rng), b10k = random_bigint(10000, rng);
  const auto a100k = random_bigint(100000, rng),
             b100k = random_bigint(100000, rng);

  ax::bench::run("bigint_mul_1k_digits", [&] { auto c = a1k * b1k; (void)c; });
  ax::bench::run("bigint_mul_10k_digits",
                 [&] { auto c = a10k * b10k; (void)c; });
  ax::bench::run("bigint_mul_100k_digits",
                 [&] { auto c = a100k * b100k; (void)c; });
  const auto a500k = random_bigint(500000, rng), b500k = random_bigint(500000, rng);
  ax::bench::run("bigint_mul_500k_digits", [&] { auto c = a500k * b500k; (void)c; });
  ax::bench::run("bigint_div_10k_by_1k", [&] { auto c = a10k / a1k; (void)c; });

  ax::thread_pool pool;
  ax::bench::run("pool_submit_overhead", [&] { pool.submit([] {}).get(); });
  return 0;
}
