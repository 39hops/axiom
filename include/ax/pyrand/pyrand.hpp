#pragma once
/** @file pyrand.hpp Bit-exact CPython random.Random (Mersenne Twister).

    Reproduces CPython's `random.Random(seed_string)` stream exactly:
    - string seeding (seed version 2): UTF-8 bytes + their SHA-512 digest,
      interpreted as a big-endian integer, decomposed into little-endian
      32-bit words for MT19937 init_by_array;
    - random() via genrand_res53;
    - getrandbits (<= 64 bits), _randbelow rejection sampling, randint,
      choice (by index), and reversed Fisher-Yates shuffle.

    Mandated by llmopt's reproducibility scar: generator seeds are strings,
    never tuple hashes. Parity fixture: tests/pyrand/fixtures/. */
#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace ax::pyrand {

/** SHA-512 (FIPS 180-4) of a byte string. */
std::array<std::uint8_t, 64> sha512(std::span<const std::uint8_t> data);

class python_random {
 public:
  /** Equivalent to random.Random(seed) with a str seed. */
  explicit python_random(std::string_view seed);

  /** [0, 1) with 53-bit resolution — CPython genrand_res53. */
  double random();
  /** k random bits, k in [1, 64] (std::invalid_argument otherwise). */
  std::uint64_t getrandbits(int k);
  /** Uniform integer in [a, b], both ends inclusive. */
  long long randint(long long a, long long b);
  /** Uniform index in [0, n): CPython choice(seq) = seq[_randbelow(n)]. */
  std::size_t choice_index(std::size_t n);
  /** In-place CPython shuffle (reversed Fisher-Yates). */
  template <class T>
  void shuffle(std::vector<T>& xs) {
    for (std::size_t i = xs.size(); i-- > 1;) {
      const std::size_t j = randbelow(i + 1);
      std::swap(xs[i], xs[j]);
    }
  }

 private:
  std::array<std::uint32_t, 624> mt_{};
  int index_ = 625;

  void init_genrand(std::uint32_t s);
  void init_by_array(std::span<const std::uint32_t> key);
  std::uint32_t genrand();
  std::uint64_t randbelow(std::uint64_t n);
};

}  // namespace ax::pyrand
