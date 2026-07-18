#include <ax/pyrand/pyrand.hpp>

#include <bit>
#include <stdexcept>

namespace ax::pyrand {

namespace {
constexpr std::uint32_t kN = 624;
constexpr std::uint32_t kM = 397;
constexpr std::uint32_t kMatrixA = 0x9908b0df;
constexpr std::uint32_t kUpperMask = 0x80000000;
constexpr std::uint32_t kLowerMask = 0x7fffffff;
}  // namespace

void python_random::init_genrand(std::uint32_t s) {
  mt_[0] = s;
  for (std::uint32_t i = 1; i < kN; ++i)
    mt_[i] = 1812433253u * (mt_[i - 1] ^ (mt_[i - 1] >> 30)) + i;
  index_ = kN;
}

void python_random::init_by_array(std::span<const std::uint32_t> key) {
  init_genrand(19650218u);
  std::size_t i = 1, j = 0;
  std::size_t k = key.size() > kN ? key.size() : kN;
  for (; k != 0; --k) {
    mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1664525u)) +
             key[j] + static_cast<std::uint32_t>(j);
    ++i; ++j;
    if (i >= kN) { mt_[0] = mt_[kN - 1]; i = 1; }
    if (j >= key.size()) j = 0;
  }
  for (k = kN - 1; k != 0; --k) {
    mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1566083941u)) -
             static_cast<std::uint32_t>(i);
    ++i;
    if (i >= kN) { mt_[0] = mt_[kN - 1]; i = 1; }
  }
  mt_[0] = 0x80000000u;
  index_ = kN;
}

python_random::python_random(std::string_view seed) {
  // CPython seed version 2 for str: key = utf8(seed) + sha512(utf8(seed)),
  // interpreted as a big-endian integer, split into little-endian 32-bit
  // words for init_by_array. Reversing the byte string gives exactly that
  // little-endian word decomposition.
  std::vector<std::uint8_t> bytes(seed.begin(), seed.end());
  const auto digest =
      sha512(std::span<const std::uint8_t>(bytes.data(), bytes.size()));
  bytes.insert(bytes.end(), digest.begin(), digest.end());

  std::vector<std::uint32_t> key;
  key.reserve(bytes.size() / 4 + 1);
  // bytes is big-endian; word i (little-endian order) covers bytes
  // [size-4(i+1), size-4i) with in-word little-endian weighting reversed:
  for (std::size_t taken = 0; taken < bytes.size();) {
    std::uint32_t w = 0;
    for (int b = 0; b < 4 && taken < bytes.size(); ++b, ++taken)
      w |= static_cast<std::uint32_t>(bytes[bytes.size() - 1 - taken])
           << (8 * b);
    key.push_back(w);
  }
  // CPython strips leading zero limbs of the integer (its _PyLong storage
  // never keeps them); an all-zero value keys as a single zero word.
  while (key.size() > 1 && key.back() == 0) key.pop_back();
  init_by_array(key);
}

std::uint32_t python_random::genrand() {
  if (index_ >= static_cast<int>(kN)) {
    for (std::uint32_t i = 0; i < kN; ++i) {
      const std::uint32_t y =
          (mt_[i] & kUpperMask) | (mt_[(i + 1) % kN] & kLowerMask);
      mt_[i] = mt_[(i + kM) % kN] ^ (y >> 1) ^ ((y & 1u) ? kMatrixA : 0u);
    }
    index_ = 0;
  }
  std::uint32_t y = mt_[static_cast<std::size_t>(index_++)];
  y ^= y >> 11;
  y ^= (y << 7) & 0x9d2c5680u;
  y ^= (y << 15) & 0xefc60000u;
  y ^= y >> 18;
  return y;
}

double python_random::random() {
  const std::uint32_t a = genrand() >> 5;
  const std::uint32_t b = genrand() >> 6;
  return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

std::uint64_t python_random::getrandbits(int k) {
  if (k < 1 || k > 64)
    throw std::invalid_argument("getrandbits: k must be in [1, 64]");
  if (k <= 32) return genrand() >> (32 - k);
  // CPython fills 32-bit words little-endian; the last word keeps k%32
  // (or 32) top bits.
  const std::uint64_t lo = genrand();
  const int rem = k - 32;
  const std::uint64_t hi = genrand() >> (32 - rem);
  return lo | (hi << 32);
}

std::uint64_t python_random::randbelow(std::uint64_t n) {
  if (n == 0) return 0;  // CPython returns 0 for n == 0
  const int k = std::bit_width(n);
  std::uint64_t r = getrandbits(k);
  while (r >= n) r = getrandbits(k);
  return r;
}

long long python_random::randint(long long a, long long b) {
  if (b < a) throw std::invalid_argument("randint: empty range");
  return a + static_cast<long long>(
                 randbelow(static_cast<std::uint64_t>(b - a + 1)));
}

std::size_t python_random::choice_index(std::size_t n) {
  if (n == 0) throw std::invalid_argument("choice: empty sequence");
  return static_cast<std::size_t>(randbelow(n));
}

}  // namespace ax::pyrand
