/** @file c1_chain_oracle.cpp RNSCHAIN C1: CPU depth-chain oracle.
 *
 * Depth-L exact matrix chain, L in {2,4,6,8,12}: each layer permutes the
 * columns of the running product by a pinned K-permutation, then multiplies
 * by that layer's weight matrix. Two independent arms compute the chain:
 *
 *   reference arm  - ax::bigint matrices, exact.
 *   rns arm        - ax::rns channels over pinned 61-bit primes, CRT exit.
 *
 * Oracle bar: entry-wise bigint equality at EVERY depth in the ladder, per
 * adversarial input class. Any mismatch throws (no soft logging).
 *
 * Chain-specific bar: K-permutation bit-identity at depth 1 and depth 8 -
 * the permutation composed 8 times via a precomposed table must equal 8
 * iterated applications, index-for-index.
 *
 * Input contract (_f24): all matrix entries are integers in
 * [-2^23, 2^23 - 1] - exactly representable in an fp32/fp64 significand.
 * Stated in the receipt header.
 *
 * Receipt: JSONL on stdout. Digests are FNV-1a over the decimal strings of
 * the final-depth entries (stdlib-independent, matches repo fixture ethos).
 */
#include <ax/core/bigint.hpp>
#include <ax/core/nt.hpp>
#include <ax/core/rns.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using ax::bigint;
using std::uint64_t;

constexpr int kN = 32;                 // matrix side
constexpr std::size_t kChannels = 8;   // 8 x 61 bits = 488-bit capacity
constexpr std::array<int, 5> kDepths{2, 4, 6, 8, 12};
constexpr int kMaxDepth = 12;
constexpr uint64_t kSeed = 0x20260810524e53ULL;
constexpr long long kF24Max = (1LL << 23) - 1;
constexpr long long kF24Min = -(1LL << 23);

// Worst-case growth: 24-bit entries, +(24 + log2 32) bits per layer:
// 24 + 12*29 = 372 bits < 488-bit RNS capacity (centered lift halves it to
// 487; still clear). The bigint arm would catch any wrap regardless.

uint64_t splitmix64(uint64_t& s) {
  s += 0x9e3779b97f4a7c15ULL;
  uint64_t z = s;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

long long rand_f24(uint64_t& s) {
  return static_cast<long long>(splitmix64(s) % (1ULL << 24)) + kF24Min;
}

// signed 64-bit entry -> residue in [0, p); integer-chain entry point on
// top of the anchor-v2 ring core (which is rational-facing via res_of)
uint64_t to_residue(long long v, uint64_t p) {
  if (v >= 0) return static_cast<uint64_t>(v) % p;
  uint64_t m = (static_cast<uint64_t>(-(v + 1)) + 1) % p;  // |v| mod p, no UB
  return m == 0 ? 0 : p - m;
}

// centered CRT exit: residues -> bigint in [-M/2, M/2), via ax::crt
bigint from_residues(const std::vector<uint64_t>& residues,
                     const std::vector<uint64_t>& primes) {
  std::vector<std::pair<bigint, bigint>> cong;
  cong.reserve(primes.size());
  for (std::size_t i = 0; i < primes.size(); ++i)
    cong.emplace_back(bigint(static_cast<long long>(residues[i])),
                      bigint(static_cast<long long>(primes[i])));
  auto [x, m] = ax::crt(cong);
  if ((x + x) >= m) x = x - m;
  return x;
}

using Mat = std::vector<long long>;  // kN*kN row-major

struct Class {
  const char* name;
  Mat x;
};

std::vector<int> pinned_permutation(uint64_t seed) {
  std::vector<int> pi(kN);
  for (int i = 0; i < kN; ++i) pi[i] = i;
  for (int i = kN - 1; i > 0; --i) {
    int j = static_cast<int>(splitmix64(seed) %
                             static_cast<uint64_t>(i + 1));
    std::swap(pi[i], pi[j]);
  }
  return pi;
}

// --- reference arm -----------------------------------------------------

using BMat = std::vector<bigint>;

BMat permute_cols(const BMat& y, const std::vector<int>& pi) {
  BMat out(kN * kN);
  for (int r = 0; r < kN; ++r)
    for (int c = 0; c < kN; ++c) out[r * kN + c] = y[r * kN + pi[c]];
  return out;
}

BMat bmul(const BMat& a, const Mat& w) {
  BMat out(kN * kN);
  for (int r = 0; r < kN; ++r)
    for (int c = 0; c < kN; ++c) {
      bigint acc;
      for (int k = 0; k < kN; ++k)
        acc = acc + a[r * kN + k] * bigint(w[k * kN + c]);
      out[r * kN + c] = std::move(acc);
    }
  return out;
}

// --- rns arm -----------------------------------------------------------

using RMat = std::vector<std::vector<uint64_t>>;  // [channel][entry]

RMat to_rns(const Mat& m, const std::vector<uint64_t>& primes) {
  RMat out(primes.size(), std::vector<uint64_t>(kN * kN));
  for (std::size_t ch = 0; ch < primes.size(); ++ch)
    for (int i = 0; i < kN * kN; ++i)
      out[ch][i] = to_residue(m[i], primes[ch]);
  return out;
}

RMat rns_permute_cols(const RMat& y, const std::vector<int>& pi) {
  RMat out(y.size(), std::vector<uint64_t>(kN * kN));
  for (std::size_t ch = 0; ch < y.size(); ++ch)
    for (int r = 0; r < kN; ++r)
      for (int c = 0; c < kN; ++c)
        out[ch][r * kN + c] = y[ch][r * kN + pi[c]];
  return out;
}

RMat rns_mul(const RMat& a, const RMat& w, const std::vector<uint64_t>& ps) {
  RMat out(ps.size(), std::vector<uint64_t>(kN * kN));
  for (std::size_t ch = 0; ch < ps.size(); ++ch) {
    uint64_t p = ps[ch];
    for (int r = 0; r < kN; ++r)
      for (int c = 0; c < kN; ++c) {
        uint64_t acc = 0;
        for (int k = 0; k < kN; ++k)
          acc = ax::rns::addm(
              acc, ax::rns::mulm(a[ch][r * kN + k], w[ch][k * kN + c], p),
              p);
        out[ch][r * kN + c] = acc;
      }
  }
  return out;
}

// --- checks ------------------------------------------------------------

void check_equal(const BMat& ref, const RMat& rns,
                 const std::vector<uint64_t>& primes, const char* cls,
                 int depth) {
  std::vector<uint64_t> res(primes.size());
  for (int i = 0; i < kN * kN; ++i) {
    for (std::size_t ch = 0; ch < primes.size(); ++ch) res[ch] = rns[ch][i];
    if (from_residues(res, primes) != ref[i])
      throw std::runtime_error(std::string("C1 MISMATCH class=") + cls +
                               " depth=" + std::to_string(depth) +
                               " index=" + std::to_string(i));
  }
}

void check_perm_bitidentity(const std::vector<int>& pi) {
  // iterated arm: apply pi 8 times
  std::vector<int> iter(kN);
  for (int i = 0; i < kN; ++i) iter[i] = i;
  for (int d = 1; d <= 8; ++d) {
    std::vector<int> next(kN);
    for (int i = 0; i < kN; ++i) next[i] = iter[pi[i]];
    iter = std::move(next);
    if (d == 1 && iter != pi)
      throw std::runtime_error("C1 PERM depth=1 bit-identity failed");
  }
  // precomposed arm: square-and-compose to pi^8
  std::vector<int> p2(kN), p4(kN), p8(kN);
  for (int i = 0; i < kN; ++i) p2[i] = pi[pi[i]];
  for (int i = 0; i < kN; ++i) p4[i] = p2[p2[i]];
  for (int i = 0; i < kN; ++i) p8[i] = p4[p4[i]];
  if (p8 != iter)
    throw std::runtime_error("C1 PERM depth=8 bit-identity failed");
}

uint64_t fnv1a(const BMat& m) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (const auto& v : m)
    for (char c : v.to_string()) {
      h ^= static_cast<unsigned char>(c);
      h *= 0x100000001b3ULL;
    }
  return h;
}

std::vector<Class> build_classes(uint64_t& s) {
  std::vector<Class> cs;
  Mat m(kN * kN);
  for (auto& v : m) v = rand_f24(s);
  cs.push_back({"random", m});
  for (auto& v : m) v = kF24Max;
  cs.push_back({"all_max", m});
  for (auto& v : m) v = kF24Min;
  cs.push_back({"all_min", m});
  for (int i = 0; i < kN * kN; ++i) m[i] = (i % 2 == 0) ? kF24Max : kF24Min;
  cs.push_back({"alt_sign_extreme", m});
  for (int i = 0; i < kN * kN; ++i)
    m[i] = (splitmix64(s) % 8 == 0) ? rand_f24(s) : 0;
  cs.push_back({"sparse_zero", m});
  for (int i = 0; i < kN * kN; ++i) m[i] = (i / kN == i % kN) ? 1 : 0;
  cs.push_back({"identity_perm_stress", m});
  return cs;
}

}  // namespace

int main() {
  auto primes = ax::rns::ctx::make(kChannels).P;
  auto pi = pinned_permutation(kSeed);
  check_perm_bitidentity(pi);

  std::printf(
      "{\"receipt\":\"rnschain-c1\",\"n\":%d,\"channels\":%zu,"
      "\"seed\":\"0x%llx\",\"input_contract\":\"_f24: all entries are "
      "integers in [-2^23, 2^23-1], exactly representable in an fp32/fp64 "
      "significand\",\"perm_bitidentity\":[1,8],\"primes\":[",
      kN, kChannels, static_cast<unsigned long long>(kSeed));
  for (std::size_t i = 0; i < primes.size(); ++i)
    std::printf("%s%llu", i ? "," : "",
                static_cast<unsigned long long>(primes[i]));
  std::printf("]}\n");

  uint64_t s = kSeed;
  auto classes = build_classes(s);
  // per-layer weights shared across classes so runs are cross-comparable
  std::vector<Mat> weights(kMaxDepth, Mat(kN * kN));
  for (auto& w : weights)
    for (auto& v : w) v = rand_f24(s);
  std::vector<RMat> weights_rns;
  weights_rns.reserve(kMaxDepth);
  for (const auto& w : weights) weights_rns.push_back(to_rns(w, primes));

  for (const auto& cls : classes) {
    auto t0 = std::chrono::steady_clock::now();
    BMat ref(kN * kN);
    for (int i = 0; i < kN * kN; ++i) ref[i] = bigint(cls.x[i]);
    RMat rns = to_rns(cls.x, primes);
    std::size_t next_depth = 0;
    for (int d = 1; d <= kMaxDepth; ++d) {
      ref = bmul(permute_cols(ref, pi), weights[d - 1]);
      rns = rns_mul(rns_permute_cols(rns, pi), weights_rns[d - 1], primes);
      if (next_depth < kDepths.size() && kDepths[next_depth] == d) {
        check_equal(ref, rns, primes, cls.name, d);
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
        std::printf(
            "{\"class\":\"%s\",\"depth\":%d,\"pass\":true,"
            "\"digest\":\"%016llx\",\"cum_ms\":%.1f}\n",
            cls.name, d, static_cast<unsigned long long>(fnv1a(ref)), ms);
        ++next_depth;
      }
    }
  }
  std::printf("{\"receipt\":\"rnschain-c1\",\"verdict\":\"PASS\"}\n");
  return 0;
}
