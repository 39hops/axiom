#include <ax/pyrand/pyrand.hpp>

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

using ax::pyrand::python_random;
using ax::pyrand::sha512;

std::string hex(std::span<const std::uint8_t> bytes, std::size_t n) {
  static const char* d = "0123456789abcdef";
  std::string out;
  for (std::size_t i = 0; i < n; ++i) {
    out += d[bytes[i] >> 4];
    out += d[bytes[i] & 0xF];
  }
  return out;
}

std::array<std::uint8_t, 64> sha512_str(std::string_view s) {
  return sha512(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(s.data()), s.size()));
}

// FIPS 180-4 test vectors (from the standard, not from CPython).
TEST(Sha512, FipsVectors) {
  EXPECT_EQ(hex(sha512_str(""), 64),
            "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce"
            "47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e");
  EXPECT_EQ(hex(sha512_str("abc"), 64),
            "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a"
            "2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f");
  EXPECT_EQ(
      hex(sha512_str("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijkl"
                     "mnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqr"
                     "stu"),
          64),
      "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018"
      "501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909");
}

// ---- fixture-driven parity vs CPython

struct fixture {
  // (seed, stream) -> values
  std::map<std::pair<std::string, std::string>, std::vector<std::string>> m;
};

const fixture& load_fixture() {
  static fixture f = [] {
    fixture out;
    std::ifstream in(std::string(AX_SOURCE_DIR) +
                     "/tests/pyrand/fixtures/pyrand_fixture.tsv");
    EXPECT_TRUE(in.good()) << "fixture missing — run "
                              "scripts/gen_pyrand_fixture.py";
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '#') continue;
      const auto t1 = line.find('\t');
      const auto t2 = line.find('\t', t1 + 1);
      std::vector<std::string> vals;
      std::istringstream vs(line.substr(t2 + 1));
      std::string v;
      while (vs >> v) vals.push_back(v);
      out.m[{line.substr(0, t1), line.substr(t1 + 1, t2 - t1 - 1)}] = vals;
    }
    return out;
  }();
  return f;
}

std::vector<std::string> vals(const std::string& seed,
                              const std::string& stream) {
  const auto it = load_fixture().m.find({seed, stream});
  EXPECT_TRUE(it != load_fixture().m.end()) << seed << "/" << stream;
  return it == load_fixture().m.end() ? std::vector<std::string>{}
                                      : it->second;
}

std::vector<std::string> seeds() {
  std::vector<std::string> out;
  for (const auto& [k, v] : load_fixture().m)
    if (k.second == "random") out.push_back(k.first);
  return out;
}

TEST(Pyrand, Sha512OfSeedsMatchesHashlib) {
  for (const auto& s : seeds())
    EXPECT_EQ(hex(sha512_str(s), 16), vals(s, "sha512_prefix").at(0)) << s;
}

TEST(Pyrand, RandomStream) {
  for (const auto& s : seeds()) {
    python_random r(s);
    for (const auto& want : vals(s, "random"))
      EXPECT_EQ(r.random(), std::strtod(want.c_str(), nullptr))
          << s << " " << want;
  }
}

TEST(Pyrand, GetrandbitsStream) {
  const int ks[] = {1, 8, 31, 32, 33, 64};
  for (const auto& s : seeds()) {
    python_random r(s);
    const auto want = vals(s, "getrandbits");
    for (std::size_t i = 0; i < want.size(); ++i)
      EXPECT_EQ(r.getrandbits(ks[i]),
                std::strtoull(want[i].c_str(), nullptr, 10))
          << s << " k=" << ks[i];
  }
}

TEST(Pyrand, RandintChoiceShuffleStreams) {
  for (const auto& s : seeds()) {
    {
      python_random r(s);
      for (const auto& w : vals(s, "randint_1_9"))
        EXPECT_EQ(r.randint(1, 9), std::atoll(w.c_str())) << s;
    }
    {
      python_random r(s);
      for (const auto& w : vals(s, "randint_m4_4"))
        EXPECT_EQ(r.randint(-4, 4), std::atoll(w.c_str())) << s;
    }
    {
      python_random r(s);
      for (const auto& w : vals(s, "choice_5"))
        EXPECT_EQ(r.choice_index(5), std::strtoull(w.c_str(), nullptr, 10))
            << s;
    }
    {
      python_random r(s);
      std::vector<int> xs{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
      r.shuffle(xs);
      const auto want = vals(s, "shuffle_10");
      for (std::size_t i = 0; i < xs.size(); ++i)
        EXPECT_EQ(xs[i], std::atoi(want[i].c_str())) << s << " i=" << i;
    }
  }
}

TEST(Pyrand, MixedCallTrace) {
  // Interleaved calls: catches any divergence in per-call state advance.
  for (const auto& s : seeds()) {
    python_random r(s);
    const auto want = vals(s, "mixed");
    EXPECT_EQ(r.randint(1, 9), std::atoll(want[0].c_str())) << s;
    EXPECT_EQ(r.random(), std::strtod(want[1].c_str(), nullptr)) << s;
    // choice([10,20,30]) recorded the chosen VALUE; index maps 0/1/2.
    const long long got_choice = 10 * (1 + static_cast<long long>(
                                               r.choice_index(3)));
    EXPECT_EQ(got_choice, std::atoll(want[2].c_str())) << s;
    EXPECT_EQ(r.getrandbits(33),
              std::strtoull(want[3].c_str(), nullptr, 10))
        << s;
    EXPECT_EQ(r.randint(-3, 3), std::atoll(want[4].c_str())) << s;
    EXPECT_EQ(r.random(), std::strtod(want[5].c_str(), nullptr)) << s;
  }
}

}  // namespace
