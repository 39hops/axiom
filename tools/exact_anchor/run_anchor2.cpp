/** @file run_anchor2.cpp ANCHOR-V2 driver (PRE-REG ANCHOR-V2):
    runs the gcd-free RNS+shadow anchor, streaming one JSONL row per
    step (loss, traj digest, pin-4 fallback counters, wall) — the
    P-HORIZON receipt. Optionally dumps per-step de-grained w9
    snapshots for byte-compare against the exact anchor's booked
    step-1 dump (P-DIGEST-EQUAL on the real cells).

    Usage:
      run_anchor2 STEPS NPRIMES               (tiny d8 fixture)
      run_anchor2 STEPS NPRIMES tables.bin init.bin [dumpdir]
                                              (d64-class r2b inputs)

    Build (from repo root; tool, not part of the suite):
      c++ -std=c++20 -O2 -Iinclude tools/exact_anchor/run_anchor2.cpp \
        build/libaxiom.a -o /tmp/run_anchor2 */
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <ax/nn/exact_anchor2.hpp>
#include <ax/nn/intbirth_core.hpp>

namespace core = ax::nn::ib::core;
namespace a2 = ax::nn::ib::anchor2;
using i64 = std::int64_t;

// ---- tiny fixture (exact_anchor_test.cpp, raw-engine draw, seed 613)
static void put_u16(std::string& b, std::uint16_t v) {
  b.append(reinterpret_cast<const char*>(&v), 2);
}
static void put_u32(std::string& b, std::uint32_t v) {
  b.append(reinterpret_cast<const char*>(&v), 4);
}
static void put_u64(std::string& b, std::uint64_t v) {
  b.append(reinterpret_cast<const char*>(&v), 8);
}
static void put_tensor(std::string& b, const std::string& name,
                       const std::vector<i64>& v) {
  put_u16(b, std::uint16_t(name.size()));
  b.append(name);
  b.push_back(char(1));
  put_u64(b, v.size());
  b.append(reinterpret_cast<const char*>(v.data()), v.size() * 8);
}
static core::birth_cfg_t tiny_cfg() {
  return {4, 8, 4, 16, 8, 12, 9, 256, 8192, 16384, 42950, 1, 1000};
}
static std::string tiny_tables(int T, int DH) {
  const i64 ts = 100, tse = 50, RS = i64{1} << 14;
  std::vector<i64> sil(2 * ts + 1), dsl(2 * ts + 1), ex(tse + 1);
  for (i64 i = 0; i < i64(sil.size()); i++) sil[i] = (i - ts) / 2;
  for (i64 i = 0; i < i64(dsl.size()); i++) dsl[i] = 256;
  for (i64 i = 0; i < i64(ex.size()); i++) ex[i] = 1 + i * 7;
  std::vector<i64> rc(std::size_t(T) * (DH / 2), RS);
  std::vector<i64> rs(std::size_t(T) * (DH / 2), 0);
  std::string b("AXP3", 4);
  put_u32(b, 5);
  put_tensor(b, "silu.tab", sil);
  put_tensor(b, "dsilu.tab", dsl);
  put_tensor(b, "exp.tab", ex);
  put_tensor(b, "rope.cos", rc);
  put_tensor(b, "rope.sin", rs);
  return b;
}
static std::string tiny_init(const core::birth_cfg_t& c) {
  std::mt19937_64 rng(613);
  const auto d = [](std::mt19937_64& r) { return i64(r() % 513) - 256; };
  const std::size_t sizes[11] = {
      std::size_t(c.DH) * c.D, std::size_t(c.DH) * c.D,
      std::size_t(c.DH) * c.D, std::size_t(c.D) * c.DH,
      std::size_t(c.F) * c.D,  std::size_t(c.F) * c.D,
      std::size_t(c.D) * c.F,  std::size_t(c.V) * c.D,
      std::size_t(c.D),        std::size_t(c.D),
      std::size_t(c.D)};
  std::string b;
  for (std::size_t n : sizes)
    for (std::size_t i = 0; i < n; i++)
      put_u64(b, std::uint64_t(d(rng)));
  for (int i = 0; i < c.T * c.D; i++)
    put_u64(b, std::uint64_t(d(rng)));
  for (int t = 0; t < c.T; t++) put_u64(b, std::uint64_t(t % c.V));
  return b;
}

static std::string slurp(const char* p) {
  FILE* f = std::fopen(p, "rb");
  if (!f) { std::perror(p); std::exit(1); }
  std::string b;
  char buf[65536];
  for (std::size_t n; (n = std::fread(buf, 1, sizeof buf, f));)
    b.append(buf, n);
  std::fclose(f);
  return b;
}

int main(int argc, char** argv) {
  const int steps = argc > 1 ? std::atoi(argv[1]) : 12;
  const int nprimes = argc > 2 ? std::atoi(argv[2]) : 256;
  core::birth_cfg_t c = tiny_cfg();
  std::string tb, in;
  if (argc > 4) {  // d64-class r2b contract defaults (ib::contract)
    c = {32, 64, 16, 128, 64, 12, 9, 256, 8192, 16384, 42950, 1, 1000};
    tb = slurp(argv[3]);
    in = slurp(argv[4]);
  } else {
    tb = tiny_tables(c.T, c.DH);
    in = tiny_init(c);
  }
  const char* dump = argc > 5 ? argv[5] : nullptr;

  a2::rx::init(nprimes, 160);
  core::birth_impl<a2::rx, a2::rx, a2::Exact2> b(tb, in, c);
  for (int s = 1; s <= steps; ++s) {
    ax::dyi::prec = 120 + 40 * s;  // pin 3: growing shadow
    const auto t0 = std::chrono::steady_clock::now();
    b.run(1);
    const std::string dig = b.mark();
    const double dt =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      t0)
            .count();
    const auto& fb = a2::rx::fb;
    std::printf(
        "{\"step\":%d,\"loss\":%lld,\"digest\":\"%s\",\"fb\":{"
        "\"eq_zero\":%ld,\"floor_exact\":%ld,\"floor_near\":%ld,"
        "\"cmp\":%ld,\"recon\":%ld,\"cache_hit\":%ld},"
        "\"prec\":%d,\"wall_s\":%.3f}\n",
        s, (long long)b.last_loss(), dig.c_str(), fb.eq_zero,
        fb.floor_exact, fb.floor_near, fb.cmp, fb.recon, fb.cache_hit,
        ax::dyi::prec, dt);
    std::fflush(stdout);
    if (dump) {
      const auto g9 = b.weights_grain9();
      char path[512];
      std::snprintf(path, sizeof path, "%s/anchor2_step%d.w9", dump, s);
      FILE* f = std::fopen(path, "wb");
      if (!f) { std::perror(path); return 1; }
      std::fwrite(g9.data(), 8, g9.size(), f);
      std::fclose(f);
    }
  }
  return 0;
}
