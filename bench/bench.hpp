#pragma once
/** @file bench.hpp Minimal wall-clock micro-benchmark harness. */
#include <chrono>
#include <cstdio>
#include <string_view>

namespace ax::bench {

/** Run fn repeatedly across timed rounds; print best time per iteration. */
template <class F>
void run(std::string_view name, F&& fn) {
  using clock = std::chrono::steady_clock;
  fn();  // warmup
  double best_ns = 1e300;
  int iters = 1;
  for (int round = 0; round < 7; ++round) {
    const auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) fn();
    const auto t1 = clock::now();
    const double ns =
        std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
    best_ns = ns < best_ns ? ns : best_ns;
    if (std::chrono::duration<double>(t1 - t0).count() < 0.1) iters *= 4;
  }
  std::printf("%-40s %14.1f ns/op\n", name.data(), best_ns);
}

}  // namespace ax::bench
