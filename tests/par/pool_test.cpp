#include <gtest/gtest.h>
#include <ax/par/pool.hpp>

#include <atomic>
#include <vector>

TEST(pool, executes_submitted_task) {
  ax::thread_pool p{2};
  auto f = p.submit([] { return 41 + 1; });
  EXPECT_EQ(f.get(), 42);
}

TEST(pool, parallel_for_covers_range_exactly_once) {
  ax::thread_pool p{4};
  std::vector<std::atomic<int>> hits(1000);
  ax::parallel_for(p, std::size_t{0}, hits.size(), 16,
                   [&](std::size_t i) { hits[i].fetch_add(1); });
  for (auto& h : hits) EXPECT_EQ(h.load(), 1);
}

TEST(pool, parallel_reduce_sums) {
  ax::thread_pool p{4};
  // sum 1..10000 = 50005000
  auto s = ax::parallel_reduce(
      p, std::size_t{1}, std::size_t{10001}, 64, 0LL,
      [](long long acc, std::size_t i) { return acc + static_cast<long long>(i); },
      [](long long a, long long b) { return a + b; });
  EXPECT_EQ(s, 50005000LL);
}
