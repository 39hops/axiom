/** Arc 3 (magic boards rung 1): persistent (rule, node) no-fire mask.
    Proposal-side only — a mask hit skips re-firing a rule that
    previously produced nothing on that node; soundness-free by
    construction (verify_edge still guards every emitted child). */
#include <ax/search/search.hpp>

#include <ax/sym/parse.hpp>

#include <gtest/gtest.h>

#include <cstdio>
#include <string>

namespace {

using ax::search::beam_options;
using ax::search::beam_search;
using ax::search::default_rules;
using ax::sym::expr;
using ax::sym::parse;

TEST(FireMask, WarmRunSameResultAndHits) {
  const std::string path = std::string(::testing::TempDir()) +
                           "firemask_test.tsv";
  std::remove(path.c_str());
  const expr root = parse("Integral(2*x*exp(x**2) + 3*sin(x), x)");
  beam_options opt;
  opt.width = 8;
  opt.max_plies = 12;
  opt.max_nodes = 200;
  opt.use_macros = true;

  ax::search::fire_mask_reset();
  ax::search::fire_mask_enable(default_rules());
  const auto cold = beam_search(root, default_rules(), opt);
  ASSERT_TRUE(cold.solved);
  EXPECT_GT(ax::search::fire_mask_size(), 0u);
  ASSERT_TRUE(ax::search::fire_mask_save(path));

  ax::search::fire_mask_reset();
  ax::search::fire_mask_enable(default_rules());
  ASSERT_TRUE(ax::search::fire_mask_load(path, default_rules()));
  ax::search::successors_cache_clear();  // simulate a fresh process
  const auto warm = beam_search(root, default_rules(), opt);
  // identical outcome (mask is proposal-side; the winning chain is
  // unchanged) and the memo actually got consulted
  ASSERT_TRUE(warm.solved);
  EXPECT_TRUE(warm.best.e.same(cold.best.e));
  EXPECT_EQ(warm.best.history, cold.best.history);
  EXPECT_GT(ax::search::fire_mask_hits(), 0u);
  ax::search::fire_mask_reset();
  std::remove(path.c_str());
}

TEST(FireMask, FingerprintMismatchRejectsLoad) {
  const std::string path = std::string(::testing::TempDir()) +
                           "firemask_fp_test.tsv";
  std::remove(path.c_str());
  ax::search::fire_mask_reset();
  ax::search::fire_mask_enable(default_rules());
  const expr root = parse("Integral(x**2, x)");
  beam_options opt;
  opt.max_nodes = 50;
  (void)beam_search(root, default_rules(), opt);
  ASSERT_TRUE(ax::search::fire_mask_save(path));
  // corrupt the fingerprint line
  {
    FILE* f = std::fopen(path.c_str(), "r+b");
    ASSERT_NE(f, nullptr);
    std::fputs("XX", f);
    std::fclose(f);
  }
  ax::search::fire_mask_reset();
  ax::search::fire_mask_enable(default_rules());
  EXPECT_FALSE(ax::search::fire_mask_load(path, default_rules()));
  EXPECT_EQ(ax::search::fire_mask_size(), 0u);
  ax::search::fire_mask_reset();
  std::remove(path.c_str());
}

}  // namespace
