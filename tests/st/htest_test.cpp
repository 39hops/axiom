// Hypothesis tests. Oracles: statistics hand-computed exactly (shown per
// test); p-values checked against published tables via brackets and against
// the (independently tested) Phase 3 distribution CDFs.
#include <ax/st/dist.hpp>
#include <ax/st/htest.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace ax::st;

// ---------- one-sample t ----------

TEST(t_test_1sample, zero_statistic_at_sample_mean) {
  std::vector<double> xs{5.1, 4.9, 5.0, 5.2, 4.8};  // mean exactly 5.0
  auto r = t_test(xs, 5.0);
  EXPECT_NEAR(r.statistic, 0.0, 1e-12);
  EXPECT_DOUBLE_EQ(r.df, 4.0);
  EXPECT_NEAR(r.p_value, 1.0, 1e-12);
}

TEST(t_test_1sample, hand_computed_statistic) {
  // mean 5.0, s^2 = 0.1/4 = 0.025, se = sqrt(0.025/5) = 0.0707107
  // t = 0.2/0.0707107 = 2.8284271, df 4; R: t.test gives p = 0.04734
  std::vector<double> xs{5.1, 4.9, 5.0, 5.2, 4.8};
  auto r = t_test(xs, 4.8);
  EXPECT_NEAR(r.statistic, 2.8284271247461903, 1e-10);
  EXPECT_DOUBLE_EQ(r.df, 4.0);
  EXPECT_GT(r.p_value, 0.04);
  EXPECT_LT(r.p_value, 0.05);
  // consistency with t_dist tail
  double tail = 1.0 - t_dist(4.0).cdf(r.statistic);
  EXPECT_NEAR(r.p_value, 2.0 * tail, 1e-12);
}

TEST(t_test_1sample, one_sided_p_values_sum_to_one) {
  std::vector<double> xs{5.1, 4.9, 5.0, 5.2, 4.8};
  auto lo = t_test(xs, 4.8, alternative::less);
  auto hi = t_test(xs, 4.8, alternative::greater);
  EXPECT_NEAR(lo.p_value + hi.p_value, 1.0, 1e-12);
  auto two = t_test(xs, 4.8, alternative::two_sided);
  EXPECT_NEAR(two.p_value, 2.0 * std::min(lo.p_value, hi.p_value), 1e-12);
}

TEST(t_test_1sample, zero_variance_throws) {
  std::vector<double> xs{2.0, 2.0, 2.0};
  EXPECT_THROW((void)t_test(xs, 2.0), std::invalid_argument);
}

TEST(t_test_2sample, both_constant_throws) {
  std::vector<double> xs{1.0, 1.0};
  std::vector<double> ys{2.0, 2.0};
  EXPECT_THROW((void)t_test(xs, ys), std::invalid_argument);
}

TEST(anova_oneway, zero_within_group_variance_throws) {
  std::vector<std::vector<double>> groups{{1, 1, 1}, {2, 2, 2}};
  EXPECT_THROW((void)anova_oneway(groups), std::invalid_argument);
}

TEST(t_test_1sample, too_small_throws) {
  std::vector<double> xs{1.0};
  EXPECT_THROW((void)t_test(xs, 0.0), std::invalid_argument);
}

// ---------- two-sample t ----------

TEST(t_test_2sample, pooled_hand_computed) {
  // xs: mean 3, var 2.5 (n=5); ys: mean 7, var 14 (n=6)
  // sp^2 = (4*2.5 + 5*14)/9 = 80/9; se^2 = 80/9 * 11/30 = 88/27
  // t = -4 / sqrt(88/27) = -4*sqrt(27/88), df = 9
  std::vector<double> xs{1, 2, 3, 4, 5};
  std::vector<double> ys{2, 4, 6, 8, 10, 12};
  auto r = t_test(xs, ys, alternative::two_sided, /*welch=*/false);
  EXPECT_NEAR(r.statistic, -4.0 * std::sqrt(27.0 / 88.0), 1e-12);
  EXPECT_DOUBLE_EQ(r.df, 9.0);
  double tail = t_dist(9.0).cdf(r.statistic);
  EXPECT_NEAR(r.p_value, 2.0 * tail, 1e-12);
}

TEST(t_test_2sample, welch_hand_computed) {
  // se^2 = 2.5/5 + 14/6 = 17/6; t = -4 / sqrt(17/6) = -4*sqrt(6/17)
  // df = (17/6)^2 / ((1/2)^2/4 + (7/3)^2/5) = (289/36) / (1/16 + 49/45)
  std::vector<double> xs{1, 2, 3, 4, 5};
  std::vector<double> ys{2, 4, 6, 8, 10, 12};
  auto r = t_test(xs, ys);  // welch default
  EXPECT_NEAR(r.statistic, -4.0 * std::sqrt(6.0 / 17.0), 1e-12);
  EXPECT_NEAR(r.df, (289.0 / 36.0) / (1.0 / 16.0 + 49.0 / 45.0), 1e-12);
  EXPECT_GT(r.p_value, 0.0);
  EXPECT_LT(r.p_value, 0.1);
}

TEST(t_test_2sample, small_sample_throws) {
  std::vector<double> xs{1.0};
  std::vector<double> ys{1.0, 2.0};
  EXPECT_THROW((void)t_test(xs, ys), std::invalid_argument);
}

// ---------- chi-square ----------

TEST(chi2_gof, fair_die) {
  // observed {8,9,19,5,8,11}, n=60, expected 10 each
  // chi2 = (4+1+81+25+4+1)/10 = 11.6, df 5; table p ~= 0.0407
  std::vector<double> obs{8, 9, 19, 5, 8, 11};
  std::vector<double> p(6, 1.0 / 6.0);
  auto r = chi2_gof(obs, p);
  EXPECT_NEAR(r.statistic, 11.6, 1e-10);
  EXPECT_DOUBLE_EQ(r.df, 5.0);
  EXPECT_GT(r.p_value, 0.03);
  EXPECT_LT(r.p_value, 0.05);
}

TEST(chi2_gof, size_mismatch_throws) {
  std::vector<double> obs{1, 2, 3};
  std::vector<double> p{0.5, 0.5};
  EXPECT_THROW((void)chi2_gof(obs, p), std::invalid_argument);
}

TEST(chi2_independence, two_by_two_hand_computed) {
  // {{10,20},{30,40}}: expected {{12,18},{28,42}}
  // chi2 = 4/12+4/18+4/28+4/42 = 0.79365079, df 1
  ax::la::mat table{{10, 20}, {30, 40}};
  auto r = chi2_independence(table);
  EXPECT_NEAR(r.statistic, 0.7936507936507936, 1e-12);
  EXPECT_DOUBLE_EQ(r.df, 1.0);
  double tail = 1.0 - chi2_dist(1.0).cdf(r.statistic);
  EXPECT_NEAR(r.p_value, tail, 1e-12);
}

TEST(chi2_independence, needs_two_by_two) {
  ax::la::mat table{{1, 2}};
  EXPECT_THROW((void)chi2_independence(table), std::invalid_argument);
}

// ---------- ANOVA ----------

TEST(anova_oneway, textbook_example) {
  // groups (n=6 each), means 5, 9, 10, grand mean 8
  // SSB = 6*(9+1+4) = 84; SSW = 16+24+28 = 68
  // F = (84/2)/(68/15) = 9.2647059, df (2, 15); p ~= 0.0024
  std::vector<std::vector<double>> groups{
      {6, 8, 4, 5, 3, 4}, {8, 12, 9, 11, 6, 8}, {13, 9, 11, 8, 7, 12}};
  auto r = anova_oneway(groups);
  EXPECT_NEAR(r.statistic, 9.264705882352942, 1e-10);
  EXPECT_DOUBLE_EQ(r.df, 2.0);
  EXPECT_DOUBLE_EQ(r.df2, 15.0);
  EXPECT_GT(r.p_value, 0.001);
  EXPECT_LT(r.p_value, 0.01);
}

TEST(anova_oneway, equal_means_large_p) {
  std::vector<std::vector<double>> groups{
      {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}};
  auto r = anova_oneway(groups);
  EXPECT_NEAR(r.statistic, 0.0, 1e-12);
  EXPECT_GT(r.p_value, 0.5);
}

TEST(anova_oneway, needs_two_groups) {
  std::vector<std::vector<double>> groups{{1, 2, 3}};
  EXPECT_THROW((void)anova_oneway(groups), std::invalid_argument);
}

// ---------- KS ----------

TEST(ks_test, uniform_sample_vs_uniform_cdf) {
  rng g{123};
  std::vector<double> xs(1000);
  for (auto& x : xs) x = g.next_double();
  auto r = ks_test(xs, [](double x) {
    return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
  });
  EXPECT_TRUE(std::isnan(r.df));
  EXPECT_GT(r.p_value, 0.01);
  EXPECT_GE(r.statistic, 0.0);
  EXPECT_LE(r.statistic, 1.0);
}

TEST(ks_test, uniform_sample_vs_normal_cdf_rejected) {
  rng g{123};
  std::vector<double> xs(1000);
  for (auto& x : xs) x = g.next_double();
  normal_dist n{0.0, 1.0};
  auto r = ks_test(xs, [&](double x) { return n.cdf(x); });
  EXPECT_LT(r.p_value, 1e-6);
}

TEST(ks_test, exact_small_case) {
  // xs = {0.5}, uniform cdf: D = max(1-0.5, 0.5-0) = 0.5
  std::vector<double> xs{0.5};
  auto r = ks_test(xs, [](double x) { return x; });
  EXPECT_NEAR(r.statistic, 0.5, 1e-15);
}

TEST(ks_test, empty_throws) {
  std::vector<double> xs;
  EXPECT_THROW((void)ks_test(xs, [](double x) { return x; }),
               std::invalid_argument);
}
