/** zx_test.cpp — ZX rewrite core: moves, serializer contract, chain. */
#include <ax/mathgen/zxchain.hpp>
#include <ax/zx/zx.hpp>

#include <gtest/gtest.h>

namespace {

using namespace ax;
using zx::etype;
using zx::vkind;

/** in - Z(a) - Z(b) - out wire with plain edges. */
zx::graph two_spider_wire(int pa, int pb) {
  zx::graph g;
  const int in = g.add_vertex(vkind::boundary);
  g.mark_input(in);
  const int a = g.add_vertex(vkind::z, pa);
  const int b = g.add_vertex(vkind::z, pb);
  const int out = g.add_vertex(vkind::boundary);
  g.mark_output(out);
  g.add_edge(in, a, etype::plain);
  g.add_edge(a, b, etype::plain);
  g.add_edge(b, out, etype::plain);
  return g;
}

TEST(ZxFuse, PhasesAddMod8) {
  zx::graph g = two_spider_wire(7, 3);
  ASSERT_TRUE(zx::check_fuse(g, 1, 2));
  zx::apply_fuse(g, 1, 2);
  EXPECT_EQ(g.phase(1), 2);  // 7 + 3 mod 8
  EXPECT_FALSE(g.alive(2));
  EXPECT_TRUE(g.connected(1, 3));  // inherited b-out edge
  EXPECT_EQ(g.spider_count(), 1);
}

TEST(ZxFuse, RefusesColorMismatchAndSharedNeighbor) {
  zx::graph g = two_spider_wire(1, 1);
  const int c = g.add_vertex(vkind::x, 0);
  g.add_edge(1, c, etype::plain);
  g.add_edge(2, c, etype::plain);
  EXPECT_FALSE(zx::check_fuse(g, 1, 2));  // shared neighbor c
  EXPECT_FALSE(zx::check_fuse(g, 1, c));  // color mismatch
}

TEST(ZxId, EdgeTypesCompose) {
  zx::graph g = two_spider_wire(0, 1);
  g.remove_edge(1, 2);
  g.add_edge(1, 2, etype::hadamard);  // in -P- Z0 -H- Z1 -P- out
  ASSERT_TRUE(zx::check_id(g, 1));
  zx::apply_id(g, 1);
  EXPECT_FALSE(g.alive(1));
  EXPECT_EQ(g.edge(0, 2), etype::hadamard);  // P * H = H
}

TEST(ZxId, RefusesNonzeroPhaseAndParallel) {
  zx::graph g = two_spider_wire(1, 0);
  EXPECT_FALSE(zx::check_id(g, 1));  // phase 1
  ASSERT_TRUE(zx::check_id(g, 2));
  g.add_edge(1, 3, etype::plain);  // a-out edge exists: removal of b
  EXPECT_FALSE(zx::check_id(g, 2));  // would create a parallel edge
}

/** Hadamard triangle of Z spiders around u, u internal. */
TEST(ZxLcomp, TogglesNeighborhoodAndShiftsPhases) {
  zx::graph g;
  const int u = g.add_vertex(vkind::z, 2);
  const int a = g.add_vertex(vkind::z, 0);
  const int b = g.add_vertex(vkind::z, 1);
  const int c = g.add_vertex(vkind::z, 4);
  g.add_edge(u, a, etype::hadamard);
  g.add_edge(u, b, etype::hadamard);
  g.add_edge(u, c, etype::hadamard);
  g.add_edge(a, b, etype::hadamard);
  ASSERT_TRUE(zx::check_lcomp(g, u));
  zx::apply_lcomp(g, u);
  EXPECT_FALSE(g.alive(u));
  EXPECT_FALSE(g.connected(a, b));               // toggled off
  EXPECT_EQ(g.edge(a, c), etype::hadamard);      // toggled on
  EXPECT_EQ(g.edge(b, c), etype::hadamard);      // toggled on
  EXPECT_EQ(g.phase(a), 6);   // 0 - 2 mod 8
  EXPECT_EQ(g.phase(b), 7);   // 1 - 2 mod 8
  EXPECT_EQ(g.phase(c), 2);   // 4 - 2 mod 8
}

TEST(ZxLcomp, RefusesBoundaryNeighborAndBadPhase) {
  zx::graph g;
  const int u = g.add_vertex(vkind::z, 1);
  const int a = g.add_vertex(vkind::z, 0);
  g.add_edge(u, a, etype::hadamard);
  EXPECT_FALSE(zx::check_lcomp(g, u));  // phase 1 is not ±pi/2
  g.set_phase(u, 6);
  EXPECT_TRUE(zx::check_lcomp(g, u));
  const int bd = g.add_vertex(vkind::boundary);
  g.mark_input(bd);
  g.add_edge(u, bd, etype::hadamard);
  EXPECT_FALSE(zx::check_lcomp(g, u));  // boundary neighbor
}

TEST(ZxPivot, ComplementsGroupsAndShiftsPhases) {
  zx::graph g;
  const int u = g.add_vertex(vkind::z, 0);
  const int v = g.add_vertex(vkind::z, 4);
  const int a = g.add_vertex(vkind::z, 0);  // A: neighbor of u only
  const int b = g.add_vertex(vkind::z, 0);  // B: neighbor of v only
  const int c = g.add_vertex(vkind::z, 1);  // C: common neighbor
  g.add_edge(u, v, etype::hadamard);
  g.add_edge(u, a, etype::hadamard);
  g.add_edge(v, b, etype::hadamard);
  g.add_edge(u, c, etype::hadamard);
  g.add_edge(v, c, etype::hadamard);
  ASSERT_TRUE(zx::check_pivot(g, u, v));
  zx::apply_pivot(g, u, v);
  EXPECT_FALSE(g.alive(u));
  EXPECT_FALSE(g.alive(v));
  EXPECT_EQ(g.edge(a, b), etype::hadamard);  // A x B toggled on
  EXPECT_EQ(g.edge(a, c), etype::hadamard);  // A x C toggled on
  EXPECT_EQ(g.edge(b, c), etype::hadamard);  // B x C toggled on
  EXPECT_EQ(g.phase(a), 4);  // + phase(v)
  EXPECT_EQ(g.phase(b), 0);  // + phase(u)
  EXPECT_EQ(g.phase(c), 1);  // 1 + 0 + 4 + 4 mod 8
}

TEST(ZxColor, FlipsIncidentEdgesAndRecolors) {
  zx::graph g;
  const int in = g.add_vertex(vkind::boundary);
  g.mark_input(in);
  const int u = g.add_vertex(vkind::x, 3);
  const int z = g.add_vertex(vkind::z, 1);
  g.add_edge(in, u, etype::plain);
  g.add_edge(u, z, etype::hadamard);
  ASSERT_TRUE(zx::check_color(g, u));
  EXPECT_FALSE(zx::check_color(g, z));  // Z spider: no
  zx::apply_color(g, u);
  EXPECT_EQ(g.kind(u), vkind::z);
  EXPECT_EQ(g.phase(u), 3);                  // phase untouched
  EXPECT_EQ(g.edge(in, u), etype::hadamard);  // flipped
  EXPECT_EQ(g.edge(u, z), etype::plain);      // flipped
  EXPECT_FALSE(zx::check_color(g, u));  // no longer X
}

TEST(ZxSerialize, BoundaryAnchoredAndLabelDriven) {
  zx::graph g = two_spider_wire(1, 5);
  std::vector<int> labels = {0, 2, 3, 1};  // in=0, out=1, a=2, b=3
  const std::string s = zx::serialize(g, labels);
  EXPECT_EQ(s, "in(0) out(1) Z(2:1) Z(3:5) P(0-2) P(1-3) P(2-3)");
  // a different internal labeling permutes names, same structure;
  // edge endpoints stay in BFS-position order (traversal-anchored,
  // never label-sorted), hence 3-2 here
  std::vector<int> labels2 = {0, 3, 2, 1};
  const std::string s2 = zx::serialize(g, labels2);
  EXPECT_EQ(s2, "in(0) out(1) Z(3:1) Z(2:5) P(0-3) P(1-2) P(3-2)");
}

TEST(ZxSerialize, RandomLabelsFixBoundaryAndAreSeedStable) {
  zx::graph g = two_spider_wire(1, 5);
  pyrand::python_random r1("t|0"), r2("t|0"), r3("t|1");
  const auto l1 = zx::random_labels(g, r1);
  const auto l2 = zx::random_labels(g, r2);
  const auto l3 = zx::random_labels(g, r3);
  EXPECT_EQ(l1[0], 0);  // input anchored
  EXPECT_EQ(l1[3], 1);  // output anchored
  EXPECT_EQ(l1, l2);    // string-seeded determinism
  EXPECT_EQ(l3[0], 0);
  // internal labels are exactly {2, 3} in every draw
  EXPECT_EQ(std::min(l1[1], l1[2]), 2);
  EXPECT_EQ(std::max(l1[1], l1[2]), 3);
}

TEST(ZxChain, DeterministicRowsWithMetadata) {
  const auto p = mathgen::make_zx_chain(1, 0);
  const auto q = mathgen::make_zx_chain(1, 0);
  ASSERT_FALSE(p.rows.empty());
  EXPECT_EQ(p.rows.size(), q.rows.size());
  for (std::size_t i = 0; i < p.rows.size(); ++i) {
    EXPECT_EQ(p.rows[i].cur, q.rows[i].cur);
    EXPECT_EQ(p.rows[i].nxt, q.rows[i].nxt);
    EXPECT_NE(p.rows[i].cur, p.rows[i].nxt);
    EXPECT_GE(p.rows[i].spiders, 1);
    EXPECT_GE(p.rows[i].tcount, 0);
    EXPECT_FALSE(p.rows[i].site.empty());
  }
  EXPECT_GE(p.level, 1);
  EXPECT_LE(p.level, 3);
}

TEST(ZxChain, DescentReducesSpidersOverall) {
  for (long long seed = 0; seed < 10; ++seed) {
    const auto p = mathgen::make_zx_chain(2, seed);
    ASSERT_FALSE(p.rows.empty()) << "seed " << seed;
    EXPECT_LT(p.rows.back().spiders, p.rows.front().spiders + 1)
        << "seed " << seed;
  }
}

}  // namespace
