/** zx/rules.cpp — the four local moves as check/apply pairs.

    Every check refuses configurations that would need multigraph
    bookkeeping, so apply is always a simple-graph rewrite and each
    move is sound by construction (pyzx-replayable at the named site).
*/
#include <ax/zx/zx.hpp>

#include <algorithm>
#include <stdexcept>

namespace ax::zx {

namespace {

bool is_spider(const graph& g, int v) {
  return g.alive(v) && g.kind(v) != vkind::boundary;
}

void require(bool ok, const char* what) {
  if (!ok) throw std::logic_error(what);
}

/** Toggle the hadamard edge inside a pair (absent <-> hadamard). */
void toggle_h(graph& g, int a, int b) {
  if (g.connected(a, b))
    g.remove_edge(a, b);
  else
    g.add_edge(a, b, etype::hadamard);
}

/** True when every pair (a in xs, b in ys), a != b, carries either no
    edge or a hadamard edge — i.e. the pair set is toggle-safe. */
bool crossing_toggle_safe(const graph& g, const std::vector<int>& xs,
                          const std::vector<int>& ys) {
  for (int a : xs)
    for (int b : ys)
      if (a != b && g.connected(a, b) && g.edge(a, b) != etype::hadamard)
        return false;
  return true;
}

}  // namespace

bool check_fuse(const graph& g, int u, int v) {
  if (u == v || !is_spider(g, u) || !is_spider(g, v)) return false;
  if (g.kind(u) != g.kind(v)) return false;
  if (!g.connected(u, v) || g.edge(u, v) != etype::plain) return false;
  // a shared neighbor would create a parallel edge on merge
  for (int n : g.neighbors(v))
    if (n != u && g.connected(u, n)) return false;
  return true;
}

void apply_fuse(graph& g, int u, int v) {
  require(check_fuse(g, u, v), "apply_fuse without check");
  g.set_phase(u, g.phase(u) + g.phase(v));
  const std::set<int> ns = g.neighbors(v);
  std::vector<std::pair<int, etype>> moved;
  for (int n : ns)
    if (n != u) moved.emplace_back(n, g.edge(v, n));
  g.remove_vertex(v);
  for (const auto& [n, t] : moved) g.add_edge(u, n, t);
}

bool check_id(const graph& g, int v) {
  if (!is_spider(g, v) || g.phase(v) != 0 || g.degree(v) != 2)
    return false;
  const auto& ns = g.neighbors(v);
  const int a = *ns.begin(), b = *std::next(ns.begin());
  return !g.connected(a, b);  // composed edge must not be parallel
}

void apply_id(graph& g, int v) {
  require(check_id(g, v), "apply_id without check");
  const auto& ns = g.neighbors(v);
  const int a = *ns.begin(), b = *std::next(ns.begin());
  const etype t = g.edge(v, a) == g.edge(v, b) ? etype::plain
                                               : etype::hadamard;
  g.remove_vertex(v);
  g.add_edge(a, b, t);
}

bool check_lcomp(const graph& g, int u) {
  if (!g.alive(u) || g.kind(u) != vkind::z) return false;
  if (g.phase(u) != 2 && g.phase(u) != 6) return false;
  const std::vector<int> ns(g.neighbors(u).begin(), g.neighbors(u).end());
  if (ns.empty()) return false;
  for (int n : ns)
    if (g.kind(n) != vkind::z || g.edge(u, n) != etype::hadamard)
      return false;
  return crossing_toggle_safe(g, ns, ns);
}

void apply_lcomp(graph& g, int u) {
  require(check_lcomp(g, u), "apply_lcomp without check");
  const std::vector<int> ns(g.neighbors(u).begin(), g.neighbors(u).end());
  const int p = g.phase(u);
  g.remove_vertex(u);
  for (std::size_t i = 0; i < ns.size(); ++i) {
    g.set_phase(ns[i], g.phase(ns[i]) - p);
    for (std::size_t j = i + 1; j < ns.size(); ++j)
      toggle_h(g, ns[i], ns[j]);
  }
}

bool check_pivot(const graph& g, int u, int v) {
  if (u == v || !g.alive(u) || !g.alive(v)) return false;
  if (g.kind(u) != vkind::z || g.kind(v) != vkind::z) return false;
  if (g.phase(u) % 4 != 0 || g.phase(v) % 4 != 0) return false;
  if (!g.connected(u, v) || g.edge(u, v) != etype::hadamard) return false;
  for (int w : {u, v})
    for (int n : g.neighbors(w)) {
      if (n == u || n == v) continue;
      if (g.kind(n) != vkind::z || g.edge(w, n) != etype::hadamard)
        return false;
    }
  std::vector<int> a, b, c;
  for (int n : g.neighbors(u)) {
    if (n == v) continue;
    (g.connected(v, n) ? c : a).push_back(n);
  }
  for (int n : g.neighbors(v))
    if (n != u && !g.connected(u, n)) b.push_back(n);
  return crossing_toggle_safe(g, a, b) && crossing_toggle_safe(g, a, c) &&
         crossing_toggle_safe(g, b, c);
}

void apply_pivot(graph& g, int u, int v) {
  require(check_pivot(g, u, v), "apply_pivot without check");
  std::vector<int> a, b, c;
  for (int n : g.neighbors(u)) {
    if (n == v) continue;
    (g.connected(v, n) ? c : a).push_back(n);
  }
  for (int n : g.neighbors(v))
    if (n != u && !g.connected(u, n)) b.push_back(n);
  const int pu = g.phase(u), pv = g.phase(v);
  g.remove_vertex(u);
  g.remove_vertex(v);
  for (int x : a)
    for (int y : b) toggle_h(g, x, y);
  for (int x : a)
    for (int y : c) toggle_h(g, x, y);
  for (int x : b)
    for (int y : c) toggle_h(g, x, y);
  for (int x : a) g.set_phase(x, g.phase(x) + pv);
  for (int x : b) g.set_phase(x, g.phase(x) + pu);
  for (int x : c) g.set_phase(x, g.phase(x) + pu + pv + 4);
}

bool check_color(const graph& g, int u) {
  return g.alive(u) && g.kind(u) == vkind::x;
}

void apply_color(graph& g, int u) {
  require(check_color(g, u), "apply_color without check");
  g.set_kind(u, vkind::z);
  const std::set<int> ns = g.neighbors(u);
  for (int n : ns) {
    const etype t = g.edge(u, n);
    g.remove_edge(u, n);
    g.add_edge(u, n, t == etype::plain ? etype::hadamard : etype::plain);
  }
}

std::vector<move> candidates(const graph& g) {
  std::vector<move> out;
  const int n = g.vertex_slots();
  for (int u = 0; u < n; ++u) {
    if (!g.alive(u)) continue;
    for (int v : g.neighbors(u))
      if (u < v && check_fuse(g, u, v)) out.push_back({"fuse", u, v});
    if (check_id(g, u)) out.push_back({"id", u, -1});
    if (check_lcomp(g, u)) out.push_back({"lcomp", u, -1});
    for (int v : g.neighbors(u))
      if (u < v && check_pivot(g, u, v)) out.push_back({"pivot", u, v});
    if (check_color(g, u)) out.push_back({"color", u, -1});
  }
  return out;
}

void apply(graph& g, const move& m) {
  if (m.kind == "fuse")
    apply_fuse(g, m.a, m.b);
  else if (m.kind == "id")
    apply_id(g, m.a);
  else if (m.kind == "lcomp")
    apply_lcomp(g, m.a);
  else if (m.kind == "pivot")
    apply_pivot(g, m.a, m.b);
  else if (m.kind == "color")
    apply_color(g, m.a);
  else
    throw std::invalid_argument("apply: unknown move " + m.kind);
}

}  // namespace ax::zx
