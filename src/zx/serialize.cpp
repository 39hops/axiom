/** zx/serialize.cpp — boundary-anchored serializer (the contract).

    Vertex order is BFS from the ordered I/O vertices with neighbors
    visited in ascending emitted label; internal labels are randomized
    per sample by random_labels (permutation augmentation over the true
    gauge). No canonical sort anywhere model-facing: the emitted order
    is a function of the anchor and the drawn labels only. */
#include <ax/zx/zx.hpp>

#include <algorithm>
#include <queue>
#include <stdexcept>

namespace ax::zx {

std::vector<int> random_labels(const graph& g,
                               pyrand::python_random& rng) {
  std::vector<int> labels(g.vertex_slots(), -1);
  int next = 0;
  for (int v : g.inputs()) labels[v] = next++;
  for (int v : g.outputs()) labels[v] = next++;
  std::vector<int> internal;
  for (int v = 0; v < g.vertex_slots(); ++v)
    if (g.alive(v) && labels[v] == -1) internal.push_back(v);
  std::vector<int> pool(internal.size());
  for (std::size_t i = 0; i < pool.size(); ++i)
    pool[i] = next + static_cast<int>(i);
  rng.shuffle(pool);
  for (std::size_t i = 0; i < internal.size(); ++i)
    labels[internal[i]] = pool[i];
  return labels;
}

std::string serialize(const graph& g, const std::vector<int>& labels) {
  // BFS order from the anchor: inputs then outputs, neighbors by label
  std::vector<int> order;
  std::vector<bool> seen(g.vertex_slots(), false);
  std::queue<int> q;
  auto push = [&](int v) {
    if (!seen[v]) {
      seen[v] = true;
      q.push(v);
      order.push_back(v);
    }
  };
  for (int v : g.inputs()) push(v);
  for (int v : g.outputs()) push(v);
  while (!q.empty()) {
    const int v = q.front();
    q.pop();
    std::vector<int> ns(g.neighbors(v).begin(), g.neighbors(v).end());
    std::sort(ns.begin(), ns.end(),
              [&](int a, int b) { return labels[a] < labels[b]; });
    for (int n : ns) push(n);
  }
  // disconnected leftovers (possible after heavy simplification):
  // appended in ascending label order, still label-driven
  std::vector<int> rest;
  for (int v = 0; v < g.vertex_slots(); ++v)
    if (g.alive(v) && !seen[v]) rest.push_back(v);
  std::sort(rest.begin(), rest.end(),
            [&](int a, int b) { return labels[a] < labels[b]; });
  for (int v : rest) order.push_back(v);

  std::vector<int> pos(g.vertex_slots(), -1);
  for (std::size_t i = 0; i < order.size(); ++i)
    pos[order[i]] = static_cast<int>(i);

  std::string s = "in(";
  for (std::size_t i = 0; i < g.inputs().size(); ++i) {
    if (i) s += ",";
    s += std::to_string(labels[g.inputs()[i]]);
  }
  s += ") out(";
  for (std::size_t i = 0; i < g.outputs().size(); ++i) {
    if (i) s += ",";
    s += std::to_string(labels[g.outputs()[i]]);
  }
  s += ")";
  for (int v : order) {
    if (g.kind(v) == vkind::boundary) continue;
    if (labels[v] < 0) throw std::invalid_argument("serialize: no label");
    s += g.kind(v) == vkind::z ? " Z(" : " X(";
    s += std::to_string(labels[v]) + ":" + std::to_string(g.phase(v)) +
         ")";
  }
  // edges in (BFS position, BFS position) order
  std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> es;
  for (int v : order)
    for (int n : g.neighbors(v)) {
      if (pos[n] < pos[v]) continue;  // each edge once, low pos first
      es.push_back({{pos[v], pos[n]}, {v, n}});
    }
  std::sort(es.begin(), es.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  for (const auto& [_, e] : es) {
    s += g.edge(e.first, e.second) == etype::plain ? " P(" : " H(";
    s += std::to_string(labels[e.first]) + "-" +
         std::to_string(labels[e.second]) + ")";
  }
  return s;
}

}  // namespace ax::zx
