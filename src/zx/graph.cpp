/** zx/graph.cpp — simple-graph substrate for the ZX rewrite core. */
#include <ax/zx/zx.hpp>

#include <stdexcept>

namespace ax::zx {

int graph::add_vertex(vkind kind, int phase) {
  verts_.push_back({kind, ((phase % 8) + 8) % 8, true});
  adj_.emplace_back();
  return static_cast<int>(verts_.size()) - 1;
}

void graph::mark_input(int v) {
  if (kind(v) != vkind::boundary)
    throw std::invalid_argument("mark_input: not a boundary vertex");
  inputs_.push_back(v);
}

void graph::mark_output(int v) {
  if (kind(v) != vkind::boundary)
    throw std::invalid_argument("mark_output: not a boundary vertex");
  outputs_.push_back(v);
}

void graph::add_edge(int u, int v, etype t) {
  if (u == v) throw std::invalid_argument("add_edge: self-loop");
  if (!alive(u) || !alive(v))
    throw std::invalid_argument("add_edge: dead endpoint");
  if (!edges_.emplace(key(u, v), t).second)
    throw std::invalid_argument("add_edge: parallel edge");
  adj_[u].insert(v);
  adj_[v].insert(u);
}

void graph::remove_edge(int u, int v) {
  if (edges_.erase(key(u, v)) == 0)
    throw std::invalid_argument("remove_edge: no such edge");
  adj_[u].erase(v);
  adj_[v].erase(u);
}

void graph::remove_vertex(int v) {
  if (!alive(v)) throw std::invalid_argument("remove_vertex: dead");
  // copy: remove_edge mutates adj_[v]
  const std::set<int> ns = adj_[v];
  for (int n : ns) remove_edge(v, n);
  verts_[v].alive = false;
}

bool graph::connected(int u, int v) const {
  return edges_.count(key(u, v)) != 0;
}

etype graph::edge(int u, int v) const {
  const auto it = edges_.find(key(u, v));
  if (it == edges_.end()) throw std::invalid_argument("edge: no edge");
  return it->second;
}

int graph::spider_count() const {
  int n = 0;
  for (const auto& w : verts_)
    if (w.alive && w.kind != vkind::boundary) ++n;
  return n;
}

int graph::t_count() const {
  int n = 0;
  for (const auto& w : verts_)
    if (w.alive && w.kind != vkind::boundary && w.phase % 2 != 0) ++n;
  return n;
}

}  // namespace ax::zx
