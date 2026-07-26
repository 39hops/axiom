#pragma once
/** @file zx.hpp ZX-calculus rewrite core (llmopt relay 2026-07-26).

    Row-factory substrate for the first GRAPH grammar: spiders (Z/X)
    with exact phases k*pi/4 stored as integers mod 8 (no floats
    anywhere), plain/hadamard edges, an ordered boundary of input and
    output vertices, and the four local moves as check/apply pairs —
    fuse, identity removal, local complementation, pivot. Soundness is
    by construction per move (the pyzx pattern): apply() is only legal
    when check() holds, and every check refuses configurations that
    would need multigraph bookkeeping (parallel edges, self-loops), so
    the diagram stays a simple graph throughout. No extraction, no
    tensor evaluation — semantic adjudication is llmopt-side (pyzx
    replay + compare_tensors at small qubit counts).

    Serialization is boundary-anchored: vertex order is BFS from the
    ordered I/O vertices, internal vertex labels are RANDOMIZED per
    emitted sample (permutation augmentation over the true gauge).
    Canonical sorts are forbidden in anything model-facing; dedup
    hashing is exempt. The serializer is the contract (Phase-B
    byte-exact discipline). */
#include <ax/pyrand/pyrand.hpp>

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace ax::zx {

enum class vkind : std::uint8_t { boundary, z, x };
enum class etype : std::uint8_t { plain, hadamard };

/** Simple undirected graph of spiders + ordered boundary. Vertex ids
    are stable across moves (moves only delete); dead vertices keep
    their slot with alive == false. Phases are integers mod 8. */
class graph {
 public:
  int add_vertex(vkind kind, int phase = 0);
  /** Register a boundary vertex as the next ordered input / output. */
  void mark_input(int v);
  void mark_output(int v);

  void add_edge(int u, int v, etype t);
  void remove_edge(int u, int v);
  void remove_vertex(int v);

  bool alive(int v) const { return verts_.at(v).alive; }
  vkind kind(int v) const { return verts_.at(v).kind; }
  int phase(int v) const { return verts_.at(v).phase; }
  void set_phase(int v, int p) { verts_.at(v).phase = ((p % 8) + 8) % 8; }

  bool connected(int u, int v) const;
  etype edge(int u, int v) const;  ///< requires connected(u, v)
  /** Neighbors in ascending vertex-id order. */
  const std::set<int>& neighbors(int v) const { return adj_.at(v); }
  int degree(int v) const { return static_cast<int>(adj_.at(v).size()); }

  const std::vector<int>& inputs() const { return inputs_; }
  const std::vector<int>& outputs() const { return outputs_; }
  int vertex_slots() const { return static_cast<int>(verts_.size()); }
  int spider_count() const;   ///< alive non-boundary vertices
  int t_count() const;        ///< alive spiders with odd phase

 private:
  struct vertex {
    vkind kind;
    int phase;
    bool alive;
  };
  std::vector<vertex> verts_;
  std::vector<std::set<int>> adj_;
  std::map<std::pair<int, int>, etype> edges_;  // key u < v
  std::vector<int> inputs_, outputs_;

  static std::pair<int, int> key(int u, int v) {
    return u < v ? std::pair{u, v} : std::pair{v, u};
  }
};

/** The four local moves. Each check_* is a pure applicability test;
    each apply_* requires its check and mutates the graph in place. */

/** Fuse spider v into spider u: same color, plain edge u-v, and no
    shared neighbor (a shared neighbor would create a parallel edge).
    Phases add mod 8; v's edges move to u with their types. */
bool check_fuse(const graph& g, int u, int v);
void apply_fuse(graph& g, int u, int v);

/** Remove a phase-0 degree-2 spider v between distinct, unconnected
    neighbors a, b; the two incident edge types compose (H*H = plain,
    H*plain = H, plain*plain = plain). */
bool check_id(const graph& g, int v);
void apply_id(graph& g, int v);

/** Local complementation at Z spider u with phase ±pi/2 (2 or 6):
    every neighbor a Z spider joined by a hadamard edge, none boundary,
    and no plain edge between any two neighbors. Removes u, toggles the
    hadamard edge inside every neighbor pair, subtracts u's phase from
    every neighbor. */
bool check_lcomp(const graph& g, int u);
void apply_lcomp(graph& g, int u);

/** Pivot along hadamard edge u-v, both Z spiders with phase 0 or pi
    (0 or 4), all other neighbors internal Z spiders on hadamard edges,
    and no plain edge inside any crossing pair of the three neighbor
    groups A = N(u)\N(v), B = N(v)\N(u), C = N(u) ∩ N(v). Removes u
    and v, complements A×B, A×C, B×C, and adds phase(v) to A,
    phase(u) to B, phase(u) + phase(v) + pi to C. */
bool check_pivot(const graph& g, int u, int v);
void apply_pivot(graph& g, int u, int v);

struct move {
  std::string kind;  ///< fuse | id | lcomp | pivot
  int a = -1;        ///< primary vertex
  int b = -1;        ///< second vertex (fuse, pivot), else -1
};

/** All moves whose check passes, in deterministic (kind, id) order. */
std::vector<move> candidates(const graph& g);
void apply(graph& g, const move& m);

/** Boundary-anchored serialization under a caller-supplied labeling.
    labels[id] is the emitted name of vertex id; ids without a label
    (dead slots) are ignored. Vertex order is BFS from inputs then
    outputs, neighbors visited in ascending label; edges are emitted in
    (BFS position, BFS position) order. Grammar:
      in(l,...) out(l,...) Z(l:p) X(l:p) ... P(l-l) H(l-l) ...
    Atoms: digits, in, out, Z, X, P, H, ( ) : - , and space. */
std::string serialize(const graph& g, const std::vector<int>& labels);

/** Random labeling for the current alive vertices: boundary vertices
    take 0..b-1 in I/O order (the anchor), internal vertices take a
    CPython-shuffle permutation of b..n-1. labels[id] == -1 for dead
    slots. */
std::vector<int> random_labels(const graph& g,
                               pyrand::python_random& rng);

}  // namespace ax::zx
