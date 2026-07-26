/** mathgen/zxchain.cpp — Clifford+T circuit generator + move descent.

    Gate -> diagram dictionary (the standard one):
      S/T/Z on q : Z spider phase 2/1/4 spliced into the wire
      X on q     : X spider phase 4 spliced into the wire
      H on q     : edge-type toggle on the wire (no box vertex)
      CZ(a, b)   : Z spider on each wire + hadamard edge between
      CNOT(c, t) : Z spider on control, X spider on target, plain edge
    Wires start at input boundary vertices and close on output boundary
    vertices with the pending hadamard toggle folded into the edge. */
#include <ax/mathgen/zxchain.hpp>

#include <string>

namespace ax::mathgen {

namespace {

struct wire_state {
  int last;      ///< vertex id at the open end of the wire
  bool pending;  ///< pending hadamard toggle for the next edge
};

zx::etype pend(bool h) {
  return h ? zx::etype::hadamard : zx::etype::plain;
}

/** Splice a new spider into wire w. */
int splice(zx::graph& g, wire_state& w, zx::vkind k, int phase) {
  const int v = g.add_vertex(k, phase);
  g.add_edge(w.last, v, pend(w.pending));
  w.last = v;
  w.pending = false;
  return v;
}

int tcount_level(int t) { return t < 5 ? 1 : t < 10 ? 2 : 3; }

}  // namespace

zx_problem make_zx_chain(int size, long long seed) {
  zx_problem p;
  p.size = size;
  p.seed = seed;
  pyrand::python_random rng("zx_chain|" + std::to_string(size) + "|" +
                            std::to_string(seed));
  const int qubits = size == 1 ? 3 : size == 2 ? 4 : 5;
  const int gates = static_cast<int>(
      rng.randint(size == 1 ? 8 : size == 2 ? 16 : 28,
                  size == 1 ? 14 : size == 2 ? 26 : 44));
  p.qubits = qubits;
  p.gates = gates;

  zx::graph g;
  std::vector<wire_state> wire;
  for (int q = 0; q < qubits; ++q) {
    const int in = g.add_vertex(zx::vkind::boundary);
    g.mark_input(in);
    wire.push_back({in, false});
  }
  // gate mix: H S T Z X CZ CNOT — CZ-heavy so graph-like Z/H patches
  // form and lcomp/pivot get real fire rates, T frequent enough to
  // spread the level buckets
  for (int i = 0; i < gates; ++i) {
    const long long r = rng.randint(0, 9);
    const int q = static_cast<int>(rng.randint(0, qubits - 1));
    if (r <= 2) {  // H
      wire[q].pending = !wire[q].pending;
    } else if (r == 3) {  // S
      splice(g, wire[q], zx::vkind::z, 2);
    } else if (r <= 5) {  // T
      splice(g, wire[q], zx::vkind::z, 1);
    } else if (r == 6) {  // X
      splice(g, wire[q], zx::vkind::x, 4);
    } else {  // CZ / CNOT on a distinct pair
      int q2 = static_cast<int>(rng.randint(0, qubits - 2));
      if (q2 >= q) ++q2;
      if (r <= 8) {
        const int a = splice(g, wire[q], zx::vkind::z, 0);
        const int b = splice(g, wire[q2], zx::vkind::z, 0);
        g.add_edge(a, b, zx::etype::hadamard);
      } else {
        const int c = splice(g, wire[q], zx::vkind::z, 0);
        const int t = splice(g, wire[q2], zx::vkind::x, 0);
        g.add_edge(c, t, zx::etype::plain);
      }
    }
  }
  for (int q = 0; q < qubits; ++q) {
    const int out = g.add_vertex(zx::vkind::boundary);
    g.mark_output(out);
    g.add_edge(wire[q].last, out, pend(wire[q].pending));
  }
  p.level = tcount_level(g.t_count());

  // descent: greedy tier (fuse/id before lcomp/pivot), uniform inside
  const int max_plies = 128;
  for (int ply = 0; ply < max_plies; ++ply) {
    const auto all = zx::candidates(g);
    if (all.empty()) break;
    std::vector<zx::move> tier;
    for (const auto& m : all)
      if (m.kind == "fuse" || m.kind == "id") tier.push_back(m);
    if (tier.empty()) tier = all;
    const auto& m = tier[rng.choice_index(tier.size())];

    pyrand::python_random label_rng(
        "zx_label|" + std::to_string(size) + "|" + std::to_string(seed) +
        "|" + std::to_string(ply));
    const auto labels = zx::random_labels(g, label_rng);
    zx_row row;
    row.kind = m.kind;
    row.site = std::to_string(labels[m.a]);
    if (m.b >= 0) row.site += " " + std::to_string(labels[m.b]);
    row.tcount = g.t_count();
    row.spiders = g.spider_count();
    row.cur = zx::serialize(g, labels);
    zx::apply(g, m);
    row.nxt = zx::serialize(g, labels);  // survivors keep their labels
    p.rows.push_back(std::move(row));
  }
  return p;
}

}  // namespace ax::mathgen
