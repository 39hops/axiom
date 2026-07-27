/** @file inverse.cpp S7 inverse-move enumeration (see inverse.hpp).

    Stage 1 proposes candidate predecessors; stage 2 settles each by a
    full forward successors() pass and keeps every (rule, p) whose child
    set contains t exactly (hash-consed equality). The forward pass is
    the soundness boundary: a bad proposal wastes a candidate slot, it
    can never emit a false predecessor.

    Site vocabulary (the reassociation problem): a forward rewrite
    replaces a carrier node with its output rw, and canonicalization
    then MERGES rw into the surrounding add/mul — rw is usually not a
    subtree of the child. Candidates therefore come in three shapes:
      - subtree sites:      p = replace(t, s, node)
      - add sub-sum sites:  p = replace(t, A, A - F + node), F a bounded
                            subset-sum of A's terms
      - mul cofactor sites: p = replace(t, M, (M/c) * node), c a bounded
                            sub-product of M's factors
    all rebuilt through the canonicalizing factories, so p is exactly
    the tree the forward engine would have held. */
#include <ax/search/inverse.hpp>

#include <ax/sym/budget.hpp>
#include <ax/sym/calc.hpp>
#include <ax/sym/count_ops.hpp>
#include <ax/sym/expand.hpp>
#include <ax/sym/oracle.hpp>
#include <ax/sym/print_sstr.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <functional>
#include <set>
#include <unordered_set>
#include <vector>

namespace ax::search {

namespace {

using sym::expr;

const expr kOne = expr::num(1);
const expr kZero = expr::num(0);

bool contains(const expr& e, const expr& target) {
  if (e.same(target)) return true;
  for (const expr& a : e.args())
    if (contains(a, target)) return true;
  return false;
}

bool is_carrier(const expr& e) {
  return e.is_fn() && (e.name() == "Integral" || e.name() == "Derivative" ||
                       e.name() == "Subs");
}

bool has_carrier(const expr& e) {
  if (is_carrier(e)) return true;
  for (const expr& a : e.args())
    if (has_carrier(a)) return true;
  return false;
}

/** Replace every occurrence of target with repl through the
    canonicalizing factories (successors.cpp's replace_node twin). */
expr replace_node(const expr& e, const expr& target, const expr& repl) {
  if (e.same(target)) return repl;
  switch (e.k()) {
    case sym::kind::num:
    case sym::kind::sym:
      return e;
    case sym::kind::fn: {
      std::vector<expr> mapped;
      mapped.reserve(e.args().size());
      bool changed = false;
      for (const expr& a : e.args()) {
        mapped.push_back(replace_node(a, target, repl));
        changed = changed || !mapped.back().same(a);
      }
      if (!changed) return e;
      return expr::fn(e.name(), std::move(mapped));
    }
    case sym::kind::add: {
      expr out = kZero;
      bool changed = false;
      for (const expr& x : e.args()) {
        const expr m = replace_node(x, target, repl);
        changed = changed || !m.same(x);
        out = out + m;
      }
      return changed ? out : e;
    }
    case sym::kind::mul: {
      expr out = kOne;
      bool changed = false;
      for (const expr& x : e.args()) {
        const expr m = replace_node(x, target, repl);
        changed = changed || !m.same(x);
        out = out * m;
      }
      return changed ? out : e;
    }
    case sym::kind::pow: {
      const expr b = replace_node(e.args()[0], target, repl);
      const expr x = replace_node(e.args()[1], target, repl);
      if (b.same(e.args()[0]) && x.same(e.args()[1])) return e;
      return b.pow(x);
    }
  }
  return e;
}

int count_occurrences(const expr& e, const expr& target) {
  if (e.same(target)) return 1;
  int n = 0;
  for (const expr& a : e.args()) n += count_occurrences(a, target);
  return n;
}

/** Replace only the k-th (pre-order) occurrence of target. Forward
    replace_node hits every occurrence of the CARRIER node, but the
    closed VALUE can collide with pre-existing content (measured:
    9*x**2 + Integral(9*x**2, x) — re-carrying both x**2 occurrences
    corrupts the untouched integrand), so the inverse also proposes
    each single-occurrence re-carry. */
expr replace_kth(const expr& e, const expr& target, const expr& repl,
                 int& k) {
  if (k < 0) return e;
  if (e.same(target)) {
    if (k-- == 0) return repl;
    return e;
  }
  switch (e.k()) {
    case sym::kind::num:
    case sym::kind::sym:
      return e;
    case sym::kind::fn: {
      std::vector<expr> mapped;
      mapped.reserve(e.args().size());
      bool changed = false;
      for (const expr& a : e.args()) {
        mapped.push_back(replace_kth(a, target, repl, k));
        changed = changed || !mapped.back().same(a);
      }
      if (!changed) return e;
      return expr::fn(e.name(), std::move(mapped));
    }
    case sym::kind::add: {
      expr out = expr::num(0);
      bool changed = false;
      for (const expr& x : e.args()) {
        const expr m = replace_kth(x, target, repl, k);
        changed = changed || !m.same(x);
        out = out + m;
      }
      return changed ? out : e;
    }
    case sym::kind::mul: {
      expr out = expr::num(1);
      bool changed = false;
      for (const expr& x : e.args()) {
        const expr m = replace_kth(x, target, repl, k);
        changed = changed || !m.same(x);
        out = out * m;
      }
      return changed ? out : e;
    }
    case sym::kind::pow: {
      const expr b = replace_kth(e.args()[0], target, repl, k);
      const expr x = replace_kth(e.args()[1], target, repl, k);
      if (b.same(e.args()[0]) && x.same(e.args()[1])) return e;
      return b.pow(x);
    }
  }
  return e;
}

/** Replace the VALUE V = c*R (R a structural mul factor) with repl in
    every mul that carries R, and every bare occurrence of V. Handles
    the flattened-coefficient shapes a subtree walk cannot see:
    x**2*(14 - 39*x)/2 holds V = x**2/2 across factors {1/2, x**2, ...}. */
expr replace_value(const expr& e, const expr& V, const expr& R,
                   const expr& repl) {
  if (e.same(V)) return repl;
  if (e.is_mul()) {
    for (const expr& f : e.args())
      if (f.same(R)) return (e / V) * repl;
    // fall through: rebuild args (nested muls are flattened, so no
    // recursion into mul-of-mul, but fn/pow args below may carry V)
  }
  switch (e.k()) {
    case sym::kind::num:
    case sym::kind::sym:
      return e;
    case sym::kind::fn: {
      std::vector<expr> mapped;
      mapped.reserve(e.args().size());
      bool changed = false;
      for (const expr& a : e.args()) {
        mapped.push_back(replace_value(a, V, R, repl));
        changed = changed || !mapped.back().same(a);
      }
      if (!changed) return e;
      return expr::fn(e.name(), std::move(mapped));
    }
    case sym::kind::add: {
      expr out = expr::num(0);
      bool changed = false;
      for (const expr& x : e.args()) {
        const expr m = replace_value(x, V, R, repl);
        changed = changed || !m.same(x);
        out = out + m;
      }
      return changed ? out : e;
    }
    case sym::kind::mul: {
      expr out = expr::num(1);
      bool changed = false;
      for (const expr& x : e.args()) {
        const expr m = replace_value(x, V, R, repl);
        changed = changed || !m.same(x);
        out = out * m;
      }
      return changed ? out : e;
    }
    case sym::kind::pow: {
      const expr b = replace_value(e.args()[0], V, R, repl);
      const expr x = replace_value(e.args()[1], V, R, repl);
      if (b.same(e.args()[0]) && x.same(e.args()[1])) return e;
      return b.pow(x);
    }
  }
  return e;
}

void free_symbols(const expr& e, std::set<std::string>& out) {
  if (e.is_sym()) {
    const std::string& n = e.name();
    if (n != "pi" && n != "E" && n != "u_") out.insert(n);
    return;
  }
  for (const expr& a : e.args()) free_symbols(a, out);
}

struct expr_hash {
  std::size_t operator()(const expr& e) const { return e.hash(); }
};
struct expr_eq {
  bool operator()(const expr& a, const expr& b) const { return a.same(b); }
};

/** Unique subtrees of t in tree order (hash-consing makes dedupe cheap). */
void collect_sites(const expr& e,
                   std::unordered_set<expr, expr_hash, expr_eq>& seen,
                   std::vector<expr>& out) {
  if (seen.insert(e).second) out.push_back(e);
  for (const expr& a : e.args()) collect_sites(a, seen, out);
}

/** Unique add / mul nodes of t (for sub-sum / cofactor sites). */
void collect_kind(const expr& e, sym::kind k,
                  std::unordered_set<expr, expr_hash, expr_eq>& seen,
                  std::vector<expr>& out) {
  if (e.k() == k && seen.insert(e).second) out.push_back(e);
  for (const expr& a : e.args()) collect_kind(a, k, seen, out);
}

/** Bare single-var Integral(f, x) test; returns f on match. */
std::optional<expr> integral_over(const expr& e, const expr& x) {
  if (e.is_fn() && e.name() == "Integral" && e.args().size() == 2 &&
      e.args()[1].same(x))
    return e.args()[0];
  return std::nullopt;
}

std::optional<expr> derivative_over(const expr& e, const expr& x) {
  if (e.is_fn() && e.name() == "Derivative" && e.args().size() == 2 &&
      e.args()[1].same(x))
    return e.args()[0];
  return std::nullopt;
}

/** Split a term into (numeric coefficient, rest). */
std::pair<expr, expr> coeff_split(const expr& term) {
  if (term.is_num()) return {term, kOne};
  if (!term.is_mul()) return {kOne, term};
  expr c = kOne;
  expr rest = kOne;
  for (const expr& f : term.args())
    (f.is_num() ? c : rest) = (f.is_num() ? c : rest) * f;
  return {c, rest};
}

// ------------------------------------------------------------ candidates

/** Fold each mul's numeric coefficient into its first add factor
    (pairwise num*add multiplication folds; n-ary flattening does not).
    The same value can live as mul{2, 7 - 39*x/2, I} or mul{14 - 39*x,
    I} depending on REBUILD ORDER — forward children carry whichever
    spelling the engine's replace produced, so every inverse candidate
    proposes both (measured: the nested i_power gate misses were all
    coefficient-spelling mismatches, never wrong values). */
expr fold_num_into_add(const expr& e) {
  switch (e.k()) {
    case sym::kind::num:
    case sym::kind::sym:
      return e;
    case sym::kind::fn: {
      std::vector<expr> mapped;
      mapped.reserve(e.args().size());
      bool changed = false;
      for (const expr& a : e.args()) {
        mapped.push_back(fold_num_into_add(a));
        changed = changed || !mapped.back().same(a);
      }
      if (!changed) return e;
      return expr::fn(e.name(), std::move(mapped));
    }
    case sym::kind::add: {
      expr out = kZero;
      for (const expr& x : e.args()) out = out + fold_num_into_add(x);
      return out;
    }
    case sym::kind::mul: {
      expr c = kOne;
      std::optional<expr> first_add;
      std::vector<expr> rest;
      for (const expr& f : e.args()) {
        const expr m = fold_num_into_add(f);
        if (m.is_num())
          c = c * m;
        else if (m.is_add() && !first_add)
          first_add = m;
        else
          rest.push_back(m);
      }
      if (!first_add || c.same(kOne)) return e;
      expr out = c * *first_add;  // pairwise: folds
      for (const expr& f : rest) out = out * f;
      return out;
    }
    case sym::kind::pow:
      return fold_num_into_add(e.args()[0])
          .pow(fold_num_into_add(e.args()[1]));
  }
  return e;
}

struct candidate_sink {
  std::vector<expr> order;  // proposal order preserved
  std::unordered_set<expr, expr_hash, expr_eq> seen;
  const expr* t = nullptr;
  std::size_t cap = 0;

  void add(const expr& p) {
    add_one(p);
    const expr folded = fold_num_into_add(p);
    if (!folded.same(p)) add_one(folded);
  }
  bool full() const { return order.size() >= cap; }

 private:
  void add_one(const expr& p) {
    if (order.size() >= cap) return;
    if (p.same(*t)) return;
    if (sym::count_ops(p) > 400) return;  // verify_edge's size gate
    if (seen.insert(p).second) order.push_back(p);
  }
};

/** Diff-trick nodes for a closed form F over x: Integrals whose
    single-shot closure IS F, for whichever rule fires. Both the raw
    derivative and its rational-canonical form are proposed — canonical()
    can rewrite a sum over a common denominator, and rules fire on
    integrand SHAPE (measured: i_linear_basis missed 4/18 gate edges when
    only the canonical form was tried). Empty when F carries carriers,
    does not depend on x, or the derivative degenerates. */
std::vector<expr> diff_trick_nodes(const expr& F, const expr& x) {
  if (has_carrier(F) || !contains(F, x)) return {};
  try {
    sym::work_budget_scope budget(std::chrono::milliseconds(200));
    const expr raw = sym::diff(F, x);
    if (raw.is_num() && raw.value() == ax::rational{}) return {};
    if (sym::count_ops(raw) > 200) return {};
    std::vector<expr> out{expr::integral(raw, x)};
    const expr cn = sym::canonical(raw, x);
    if (!cn.same(raw) && sym::count_ops(cn) <= 200)
      out.push_back(expr::integral(cn, x));
    return out;
  } catch (const std::exception&) {
    return {};
  }
}

/** Factor shared carrier atoms back out of adds, innermost-first:
    sum_i ci*V + rest -> (sum_i ci)*V + rest. The cancel algebra move
    treats carrier atoms as opaque polynomial generators and distributes
    their cofactors; this is its structural inverse. */
expr collect_carriers(const expr& e) {
  switch (e.k()) {
    case sym::kind::num:
    case sym::kind::sym:
      return e;
    case sym::kind::fn: {
      std::vector<expr> mapped;
      mapped.reserve(e.args().size());
      bool changed = false;
      for (const expr& a : e.args()) {
        mapped.push_back(collect_carriers(a));
        changed = changed || !mapped.back().same(a);
      }
      if (!changed) return e;
      return expr::fn(e.name(), std::move(mapped));
    }
    case sym::kind::mul: {
      expr out = kOne;
      for (const expr& f : e.args()) out = out * collect_carriers(f);
      return out;
    }
    case sym::kind::pow:
      return collect_carriers(e.args()[0])
          .pow(collect_carriers(e.args()[1]));
    case sym::kind::add: {
      std::vector<expr> terms;
      for (const expr& x : e.args()) terms.push_back(collect_carriers(x));
      // group terms by their first carrier factor
      std::vector<std::pair<expr, expr>> groups;  // (V, cofactor sum)
      expr rest = kZero;
      for (const expr& term : terms) {
        std::optional<expr> V;
        if (is_carrier(term)) V = term;
        else if (term.is_mul())
          for (const expr& f : term.args())
            if (is_carrier(f)) {
              V = f;
              break;
            }
        if (!V) {
          rest = rest + term;
          continue;
        }
        const expr cof = term / *V;
        bool merged = false;
        for (auto& [gv, gsum] : groups)
          if (gv.same(*V)) {
            gsum = gsum + cof;
            merged = true;
          }
        if (!merged) groups.emplace_back(*V, cof);
      }
      expr out = rest;
      for (const auto& [gv, gsum] : groups) out = out + gsum * gv;
      return out;
    }
  }
  return e;
}

/** Stage 1: propose candidate predecessors of t. */
void propose(const expr& t, const std::vector<expr>& vars,
             candidate_sink& sink) {
  std::unordered_set<expr, expr_hash, expr_eq> site_seen;
  std::vector<expr> sites;
  collect_sites(t, site_seen, sites);

  std::unordered_set<expr, expr_hash, expr_eq> add_seen, mul_seen;
  std::vector<expr> adds, muls;
  collect_kind(t, sym::kind::add, add_seen, adds);
  collect_kind(t, sym::kind::mul, mul_seen, muls);

  // ---- whole-expression algebra near-involutions (cancel <-> expand)
  try {
    sym::work_budget_scope budget(std::chrono::milliseconds(400));
    const expr ex = sym::expand(t);
    if (!ex.same(t)) sink.add(ex);  // cancel(ex) may reach t
    const expr col = collect_carriers(t);
    if (!col.same(t)) sink.add(col);  // cancel(col) may reach t
    for (const expr& x : vars) {
      const expr cn = sym::canonical(t, x);
      if (!cn.same(t)) sink.add(cn);  // expand(cn) may reach t
    }
  } catch (const std::exception&) {
  }

  // ---- subtree sites: generic diff-trick (covers i_power / i_table /
  // tranche-2/3 closers) + structural carriers (i_usub, i_sum, ...)
  for (const expr& s : sites) {
    if (sink.full()) return;
    if (s.is_num()) continue;

    // i_usub inverse: Subs(Integral(q, u_), u_, g) came from
    // Integral(q[u_ := g] * dg/dx, x)
    if (s.is_fn() && s.name() == "Subs" && s.args().size() == 3) {
      const expr& body = s.args()[0];
      const expr& u = s.args()[1];
      const expr& g = s.args()[2];
      if (body.is_fn() && body.name() == "Integral" &&
          body.args().size() == 2 && body.args()[1].same(u)) {
        std::set<std::string> gf;
        free_symbols(g, gf);
        for (const std::string& xn : gf) {
          const expr x = expr::symbol(xn);
          try {
            sym::work_budget_scope budget(std::chrono::milliseconds(400));
            const expr q = body.args()[0].subs(u, g);
            const expr f = sym::canonical(q * sym::diff(g, x), x);
            sink.add(replace_node(t, s, expr::integral(f, x)));
            const expr fe = sym::expand(f);
            if (!fe.same(f))
              sink.add(replace_node(t, s, expr::integral(fe, x)));
          } catch (const std::exception&) {
          }
        }
      }
      continue;
    }

    for (const expr& x : vars) {
      for (const expr& node : diff_trick_nodes(s, x))
        sink.add(replace_node(t, s, node));

      // cofactor-absorbed closer: c*Integral(frest, x) forwards to
      // (c*fc)*G where diff(s) = fc*frest and G = s/... is the unit
      // antiderivative. Pull the derivative's numeric content fc back
      // OUT of the integrand and absorb it into the site's cofactor —
      // mul flattening makes s a subtree of c*s, so the canonical
      // rebuild merges fc with any surrounding coefficient (measured:
      // i_power went 0/52 on the gate without this — every merged-
      // coefficient fire looked like an unreachable unit split).
      if (!has_carrier(s) && contains(s, x)) {
        try {
          sym::work_budget_scope budget(std::chrono::milliseconds(200));
          const expr raw = sym::diff(s, x);
          const auto [fc, frest] = coeff_split(raw);
          if (!(raw.is_num() && raw.value() == ax::rational{}) &&
              !fc.same(kOne) && sym::count_ops(frest) <= 200) {
            const expr node = fc * expr::integral(frest, x);
            sink.add(replace_node(t, s, node));
            // value-collision variants: re-carry each occurrence alone
            const int occ = count_occurrences(t, s);
            if (occ >= 2 && occ <= 4)
              for (int i = 0; i < occ; ++i) {
                int k = i;
                sink.add(replace_kth(t, s, node, k));
              }
          }
        } catch (const std::exception&) {
        }
      }

      // flattened-coefficient value sites: a closed v**m/m hides across
      // mul factors ({1/m', v**m, cofactor...}) at EVERY occurrence the
      // forward replace-all touched, including inside other integrands
      // (the i_parts nesting shape). Re-carry the VALUE everywhere.
      if (s.is_pow() && s.args()[0].same(x) && s.args()[1].is_num()) {
        const expr& m = s.args()[1];
        if (!(m.value() == ax::rational{})) {
          const expr V = s / m;
          const expr integrand = s / x;  // v**(m-1)
          const expr node = expr::integral(integrand, x);
          const expr p = replace_value(t, V, s, node);
          if (!p.same(t)) sink.add(p);
        }
      }
    }
  }

  // ---- add nodes: sub-sum sites
  for (const expr& A : adds) {
    if (sink.full()) return;
    const auto terms = A.args();
    const std::size_t k = terms.size();

    for (const expr& x : vars) {
      // i_sum inverse: the bare-Integral terms recombine into one
      // Integral of their summed integrands. All-of-them is the common
      // shape (forward i_sum splits an entire Add); size-2 subsets keep
      // partial-split predecessors reachable on small sums.
      std::vector<std::size_t> ints;
      for (std::size_t i = 0; i < k; ++i)
        if (integral_over(terms[i], x)) ints.push_back(i);
      const auto recombine = [&](const std::vector<std::size_t>& pick) {
        expr F = kZero;
        expr f = kZero;
        for (const std::size_t i : pick) {
          F = F + terms[i];
          f = f + *integral_over(terms[i], x);
        }
        sink.add(replace_node(t, A, A - F + expr::integral(f, x)));
      };
      if (ints.size() >= 2) {
        recombine(ints);
        if (ints.size() > 2 && ints.size() <= 6)
          for (std::size_t a = 0; a < ints.size(); ++a)
            for (std::size_t b = a + 1; b < ints.size(); ++b)
              recombine({ints[a], ints[b]});
      }

      // partial-closure inverse (i_unprod family): a closed x-dependent
      // part plus residual Integral terms recombine into one Integral of
      // (residual integrands + d/dx closed part). x-free terms stay in
      // context — diff would erase them from the reconstruction.
      if (!ints.empty()) {
        expr closed = kZero;
        bool any_closed = false;
        for (std::size_t i = 0; i < k; ++i) {
          if (integral_over(terms[i], x)) continue;
          if (!contains(terms[i], x) || has_carrier(terms[i])) continue;
          closed = closed + terms[i];
          any_closed = true;
        }
        if (any_closed) {
          try {
            sym::work_budget_scope budget(std::chrono::milliseconds(200));
            expr f = sym::canonical(sym::diff(closed, x), x);
            expr F = closed;
            for (const std::size_t i : ints) {
              F = F + terms[i];
              f = f + *integral_over(terms[i], x);
            }
            sink.add(replace_node(t, A, A - F + expr::integral(f, x)));
          } catch (const std::exception&) {
          }
        }
      }

      // i_table log inverse with merged linear term: the closer's
      // x*log(x) - x lands with its -x absorbed by context. Spot the
      // unit v*log(v) term and re-split F = v*log(v) - v.
      for (std::size_t i = 0; i < k; ++i) {
        const auto [tc, trest] = coeff_split(terms[i]);
        if (!tc.same(kOne) || !trest.is_mul() ||
            trest.args().size() != 2)
          continue;
        const expr& f0 = trest.args()[0];
        const expr& f1 = trest.args()[1];
        const expr* v = nullptr;
        if (f1.is_fn() && f1.name() == "log" && f1.args()[0].same(f0))
          v = &f0;
        else if (f0.is_fn() && f0.name() == "log" && f0.args()[0].same(f1))
          v = &f1;
        if (v == nullptr || !v->is_sym()) continue;
        const expr F = trest - *v;
        sink.add(replace_node(
            t, A, A - F + expr::integral(expr::fn("log", *v), *v)));
      }

      // d_sum inverse: bare-Derivative terms recombine likewise.
      std::vector<std::size_t> ders;
      for (std::size_t i = 0; i < k; ++i)
        if (derivative_over(terms[i], x)) ders.push_back(i);
      if (ders.size() >= 2) {
        expr F = kZero;
        expr f = kZero;
        for (const std::size_t i : ders) {
          F = F + terms[i];
          f = f + *derivative_over(terms[i], x);
        }
        sink.add(replace_node(t, A, A - F + expr::derivative(f, x)));
      }

      // d_product inverse: D(u,x)*v + u*D(v,x) recombines into
      // Derivative(u*v, x). Cross-matched pairs: T1's Derivative
      // argument is T2's cofactor and vice versa.
      for (std::size_t ai = 0; ai < k && !sink.full(); ++ai) {
        if (!terms[ai].is_mul()) continue;
        for (const expr& f : terms[ai].args()) {
          const auto du = derivative_over(f, x);
          if (!du) continue;
          const expr v = terms[ai] / f;  // T1 = D(u,x) * v
          for (std::size_t bi = 0; bi < k; ++bi) {
            if (bi == ai || !terms[bi].is_mul()) continue;
            bool cross = false;
            for (const expr& g : terms[bi].args())
              if (const auto dv = derivative_over(g, x))
                cross = cross || (dv->same(v) && (terms[bi] / g).same(*du));
            if (!cross) continue;
            const expr F = terms[ai] + terms[bi];
            sink.add(replace_node(
                t, A, A - F + expr::derivative(*du * v, x)));
          }
        }
      }

      // i_parts inverse: u*V - Integral(V*du, x) with V = Integral(dv, x)
      // recombines into Integral(u*dv, x). Scan (positive, integral-term)
      // ordered pairs; the Integral factor V must appear in both.
      for (std::size_t bi = 0; bi < k && !sink.full(); ++bi) {
        const auto [bc, brest] = coeff_split(terms[bi]);
        if (!bc.is_num()) continue;
        // brest = Integral(V * du, x) (possibly bare Integral); the
        // context coefficient c = -bc carries any surrounding sign, so
        // an i_parts fire under a negated site inverts too
        const auto big = integral_over(brest, x);
        if (!big) continue;
        const expr c = -bc;
        // find the Integral(dv, x) factor V inside the integrand
        std::vector<expr> vfactors;
        if (big->is_mul()) {
          for (const expr& f : big->args())
            if (integral_over(f, x)) vfactors.push_back(f);
        } else if (integral_over(*big, x)) {
          vfactors.push_back(*big);
        }
        for (const expr& V : vfactors) {
          const expr du = *big / V;  // may be 1 (invisible-*1 case)
          const expr dv = *integral_over(V, x);
          for (std::size_t ai = 0; ai < k; ++ai) {
            if (ai == bi) continue;
            if (!contains(terms[ai], V)) continue;
            const expr u = terms[ai] / V / c;  // unit-context u
            if (contains(u, V)) continue;  // V^2 shapes: not i_parts
            const expr F = terms[ai] + terms[bi];
            sink.add(replace_node(
                t, A, A - F + c * expr::integral(u * dv, x)));
            // context-coefficient variant: uc*Integral(u'*dv, x) with
            // u = uc*u' — the fire happened under a numeric cofactor
            const auto [uc, urest] = coeff_split(u);
            if (!uc.same(kOne))
              sink.add(replace_node(
                  t, A, A - F + c * uc * expr::integral(urest * dv, x)));
          }
        }
      }

      // generic diff-trick over bounded sub-sums (i_table's log closer
      // x*log(x) - x and friends land as 2-term merges)
      if (k <= 8) {
        for (std::size_t a = 0; a < k && !sink.full(); ++a)
          for (std::size_t b = a + 1; b < k; ++b) {
            const expr F = terms[a] + terms[b];
            for (const expr& node : diff_trick_nodes(F, x))
              sink.add(replace_node(t, A, A - F + node));
          }
      }
    }
  }

  // ---- mul nodes: cofactor sites + coefficient splits
  for (const expr& M : muls) {
    if (sink.full()) return;
    const auto factors = M.args();
    const auto [mc, mrest] = coeff_split(M);

    for (const expr& x : vars) {
      // i_const inverse: M with factor x and the rest x-free came from
      // Integral(M/x, x)
      bool has_x = false;
      bool rest_free = true;
      for (const expr& f : factors) {
        if (f.same(x)) has_x = true;
        else if (contains(f, x)) rest_free = false;
      }
      if (has_x && rest_free)
        sink.add(replace_node(t, M, expr::integral(M / x, x)));

      // i_const_factor / d_const_factor inverse: an Integral/Derivative
      // factor re-absorbs a sub-product of its x-free cofactor. Bounded:
      // the full cofactor, each single factor, and the numeric part.
      for (const expr& f : factors) {
        const auto fi = integral_over(f, x);
        const auto fd = derivative_over(f, x);
        if (!fi && !fd) continue;
        const expr cof = M / f;
        if (cof.same(kOne) || contains(cof, x)) continue;
        std::vector<expr> absorbs{cof, -cof};
        if (cof.is_mul())
          for (const expr& c : cof.args()) {
            absorbs.push_back(c);
            absorbs.push_back(-c);  // sign stays in context (a - I shape)
          }
        for (const expr& c : absorbs) {
          if (c.same(kOne)) continue;
          const expr keep = cof / c;
          const expr inner = fi ? expr::integral(c * *fi, x)
                                : expr::derivative(c * *fd, x);
          sink.add(replace_node(t, M, keep * inner));
        }
      }

      // coefficient-split closers: the closer's output merged its
      // numeric coefficient with a pre-existing like term. Re-split the
      // unit-content shapes: x^m/m (i_power), log/trig/exp tables.
      if (!mc.same(kOne) && !mrest.same(kOne) && !has_carrier(mrest)) {
        std::vector<expr> Fs;
        if (mrest.is_pow() && mrest.args()[0].same(x) &&
            mrest.args()[1].is_num())
          Fs.push_back(mrest / mrest.args()[1]);  // x^m/m
        if (mrest.same(x)) Fs.push_back(x);       // Integral(1, x) -> x
        if (mrest.is_fn() && mrest.args().size() == 1) {
          if (mrest.name() == "cos") Fs.push_back(-mrest);  // -cos <- sin
          if (mrest.name() == "sin" || mrest.name() == "exp" ||
              mrest.name() == "log")
            Fs.push_back(mrest);
        }
        for (const expr& F : Fs) {
          // guard the no-op re-split (M == F): plain subtree site
          // already proposes it
          if (M.same(F)) continue;
          for (const expr& node : diff_trick_nodes(F, x))
            sink.add(replace_node(t, M, M - F + node));
        }
      }

      // d_power / d_chain_table inverse: a Derivative(g, x) factor with
      // its table/power cofactor recombines into one Derivative carrier.
      for (const expr& f : factors) {
        const auto fd = derivative_over(f, x);
        if (!fd) continue;
        const expr& g = *fd;
        const expr cof = M / f;
        // d_chain_table: cof == outer'(g)
        if (cof.is_fn() && cof.args().size() == 1 &&
            cof.args()[0].same(g)) {
          if (cof.name() == "cos")
            sink.add(replace_node(t, M, expr::derivative(expr::fn("sin", g), x)));
          if (cof.name() == "exp")
            sink.add(replace_node(t, M, expr::derivative(expr::fn("exp", g), x)));
        }
        const auto [cc, crest] = coeff_split(cof);
        if (crest.is_fn() && crest.args().size() == 1 &&
            crest.args()[0].same(g) && crest.name() == "sin" &&
            cc.is_num() && cc.value() < ax::rational{})
          sink.add(replace_node(
              t, M, (-cc) * expr::derivative(expr::fn("cos", g), x)));
        if (cof.is_pow() && cof.args()[0].same(g) &&
            cof.args()[1].is_num() &&
            cof.args()[1].value() ==
                ax::rational(ax::bigint(-1)))  // 1/g <- log
          sink.add(replace_node(t, M, expr::derivative(expr::fn("log", g), x)));
        // d_power: cof == n * g^(n-1)
        if (crest.is_pow() && crest.args()[0].same(g) &&
            crest.args()[1].is_num() && cc.is_num()) {
          const expr n = expr::num(crest.args()[1].value()) + kOne;
          if (n.is_num() && cc.value() == n.value() &&
              !(n.value() == ax::rational{}))
            sink.add(replace_node(t, M, expr::derivative(g.pow(n), x)));
        }
        if (crest.same(g) && cc.is_num() &&
            cc.value() == ax::rational(ax::bigint(2)))  // n == 2
          sink.add(replace_node(
              t, M, expr::derivative(g.pow(expr::num(2)), x)));
      }
    }
  }
}

}  // namespace

std::vector<predecessor> predecessors(const expr& t, const rule_set& rules,
                                      const inverse_options& opt) {
  std::vector<predecessor> out;

  std::set<std::string> names;
  free_symbols(t, names);
  std::vector<expr> vars;
  for (const std::string& n : names) vars.push_back(expr::symbol(n));
  if (vars.empty()) vars.push_back(expr::symbol("x"));

  candidate_sink sink;
  sink.t = &t;
  sink.cap = opt.max_candidates;
  propose(t, vars, sink);

  // Stage 2: forward-verify funnel. successors() runs the full rule set
  // with verify_p = 1, so every kept (rule, p) is an oracle-verified
  // legal edge whose child is exactly t.
  successor_options sopt;
  sopt.use_macros = opt.use_macros;
  sopt.verify_p = 1.0;
  sopt.deadline = opt.deadline;
  static const bool debug = std::getenv("AX_INV_DEBUG") != nullptr;
  for (const expr& p : sink.order) {
    if (opt.deadline && std::chrono::steady_clock::now() > *opt.deadline)
      break;
    if (debug)
      std::cerr << "[inv-cand] " << sym::to_sstr(p) << "\n";
    state s{p};
    // successors() dedupes children per call, so each (rule, p) match
    // appears at most once; candidate p's are unique by construction.
    for (const auto& [name, child] : successors(s, rules, sopt))
      if (child.e.same(t)) out.push_back(predecessor{name, p});
  }
  return out;
}

}  // namespace ax::search
