#pragma once
/** @file expr.hpp Immutable hash-consed symbolic expression DAG.

    Expressions are canonicalized at construction: add/mul are flattened,
    sorted, constant-folded, and like terms/factors are merged. Structural
    equality is therefore pointer equality (same()). Handles are immutable
    and safe to share across threads once built. */
#include <ax/core/rational.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ax::sym {

enum class kind : std::uint8_t { num, sym, add, mul, pow, fn };

struct node;

/** Immutable expression handle; cheap to copy. */
class expr {
 public:
  // ---- factories
  static expr num(std::int64_t n);
  static expr num(const rational& q);
  static expr symbol(std::string name);
  /** Named unary function: sin, cos, tan, exp, log, sqrt (extensible). */
  static expr fn(std::string name, expr arg);
  /** N-ary named node. Used for the unevaluated search carriers:
      Integral(f, x[, x...]), Derivative(f, x[, x...]), Subs(e, x, r).
      Carriers are opaque to canonicalization and throw on eval. */
  static expr fn(std::string name, std::vector<expr> args);
  static expr integral(expr f, expr x);
  static expr derivative(expr f, expr x);
  static expr subs_carrier(expr e, expr x, expr r);

  // ---- observers
  kind k() const;
  bool is_num() const { return k() == kind::num; }
  bool is_sym() const { return k() == kind::sym; }
  bool is_add() const { return k() == kind::add; }
  bool is_mul() const { return k() == kind::mul; }
  bool is_pow() const { return k() == kind::pow; }
  bool is_fn() const { return k() == kind::fn; }
  /** Numeric value; std::logic_error unless is_num(). */
  const rational& value() const;
  /** Symbol or function name; std::logic_error otherwise. */
  const std::string& name() const;
  /** add/mul: operands; pow: {base, exponent}; fn: {argument}. */
  std::span<const expr> args() const;
  /** Structural equality — pointer equality thanks to hash-consing. */
  bool same(const expr& o) const { return p_ == o.p_; }
  std::size_t hash() const;
  /** Total order for canonical sorting: negative/zero/positive like strcmp. */
  static int compare(const expr& a, const expr& b);

  // ---- algebra (canonicalizing)
  friend expr operator+(const expr& a, const expr& b);
  friend expr operator-(const expr& a, const expr& b);
  friend expr operator*(const expr& a, const expr& b);
  friend expr operator/(const expr& a, const expr& b);
  friend expr operator-(const expr& a);
  expr pow(const expr& exponent) const;

  // ---- operations
  /** Replace every occurrence of sym (a symbol) with replacement. */
  expr subs(const expr& sym, const expr& replacement) const;
  /** Numeric evaluation. Throws std::logic_error on unbound symbol or
      unknown function. */
  double eval(const std::map<std::string, double>& env = {}) const;

 private:
  explicit expr(std::shared_ptr<const node> p) : p_(std::move(p)) {}
  std::shared_ptr<const node> p_;
  friend struct node_access;
};

/** Re-canonicalize (rebuild through factories). Extra rewrite rules land in
    Phase 7; v1 simplification == canonical form. */
expr simplify(const expr& e);

}  // namespace ax::sym
