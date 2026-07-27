#pragma once
/** @file exact.hpp AXNN FX-V1 exact inference (rung 2b; spec:
    docs/specs/2026-07-27-axnn-exact.md, precision-doctrine amendment
    2026-07-27).

    Integer-only forward over a declared fixed-point model definition:
    exact GEMM (Q.16 operands, int64 accumulation — associative, so
    order-free), container-shipped tables for every transcendental
    (fx.act / fx.exp / fx.rsqrt / fx.rope.*), declared floor/RNE
    rounding everywhere. No float touches the forward path after load;
    bit-identity across platforms is a property of the arithmetic, not
    a testing goal. */
#include <ax/nn/model.hpp>

#include <cstdint>

namespace ax::nn {

class exact_model {
 public:
  /** Load an AXNN container carrying the FX-V1 tables. Throws on
      missing tables or shape violations (fail at load, never at
      forward). */
  static exact_model load(const std::string& path);
  static exact_model from_parts(config cfg,
                                const std::map<std::string, tensor>& t);

  /** Last-position logits, Q.32 int64 (the readout is never
      rescaled). Deterministic and platform-free. */
  std::vector<std::int64_t> logits_q32(const std::vector<int>& tokens) const;
  /** Greedy readout: argmax over exact logits, ties to LOWEST id. */
  int argmax(const std::vector<int>& tokens) const;
  /** FNV-1a 64 over the last-position logits (little-endian int64). */
  std::uint64_t logits_hash(const std::vector<int>& tokens) const;

  /** Greedy decode with a per-layer KV cache. Bit-exact with the
      full forward by construction: the stepper IS the forward (a
      position's integer ops never depend on later positions), so
      caching changes cost, never values. Returns the generated ids
      (prompt excluded); stops at stop_id (when >= 0), max_new, or
      max_seq. */
  std::vector<int> generate(const std::vector<int>& prompt, int max_new,
                            int stop_id = -1) const;

  /** Table certification (relay -4 ask 6): bounded-error vs a float
      reference, monotonicity where argmax correctness depends on it,
      seeded midpoint fuzz. Returns "" when certified, else the first
      violation. The reference uses doubles — allowed here because
      this is a CHECKER of the artifact, not part of the forward. */
  std::string certify_tables() const;

  const config& cfg() const { return cfg_; }

 private:
  void build(const std::map<std::string, tensor>& t);
  config cfg_;
  // Q.16 int32 weight tensors and int64 tables, keyed by tensor name.
  std::map<std::string, std::vector<std::int32_t>> w_;
  std::map<std::string, std::vector<std::int64_t>> tab_;
};

/** FX-V1 profile constants (also mirrored in scripts/nn_exact_ref.py;
    changing any of these is a PROFILE change, not a tweak). */
namespace fxv1 {
inline constexpr int kFrac = 16;                     // Q.16
inline constexpr std::int64_t kActSat = 2048LL << 16;   // Q11.16
inline constexpr std::int32_t kWeightSat = 128 << 16;   // Q7.16
inline constexpr std::int64_t kCentSat = 512LL << 16;   // Q9.16
inline constexpr std::int64_t kEpsQ32 = 42950;          // ~1e-5
inline constexpr int kActTableN = 2049;   // [-32, 32], step 1/32
inline constexpr int kExpTableN = 2049;   // [-16, 0], step 1/128
inline constexpr int kRsqrtTableN = 385;  // m in [1, 4], step 1/128
}  // namespace fxv1

}  // namespace ax::nn
