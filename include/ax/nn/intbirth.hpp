#pragma once
/** @file intbirth.hpp The integer-birth engine (relay 2026-08-01
    engine ask). The R2b full-block deterministic birth — int64
    forward/backward/IntAdamWQw, shipped-table nonlinearities, exact
    big-int bias correction — as a library, so house Python can spec
    and verify while this engine does the heavy running.

    Doctrine unchanged from the tool legs: tables and init are
    OPAQUE BYTES (AXP3 container / raw KEYS-order dump), never
    regenerated here; the contract carries every scale constant;
    digests are computed engine-side and compared house-side.
    Certification: the library-backed driver reproduces the R2b
    reference milestones (r2b_ref.json, final sha efe3557c...)
    bit-for-bit — the refactor is certified by the old digests.

    Rounding placement is part of the contract (booked spec rule):
    every multi-int_mm sum here rounds ONCE after the sum, exactly
    where the Python reference does. */
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ax::nn::ib {

using i64 = std::int64_t;

namespace detail {
/** Copyable running sha256 (FIPS 180-4); copy + hex() peeks the
    running digest without disturbing the stream. */
struct sha256 {
  std::uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                        0xa54ff53a, 0x510e527f, 0x9b05688c,
                        0x1f83d9ab, 0x5be0cd19};
  std::uint8_t buf[64];
  std::uint64_t len = 0;
  std::size_t fill = 0;
  void update(const void* data, std::size_t n);
  std::string hex();  ///< finalizes THIS copy; peek via a copy
 private:
  void block(const std::uint8_t* p);
};
}  // namespace detail

/** The r2b_ref.json contract, plus the block dims. All scales are
    explicit; nothing is derived from libm (attn scale = the integer
    rounding of Q*sqrt(DH), computed with exact isqrt). */
struct contract {
  int T = 32, D = 64, DH = 16, F = 128, V = 64;
  int shift = 12;             // Q_w = Q << shift
  i64 gboost = 256;           // backward boost (unboost at optimizer)
  i64 pq = 8192;              // attention-prob carry scale
  i64 act_clamp = 16384;      // residual clamp, Q units
  i64 eps32 = 42950;          // rmsnorm eps at 2^32
  i64 lrn = 1, lrd = 1000;    // lr = lrn/lrd
};

/** One full-block birth. */
class full_birth {
 public:
  /** tables_bytes: AXP3 with silu.tab / dsilu.tab / exp.tab /
      rope.cos / rope.sin (ranges derived from table sizes).
      init_bytes: 11 weight tensors in KEYS order, then x [T,D],
      then tgt [T], int64 LE. Throws std::runtime_error on
      malformed bytes or size mismatch (fail at load). */
  full_birth(const std::string& tables_bytes,
             const std::string& init_bytes, const contract& c);

  void run(int steps);              ///< advance n training steps
  int step_count() const { return step_; }
  i64 last_loss() const { return loss_; }
  double nz_last() const { return nz_; }

  /** Feed the current wide weights (KEYS order) into the running
      trajectory hash — the milestone protocol — and return its
      current hex digest. */
  std::string mark();
  /** Current running trajectory digest without marking. */
  std::string traj_sha() const;
  /** Wide (Q_w-scale) weights, KEYS order, int64 LE — for
      house-side gating and probes. */
  std::string weights_bytes() const;

  static const char* const KEYS[11];

 private:
  void step_once();
  contract c_;
  i64 scale_ = 0, ts_ = 0, tse_ = 0;  // attn scale, table ranges
  int step_ = 0;
  i64 loss_ = 0;
  double nz_ = 0;
  std::map<std::string, std::vector<i64>> w_;   // wide, Q_w scale
  std::vector<i64> x_, tgt_;
  std::map<std::string, std::vector<i64>> tab_;
  // optimizer state
  std::vector<std::vector<i64>> m_, v_;
  int t_ = 0;
  std::vector<std::uint32_t> p10_, p9_, p1000_, p999_;  // big-uints
  detail::sha256 th_;   // running trajectory hash
};

}  // namespace ax::nn::ib
