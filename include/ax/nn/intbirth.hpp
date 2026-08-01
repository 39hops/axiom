#pragma once
/** @file intbirth.hpp The integer-birth engine (relay 2026-08-01
    engine ask; primitives split per the follow-up). Layers:

      int_gemm forms (mmT / mm / xty)     — free functions
      block                               — R2b fwd/bwd + int softmax
      adamw                               — IntAdamWQw, exact big-int
                                            bias correction
      full_birth                          — the composed loop (the
                                            certified R2b trajectory)

    full_birth is BUILT FROM block + adamw, so the primitive layer is
    certified by the same r2b_ref.json digests as the composed one
    (final sha efe3557c...). Doctrine unchanged: tables and init are
    opaque bytes, never regenerated here; the contract carries every
    scale constant; digests engine-side, comparison house-side.

    Rounding placement is part of the contract (booked spec rule):
    every multi-int_gemm sum rounds ONCE after the sum, where the
    Python reference does. */
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ax::nn::ib {

using i64 = std::int64_t;
using Mat = std::vector<i64>;  // row-major

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

// ---- int_gemm forms (exact int64 sum-reduce; caller rounds) ----
/** a[rows,K] @ w[N,K]^T -> [rows,N] (the torch int_mm convention). */
Mat int_gemm(const Mat& a, int rows, int K, const Mat& w, int N);
/** a[rows,K] @ w[K,N] -> [rows,N] (int_mm against w.T). */
Mat int_gemm_nt(const Mat& a, int rows, int K, const Mat& w, int N);
/** x[rows,K]^T @ y[rows,N] -> [K,N] (the dW outer form). */
Mat int_gemm_xty(const Mat& x, int rows, int K, const Mat& y, int N);
/** Round-half-away division, elementwise (the program-wide rdiv). */
void rdiv_inplace(Mat& m, i64 d);

/** The r2b_ref.json contract, plus the block dims. All scales are
    explicit; nothing is derived from libm (attn scale = the integer
    rounding of Q*sqrt(DH), computed with exact isqrt). */
struct contract {
  int T = 32, D = 64, DH = 16, F = 128, V = 64;
  int n_blocks = 1;           // multi-block anatomy (mb spec)
  int shift = 12;             // Q_w = Q << shift
  i64 gboost = 256;           // backward boost (unboost at optimizer)
  i64 pq = 8192;              // attention-prob carry scale
  i64 act_clamp = 16384;      // residual clamp, Q units
  i64 eps32 = 42950;          // rmsnorm eps at 2^32
  i64 lrn = 1, lrd = 1000;    // lr = lrn/lrd
};

/** Forward activations/masks the backward needs; opaque to Python. */
struct block_cache {
  Mat x, h1, i1, q0, k0, v0, qr, kr, p, a, m1, x1, h2, i2, gp, u, sg,
      f, m2, x2, h3, i3;
};

/** The R2b block: fwd/bwd/softmax over Q-scale weights (KEYS-order
    names). Holds the parsed shipped tables; stateless otherwise. */
class block {
 public:
  block(const std::string& tables_bytes, const contract& c);
  /** logits [T,V]; fills the cache for bwd. Weights at Q scale. */
  Mat fwd(const std::map<std::string, Mat>& w, const Mat& x,
          block_cache& cache) const;
  /** Param grads (boosted scale, keyed like weights) + dx0 in
      *dx0_out if non-null (the multi-block chain point). */
  std::map<std::string, Mat> bwd(const std::map<std::string, Mat>& w,
                                 const Mat& dlogits,
                                 const block_cache& cache,
                                 Mat* dx0_out = nullptr) const;
  /** Integer row softmax at `scale` units via the shipped exp
      table (the r1b construction). rows x C row-major. */
  Mat softmax_rows(const Mat& s, int rows, int C, i64 scale) const;
  const contract& cfg() const { return c_; }
  /** Validate a weight map against the contract shapes. */
  void check_weights(const std::map<std::string, Mat>& w) const;

  // ---- Body layer (mb spec): the Block minus g3/wh, chainable.
  /** x -> x2 (post-clamp); 9-param weight map (BODY_KEYS). */
  Mat body_fwd(const std::map<std::string, Mat>& w, const Mat& x,
               block_cache& cache) const;
  /** dxin = grad wrt this body's OUTPUT x2 (post-clamp; first op
      is the m2 clamp mask). Returns the 9 body grads; *dx0_out
      gets the residual-chained input grad. */
  std::map<std::string, Mat> body_bwd(
      const std::map<std::string, Mat>& w, const Mat& dxin,
      const block_cache& cache, Mat* dx0_out) const;
  /** rmsnorm fwd/bwd over [T, D] (the mb head norm needs them). */
  Mat rms_fwd(const Mat& x, const Mat& g, Mat& isq_out) const;
  Mat rms_bwd(const Mat& dy, const Mat& x, const Mat& g,
              const Mat& isq, Mat& dg_out) const;

  static const char* const KEYS[11];
  static const char* const BODY_KEYS[9];

 private:
  // Body halves (gravmoe seam): body = ffn ∘ attn; the MoE body
  // swaps only the ffn half. Residual adds + clamp masks stay in
  // the composing caller so rounding placement is unchanged.
  Mat attn_fwd(const std::map<std::string, Mat>& w, const Mat& x,
               block_cache& c) const;
  Mat ffn_fwd(const std::map<std::string, Mat>& w, const Mat& x1,
              block_cache& c) const;
  Mat ffn_bwd(const std::map<std::string, Mat>& w,
              const Mat& dx2_masked, const block_cache& c,
              std::map<std::string, Mat>& G) const;
  Mat attn_bwd(const std::map<std::string, Mat>& w,
               const Mat& dx1_masked, const block_cache& c,
               std::map<std::string, Mat>& G) const;

  contract c_;
  i64 scale_ = 0, ts_ = 0, tse_ = 0;
  std::map<std::string, Mat> tab_;
};

/** IntAdamWQw: weights at Q_w = Q << shift, exact big-int bias
    correction, decoupled decay. State keyed by parameter index. */
class adamw {
 public:
  adamw(int shift, i64 lrn, i64 lrd);
  /** One step over parallel param/grad lists; params (Q_w scale)
      are updated in place. Grads at the unboosted Q scale. */
  void step(const std::vector<Mat*>& params,
            const std::vector<const Mat*>& grads);
  double nz_last() const { return nz_; }
  int step_count() const { return t_; }

 private:
  int shift_, t_ = 0;
  i64 lrn_, lrd_;
  double nz_ = 0;
  std::vector<Mat> m_, v_;
  std::vector<std::uint32_t> p10_, p9_, p1000_, p999_;
};

/** The composed R2b loop (certified trajectory efe3557c...). */
class full_birth {
 public:
  /** tables_bytes: AXP3 with silu.tab / dsilu.tab / exp.tab /
      rope.cos / rope.sin. init_bytes: 11 weight tensors in KEYS
      order, then x [T,D], then tgt [T], int64 LE. Throws
      std::runtime_error on malformed bytes (fail at load). */
  full_birth(const std::string& tables_bytes,
             const std::string& init_bytes, const contract& c);

  void run(int steps);              ///< advance n training steps
  int step_count() const { return step_; }
  i64 last_loss() const { return loss_; }
  double nz_last() const { return opt_.nz_last(); }

  /** Feed the current wide weights (KEYS order) into the running
      trajectory hash — the milestone protocol — and return its
      current hex digest. */
  std::string mark();
  /** Current running trajectory digest without marking. */
  std::string traj_sha() const;
  /** Wide (Q_w-scale) weights, KEYS order, int64 LE. */
  std::string weights_bytes() const;

 private:
  void step_once();
  block blk_;
  adamw opt_;
  int step_ = 0;
  i64 loss_ = 0;
  std::map<std::string, Mat> w_;   // wide, Q_w scale
  std::vector<i64> x_, tgt_;
  detail::sha256 th_;
};

/** The multi-block composed loop (mb spec: emb -> Body x n_blocks
    -> rmsnorm(g_f) -> TIED head; embedding grad = rounded head
    part + exact scatter-add of dx0 by token). init_bytes: params
    in mb_ref.json param_order (emb, b{i}.*, g_f), then tok [T],
    then tgt [T], int64 LE. */
class multi_birth {
 public:
  multi_birth(const std::string& tables_bytes,
              const std::string& init_bytes, const contract& c);
  void run(int steps);
  int step_count() const { return step_; }
  i64 last_loss() const { return loss_; }
  double nz_last() const { return opt_.nz_last(); }
  std::string mark();       ///< milestone protocol, param_order
  std::string traj_sha() const;
  std::string weights_bytes() const;  ///< wide, param_order, i64 LE
  const std::vector<std::string>& param_order() const {
    return order_;
  }

 private:
  void step_once();
  block blk_;
  adamw opt_;
  int step_ = 0;
  i64 loss_ = 0;
  std::vector<std::string> order_;
  std::map<std::string, Mat> w_;   // wide, Q_w scale
  std::vector<i64> tok_, tgt_;
  detail::sha256 th_;
};

}  // namespace ax::nn::ib
