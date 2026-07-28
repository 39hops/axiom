#pragma once
/** @file model.hpp NNUE homecoming rung 1: micro-inference for llmopt
    crystals (relay 2026-07-27-2 ask 2). INFERENCE ONLY by doctrine —
    training stays torch; no optimizers, no autograd, ever.

    Scope: decoder-only pre-LN transformer, the d256/8L/ffn1024/h4
    crystal family. Every architectural convention is DECLARED in the
    container config, not assumed: norm (layernorm|rmsnorm), activation
    (gelu|gelu_tanh|silu|relu), positional (learned|rope|none, rope
    half|interleaved), tied or separate head. fp32 weights, double
    accumulation (acceptance bar: logits within 1e-4 of torch fp32).

    AXNN container v1 (little-endian):
      magic  "AXNN"            4 bytes
      u32    version           == 1
      u32    cfg_len; cfg      flat JSON object (jsonl codec grammar)
      repeat until EOF:
        u32  name_len; name
        u32  ndim; u64 dims[ndim]
        f32  data[prod(dims)]

    Tensor names (torch Linear layout: weight [out, in], y = x W^T + b;
    absent bias = zero; absent head.weight = tied to tok_emb):
      tok_emb.weight [V,D]      pos_emb.weight [max_seq,D]
      layers.{i}.ln1.weight/.bias [D]
      layers.{i}.attn.{q,k,v,o}.weight [D,D] /.bias [D]
      layers.{i}.ln2.weight/.bias [D]
      layers.{i}.ffn.fc1.weight [F,D] /.bias [F]
      layers.{i}.ffn.fc2.weight [D,F] /.bias [D]
      ln_f.weight/.bias [D]     head.weight [V,D] (optional)

    v1.1 (axnn_minor 1, relay 2026-07-28-4 — byte-compatible header/
    tensor encoding; the delta is cfg keys + alternate tensor names):
      cfg ffn = "swiglu": three-tensor FFN, down(act(gate(x)) * up(x))
        layers.{i}.ffn.gate.weight [F,D] /.bias [F]
        layers.{i}.ffn.up.weight   [F,D] /.bias [F]
        layers.{i}.ffn.down.weight [D,F] /.bias [D]
      cfg attn_fused = 1: q,k,v stacked row-wise in one tensor
        layers.{i}.attn.qkv.weight [3D,D] /.bias [3D]
      cfg head = "tied"|"separate": declared (vs inferred-by-absence);
      validated against tensor presence.
    The house state-dict name dialect (relay -5: emb.weight, norm.g,
    blocks.{i}.{n1.g,n2.g,qkv,o,gate,up,down}) is remapped to the
    canonical names at read time by load_container; attn_fused accepts
    the house string spelling ("qkv") as truthy. */
#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ax::nn {

struct config {
  int d_model = 0, n_layers = 0, n_heads = 0, d_ff = 0, vocab = 0,
      max_seq = 0;
  std::string norm = "layernorm";   // layernorm | rmsnorm
  std::string act = "gelu";         // gelu | gelu_tanh | silu | relu
  std::string pos = "learned";      // learned | rope | none
  std::string rope_style = "half";  // half | interleaved
  double eps = 1e-5;
  double rope_theta = 10000.0;
  // v1.1 (axnn_minor 1) declarations; v1 defaults are the fallbacks
  std::string ffn = "fc";     // fc | swiglu
  bool attn_fused = false;    // one qkv tensor [3D,D] vs separate q/k/v
  std::string head = "auto";  // auto (infer by presence) | tied | separate
  int axnn_minor = 0;
};

struct tensor {
  std::vector<std::size_t> dims;
  std::vector<float> data;
};

/** Raw container read (config + all tensors, no validation) — shared
    by model::load and the FX-V1 exact loader. */
std::pair<config, std::map<std::string, tensor>> load_container(
    const std::string& path);

class model {
 public:
  /** Load an AXNN container. Throws std::runtime_error on malformed
      files, shape mismatches, or unknown config values (fail at load,
      never at forward — the friendly-fire doctrine). */
  static model load(const std::string& path);
  /** Assemble from parts (tests, in-memory weight conversion). Runs
      the same shape validation as load(). */
  static model from_parts(config cfg, std::map<std::string, tensor> t);

  /** Full logits, row-major [tokens.size() x vocab]. Deterministic:
      plain loops, double accumulators, no threading. */
  std::vector<float> logits(const std::vector<int>& tokens) const;
  /** Last-position logits only (the scorer/readout shape). */
  std::vector<float> logits_last(const std::vector<int>& tokens) const;

  const config& cfg() const { return cfg_; }
  bool has(const std::string& name) const { return t_.count(name) != 0; }

 private:
  void validate() const;
  config cfg_;
  std::map<std::string, tensor> t_;
};

}  // namespace ax::nn
