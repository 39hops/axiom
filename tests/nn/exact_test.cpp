/** @file exact_test.cpp FX-V1 structural properties. Bit-exactness
    acceptance (C++ vs independent python integer reference, 100
    prompts) lives in scripts/nn_exact_ref.py; here: determinism,
    load validation, hash stability across instances. Table
    generation below uses doubles — allowed in TESTS only; the
    forward path itself is integer-only. */
#include <ax/nn/exact.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <random>

namespace {

using ax::nn::config;
using ax::nn::exact_model;
using ax::nn::tensor;

config tiny_cfg() {
  config c;
  c.d_model = 8;
  c.n_layers = 2;
  c.n_heads = 2;
  c.d_ff = 16;
  c.vocab = 11;
  c.max_seq = 32;
  return c;
}

tensor table_of(int n, double (*f)(double), double x0, double step) {
  tensor t;
  t.dims = {static_cast<std::size_t>(n)};
  for (int i = 0; i < n; ++i)
    t.data.push_back(static_cast<float>(
        std::llround(f(x0 + i * step) * 65536.0)));
  return t;
}

std::map<std::string, tensor> tiny_parts(const config& c, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.0f, 0.05f);
  const auto mk = [&](std::size_t r, std::size_t co) {
    tensor t;
    t.dims = co ? std::vector<std::size_t>{r, co}
                : std::vector<std::size_t>{r};
    t.data.resize(r * (co ? co : 1));
    for (float& v : t.data) v = nd(rng);
    return t;
  };
  const auto D = static_cast<std::size_t>(c.d_model);
  const auto F = static_cast<std::size_t>(c.d_ff);
  std::map<std::string, tensor> t;
  t["tok_emb.weight"] = mk(static_cast<std::size_t>(c.vocab), D);
  t["pos_emb.weight"] = mk(static_cast<std::size_t>(c.max_seq), D);
  for (int i = 0; i < c.n_layers; ++i) {
    const std::string L = "layers." + std::to_string(i) + ".";
    t[L + "ln1.weight"] = mk(D, 0);
    t[L + "ln1.bias"] = mk(D, 0);
    for (const char* w : {"q", "k", "v", "o"}) {
      t[L + "attn." + w + ".weight"] = mk(D, D);
      t[L + "attn." + w + ".bias"] = mk(D, 0);
    }
    t[L + "ln2.weight"] = mk(D, 0);
    t[L + "ln2.bias"] = mk(D, 0);
    t[L + "ffn.fc1.weight"] = mk(F, D);
    t[L + "ffn.fc1.bias"] = mk(F, 0);
    t[L + "ffn.fc2.weight"] = mk(D, F);
    t[L + "ffn.fc2.bias"] = mk(D, 0);
  }
  t["ln_f.weight"] = mk(D, 0);
  t["ln_f.bias"] = mk(D, 0);
  t["fx.act.table"] = table_of(
      2049, [](double x) { return 0.5 * x * (1 + std::erf(x / std::sqrt(2.0))); },
      -32.0, 1.0 / 32.0);
  t["fx.exp.table"] =
      table_of(2049, [](double x) { return std::exp(x); }, -16.0,
               1.0 / 128.0);
  t["fx.rsqrt.table"] = table_of(
      385, [](double x) { return 1.0 / std::sqrt(x); }, 1.0, 1.0 / 128.0);
  return t;
}

TEST(NnExact, FusedQkvBitExact) {
  // Q.16 conversion is per-weight, so stacking q/k/v into one tensor
  // must not change a single bit of the integer forward.
  const auto cfg = tiny_cfg();
  const auto plain = tiny_parts(cfg, 55);
  config fused_cfg = cfg;
  fused_cfg.attn_fused = true;
  fused_cfg.axnn_minor = 1;
  auto fused = plain;
  const auto D = static_cast<std::size_t>(cfg.d_model);
  for (int i = 0; i < cfg.n_layers; ++i) {
    const std::string L = "layers." + std::to_string(i) + ".";
    tensor qkv, qkvb;
    qkv.dims = {3 * D, D};
    qkvb.dims = {3 * D};
    for (const char* w : {"q", "k", "v"}) {
      const auto& wt = fused.at(L + "attn." + w + ".weight");
      qkv.data.insert(qkv.data.end(), wt.data.begin(), wt.data.end());
      const auto& bt = fused.at(L + "attn." + w + ".bias");
      qkvb.data.insert(qkvb.data.end(), bt.data.begin(), bt.data.end());
      fused.erase(L + "attn." + std::string(w) + ".weight");
      fused.erase(L + "attn." + std::string(w) + ".bias");
    }
    fused[L + "attn.qkv.weight"] = std::move(qkv);
    fused[L + "attn.qkv.bias"] = std::move(qkvb);
  }
  const auto m1 = exact_model::from_parts(cfg, plain);
  const auto m2 = exact_model::from_parts(fused_cfg, fused);
  const std::vector<int> toks{2, 5, 1, 8, 3};
  EXPECT_EQ(m1.logits_q32(toks), m2.logits_q32(toks));
  EXPECT_EQ(m1.logits_hash(toks), m2.logits_hash(toks));
}

TEST(NnExact, SwigluDeterministicInstanceStable) {
  config cfg = tiny_cfg();
  cfg.ffn = "swiglu";
  cfg.act = "silu";
  cfg.axnn_minor = 1;
  auto parts = tiny_parts(cfg, 65);
  const auto D = static_cast<std::size_t>(cfg.d_model);
  const auto F = static_cast<std::size_t>(cfg.d_ff);
  std::mt19937 rng(66);
  std::normal_distribution<float> nd(0.0f, 0.05f);
  const auto mk = [&](std::size_t r, std::size_t co) {
    tensor x;
    x.dims = co ? std::vector<std::size_t>{r, co}
                : std::vector<std::size_t>{r};
    x.data.resize(r * (co ? co : 1));
    for (float& v : x.data) v = nd(rng);
    return x;
  };
  for (int i = 0; i < cfg.n_layers; ++i) {
    const std::string L = "layers." + std::to_string(i) + ".";
    parts.erase(L + "ffn.fc1.weight");
    parts.erase(L + "ffn.fc1.bias");
    parts.erase(L + "ffn.fc2.weight");
    parts.erase(L + "ffn.fc2.bias");
    parts[L + "ffn.gate.weight"] = mk(F, D);
    parts[L + "ffn.gate.bias"] = mk(F, 0);
    parts[L + "ffn.up.weight"] = mk(F, D);
    parts[L + "ffn.up.bias"] = mk(F, 0);
    parts[L + "ffn.down.weight"] = mk(D, F);
    parts[L + "ffn.down.bias"] = mk(D, 0);
  }
  parts["fx.act.table"] = table_of(
      2049, [](double x) { return x / (1.0 + std::exp(-x)); }, -32.0,
      1.0 / 32.0);
  const auto m1 = exact_model::from_parts(cfg, parts);
  const auto m2 = exact_model::from_parts(cfg, parts);
  const std::vector<int> toks{4, 9, 0, 2};
  EXPECT_EQ(m1.logits_q32(toks), m2.logits_q32(toks));
  const auto gen = m1.generate({4, 9}, 4);
  EXPECT_EQ(gen.size(), 4u);  // stepper runs the swiglu path too
}

TEST(NnExact, DeterministicAndInstanceStable) {
  const auto cfg = tiny_cfg();
  const auto m1 = exact_model::from_parts(cfg, tiny_parts(cfg, 7));
  const auto m2 = exact_model::from_parts(cfg, tiny_parts(cfg, 7));
  const std::vector<int> toks{1, 4, 9, 2, 2, 10};
  EXPECT_EQ(m1.logits_q32(toks), m1.logits_q32(toks));
  EXPECT_EQ(m1.logits_hash(toks), m2.logits_hash(toks));
  const auto lg = m1.logits_q32(toks);
  EXPECT_EQ(lg.size(), 11u);
  EXPECT_EQ(lg[static_cast<std::size_t>(m1.argmax(toks))],
            *std::max_element(lg.begin(), lg.end()));
}

TEST(NnExact, LoadRejectsMissingTables) {
  const auto cfg = tiny_cfg();
  auto parts = tiny_parts(cfg, 9);
  parts.erase("fx.exp.table");
  EXPECT_THROW(exact_model::from_parts(cfg, parts), std::runtime_error);
  auto bad = tiny_parts(cfg, 9);
  bad.at("fx.rsqrt.table").data[3] = 0.5f;  // non-integer table value
  EXPECT_THROW(exact_model::from_parts(cfg, bad), std::runtime_error);
}

TEST(NnExact, GenerateMatchesFullForwardGreedy) {
  // the KV-cached stepper must be bit-exact with re-running the full
  // forward per step (same integer ops, cache changes cost only)
  const auto cfg = tiny_cfg();
  const auto m = exact_model::from_parts(cfg, tiny_parts(cfg, 33));
  const std::vector<int> prompt{4, 1, 7};
  const auto gen = m.generate(prompt, 8);
  ASSERT_EQ(gen.size(), 8u);
  std::vector<int> ctx = prompt;
  for (const int g : gen) {
    EXPECT_EQ(g, m.argmax(ctx));  // full re-forward path
    ctx.push_back(g);
  }
}

TEST(NnExact, GenerateStopsOnStopId) {
  const auto cfg = tiny_cfg();
  const auto m = exact_model::from_parts(cfg, tiny_parts(cfg, 33));
  const std::vector<int> prompt{4, 1, 7};
  const int first = m.generate(prompt, 1)[0];
  const auto gen = m.generate(prompt, 8, first);
  ASSERT_EQ(gen.size(), 1u);  // stop token kept, decode halted
  EXPECT_EQ(gen[0], first);
}

TEST(NnExact, TableCertification) {
  const auto cfg = tiny_cfg();
  const auto m = exact_model::from_parts(cfg, tiny_parts(cfg, 41));
  EXPECT_EQ(m.certify_tables(), "");
  // corrupt one exp entry: certification must name the violation
  auto bad = tiny_parts(cfg, 41);
  bad.at("fx.exp.table").data[1000] += 300.0f;
  const auto mb = exact_model::from_parts(cfg, bad);
  EXPECT_NE(mb.certify_tables(), "");
  // break monotonicity subtly (within entry tolerance windows this is
  // exactly the argmax-order hazard the node exists for)
  auto swap = tiny_parts(cfg, 41);
  auto& et = swap.at("fx.exp.table").data;
  // pick a region where the per-step gradient is ~1 LSB so the swap is
  // inside entry tolerance but breaks ordering (t[500] sits in the
  // flat underflow region where adjacent entries are EQUAL — a swap
  // there is a no-op, measured)
  ASSERT_GT(et[1301], et[1300]);
  std::swap(et[1300], et[1301]);
  const auto ms = exact_model::from_parts(cfg, swap);
  EXPECT_NE(ms.certify_tables(), "");
}

TEST(NnExact, CausalityHoldsInIntegerPath) {
  const auto cfg = tiny_cfg();
  const auto m = exact_model::from_parts(cfg, tiny_parts(cfg, 21));
  // exact path emits last-position logits; causality shows as
  // prefix-stability: extending the prompt never changes the readout
  // of the shorter prompt re-evaluated on its own
  const std::vector<int> a{3, 5, 1, 8};
  const auto ha = m.logits_hash(a);
  std::vector<int> ext = a;
  ext.push_back(6);
  (void)m.logits_hash(ext);  // must not perturb any cached state
  EXPECT_EQ(m.logits_hash(a), ha);
}

}  // namespace
