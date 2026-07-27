/** @file model_test.cpp AXNN structural properties (torch-free CI
    side; the numeric acceptance vs torch fp32 lives in
    scripts/nn_crosscheck.py). */
#include <ax/nn/model.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>

namespace {

using ax::nn::config;
using ax::nn::model;
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

std::map<std::string, tensor> tiny_tensors(const config& c, unsigned seed) {
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
  if (c.pos == "learned")
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
  return t;
}

TEST(Nn, DeterministicForward) {
  const auto m = model::from_parts(tiny_cfg(), tiny_tensors(tiny_cfg(), 7));
  const std::vector<int> toks{1, 4, 9, 2, 2, 10};
  EXPECT_EQ(m.logits(toks), m.logits(toks));
  const auto last = m.logits_last(toks);
  const auto all = m.logits(toks);
  ASSERT_EQ(last.size(), 11u);
  for (std::size_t i = 0; i < last.size(); ++i)
    EXPECT_EQ(last[i], all[(toks.size() - 1) * 11 + i]);
}

TEST(Nn, CausalMaskHolds) {
  const auto m = model::from_parts(tiny_cfg(), tiny_tensors(tiny_cfg(), 9));
  std::vector<int> a{3, 5, 1, 8, 2};
  std::vector<int> b = a;
  b[4] = 7;  // change only the LAST token
  const auto la = m.logits(a);
  const auto lb = m.logits(b);
  // positions 0..3 must be bit-identical (causality)
  for (std::size_t i = 0; i < 4 * 11; ++i) EXPECT_EQ(la[i], lb[i]);
  // and the last position must differ (the change is visible there)
  bool differs = false;
  for (std::size_t i = 4 * 11; i < 5 * 11; ++i)
    differs = differs || la[i] != lb[i];
  EXPECT_TRUE(differs);
}

TEST(Nn, TiedHeadEqualsExplicitCopy) {
  const auto cfg = tiny_cfg();
  auto tied = tiny_tensors(cfg, 11);
  auto split = tied;
  split["head.weight"] = tied.at("tok_emb.weight");
  const auto m1 = model::from_parts(cfg, tied);
  const auto m2 = model::from_parts(cfg, split);
  const std::vector<int> toks{0, 6, 3};
  EXPECT_EQ(m1.logits(toks), m2.logits(toks));
}

TEST(Nn, ConfigVariantsForward) {
  for (const char* norm : {"layernorm", "rmsnorm"})
    for (const char* pos : {"learned", "rope", "none"})
      for (const char* act : {"gelu", "gelu_tanh", "silu", "relu"}) {
        config c = tiny_cfg();
        c.norm = norm;
        c.pos = pos;
        c.act = act;
        auto t = tiny_tensors(c, 13);
        if (std::string(norm) == "rmsnorm")
          for (int i = 0; i < c.n_layers; ++i) {
            t.erase("layers." + std::to_string(i) + ".ln1.bias");
            t.erase("layers." + std::to_string(i) + ".ln2.bias");
          }
        const auto m = model::from_parts(c, std::move(t));
        const auto lg = m.logits({1, 2, 3, 4});
        ASSERT_EQ(lg.size(), 4u * 11u);
        for (const float v : lg) EXPECT_TRUE(std::isfinite(v));
      }
}

TEST(Nn, ValidationRejectsBadShapesAndTokens) {
  const auto cfg = tiny_cfg();
  auto bad = tiny_tensors(cfg, 15);
  bad.at("layers.0.attn.q.weight").dims = {4, 4};
  EXPECT_THROW(model::from_parts(cfg, bad), std::runtime_error);
  auto missing = tiny_tensors(cfg, 15);
  missing.erase("ln_f.weight");
  EXPECT_THROW(model::from_parts(cfg, missing), std::runtime_error);
  const auto m = model::from_parts(cfg, tiny_tensors(cfg, 15));
  EXPECT_THROW(m.logits({0, 11}), std::runtime_error);  // out of vocab
  EXPECT_THROW(m.logits(std::vector<int>(33, 1)), std::runtime_error);
}

TEST(Nn, ContainerRoundTrip) {
  const auto cfg = tiny_cfg();
  const auto tens = tiny_tensors(cfg, 21);
  const auto m1 = model::from_parts(cfg, tens);
  const std::string path = "tiny_roundtrip.axnn";
  {
    std::ofstream f(path, std::ios::binary);
    f.write("AXNN", 4);
    const std::uint32_t ver = 1;
    f.write(reinterpret_cast<const char*>(&ver), 4);
    const std::string cj =
        "{\"d_model\": 8, \"n_layers\": 2, \"n_heads\": 2, \"d_ff\": 16,"
        " \"vocab\": 11, \"max_seq\": 32}";
    const std::uint32_t cl = static_cast<std::uint32_t>(cj.size());
    f.write(reinterpret_cast<const char*>(&cl), 4);
    f.write(cj.data(), static_cast<std::streamsize>(cj.size()));
    for (const auto& [name, t] : tens) {
      const std::uint32_t nl = static_cast<std::uint32_t>(name.size());
      f.write(reinterpret_cast<const char*>(&nl), 4);
      f.write(name.data(), static_cast<std::streamsize>(name.size()));
      const std::uint32_t nd = static_cast<std::uint32_t>(t.dims.size());
      f.write(reinterpret_cast<const char*>(&nd), 4);
      for (const std::size_t d : t.dims) {
        const std::uint64_t dd = d;
        f.write(reinterpret_cast<const char*>(&dd), 8);
      }
      f.write(reinterpret_cast<const char*>(t.data.data()),
              static_cast<std::streamsize>(t.data.size() * sizeof(float)));
    }
  }
  const auto m2 = model::load(path);
  std::remove(path.c_str());
  const std::vector<int> toks{2, 7, 5, 5, 1};
  EXPECT_EQ(m1.logits(toks), m2.logits(toks));
}

}  // namespace
