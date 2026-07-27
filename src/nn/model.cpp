/** @file model.cpp AXNN micro-inference (see model.hpp). Plain loops,
    double accumulation, zero dependencies — determinism and audit-
    ability over speed; the consumers (gate batteries, S2 listwise
    scorer, miner loops) are latency-shaped, not throughput-shaped. */
#include <ax/nn/model.hpp>

#include <ax/sym/jsonl.hpp>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace ax::nn {

namespace {

// ------------------------------------------------------------ loader

template <class T>
T read_pod(std::ifstream& in) {
  T v{};
  in.read(reinterpret_cast<char*>(&v), sizeof(T));
  if (!in) throw std::runtime_error("axnn: truncated file");
  return v;
}

std::string read_str(std::ifstream& in, std::size_t n) {
  std::string s(n, '\0');
  in.read(s.data(), static_cast<std::streamsize>(n));
  if (!in) throw std::runtime_error("axnn: truncated string");
  return s;
}

int cfg_int(const sym::jsonl::object& o, const std::string& k) {
  const auto it = o.find(k);
  if (it == o.end()) throw std::runtime_error("axnn config missing " + k);
  return std::stoi(it->second);
}

double cfg_num(const sym::jsonl::object& o, const std::string& k,
               double dflt) {
  const auto it = o.find(k);
  return it == o.end() ? dflt : std::stod(it->second);
}

std::string cfg_str(const sym::jsonl::object& o, const std::string& k,
                    const std::string& dflt) {
  const auto it = o.find(k);
  return it == o.end() ? dflt : it->second;
}

// ------------------------------------------------------------ kernels

double act_of(double x, int act) {
  switch (act) {
    case 0:  // gelu (exact erf — torch approximate='none')
      return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
    case 1: {  // gelu_tanh (torch approximate='tanh')
      const double c = std::sqrt(2.0 / 3.14159265358979323846);
      return 0.5 * x * (1.0 + std::tanh(c * (x + 0.044715 * x * x * x)));
    }
    case 2:  // silu
      return x / (1.0 + std::exp(-x));
    default:  // relu
      return x > 0 ? x : 0;
  }
}

}  // namespace

void model::validate() const {
  const auto need = [&](const std::string& n, std::size_t r,
                        std::size_t c) {
    const auto it = t_.find(n);
    if (it == t_.end()) throw std::runtime_error("axnn: missing " + n);
    const auto& d = it->second.dims;
    const bool ok = c == 0 ? (d.size() == 1 && d[0] == r)
                           : (d.size() == 2 && d[0] == r && d[1] == c);
    if (!ok) throw std::runtime_error("axnn: bad shape for " + n);
    if (it->second.data.size() != r * (c == 0 ? 1 : c))
      throw std::runtime_error("axnn: data size mismatch " + n);
  };
  const auto optional = [&](const std::string& n, std::size_t r) {
    if (t_.count(n)) need(n, r, 0);
  };
  const std::size_t D = static_cast<std::size_t>(cfg_.d_model);
  const std::size_t F = static_cast<std::size_t>(cfg_.d_ff);
  const std::size_t V = static_cast<std::size_t>(cfg_.vocab);
  if (cfg_.d_model <= 0 || cfg_.n_layers <= 0 || cfg_.n_heads <= 0 ||
      cfg_.d_ff <= 0 || cfg_.vocab <= 0)
    throw std::runtime_error("axnn: non-positive config dimension");
  if (cfg_.d_model % cfg_.n_heads)
    throw std::runtime_error("axnn: d_model % n_heads != 0");
  if (cfg_.norm != "layernorm" && cfg_.norm != "rmsnorm")
    throw std::runtime_error("axnn: unknown norm " + cfg_.norm);
  if (cfg_.act != "gelu" && cfg_.act != "gelu_tanh" &&
      cfg_.act != "silu" && cfg_.act != "relu")
    throw std::runtime_error("axnn: unknown act " + cfg_.act);
  if (cfg_.pos != "learned" && cfg_.pos != "rope" && cfg_.pos != "none")
    throw std::runtime_error("axnn: unknown pos " + cfg_.pos);
  if (cfg_.rope_style != "half" && cfg_.rope_style != "interleaved")
    throw std::runtime_error("axnn: unknown rope_style " +
                             cfg_.rope_style);
  need("tok_emb.weight", V, D);
  if (cfg_.pos == "learned")
    need("pos_emb.weight", static_cast<std::size_t>(cfg_.max_seq), D);
  for (int i = 0; i < cfg_.n_layers; ++i) {
    const std::string L = "layers." + std::to_string(i) + ".";
    need(L + "ln1.weight", D, 0);
    optional(L + "ln1.bias", D);
    for (const char* w : {"q", "k", "v", "o"}) {
      need(L + "attn." + w + ".weight", D, D);
      optional(L + "attn." + w + ".bias", D);
    }
    need(L + "ln2.weight", D, 0);
    optional(L + "ln2.bias", D);
    need(L + "ffn.fc1.weight", F, D);
    optional(L + "ffn.fc1.bias", F);
    need(L + "ffn.fc2.weight", D, F);
    optional(L + "ffn.fc2.bias", D);
  }
  need("ln_f.weight", D, 0);
  optional("ln_f.bias", D);
  if (t_.count("head.weight")) need("head.weight", V, D);
}

model model::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.good()) throw std::runtime_error("axnn: cannot open " + path);
  if (read_str(in, 4) != "AXNN")
    throw std::runtime_error("axnn: bad magic");
  if (read_pod<std::uint32_t>(in) != 1)
    throw std::runtime_error("axnn: unsupported version");
  const auto cfg_len = read_pod<std::uint32_t>(in);
  const auto cfg_json = sym::jsonl::parse_line(read_str(in, cfg_len));

  model m;
  m.cfg_.d_model = cfg_int(cfg_json, "d_model");
  m.cfg_.n_layers = cfg_int(cfg_json, "n_layers");
  m.cfg_.n_heads = cfg_int(cfg_json, "n_heads");
  m.cfg_.d_ff = cfg_int(cfg_json, "d_ff");
  m.cfg_.vocab = cfg_int(cfg_json, "vocab");
  m.cfg_.max_seq = cfg_int(cfg_json, "max_seq");
  m.cfg_.norm = cfg_str(cfg_json, "norm", "layernorm");
  m.cfg_.act = cfg_str(cfg_json, "act", "gelu");
  m.cfg_.pos = cfg_str(cfg_json, "pos", "learned");
  m.cfg_.rope_style = cfg_str(cfg_json, "rope_style", "half");
  m.cfg_.eps = cfg_num(cfg_json, "eps", 1e-5);
  m.cfg_.rope_theta = cfg_num(cfg_json, "rope_theta", 10000.0);

  while (in.peek() != std::ifstream::traits_type::eof()) {
    const auto name_len = read_pod<std::uint32_t>(in);
    const std::string name = read_str(in, name_len);
    const auto ndim = read_pod<std::uint32_t>(in);
    if (ndim == 0 || ndim > 4)
      throw std::runtime_error("axnn: bad ndim for " + name);
    tensor t;
    std::size_t total = 1;
    for (std::uint32_t d = 0; d < ndim; ++d) {
      const auto dim = read_pod<std::uint64_t>(in);
      t.dims.push_back(static_cast<std::size_t>(dim));
      total *= static_cast<std::size_t>(dim);
    }
    if (total > (1ull << 31))
      throw std::runtime_error("axnn: tensor too large " + name);
    t.data.resize(total);
    in.read(reinterpret_cast<char*>(t.data.data()),
            static_cast<std::streamsize>(total * sizeof(float)));
    if (!in) throw std::runtime_error("axnn: truncated tensor " + name);
    m.t_.emplace(name, std::move(t));
  }
  m.validate();
  return m;
}

model model::from_parts(config cfg, std::map<std::string, tensor> t) {
  model m;
  m.cfg_ = std::move(cfg);
  m.t_ = std::move(t);
  m.validate();
  return m;
}

std::vector<float> model::logits(const std::vector<int>& tokens) const {
  const std::size_t T = tokens.size();
  const std::size_t D = static_cast<std::size_t>(cfg_.d_model);
  const std::size_t H = static_cast<std::size_t>(cfg_.n_heads);
  const std::size_t dh = D / H;
  const std::size_t F = static_cast<std::size_t>(cfg_.d_ff);
  const std::size_t V = static_cast<std::size_t>(cfg_.vocab);
  if (T == 0) return {};
  if (cfg_.max_seq > 0 && T > static_cast<std::size_t>(cfg_.max_seq))
    throw std::runtime_error("axnn: sequence exceeds max_seq");
  for (const int tok : tokens)
    if (tok < 0 || tok >= cfg_.vocab)
      throw std::runtime_error("axnn: token out of vocab");

  const int act =
      cfg_.act == "gelu" ? 0 : cfg_.act == "gelu_tanh" ? 1
      : cfg_.act == "silu" ? 2 : 3;
  const auto& W = [&](const std::string& n) -> const std::vector<float>& {
    return t_.at(n).data;
  };
  const auto bias = [&](const std::string& n) -> const float* {
    const auto it = t_.find(n);
    return it == t_.end() ? nullptr : it->second.data.data();
  };
  // y[T,out] = x[T,in] W^T + b, torch Linear layout W[out,in]
  const auto linear = [&](const std::vector<double>& x, std::size_t in_d,
                          std::size_t out_d, const std::vector<float>& w,
                          const float* b) {
    std::vector<double> y(T * out_d);
    for (std::size_t t = 0; t < T; ++t)
      for (std::size_t o = 0; o < out_d; ++o) {
        double acc = b ? b[o] : 0.0;
        const float* wr = &w[o * in_d];
        const double* xr = &x[t * in_d];
        for (std::size_t i = 0; i < in_d; ++i) acc += xr[i] * wr[i];
        y[t * out_d + o] = acc;
      }
    return y;
  };
  const auto normed = [&](const std::vector<double>& x,
                          const std::string& prefix) {
    const auto& g = W(prefix + ".weight");
    const float* b = bias(prefix + ".bias");
    std::vector<double> y(T * D);
    for (std::size_t t = 0; t < T; ++t) {
      const double* row = &x[t * D];
      double mean = 0, ms = 0;
      if (cfg_.norm == "layernorm") {
        for (std::size_t d = 0; d < D; ++d) mean += row[d];
        mean /= static_cast<double>(D);
      }
      for (std::size_t d = 0; d < D; ++d)
        ms += (row[d] - mean) * (row[d] - mean);
      ms /= static_cast<double>(D);
      const double inv = 1.0 / std::sqrt(ms + cfg_.eps);
      for (std::size_t d = 0; d < D; ++d)
        y[t * D + d] =
            (row[d] - mean) * inv * g[d] + (b ? b[d] : 0.0);
    }
    return y;
  };
  const auto rope = [&](std::vector<double>& q) {
    for (std::size_t t = 0; t < T; ++t)
      for (std::size_t h = 0; h < H; ++h) {
        double* head = &q[t * D + h * dh];
        for (std::size_t p = 0; p < dh / 2; ++p) {
          const double freq = std::pow(
              cfg_.rope_theta, -2.0 * static_cast<double>(p) /
                                   static_cast<double>(dh));
          const double c = std::cos(static_cast<double>(t) * freq);
          const double s = std::sin(static_cast<double>(t) * freq);
          const std::size_t i0 =
              cfg_.rope_style == "half" ? p : 2 * p;
          const std::size_t i1 =
              cfg_.rope_style == "half" ? p + dh / 2 : 2 * p + 1;
          const double a = head[i0], b2 = head[i1];
          head[i0] = a * c - b2 * s;
          head[i1] = a * s + b2 * c;
        }
      }
  };

  // embeddings
  std::vector<double> x(T * D);
  const auto& emb = W("tok_emb.weight");
  for (std::size_t t = 0; t < T; ++t)
    for (std::size_t d = 0; d < D; ++d)
      x[t * D + d] = emb[static_cast<std::size_t>(tokens[t]) * D + d];
  if (cfg_.pos == "learned") {
    const auto& pe = W("pos_emb.weight");
    for (std::size_t t = 0; t < T; ++t)
      for (std::size_t d = 0; d < D; ++d) x[t * D + d] += pe[t * D + d];
  }

  for (int layer = 0; layer < cfg_.n_layers; ++layer) {
    const std::string L = "layers." + std::to_string(layer) + ".";
    // ---- attention block (pre-LN)
    const auto h1 = normed(x, L + "ln1");
    auto q = linear(h1, D, D, W(L + "attn.q.weight"),
                    bias(L + "attn.q.bias"));
    auto k = linear(h1, D, D, W(L + "attn.k.weight"),
                    bias(L + "attn.k.bias"));
    const auto v = linear(h1, D, D, W(L + "attn.v.weight"),
                          bias(L + "attn.v.bias"));
    if (cfg_.pos == "rope") {
      rope(q);
      rope(k);
    }
    std::vector<double> attn(T * D, 0.0);
    const double scale = 1.0 / std::sqrt(static_cast<double>(dh));
    std::vector<double> score(T);
    for (std::size_t h = 0; h < H; ++h)
      for (std::size_t t = 0; t < T; ++t) {
        double mx = -1e300;
        for (std::size_t u = 0; u <= t; ++u) {  // causal
          double s = 0;
          for (std::size_t d = 0; d < dh; ++d)
            s += q[t * D + h * dh + d] * k[u * D + h * dh + d];
          score[u] = s * scale;
          mx = std::max(mx, score[u]);
        }
        double z = 0;
        for (std::size_t u = 0; u <= t; ++u) {
          score[u] = std::exp(score[u] - mx);
          z += score[u];
        }
        for (std::size_t u = 0; u <= t; ++u) {
          const double w2 = score[u] / z;
          for (std::size_t d = 0; d < dh; ++d)
            attn[t * D + h * dh + d] += w2 * v[u * D + h * dh + d];
        }
      }
    const auto proj = linear(attn, D, D, W(L + "attn.o.weight"),
                             bias(L + "attn.o.bias"));
    for (std::size_t i = 0; i < T * D; ++i) x[i] += proj[i];
    // ---- ffn block (pre-LN)
    const auto h2 = normed(x, L + "ln2");
    auto f1 = linear(h2, D, F, W(L + "ffn.fc1.weight"),
                     bias(L + "ffn.fc1.bias"));
    for (double& v2 : f1) v2 = act_of(v2, act);
    const auto f2 = linear(f1, F, D, W(L + "ffn.fc2.weight"),
                           bias(L + "ffn.fc2.bias"));
    for (std::size_t i = 0; i < T * D; ++i) x[i] += f2[i];
  }

  const auto xf = normed(x, "ln_f");
  const auto& head =
      t_.count("head.weight") ? W("head.weight") : W("tok_emb.weight");
  std::vector<float> out(T * V);
  for (std::size_t t = 0; t < T; ++t)
    for (std::size_t vv = 0; vv < V; ++vv) {
      double acc = 0;
      for (std::size_t d = 0; d < D; ++d)
        acc += xf[t * D + d] * head[vv * D + d];
      out[t * V + vv] = static_cast<float>(acc);
    }
  return out;
}

std::vector<float> model::logits_last(const std::vector<int>& tokens) const {
  const auto all = logits(tokens);
  const std::size_t V = static_cast<std::size_t>(cfg_.vocab);
  if (all.empty()) return {};
  return {all.end() - static_cast<std::ptrdiff_t>(V), all.end()};
}

}  // namespace ax::nn
