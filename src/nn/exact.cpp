/** @file exact.cpp FX-V1 integer forward (see exact.hpp and the spec).

    Implementation discipline: every arithmetic step below is one of
    the declared operations — int add/mul, arithmetic shift (floor),
    floor division, saturation, table lerp, exact comparison. Any
    float appearing past build() is a spec violation. */
#include <ax/nn/exact.hpp>

#include <bit>
#include <cstring>
#include <stdexcept>

namespace ax::nn {

namespace {

using std::int32_t;
using std::int64_t;
using std::uint32_t;
using std::uint64_t;

// ---------------------------------------------- declared primitives

/** fp32 bits -> Q.16, round-half-even, bit-exact (no float ops). */
int64_t f32_to_q16(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  const bool neg = (b >> 31) != 0;
  const int exp = static_cast<int>((b >> 23) & 0xff);
  const uint64_t frac = b & 0x7fffffu;
  if (exp == 0xff) throw std::runtime_error("fxv1: nan/inf weight");
  uint64_t mant = exp == 0 ? frac : (frac | (1ull << 23));
  if (mant == 0) return 0;
  // value = mant * 2^(exp-150); Q.16 target adds 16
  const int shift = exp - 150 + 16;
  uint64_t mag;
  if (shift >= 0) {
    if (shift > 38) return neg ? INT64_MIN / 2 : INT64_MAX / 2;  // sat later
    mag = mant << shift;
  } else {
    const int r = -shift;
    if (r > 63) return 0;
    const uint64_t keep = mant >> r;
    const uint64_t rem = mant & ((1ull << r) - 1);
    const uint64_t half = 1ull << (r - 1);
    mag = keep;
    if (rem > half || (rem == half && (keep & 1))) ++mag;
  }
  return neg ? -static_cast<int64_t>(mag) : static_cast<int64_t>(mag);
}

int64_t sat(int64_t v, int64_t bound) {
  return v > bound ? bound : v < -bound ? -bound : v;
}

/** Floor division (python // semantics), declared rule 3. */
int64_t floor_div(int64_t a, int64_t b) {
  int64_t q = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
  return q;
}

/** Table lerp, declared rule 4. t has n entries; u in [0, (n-1)<<fbits]. */
int64_t lerp(const std::vector<int64_t>& t, int64_t u, int fbits) {
  const int64_t idx = u >> fbits;
  const int64_t frac = u & ((1ll << fbits) - 1);
  if (idx >= static_cast<int64_t>(t.size()) - 1) return t.back();
  return t[static_cast<std::size_t>(idx)] +
         (((t[static_cast<std::size_t>(idx) + 1] -
            t[static_cast<std::size_t>(idx)]) *
           frac) >>
          fbits);
}

}  // namespace

void exact_model::build(const std::map<std::string, tensor>& src) {
  // separate fx tables (int64, values are exact integers in f32) from
  // weight tensors (Q.16 int32, RNE + weight saturation)
  for (const auto& [name, t] : src) {
    if (name.rfind("fx.", 0) == 0) {
      std::vector<int64_t> v;
      v.reserve(t.data.size());
      for (const float f : t.data) {
        // table values are integers < 2^24: f32 carries them exactly;
        // any fractional residue means a mis-generated table
        const int64_t raw = f32_to_q16(f);
        if (raw & 0xffff)
          throw std::runtime_error("fxv1: non-integer table value in " +
                                   name);
        v.push_back(raw >> 16);
      }
      tab_.emplace(name, std::move(v));
    } else {
      std::vector<int32_t> v;
      v.reserve(t.data.size());
      for (const float f : t.data)
        v.push_back(static_cast<int32_t>(
            sat(f32_to_q16(f), fxv1::kWeightSat)));
      w_.emplace(name, std::move(v));
    }
  }
  // required tables
  const auto need_tab = [&](const std::string& n, std::size_t sz) {
    const auto it = tab_.find(n);
    if (it == tab_.end() || it->second.size() != sz)
      throw std::runtime_error("fxv1: missing/mis-sized table " + n);
  };
  if (cfg_.act != "relu")
    need_tab("fx.act.table", fxv1::kActTableN);
  need_tab("fx.exp.table", fxv1::kExpTableN);
  need_tab("fx.rsqrt.table", fxv1::kRsqrtTableN);
  if (cfg_.pos == "rope") {
    const std::size_t half =
        static_cast<std::size_t>(cfg_.d_model / cfg_.n_heads / 2);
    need_tab("fx.rope.cos", static_cast<std::size_t>(cfg_.max_seq) * half);
    need_tab("fx.rope.sin", static_cast<std::size_t>(cfg_.max_seq) * half);
  }
}

exact_model exact_model::load(const std::string& path) {
  auto [cfg, tensors] = load_container(path);
  // reuse the rounded model's shape validation for the weight side
  {
    std::map<std::string, tensor> weights_only;
    for (const auto& [n, t] : tensors)
      if (n.rfind("fx.", 0) != 0) weights_only.emplace(n, t);
    (void)model::from_parts(cfg, std::move(weights_only));
  }
  exact_model m;
  m.cfg_ = std::move(cfg);
  m.build(tensors);
  return m;
}

exact_model exact_model::from_parts(config cfg,
                                    const std::map<std::string, tensor>& t) {
  {
    std::map<std::string, tensor> weights_only;
    for (const auto& [n, tt] : t)
      if (n.rfind("fx.", 0) != 0) weights_only.emplace(n, tt);
    (void)model::from_parts(cfg, std::move(weights_only));
  }
  exact_model m;
  m.cfg_ = std::move(cfg);
  m.build(t);
  return m;
}

namespace {

/** rsqrt_fx per spec: v Q.32 > 0 -> Q.16. */
int64_t rsqrt_fx(int64_t v, const std::vector<int64_t>& table) {
  const int k = std::bit_width(static_cast<uint64_t>(v)) - 1;
  int s = k - 31;
  if (s & 1) ++s;  // even
  const int64_t m = s >= 0 ? (v >> s) : (v << -s);
  const int64_t u = m - (1ll << 30);  // offset into [1, 4), step 2^23
  const int64_t r = lerp(table, u, 23);
  const int sh = s / 2 - 1;
  return sh >= 0 ? (r >> sh) : (r << -sh);
}

}  // namespace

std::vector<int64_t> exact_model::logits_q32(
    const std::vector<int>& tokens) const {
  const std::size_t T = tokens.size();
  const std::size_t D = static_cast<std::size_t>(cfg_.d_model);
  const std::size_t H = static_cast<std::size_t>(cfg_.n_heads);
  const std::size_t dh = D / H;
  const std::size_t F = static_cast<std::size_t>(cfg_.d_ff);
  const std::size_t V = static_cast<std::size_t>(cfg_.vocab);
  if (T == 0) return {};
  if (cfg_.max_seq > 0 && T > static_cast<std::size_t>(cfg_.max_seq))
    throw std::runtime_error("fxv1: sequence exceeds max_seq");
  for (const int tok : tokens)
    if (tok < 0 || tok >= cfg_.vocab)
      throw std::runtime_error("fxv1: token out of vocab");

  const auto& W = [&](const std::string& n) -> const std::vector<int32_t>& {
    return w_.at(n);
  };
  const auto bias = [&](const std::string& n) -> const int32_t* {
    const auto it = w_.find(n);
    return it == w_.end() ? nullptr : it->second.data();
  };
  const auto linear = [&](const std::vector<int64_t>& x, std::size_t in_d,
                          std::size_t out_d, const std::vector<int32_t>& w,
                          const int32_t* b) {
    std::vector<int64_t> y(T * out_d);
    for (std::size_t t = 0; t < T; ++t)
      for (std::size_t o = 0; o < out_d; ++o) {
        int64_t acc = b ? (static_cast<int64_t>(b[o]) << 16) : 0;
        const int32_t* wr = &w[o * in_d];
        const int64_t* xr = &x[t * in_d];
        for (std::size_t i = 0; i < in_d; ++i)
          acc += xr[i] * static_cast<int64_t>(wr[i]);
        y[t * out_d + o] = sat(acc >> 16, fxv1::kActSat);
      }
    return y;
  };
  const auto normed = [&](const std::vector<int64_t>& x,
                          const std::string& prefix) {
    const auto& g = W(prefix + ".weight");
    const int32_t* b = bias(prefix + ".bias");
    const auto& rt = tab_.at("fx.rsqrt.table");
    std::vector<int64_t> y(T * D);
    const auto Di = static_cast<int64_t>(D);
    for (std::size_t t = 0; t < T; ++t) {
      const int64_t* row = &x[t * D];
      int64_t mean = 0;
      if (cfg_.norm == "layernorm") {
        int64_t S = 0;
        for (std::size_t d = 0; d < D; ++d) S += row[d];
        mean = floor_div(S, Di);
      }
      int64_t sq = 0;
      std::vector<int64_t> c(D);
      for (std::size_t d = 0; d < D; ++d) {
        c[d] = sat(row[d] - mean, fxv1::kCentSat);
        sq += c[d] * c[d];
      }
      const int64_t var = floor_div(sq, Di);  // Q.32
      const int64_t inv = rsqrt_fx(var + fxv1::kEpsQ32, rt);
      for (std::size_t d = 0; d < D; ++d) {
        const int64_t n1 = (c[d] * inv) >> 16;
        const int64_t n2 = (n1 * static_cast<int64_t>(g[d])) >> 16;
        y[t * D + d] = sat(n2 + (b ? static_cast<int64_t>(b[d]) : 0),
                           fxv1::kActSat);
      }
    }
    return y;
  };
  const auto rope = [&](std::vector<int64_t>& q) {
    const auto& tc = tab_.at("fx.rope.cos");
    const auto& ts = tab_.at("fx.rope.sin");
    const std::size_t half = dh / 2;
    for (std::size_t t = 0; t < T; ++t)
      for (std::size_t h = 0; h < H; ++h) {
        int64_t* head = &q[t * D + h * dh];
        for (std::size_t p = 0; p < half; ++p) {
          const int64_t c = tc[t * half + p];
          const int64_t s = ts[t * half + p];
          const std::size_t i0 = cfg_.rope_style == "half" ? p : 2 * p;
          const std::size_t i1 =
              cfg_.rope_style == "half" ? p + half : 2 * p + 1;
          const int64_t a = head[i0], b2 = head[i1];
          head[i0] = sat((a * c - b2 * s) >> 16, fxv1::kActSat);
          head[i1] = sat((a * s + b2 * c) >> 16, fxv1::kActSat);
        }
      }
  };
  const auto activate = [&](std::vector<int64_t>& v) {
    if (cfg_.act == "relu") {
      for (int64_t& x : v) x = x > 0 ? x : 0;
      return;
    }
    const auto& at = tab_.at("fx.act.table");
    const int64_t lo = -(32ll << 16), hi = (32ll << 16) - 1;
    for (int64_t& x : v) {
      if (x >= hi + 1) continue;         // identity tail (declared)
      if (x < lo) {
        x = 0;                            // zero tail (declared)
        continue;
      }
      x = lerp(at, x - lo, 11);
    }
  };

  // embeddings
  std::vector<int64_t> x(T * D);
  const auto& emb = W("tok_emb.weight");
  for (std::size_t t = 0; t < T; ++t)
    for (std::size_t d = 0; d < D; ++d)
      x[t * D + d] = emb[static_cast<std::size_t>(tokens[t]) * D + d];
  if (cfg_.pos == "learned") {
    const auto& pe = W("pos_emb.weight");
    for (std::size_t t = 0; t < T; ++t)
      for (std::size_t d = 0; d < D; ++d)
        x[t * D + d] =
            sat(x[t * D + d] + pe[t * D + d], fxv1::kActSat);
  } else {
    for (int64_t& v : x) v = sat(v, fxv1::kActSat);
  }

  const auto& et = tab_.at("fx.exp.table");
  const auto& rt = tab_.at("fx.rsqrt.table");
  const int64_t scale =
      rsqrt_fx(static_cast<int64_t>(dh) << 32, rt);  // Q.16

  for (int layer = 0; layer < cfg_.n_layers; ++layer) {
    const std::string L = "layers." + std::to_string(layer) + ".";
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
    std::vector<int64_t> attn(T * D, 0);
    std::vector<int64_t> score(T), ew(T);
    for (std::size_t h = 0; h < H; ++h)
      for (std::size_t t = 0; t < T; ++t) {
        int64_t mx = INT64_MIN;
        for (std::size_t u = 0; u <= t; ++u) {
          int64_t dot = 0;
          for (std::size_t d = 0; d < dh; ++d)
            dot += q[t * D + h * dh + d] * k[u * D + h * dh + d];
          score[u] = ((dot >> 16) * scale) >> 16;  // Q.16 (declared)
          mx = std::max(mx, score[u]);
        }
        int64_t Z = 0;
        for (std::size_t u = 0; u <= t; ++u) {
          int64_t d2 = score[u] - mx;  // <= 0
          if (d2 < -(16ll << 16)) d2 = -(16ll << 16);
          ew[u] = lerp(et, d2 + (16ll << 16), 9);
          Z += ew[u];
        }
        if (Z == 0) Z = 1;  // all-underflow guard (declared: uniform 0)
        for (std::size_t u = 0; u <= t; ++u) {
          const int64_t w2 = floor_div(ew[u] << 16, Z);  // Q.16
          for (std::size_t d = 0; d < dh; ++d)
            attn[t * D + h * dh + d] += w2 * v[u * D + h * dh + d];
        }
      }
    for (int64_t& a : attn) a = sat(a >> 16, fxv1::kActSat);
    const auto proj = linear(attn, D, D, W(L + "attn.o.weight"),
                             bias(L + "attn.o.bias"));
    for (std::size_t i = 0; i < T * D; ++i)
      x[i] = sat(x[i] + proj[i], fxv1::kActSat);
    const auto h2 = normed(x, L + "ln2");
    auto f1 = linear(h2, D, F, W(L + "ffn.fc1.weight"),
                     bias(L + "ffn.fc1.bias"));
    activate(f1);
    const auto f2 = linear(f1, F, D, W(L + "ffn.fc2.weight"),
                           bias(L + "ffn.fc2.bias"));
    for (std::size_t i = 0; i < T * D; ++i)
      x[i] = sat(x[i] + f2[i], fxv1::kActSat);
  }

  const auto xf = normed(x, "ln_f");
  const auto& head =
      w_.count("head.weight") ? W("head.weight") : W("tok_emb.weight");
  std::vector<int64_t> out(V);
  const std::size_t t = T - 1;
  for (std::size_t vv = 0; vv < V; ++vv) {
    int64_t acc = 0;
    for (std::size_t d = 0; d < D; ++d)
      acc += xf[t * D + d] * static_cast<int64_t>(head[vv * D + d]);
    out[vv] = acc;  // Q.32, never rescaled (declared)
  }
  return out;
}

int exact_model::argmax(const std::vector<int>& tokens) const {
  const auto lg = logits_q32(tokens);
  int best = 0;
  for (std::size_t i = 1; i < lg.size(); ++i)
    if (lg[i] > lg[best]) best = static_cast<int>(i);  // ties: lowest id
  return best;
}

std::uint64_t exact_model::logits_hash(const std::vector<int>& tokens) const {
  const auto lg = logits_q32(tokens);
  uint64_t h = 14695981039346656037ull;
  for (const int64_t v : lg) {
    uint64_t u = static_cast<uint64_t>(v);
    for (int i = 0; i < 8; ++i) {
      h ^= (u >> (8 * i)) & 0xff;
      h *= 1099511628211ull;
    }
  }
  return h;
}

}  // namespace ax::nn
