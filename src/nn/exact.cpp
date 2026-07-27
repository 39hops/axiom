/** @file exact.cpp FX-V1 integer forward (see exact.hpp and the spec).

    Implementation discipline: every arithmetic step below is one of
    the declared operations — int add/mul, arithmetic shift (floor),
    floor division, saturation, table lerp, exact comparison. Any
    float appearing past build() is a spec violation. */
#include <ax/nn/exact.hpp>

#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
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

namespace {

/** Per-position stepper with per-layer KV caches. This IS the forward
    (logits_q32 runs it over the whole prompt), so generate() with the
    cache is bit-exact with the batch path by construction — a
    position's integer ops never read later positions. */
struct fx_stepper {
  const config& cfg;
  const std::map<std::string, std::vector<int32_t>>& w;
  const std::map<std::string, std::vector<int64_t>>& tab;
  std::size_t D, H, dh, F, V;
  int64_t scale;
  std::vector<std::vector<int64_t>> kc, vc;  // [layer], D per position
  std::vector<int64_t> x;                    // residual row of last pos
  std::size_t pos = 0;

  fx_stepper(const config& c,
             const std::map<std::string, std::vector<int32_t>>& ww,
             const std::map<std::string, std::vector<int64_t>>& tt)
      : cfg(c), w(ww), tab(tt) {
    D = static_cast<std::size_t>(cfg.d_model);
    H = static_cast<std::size_t>(cfg.n_heads);
    dh = D / H;
    F = static_cast<std::size_t>(cfg.d_ff);
    V = static_cast<std::size_t>(cfg.vocab);
    scale = rsqrt_fx(static_cast<int64_t>(dh) << 32,
                     tab.at("fx.rsqrt.table"));
    kc.resize(static_cast<std::size_t>(cfg.n_layers));
    vc.resize(static_cast<std::size_t>(cfg.n_layers));
  }

  const std::vector<int32_t>& W(const std::string& n) const {
    return w.at(n);
  }
  const int32_t* bias(const std::string& n) const {
    const auto it = w.find(n);
    return it == w.end() ? nullptr : it->second.data();
  }

  std::vector<int64_t> linear_row(const std::vector<int64_t>& in,
                                  std::size_t in_d, std::size_t out_d,
                                  const std::vector<int32_t>& ww,
                                  const int32_t* b) const {
    std::vector<int64_t> y(out_d);
    for (std::size_t o = 0; o < out_d; ++o) {
      int64_t acc = b ? (static_cast<int64_t>(b[o]) << 16) : 0;
      const int32_t* wr = &ww[o * in_d];
      for (std::size_t i = 0; i < in_d; ++i)
        acc += in[i] * static_cast<int64_t>(wr[i]);
      y[o] = sat(acc >> 16, fxv1::kActSat);
    }
    return y;
  }

  std::vector<int64_t> norm_row(const std::vector<int64_t>& row,
                                const std::string& prefix) const {
    const auto& g = W(prefix + ".weight");
    const int32_t* b = bias(prefix + ".bias");
    const auto& rt = tab.at("fx.rsqrt.table");
    const auto Di = static_cast<int64_t>(D);
    int64_t mean = 0;
    if (cfg.norm == "layernorm") {
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
    std::vector<int64_t> y(D);
    for (std::size_t d = 0; d < D; ++d) {
      const int64_t n1 = (c[d] * inv) >> 16;
      const int64_t n2 = (n1 * static_cast<int64_t>(g[d])) >> 16;
      y[d] = sat(n2 + (b ? static_cast<int64_t>(b[d]) : 0),
                 fxv1::kActSat);
    }
    return y;
  }

  void rope_row(std::vector<int64_t>& q) const {
    const auto& tc = tab.at("fx.rope.cos");
    const auto& ts = tab.at("fx.rope.sin");
    const std::size_t half = dh / 2;
    for (std::size_t h = 0; h < H; ++h) {
      int64_t* head = &q[h * dh];
      for (std::size_t p = 0; p < half; ++p) {
        const int64_t c = tc[pos * half + p];
        const int64_t s = ts[pos * half + p];
        const std::size_t i0 = cfg.rope_style == "half" ? p : 2 * p;
        const std::size_t i1 =
            cfg.rope_style == "half" ? p + half : 2 * p + 1;
        const int64_t a = head[i0], b2 = head[i1];
        head[i0] = sat((a * c - b2 * s) >> 16, fxv1::kActSat);
        head[i1] = sat((a * s + b2 * c) >> 16, fxv1::kActSat);
      }
    }
  }

  void activate(std::vector<int64_t>& v) const {
    if (cfg.act == "relu") {
      for (int64_t& x2 : v) x2 = x2 > 0 ? x2 : 0;
      return;
    }
    const auto& at = tab.at("fx.act.table");
    const int64_t lo = -(32ll << 16), hi = (32ll << 16) - 1;
    for (int64_t& x2 : v) {
      if (x2 >= hi + 1) continue;  // identity tail (declared)
      if (x2 < lo) {
        x2 = 0;  // zero tail (declared)
        continue;
      }
      x2 = lerp(at, x2 - lo, 11);
    }
  }

  void feed(int token) {
    if (token < 0 || token >= cfg.vocab)
      throw std::runtime_error("fxv1: token out of vocab");
    if (cfg.max_seq > 0 && pos >= static_cast<std::size_t>(cfg.max_seq))
      throw std::runtime_error("fxv1: sequence exceeds max_seq");
    const auto& et = tab.at("fx.exp.table");
    x.assign(D, 0);
    const auto& emb = W("tok_emb.weight");
    for (std::size_t d = 0; d < D; ++d)
      x[d] = emb[static_cast<std::size_t>(token) * D + d];
    if (cfg.pos == "learned") {
      const auto& pe = W("pos_emb.weight");
      for (std::size_t d = 0; d < D; ++d)
        x[d] = sat(x[d] + pe[pos * D + d], fxv1::kActSat);
    } else {
      for (int64_t& v2 : x) v2 = sat(v2, fxv1::kActSat);
    }
    for (int layer = 0; layer < cfg.n_layers; ++layer) {
      const std::string L = "layers." + std::to_string(layer) + ".";
      const auto h1 = norm_row(x, L + "ln1");
      auto q = linear_row(h1, D, D, W(L + "attn.q.weight"),
                          bias(L + "attn.q.bias"));
      auto k = linear_row(h1, D, D, W(L + "attn.k.weight"),
                          bias(L + "attn.k.bias"));
      const auto v = linear_row(h1, D, D, W(L + "attn.v.weight"),
                                bias(L + "attn.v.bias"));
      if (cfg.pos == "rope") {
        rope_row(q);
        rope_row(k);
      }
      auto& kl = kc[static_cast<std::size_t>(layer)];
      auto& vl = vc[static_cast<std::size_t>(layer)];
      kl.insert(kl.end(), k.begin(), k.end());
      vl.insert(vl.end(), v.begin(), v.end());
      const std::size_t T = pos + 1;
      std::vector<int64_t> attn(D, 0);
      std::vector<int64_t> score(T), ew(T);
      for (std::size_t h = 0; h < H; ++h) {
        int64_t mx = INT64_MIN;
        for (std::size_t u = 0; u < T; ++u) {
          int64_t dot = 0;
          for (std::size_t d = 0; d < dh; ++d)
            dot += q[h * dh + d] * kl[u * D + h * dh + d];
          score[u] = ((dot >> 16) * scale) >> 16;  // Q.16 (declared)
          mx = std::max(mx, score[u]);
        }
        int64_t Z = 0;
        for (std::size_t u = 0; u < T; ++u) {
          int64_t d2 = score[u] - mx;  // <= 0
          if (d2 < -(16ll << 16)) d2 = -(16ll << 16);
          ew[u] = lerp(et, d2 + (16ll << 16), 9);
          Z += ew[u];
        }
        if (Z == 0) Z = 1;  // all-underflow guard (declared)
        for (std::size_t u = 0; u < T; ++u) {
          const int64_t w2 = floor_div(ew[u] << 16, Z);  // Q.16
          for (std::size_t d = 0; d < dh; ++d)
            attn[h * dh + d] += w2 * vl[u * D + h * dh + d];
        }
      }
      for (int64_t& a : attn) a = sat(a >> 16, fxv1::kActSat);
      const auto proj = linear_row(attn, D, D, W(L + "attn.o.weight"),
                                   bias(L + "attn.o.bias"));
      for (std::size_t d = 0; d < D; ++d)
        x[d] = sat(x[d] + proj[d], fxv1::kActSat);
      const auto h2 = norm_row(x, L + "ln2");
      auto f1 = linear_row(h2, D, F, W(L + "ffn.fc1.weight"),
                           bias(L + "ffn.fc1.bias"));
      activate(f1);
      const auto f2 = linear_row(f1, F, D, W(L + "ffn.fc2.weight"),
                                 bias(L + "ffn.fc2.bias"));
      for (std::size_t d = 0; d < D; ++d)
        x[d] = sat(x[d] + f2[d], fxv1::kActSat);
    }
    ++pos;
  }

  std::vector<int64_t> readout() const {
    const auto xf = norm_row(x, "ln_f");
    const auto& head =
        w.count("head.weight") ? W("head.weight") : W("tok_emb.weight");
    std::vector<int64_t> out(V);
    for (std::size_t vv = 0; vv < V; ++vv) {
      int64_t acc = 0;
      for (std::size_t d = 0; d < D; ++d)
        acc += xf[d] * static_cast<int64_t>(head[vv * D + d]);
      out[vv] = acc;  // Q.32, never rescaled (declared)
    }
    return out;
  }
};

int argmax_lowest(const std::vector<int64_t>& lg) {
  int best = 0;
  for (std::size_t i = 1; i < lg.size(); ++i)
    if (lg[i] > lg[best]) best = static_cast<int>(i);  // ties: lowest id
  return best;
}

}  // namespace

std::vector<int64_t> exact_model::logits_q32(
    const std::vector<int>& tokens) const {
  if (tokens.empty()) return {};
  fx_stepper s(cfg_, w_, tab_);
  for (const int t : tokens) s.feed(t);
  return s.readout();
}

std::vector<int> exact_model::generate(const std::vector<int>& prompt,
                                       int max_new, int stop_id) const {
  std::vector<int> out;
  if (prompt.empty()) return out;
  fx_stepper s(cfg_, w_, tab_);
  for (const int t : prompt) s.feed(t);
  for (int i = 0; i < max_new; ++i) {
    if (cfg_.max_seq > 0 &&
        s.pos >= static_cast<std::size_t>(cfg_.max_seq))
      break;
    const int next = argmax_lowest(s.readout());
    out.push_back(next);
    if (next == stop_id) break;
    if (i + 1 < max_new) s.feed(next);
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

std::string exact_model::certify_tables() const {
  char buf[160];
  const auto fail = [&](const char* fmt, const char* name, std::size_t i,
                        double got, double want) {
    std::snprintf(buf, sizeof buf, fmt, name, i, got, want);
    return std::string(buf);
  };
  // per-table float references (checker-side doubles: allowed)
  const auto check = [&](const std::string& name, std::size_t n,
                         double x0, double step,
                         const std::function<double(double)>& f,
                         int mono,          // +1 nondecr, -1 nonincr, 0 none
                         std::size_t mono_from, double entry_tol,
                         double mid_tol) -> std::string {
    const auto it = tab_.find(name);
    if (it == tab_.end()) return "missing " + name;
    const auto& t = it->second;
    if (t.size() != n) return "mis-sized " + name;
    for (std::size_t i = 0; i < n; ++i) {
      const double ref = f(x0 + static_cast<double>(i) * step) * 65536.0;
      if (std::abs(static_cast<double>(t[i]) - ref) > entry_tol)
        return fail("entry-error %s[%zu] got %.1f want %.1f", name.c_str(),
                    i, static_cast<double>(t[i]), ref);
      if (i > mono_from && mono != 0) {
        const int64_t d = t[i] - t[i - 1];
        if ((mono > 0 && d < 0) || (mono < 0 && d > 0))
          return fail("monotonicity %s[%zu] got %.0f prev %.0f",
                      name.c_str(), i, static_cast<double>(t[i]),
                      static_cast<double>(t[i - 1]));
      }
    }
    // seeded midpoint fuzz: interpolation error stays inside the
    // curvature + rounding bound (argmax correctness rides on this)
    std::mt19937 rng(20260727u);
    std::uniform_int_distribution<std::size_t> pick(0, n - 2);
    const int fbits = name == "fx.act.table" ? 11
                      : name == "fx.exp.table" ? 9 : 23;
    for (int s = 0; s < 1000; ++s) {
      const std::size_t i = pick(rng);
      const int64_t frac = static_cast<int64_t>(rng() & ((1u << fbits) - 1));
      const int64_t u = (static_cast<int64_t>(i) << fbits) + frac;
      const int64_t got = lerp(t, u, fbits);
      const double xx = x0 + (static_cast<double>(i) +
                              static_cast<double>(frac) /
                                  static_cast<double>(1ll << fbits)) *
                                 step;
      if (std::abs(static_cast<double>(got) - f(xx) * 65536.0) > mid_tol)
        return fail("midpoint %s[%zu] got %.1f want %.1f", name.c_str(), i,
                    static_cast<double>(got), f(xx) * 65536.0);
    }
    return "";
  };

  std::string r = check(
      "fx.exp.table", fxv1::kExpTableN, -16.0, 1.0 / 128.0,
      [](double x) { return std::exp(x); }, +1, 0, 1.0, 4.0);
  if (!r.empty()) return r;
  if (tab_.at("fx.exp.table").back() != 65536)
    return "fx.exp.table endpoint != 1.0 (softmax max element rides it)";
  r = check(
      "fx.rsqrt.table", fxv1::kRsqrtTableN, 1.0, 1.0 / 128.0,
      [](double x) { return 1.0 / std::sqrt(x); }, -1, 0, 1.0, 4.0);
  if (!r.empty()) return r;
  if (cfg_.act != "relu") {
    std::function<double(double)> f;
    if (cfg_.act == "gelu")
      f = [](double x) {
        return 0.5 * x * (1 + std::erf(x / std::sqrt(2.0)));
      };
    else if (cfg_.act == "gelu_tanh")
      f = [](double x) {
        const double c = std::sqrt(2.0 / 3.14159265358979323846);
        return 0.5 * x * (1 + std::tanh(c * (x + 0.044715 * x * x * x)));
      };
    else
      f = [](double x) { return x / (1.0 + std::exp(-x)); };
    // monotone from x = 0 (index 1024); the negative lobe is checked by
    // entry error only (gelu/silu genuinely dip there)
    r = check("fx.act.table", fxv1::kActTableN, -32.0, 1.0 / 32.0, f, +1,
              1024, 1.0, 16.0);
    if (!r.empty()) return r;
  }
  if (cfg_.pos == "rope") {
    const auto& tc = tab_.at("fx.rope.cos");
    const auto& ts = tab_.at("fx.rope.sin");
    for (std::size_t i = 0; i < tc.size(); ++i) {
      if (std::abs(tc[i]) > 65536 || std::abs(ts[i]) > 65536)
        return "rope entry out of unit range";
      const int64_t mag = tc[i] * tc[i] + ts[i] * ts[i];
      if (std::abs(mag - (1ll << 32)) > 3ll * 65536)
        return "rope pair off the unit circle at " + std::to_string(i);
    }
  }
  return "";
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
