# Gravmoe Engine Leg (MoE Body + Window Cycling) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Grow the certified intbirth engine to the gravmoe anatomy — an E-expert MoE FFN at the Body seam with top-1 routing, gravity events, and 4× backward boost — plus window cycling in the training loop, so the C++ leg can reproduce the house pins RB1 (`c6766da2...`) and RB1-S16 (`14981553...`) once Artin carries over the `detbwd_gmoe_ref` artifacts.

**Architecture:** Split `block::body_fwd/body_bwd` into attention and FFN halves (pure refactor, regression-gated by the existing certified digests), add MoE FFN + router methods on `block`, and a new composed loop class `moe_birth` (bound as `MoeBirth`). Window cycling is added to both `multi_birth` and `moe_birth` via an optional `windows_bytes` constructor argument; the no-windows path stays byte-identical to the certified mb trajectory.

**Tech Stack:** C++17 (`src/nn/intbirth.cpp`, `include/ax/nn/intbirth.hpp`), pybind11 (`bindings/intbirth.cpp`), Python acceptance scripts in `tools/int_adamw/`.

## Global Constraints

- Contract defaults for the gravmoe base (relay 2026-08-01-5): V=40, T=32, D=64, DH=16, F=128, NBLK=2, seed 17, SHIFT=14, STEPS=2000, E=4, K=100, LN=0 default (no gravity), GB = 4 × GBOOST = 1024 when GBOOST=256... **NOTE:** relay says "GB = 4 x GBOOST (=1024)", i.e. GB=1024 with the r2b GBOOST=256. Engine rule: MoE loop uses `gboost * 4` everywhere the dense loop uses `gboost`.
- COND=1/QK=1 are **init-draw bounds, house-side**: the engine never draws weights; init arrives as opaque bytes. The engine's verify-the-knob obligation is fulfilled by the acceptance script printing the per-family bounds read from `contract.json` (Task 6).
- Rounding placement is contract text: every multi-term int sum rounds ONCE after the sum; gravity mean finalized ONCE via one rdiv BEFORE per-expert pulls; embedding grad = rounded head part + exact scatter-add (unchanged).
- Router convention (relay text, verbatim semantics): `r = rdiv(int_mm(h2, wr), Q)` [T,E]; `p_r = softmax_rows(r, PQ)`; top-1 with fixed tie-break, **lowest expert index wins**; `y = rdiv(out_top * top_p, PQ)`. Backward: d(out_e) via top_p, d(top_p) scattered into dp_r, softmax_bwd, then router + h2 paths.
- Gravity event: every K=100 **optimizer** steps (i.e. after steps 100, 200, …), in wide Q_w space, per body in index order, kinds in order (wg, wu, wd), experts in index order: `mean = rdiv(sum_e w_e, E)` once, then each `w_e += rdiv((mean - w_e) * LN, LD)`.
- Draw order / param_order (the contract): `emb`, per body `[wq wk wv wo g1 g2, wr, e0.wg e0.wu e0.wd, e1.wg, …, e3.wd]`, `g_f`. wr shape [E, D]. Total params at defaults: 208,192 (verified against the relay).
- OUT OF SCOPE for this plan: TAU (RB3 pin), GATE mode, scheduled sampling (S1) — house/Python-side readouts or knobs whose exact contract text travels with the artifacts; batching/data-loaders (explicitly excluded by the relay). Final-sha verification against pins is Task 6 and can only PASS after artifacts land in `scratch/detbwd_gmoe_ref/` locally.
- Regression gates that must stay green after EVERY task (run all three):
  ```bash
  cd /Users/artin/code/axiom/tools/int_adamw
  python verify_intbirth.py ../../build-rel/bindings
  python verify_primitives.py ../../build-rel/bindings
  python verify_multiblock.py ../../build-rel/bindings <mb_ref.json path if not default>
  ```
  (verify_multiblock defaults to `/Users/artin/code/llmopt/scratch/detbwd_mb_ref/mb_ref.json`; if that path is absent on this machine, skip it only with an explicit note in the commit message.)
- Build command (existing CMake target):
  ```bash
  cmake --build /Users/artin/code/axiom/build-rel --target intbirth -j
  ```

---

### Task 1: Split body_fwd/body_bwd into attention + FFN halves (pure refactor)

**Files:**
- Modify: `include/ax/nn/intbirth.hpp` (block class, ~line 99–112)
- Modify: `src/nn/intbirth.cpp` (block::body_fwd ~389–463, block::body_bwd ~475–581)

**Interfaces:**
- Produces (private methods on `block`, used by Tasks 3–4):
  - `Mat attn_fwd(const std::map<std::string, Mat>& w, const Mat& x, block_cache& c) const;` — x → x1 (fills c.x…c.x1 exactly as the first half of body_fwd today; returns c.x1).
  - `Mat ffn_fwd(const std::map<std::string, Mat>& w, const Mat& x1, block_cache& c) const;` — x1 → x2 (fills c.h2…c.x2; `w` needs wg/wu/wd/g2).
  - `Mat ffn_bwd(const std::map<std::string, Mat>& w, const Mat& dx2_masked, const block_cache& c, std::map<std::string, Mat>& G) const;` — takes dx2 **already multiplied by c.m2**, fills G["wd"/"wu"/"wg"/"g2"], returns dx1 contribution from the FFN path (the `rms_b(dh2, …)` result, WITHOUT the `+ dx2` residual add or m1 mask — those stay in the caller).
  - `Mat attn_bwd(const std::map<std::string, Mat>& w, const Mat& dx1_masked, const block_cache& c, std::map<std::string, Mat>& G) const;` — takes dx1 already `(dx_ffn + dx2)*m1`-combined, fills G for wq/wk/wv/wo/g1, returns dx0 (rms output grad, WITHOUT the `+ dx1` residual add — caller adds).
- `body_fwd` becomes `{ check_keys; return ffn_fwd(w, attn_fwd(w, x, c), c); }` and `body_bwd` composes the two bwd halves with the residual adds/masks exactly where they are today.

- [ ] **Step 1: Declare the four private methods in the header**

Add under the existing private section of `class block` in `intbirth.hpp`:

```cpp
 private:
  // body halves (gravmoe seam): body = attn ∘ ffn; the MoE body
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
```

- [ ] **Step 2: Move the code**

In `intbirth.cpp`, cut body_fwd lines from `c.x = x;` through the `c.x1[i] = clampi(...)` loop into `attn_fwd` (it ends `return c.x1;`), and the remainder (from `c.h2 = rms_fwd(...)` through `return c.x2;`) into `ffn_fwd(w, x1, c)` — note `ffn_fwd` reads the residual base from `c.x1`, so its body is verbatim today's code. `body_fwd` becomes:

```cpp
Mat block::body_fwd(const std::map<std::string, Mat>& w, const Mat& x,
                    block_cache& c) const {
  check_keys(c_, w, BODY_KEYS, 9);
  return ffn_fwd(w, attn_fwd(w, x, c), c);
}
```

(Move the `if (x.size() != T*D) throw` and the rope/table lookups into `attn_fwd`; `ffn_fwd` re-fetches `sil = tab_.at("silu.tab")` itself.)

For backward: `ffn_bwd` gets today's code from `Mat df = int_gemm_nt(dx2, ...)` through `Mat dx1 = rms_b(dh2, c.x1, w.at("g2"), c.i2, G["g2"]);` and returns that `dx1` (pre-residual). `attn_bwd` gets everything from `Mat da = int_gemm_nt(dx1, ...)` through `Mat dx0 = rms_b(dh1, ...)` and returns `dx0` (pre-residual). `body_bwd` becomes:

```cpp
std::map<std::string, Mat> block::body_bwd(
    const std::map<std::string, Mat>& w, const Mat& dxin,
    const block_cache& c, Mat* dx0_out) const {
  check_keys(c_, w, BODY_KEYS, 9);
  if (i64(dxin.size()) != i64(c_.T) * c_.D)
    throw std::runtime_error("intbirth: bad dxin shape");
  std::map<std::string, Mat> G;
  Mat dx2 = dxin;
  for (std::size_t i = 0; i < dx2.size(); i++) dx2[i] *= c.m2[i];
  Mat dx1 = ffn_bwd(w, dx2, c, G);
  for (std::size_t i = 0; i < dx1.size(); i++)
    dx1[i] = (dx1[i] + dx2[i]) * c.m1[i];
  Mat dx0 = attn_bwd(w, dx1, c, G);
  if (dx0_out) {
    for (std::size_t i = 0; i < dx0.size(); i++) dx0[i] += dx1[i];
    *dx0_out = std::move(dx0);
  }
  return G;
}
```

- [ ] **Step 3: Build**

Run: `cmake --build /Users/artin/code/axiom/build-rel --target intbirth -j` — expect clean build.

- [ ] **Step 4: Run all three regression gates (Global Constraints)**

Expected: verify_intbirth, verify_primitives, verify_multiblock all PASS with unchanged digests. This is the certification that the refactor is pure.

- [ ] **Step 5: Commit**

```bash
git add include/ax/nn/intbirth.hpp src/nn/intbirth.cpp
git commit -m "refactor: split Body into attn/ffn halves (gravmoe seam) - all certified digests unchanged"
```

---

### Task 2: Window cycling in multi_birth

**Files:**
- Modify: `include/ax/nn/intbirth.hpp` (multi_birth)
- Modify: `src/nn/intbirth.cpp` (multi_birth ctor + step_once)
- Modify: `bindings/intbirth.cpp` (MultiBirth init)
- Create: `tools/int_adamw/test_windows.py`

**Interfaces:**
- Produces: `multi_birth(tables_bytes, init_bytes, contract, windows_bytes = "")`. When `windows_bytes` is empty: behavior exactly as today (tok/tgt read from init tail). When non-empty: init_bytes contains **params only** (no tok/tgt tail); `windows_bytes` is NW consecutive records of `tok[T] ++ tgt[T]`, int64 LE (NW inferred from length, must divide evenly, NW ≥ 1); step i trains on window `i mod NW`.
- Python: `intbirth.MultiBirth(tables, init, contract, windows_bytes=b"...")` keyword-optional.

- [ ] **Step 1: Write the failing test**

`tools/int_adamw/test_windows.py`:

```python
"""Window-cycling gate: NW=1 windows path must reproduce the
certified no-windows mb trajectory digest exactly, and NW=2 with
distinct windows must diverge from it."""
import json, sys
sys.path.insert(0, sys.argv[1])
import intbirth  # noqa: E402

C = {"V": 64, "T": 32, "D": 64, "DH": 16, "F": 128,
     "n_blocks": 2, "SHIFT": 14}
tables = open("r2b_tables.bin", "rb").read()
init = open("mb_init.bin", "rb").read()
T = C["T"]
tail = 2 * T * 8                      # tok + tgt
params, win = init[:-tail], init[-tail:]

a = intbirth.MultiBirth(tables, init, C)
b = intbirth.MultiBirth(tables, params, C, windows_bytes=win)
a.run(20); b.run(20)
assert a.mark() == b.mark(), "NW=1 must equal no-windows path"

# two distinct windows -> different trajectory
import struct
tok2 = struct.pack("<%dq" % T, *([1] * T))
tgt2 = struct.pack("<%dq" % T, *([2] * T))
c = intbirth.MultiBirth(tables, params, C,
                        windows_bytes=win + tok2 + tgt2)
c.run(20)
assert c.mark() != a.mark(), "NW=2 must diverge"
print("WINDOWS PASS")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd tools/int_adamw && python test_windows.py ../../build-rel/bindings`
Expected: TypeError (MultiBirth takes no windows_bytes) or similar.

- [ ] **Step 3: Implement**

Header — replace multi_birth's `std::vector<i64> tok_, tgt_;` with:

```cpp
  std::vector<Mat> wtok_, wtgt_;  // NW windows; step i uses i % NW
```

and the ctor signature gains `const std::string& windows_bytes = ""`. Ctor: after reading params (unchanged), if `windows_bytes.empty()` read tok/tgt from init tail into `wtok_/wtgt_` as a single window (exactly today's reads + validation). Else require `off == init_bytes.size()` right after params ("trailing init bytes"), then:

```cpp
  const std::size_t rec = std::size_t(c.T) * 2 * 8;
  if (windows_bytes.empty() || windows_bytes.size() % rec)
    throw std::runtime_error("intbirth: bad windows length");
  const std::size_t nw = windows_bytes.size() / rec;
  for (std::size_t i = 0; i < nw; i++) {
    Mat tk(c.T), tg(c.T);
    std::memcpy(tk.data(), windows_bytes.data() + i * rec, c.T * 8);
    std::memcpy(tg.data(), windows_bytes.data() + i * rec + c.T * 8,
                c.T * 8);
    for (const i64 t : tk)
      if (t < 0 || t >= c.V)
        throw std::runtime_error("intbirth: token out of vocab");
    for (const i64 t : tg)
      if (t < 0 || t >= c.V)
        throw std::runtime_error("intbirth: target out of vocab");
    wtok_.push_back(std::move(tk));
    wtgt_.push_back(std::move(tg));
  }
```

step_once: at the top, `const Mat& tok = wtok_[step_ % wtok_.size()]; const Mat& tgt = wtgt_[step_ % wtgt_.size()];` and replace all `tok_`/`tgt_` uses in the function body with `tok`/`tgt` (embedding lookup, loss, dlogits, embedding scatter-add).

Binding — MultiBirth init lambda gains `const py::bytes& windows` with default:

```cpp
      .def(py::init([](const py::bytes& tables, const py::bytes& init,
                       const py::dict& c, const py::bytes& windows) {
             return new multi_birth(std::string(tables),
                                    std::string(init),
                                    contract_from_dict(c),
                                    std::string(windows));
           }),
           py::arg("tables_bytes"), py::arg("init_bytes"),
           py::arg("contract"), py::arg("windows_bytes") = py::bytes(),
           ...)
```

- [ ] **Step 4: Build, run test_windows.py (PASS) and all three regression gates (PASS, digests unchanged)**

- [ ] **Step 5: Commit**

```bash
git add include/ax/nn/intbirth.hpp src/nn/intbirth.cpp bindings/intbirth.cpp tools/int_adamw/test_windows.py
git commit -m "feat: window cycling in MultiBirth (step i uses window i mod NW) - no-windows path digest-identical"
```

---

### Task 3: MoE FFN + router on block (moe_body_fwd / moe_body_bwd)

**Files:**
- Modify: `include/ax/nn/intbirth.hpp` (contract, block_cache, block)
- Modify: `src/nn/intbirth.cpp`
- Create: `tools/int_adamw/test_moe_parity.py`

**Interfaces:**
- `contract` gains `int n_experts = 0;` (0/1-with-no-router = dense; the MoE path requires ≥ 1). Binding key `"E"`.
- MoE weight names inside a body map: `wq wk wv wo g1 g2` + `wr` ([E,D]) + `e{j}.wg / e{j}.wu / e{j}.wd` for j in 0..E-1.
- `block_cache` gains MoE fields: `Mat r, pr; std::vector<int> top; Mat top_p; Mat egp, eu, esg, ef, eout;` — per-token router logits/probs, chosen expert index per row, its prob, and the selected expert's FFN intermediates laid out [T,F]/[T,D] (row t holds expert top[t]'s values).
- Public methods on `block`:
  - `Mat moe_body_fwd(const std::map<std::string, Mat>& w, const Mat& x, block_cache& c) const;`
  - `std::map<std::string, Mat> moe_body_bwd(const std::map<std::string, Mat>& w, const Mat& dxin, const block_cache& c, Mat* dx0_out) const;` — grads keyed with the MoE names above.

- [ ] **Step 1: Write the failing parity test**

E=1 MoE is exactly dense: softmax over one expert gives `p = rdiv(e*PQ, e) = PQ`, so `y = rdiv(out*PQ, PQ) = out`, and the gate backward is the identity at PQ scale. `tools/int_adamw/test_moe_parity.py`:

```python
"""E=1 MoE body must match the dense body bit-exactly (fwd + all
shared grads + dx0), and its expert grads must equal the dense
wg/wu/wd grads. Runs on the r2b fixture weights."""
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import intbirth  # noqa: E402

C = {"V": 64, "T": 32, "D": 64, "DH": 16, "F": 128, "SHIFT": 12}
tables = open("r2b_tables.bin", "rb").read()
init = open("r2b_init.bin", "rb").read()

# unpack the 11 KEYS tensors at Q scale (init is narrow ints)
shapes = [("wq", (16, 64)), ("wk", (16, 64)), ("wv", (16, 64)),
          ("wo", (64, 16)), ("wg", (128, 64)), ("wu", (128, 64)),
          ("wd", (64, 128)), ("wh", (64, 64)), ("g1", (64,)),
          ("g2", (64,)), ("g3", (64,))]
w, off = {}, 0
buf = np.frombuffer(init, dtype="<i8")
for k, s in shapes:
    n = int(np.prod(s))
    w[k] = buf[off:off + n].reshape(s).copy()
    off += n
x = buf[off:off + 32 * 64].reshape(32, 64).copy()

blk = intbirth.Block(tables, C)
Cm = dict(C); Cm["E"] = 1
blkm = intbirth.Block(tables, Cm)

wm = {k: w[k] for k in ("wq", "wk", "wv", "wo", "g1", "g2")}
wm["wr"] = np.ones((1, 64), dtype=np.int64)
wm["e0.wg"], wm["e0.wu"], wm["e0.wd"] = w["wg"], w["wu"], w["wd"]
wd = {k: w[k] for k in ("wq", "wk", "wv", "wo", "g1", "g2",
                        "wg", "wu", "wd")}

x2d, cd = blk.body_fwd(wd, x)
x2m, cm = blkm.moe_body_fwd(wm, x)
assert (x2d == x2m).all(), "fwd parity"

rng = np.random.RandomState(7)
dxin = rng.randint(-512, 512, size=(32, 64)).astype(np.int64)
Gd, dx0d = blk.body_bwd(wd, dxin, cd)
Gm, dx0m = blkm.moe_body_bwd(wm, dxin, cm)
assert (dx0d == dx0m).all(), "dx0 parity"
for k in ("wq", "wk", "wv", "wo", "g1", "g2"):
    assert (Gd[k] == Gm[k]).all(), k
for a, b in (("wg", "e0.wg"), ("wu", "e0.wu"), ("wd", "e0.wd")):
    assert (Gd[a] == Gm[b]).all(), b
print("MOE PARITY PASS")
```

NOTE for implementer: with E=1 the router grad path contributes `d(top_p)`-driven terms into dh2 and G["wr"]; softmax_bwd over a single class is `p*(dp - inner)/PQ` with `inner = rdiv(p*dp, PQ)` and `p = PQ`, which gives `ds = dp - inner`; this is NOT automatically zero under integer rounding for arbitrary dp, but `inner = rdiv(PQ*dp, PQ) = dp`, so `ds = 0` exactly, hence zero router leakage into dh2/wr and the parity above is exact. If parity fails, check this chain first.

- [ ] **Step 2: Run it to verify it fails**

Run: `cd tools/int_adamw && python test_moe_parity.py ../../build-rel/bindings`
Expected: FAIL — `Block` has no `moe_body_fwd` / `"E"` rejected.

- [ ] **Step 3: Implement moe_body_fwd**

Add `int n_experts = 0;` to contract (binding: `c.n_experts = geti("E", c.n_experts);`). Add cache fields. In `intbirth.cpp`:

```cpp
namespace {
std::vector<std::string> moe_body_keys(int E) {
  std::vector<std::string> k = {"wq", "wk", "wv", "wo",
                                "g1", "g2", "wr"};
  for (int e = 0; e < E; e++)
    for (const char* s : {".wg", ".wu", ".wd"})
      k.push_back("e" + std::to_string(e) + s);
  return k;
}
}  // namespace

Mat block::moe_body_fwd(const std::map<std::string, Mat>& w,
                        const Mat& x, block_cache& c) const {
  const int T = c_.T, D = c_.D, F = c_.F, E = c_.n_experts;
  const i64 PQ = c_.pq, CL = c_.act_clamp;
  if (E < 1) throw std::runtime_error("intbirth: E < 1");
  for (const auto& k : moe_body_keys(E))
    if (!w.count(k))
      throw std::runtime_error("intbirth: missing " + k);
  if (i64(w.at("wr").size()) != i64(E) * D)
    throw std::runtime_error("intbirth: bad wr shape");
  const Mat& sil = tab_.at("silu.tab");

  attn_fwd(w, x, c);                       // fills c.x .. c.x1
  c.h2 = rms_fwd(c.x1, w.at("g2"), c.i2);

  // router: r = rdiv(h2 @ wr^T, Q); p_r = softmax(PQ); top-1,
  // lowest index wins ties (strict > when scanning up)
  c.r = int_gemm(c.h2, T, D, w.at("wr"), E);
  rdiv_inplace(c.r, Q);
  c.pr = softmax_rows(c.r, T, E, PQ);
  c.top.assign(T, 0);
  c.top_p.assign(T, 0);
  for (int t = 0; t < T; t++) {
    int best = 0;
    for (int e = 1; e < E; e++)
      if (c.pr[std::size_t(t) * E + e] >
          c.pr[std::size_t(t) * E + best])
        best = e;
    c.top[t] = best;
    c.top_p[t] = c.pr[std::size_t(t) * E + best];
  }

  // selected expert's FFN per row (row t computed with expert
  // top[t]'s weights; per-row independent so this is exact)
  c.egp.assign(std::size_t(T) * F, 0);
  c.eu.assign(std::size_t(T) * F, 0);
  c.esg.assign(std::size_t(T) * F, 0);
  c.ef.assign(std::size_t(T) * F, 0);
  c.eout.assign(std::size_t(T) * D, 0);
  for (int t = 0; t < T; t++) {
    const std::string p = "e" + std::to_string(c.top[t]);
    const Mat& wg = w.at(p + ".wg");
    const Mat& wu = w.at(p + ".wu");
    const Mat& wd = w.at(p + ".wd");
    for (int f = 0; f < F; f++) {
      i64 ag = 0, au = 0;
      for (int d = 0; d < D; d++) {
        const i64 h = c.h2[std::size_t(t) * D + d];
        ag += h * wg[std::size_t(f) * D + d];
        au += h * wu[std::size_t(f) * D + d];
      }
      const i64 z = rdiv(ag, Q);
      c.egp[std::size_t(t) * F + f] = z;
      c.eu[std::size_t(t) * F + f] = rdiv(au, Q);
      c.esg[std::size_t(t) * F + f] =
          z > ts_ ? z : z < -ts_ ? 0 : sil[z + ts_];
      c.ef[std::size_t(t) * F + f] =
          rdiv(c.esg[std::size_t(t) * F + f] *
                   c.eu[std::size_t(t) * F + f],
               Q);
    }
    for (int d = 0; d < D; d++) {
      i64 acc = 0;
      for (int f = 0; f < F; f++)
        acc += c.ef[std::size_t(t) * F + f] *
               wd[std::size_t(d) * F + f];
      c.eout[std::size_t(t) * D + d] = rdiv(acc, Q);
    }
  }

  // gate + residual + clamp (fx3 multiplicative-gate convention)
  Mat pre2(std::size_t(T) * D);
  for (int t = 0; t < T; t++)
    for (int d = 0; d < D; d++)
      pre2[std::size_t(t) * D + d] =
          rdiv(c.eout[std::size_t(t) * D + d] * c.top_p[t], PQ) +
          c.x1[std::size_t(t) * D + d];
  c.m2.assign(pre2.size(), 0);
  c.x2.assign(pre2.size(), 0);
  for (std::size_t i = 0; i < pre2.size(); i++) {
    c.m2[i] = (pre2[i] <= CL && pre2[i] >= -CL);
    c.x2[i] = clampi(pre2[i], -CL, CL);
  }
  return c.x2;
}
```

- [ ] **Step 4: Implement moe_body_bwd**

Mirror of the dense ffn_bwd with the gate chain first, per-row expert weights, and dW accumulated per expert. Then softmax_bwd on the router, then wr/h2 paths, then the shared attention half via `attn_bwd`:

```cpp
std::map<std::string, Mat> block::moe_body_bwd(
    const std::map<std::string, Mat>& w, const Mat& dxin,
    const block_cache& c, Mat* dx0_out) const {
  const int T = c_.T, D = c_.D, F = c_.F, E = c_.n_experts;
  const i64 PQ = c_.pq;
  if (E < 1) throw std::runtime_error("intbirth: E < 1");
  if (i64(dxin.size()) != i64(T) * D)
    throw std::runtime_error("intbirth: bad dxin shape");
  const Mat& dsl = tab_.at("dsilu.tab");
  std::map<std::string, Mat> G;

  Mat dx2 = dxin;
  for (std::size_t i = 0; i < dx2.size(); i++) dx2[i] *= c.m2[i];

  // gate chain: d(out) = rdiv(dx2 * top_p, PQ);
  // d(top_p)[t] = rdiv(sum_d out[t,d]*dx2[t,d], PQ)
  Mat dout(std::size_t(T) * D);
  Mat dtp(T);
  for (int t = 0; t < T; t++) {
    i64 acc = 0;
    for (int d = 0; d < D; d++) {
      const std::size_t i = std::size_t(t) * D + d;
      dout[i] = rdiv(dx2[i] * c.top_p[t], PQ);
      acc += c.eout[i] * dx2[i];
    }
    dtp[t] = rdiv(acc, PQ);
  }

  // expert FFN backward, per row, grads accumulated per expert
  for (int e = 0; e < E; e++) {
    const std::string p = "e" + std::to_string(e);
    G[p + ".wg"].assign(std::size_t(F) * D, 0);
    G[p + ".wu"].assign(std::size_t(F) * D, 0);
    G[p + ".wd"].assign(std::size_t(D) * F, 0);
  }
  G["wr"].assign(std::size_t(E) * D, 0);
  Mat dh2(std::size_t(T) * D, 0);
  Mat df(F), du(F), dgp(F);
  for (int t = 0; t < T; t++) {
    const std::string p = "e" + std::to_string(c.top[t]);
    const Mat& wg = w.at(p + ".wg");
    const Mat& wu = w.at(p + ".wu");
    const Mat& wd = w.at(p + ".wd");
    Mat& Gwg = G.at(p + ".wg");
    Mat& Gwu = G.at(p + ".wu");
    Mat& Gwd = G.at(p + ".wd");
    for (int f = 0; f < F; f++) {
      i64 acc = 0;
      for (int d = 0; d < D; d++)
        acc += dout[std::size_t(t) * D + d] * wd[std::size_t(d) * F + f];
      df[f] = rdiv(acc, Q);
      const i64 z = c.egp[std::size_t(t) * F + f];
      const i64 dsv = z > ts_ ? Q : z < -ts_ ? 0 : dsl[z + ts_];
      du[f] = rdiv(c.esg[std::size_t(t) * F + f] * df[f], Q);
      dgp[f] = rdiv(
          rdiv(c.eu[std::size_t(t) * F + f] * df[f], Q) * dsv, Q);
    }
    // dW: accumulate RAW over this expert's rows; one rdiv(Q) per
    // expert AFTER the token loop (dense placement sums all rows in
    // xty before its single rdiv — same grouping, per expert)
    for (int f = 0; f < F; f++)
      for (int d = 0; d < D; d++) {
        Gwg[std::size_t(f) * D + d] +=
            dgp[f] * c.h2[std::size_t(t) * D + d];
        Gwu[std::size_t(f) * D + d] +=
            du[f] * c.h2[std::size_t(t) * D + d];
      }
    for (int d = 0; d < D; d++)
      for (int f = 0; f < F; f++)
        Gwd[std::size_t(d) * F + f] +=
            dout[std::size_t(t) * D + d] *
            c.ef[std::size_t(t) * F + f];
    // dh2 from expert weights (two-term sum, one rdiv after)
    for (int d = 0; d < D; d++) {
      i64 acc = 0;
      for (int f = 0; f < F; f++)
        acc += du[f] * wu[std::size_t(f) * D + d] +
               dgp[f] * wg[std::size_t(f) * D + d];
      dh2[std::size_t(t) * D + d] = acc;  // raw; rdiv(Q) after loop
    }
  }
```

Then the router path:

```cpp
  for (auto& kv : G) rdiv_inplace(kv.second, Q);   // expert + wr? NO —
  // round ONLY the expert grads here (G currently holds just experts
  // and the zeroed wr); do it explicitly:
  for (int e = 0; e < E; e++) {
    const std::string p = "e" + std::to_string(e);
    rdiv_inplace(G.at(p + ".wg"), Q);
    rdiv_inplace(G.at(p + ".wu"), Q);
    rdiv_inplace(G.at(p + ".wd"), Q);
  }
  rdiv_inplace(dh2, Q);

  // scatter d(top_p) into dp_r, softmax_bwd at PQ, router paths
  Mat dpr(std::size_t(T) * E, 0);
  for (int t = 0; t < T; t++)
    dpr[std::size_t(t) * E + c.top[t]] = dtp[t];
  Mat dr(std::size_t(T) * E);
  for (int t = 0; t < T; t++) {
    i64 inner = 0;
    for (int e = 0; e < E; e++)
      inner += rdiv(c.pr[std::size_t(t) * E + e] *
                        dpr[std::size_t(t) * E + e],
                    PQ);
    for (int e = 0; e < E; e++)
      dr[std::size_t(t) * E + e] =
          rdiv(c.pr[std::size_t(t) * E + e] *
                   (dpr[std::size_t(t) * E + e] - inner),
               PQ);
  }
  G["wr"] = int_gemm_xty(dr, T, E, c.h2, D);
  rdiv_inplace(G["wr"], Q);
  {
    const Mat t2 = int_gemm_nt(dr, T, E, w.at("wr"), D);
    for (std::size_t i = 0; i < dh2.size(); i++) dh2[i] += t2[i];
    // dh2 already rounded; the router term is rounded inside
    // int_gemm_nt? NO — int_gemm_nt is raw. Round the router term
    // separately then add: this is a TWO-GROUP sum (ffn group and
    // router group each finalized once), the same grouping the
    // house reference uses for "router + h2 paths".
  }
```

**Write the router/h2 merge as two finalized groups:** compute `Mat dh2r = int_gemm_nt(dr, T, E, w.at("wr"), D); rdiv_inplace(dh2r, Q);` and add into the already-rounded FFN `dh2`. (If the P4 pins later disagree, the alternative single-group rounding — accumulate raw and round once — is the first thing to flip; both are one-line changes and the parity test in this task passes under either since E=1 makes dr ≡ 0.)

Then finish exactly like the dense path:

```cpp
  Mat dx1 = rms_bwd(dh2, c.x1, w.at("g2"), c.i2, G["g2"]);
  for (std::size_t i = 0; i < dx1.size(); i++)
    dx1[i] = (dx1[i] + dx2[i]) * c.m1[i];
  Mat dx0 = attn_bwd(w, dx1, c, G);
  if (dx0_out) {
    for (std::size_t i = 0; i < dx0.size(); i++) dx0[i] += dx1[i];
    *dx0_out = std::move(dx0);
  }
  return G;
}
```

Bindings: add `moe_body_fwd` / `moe_body_bwd` on Block mirroring `body_fwd`/`body_bwd` (grads dict must pass through MoE names — extend `grads_to_dict` to fall back to a flat shape `{int(m.size())}` for names not in `key_shapes`, and add `"wr" -> {E, D}`).

- [ ] **Step 5: Build, run test_moe_parity.py (PASS) and the three regression gates (PASS)**

- [ ] **Step 6: Commit**

```bash
git add include/ax/nn/intbirth.hpp src/nn/intbirth.cpp bindings/intbirth.cpp tools/int_adamw/test_moe_parity.py
git commit -m "feat: MoE Body (top-1 router, fx3 gate, per-expert FFN) - E=1 bit-identical to dense body"
```

---

### Task 4: moe_birth composed loop (param order, GB boost, gravity, windows)

**Files:**
- Modify: `include/ax/nn/intbirth.hpp`
- Modify: `src/nn/intbirth.cpp`
- Modify: `bindings/intbirth.cpp`
- Create: `tools/int_adamw/test_moe_birth.py`

**Interfaces:**
- `contract` gains `i64 grav_k = 100, grav_ln = 0, grav_ld = 1;` (binding keys `"K"`, `"LN"`, `"LD"`).
- `class moe_birth` — same shape as `multi_birth` (run/mark/traj_sha/weights_bytes/param_order/loss/nz/step_count), ctor `(tables_bytes, init_bytes, contract, windows_bytes = "")`. param_order: `emb`, per body `b{i}.{wq,wk,wv,wo,g1,g2,wr}`, `b{i}.e{j}.{wg,wu,wd}` j ascending, then `g_f`. init_bytes = params in param_order (+ tok/tgt tail iff windows_bytes empty). Backward boost `gb = c.gboost * 4` (dlogits × gb, unboost by Q*gb). Gravity: after each optimizer step, if `step_ % grav_k == 0 && grav_ln != 0`, in wide space, per body / kind (wg,wu,wd) / feature index: `mean = rdiv(Σ_e w_e[i], E)` then `w_e[i] += rdiv((mean - w_e[i]) * grav_ln, grav_ld)`.
- Binding class `MoeBirth`.

- [ ] **Step 1: Write the failing test**

`tools/int_adamw/test_moe_birth.py` — no reference pins yet, so gate on internal invariants: (a) param_order matches the relay contract; (b) param count at gravmoe defaults = 208,192; (c) two identical runs give identical marks (determinism); (d) LN=0 vs (LN=1, LD=10^9) at K=100 over 120 steps DIVERGE only if pulls are nonzero — with LD=1e9 every pull rounds to 0, so digests must be EQUAL (gravity rdiv placement gate); with (LN=1, LD=2) they must DIFFER.

```python
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import intbirth  # noqa: E402

C = {"V": 40, "T": 32, "D": 64, "DH": 16, "F": 128, "n_blocks": 2,
     "SHIFT": 14, "E": 4, "K": 100, "LN": 0, "LD": 1}
tables = open("r2b_tables.bin", "rb").read()

rng = np.random.RandomState(17)
order = ["emb"]
sizes = {"emb": 40 * 64}
for b in range(2):
    for k, n in (("wq", 16 * 64), ("wk", 16 * 64), ("wv", 16 * 64),
                 ("wo", 64 * 16), ("g1", 64), ("g2", 64),
                 ("wr", 4 * 64)):
        order.append(f"b{b}.{k}"); sizes[f"b{b}.{k}"] = n
    for e in range(4):
        for k, n in (("wg", 128 * 64), ("wu", 128 * 64),
                     ("wd", 64 * 128)):
            order.append(f"b{b}.e{e}.{k}")
            sizes[f"b{b}.e{e}.{k}"] = n
order.append("g_f"); sizes["g_f"] = 64
assert sum(sizes.values()) == 208192

parts = []
for name in order:
    if name.startswith("g"):  # g1/g2/g_f draw at Q
        v = np.full(sizes[name], 512, dtype=np.int64)
    else:
        v = rng.randint(-64, 65, size=sizes[name]).astype(np.int64)
    parts.append(v.tobytes())
tok = rng.randint(0, 40, size=32).astype(np.int64)
tgt = rng.randint(0, 40, size=32).astype(np.int64)
win = tok.tobytes() + tgt.tobytes()
init = b"".join(parts)

def run(c, steps=120):
    m = intbirth.MoeBirth(tables, init, c, windows_bytes=win)
    assert list(m.param_order) == order
    m.run(steps)
    return m.mark()

a = run(C)
assert a == run(C), "determinism"
c2 = dict(C); c2["LN"] = 1; c2["LD"] = 10**9
assert run(c2) == a, "all-zero pulls must not move the trajectory"
c3 = dict(C); c3["LN"] = 1; c3["LD"] = 2
assert run(c3) != a, "real gravity must move the trajectory"
print("MOEBIRTH PASS")
```

- [ ] **Step 2: Run to verify it fails** (`MoeBirth` missing).

- [ ] **Step 3: Implement**

`moe_birth` mirrors `multi_birth` (windows form from Task 2) with these deltas in `step_once`:
- per-body weight views use `moe_body_keys(E)` names; forward chain calls `blk_.moe_body_fwd`, backward `blk_.moe_body_bwd`;
- `const i64 gb = c.gboost * 4;` for dlogits and unboost;
- after `opt_.step(...); step_ += 1;` run gravity:

```cpp
  if (c.grav_k > 0 && step_ % c.grav_k == 0 && c.grav_ln != 0) {
    static const char* const KINDS[3] = {"wg", "wu", "wd"};
    for (int b = 0; b < c.n_blocks; b++)
      for (const char* kind : KINDS) {
        std::vector<Mat*> ws;
        for (int e = 0; e < c.n_experts; e++)
          ws.push_back(&w_.at("b" + std::to_string(b) + ".e" +
                              std::to_string(e) + "." + kind));
        const std::size_t n = ws[0]->size();
        for (std::size_t i = 0; i < n; i++) {
          i64 s = 0;
          for (Mat* wp : ws) s += (*wp)[i];
          const i64 mean = rdiv(s, c.n_experts);  // finalized ONCE
          for (Mat* wp : ws)
            (*wp)[i] += rdiv((mean - (*wp)[i]) * c.grav_ln, c.grav_ld);
        }
      }
  }
```

Bindings: `MoeBirth` class mirroring `MultiBirth` exactly (init with windows_bytes default, run with gil release, mark/traj_sha/milestone_sha/weights_bytes/param_order/step_count/loss/nz).

- [ ] **Step 4: Build, run test_moe_birth.py (PASS), test_moe_parity.py, test_windows.py, and the three regression gates (PASS)**

- [ ] **Step 5: Commit**

```bash
git add include/ax/nn/intbirth.hpp src/nn/intbirth.cpp bindings/intbirth.cpp tools/int_adamw/test_moe_birth.py
git commit -m "feat: MoeBirth composed loop - gravmoe anatomy, GB=4x boost, gravity events, window cycling"
```

---

### Task 5: Acceptance script (artifact-driven, verify-the-knob)

**Files:**
- Create: `tools/int_adamw/verify_gravmoe.py`

**Interfaces:**
- Consumes: `MoeBirth` from Task 4; artifact layout from the relay postscript: `scratch/detbwd_gmoe_ref/{a0,ca0,rb1,grb1}_{init.bin,windows.bin,contract.json} + pins.json`.
- Usage: `python verify_gravmoe.py <build_dir> <ref_dir> [arm ...]` — defaults to the engine-reproducible default arms in pins.json (skips GATE/SS/TAU arms with an explicit SKIP line).

- [ ] **Step 1: Write the script**

```python
"""Gravmoe acceptance: assert init sha + param_order + per-family
draw bounds (verify-the-knob: PRINT the bound per weight family at
arm start; a reproduction that cannot show its bounds is not a
reproduction), then reproduce FINAL trajectory shas from pins.json.

Refuse-if-disagree: any mismatch in param_order or init/windows sha
aborts before running a single step.

Usage: python verify_gravmoe.py <build_dir> <ref_dir> [arm ...]
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, sys.argv[1])
import intbirth  # noqa: E402

ref_dir = sys.argv[2]
pins = json.load(open(os.path.join(ref_dir, "pins.json")))
tables = open("r2b_tables.bin", "rb").read()

# engine-side arms only: no TAU / GATE / SS knobs in this leg
SKIP_KEYS = ("TAU", "GATE", "SS")
arms = sys.argv[3:] or [a for a, e in pins.items()
                        if not any(int(e.get(k, 0)) for k in SKIP_KEYS)]

ok = True
for arm in arms:
    env = pins[arm]
    if any(int(env.get(k, 0)) for k in SKIP_KEYS):
        print(f"{arm}: SKIP (house-side knob)")
        continue
    cell = env["cell"]  # which {init,windows,contract} triple
    contract = json.load(
        open(os.path.join(ref_dir, f"{cell}_contract.json")))
    init = open(os.path.join(ref_dir, f"{cell}_init.bin"), "rb").read()
    win = open(os.path.join(ref_dir, f"{cell}_windows.bin"),
               "rb").read()
    assert hashlib.sha256(init).hexdigest() == contract["init_sha"], \
        f"{arm}: init sha DISAGREE - refusing"
    assert hashlib.sha256(win).hexdigest() == contract["windows_sha"], \
        f"{arm}: windows sha DISAGREE - refusing"
    print(f"{arm}: draw bounds per family:")
    for fam, bound in contract["draw_bounds"].items():
        print(f"  {fam}: +-{bound}")
    c = dict(contract["contract"])
    for k in ("E", "K", "LN", "LD", "SHIFT", "STEPS"):
        if k in env:
            c[k] = int(env[k])
    steps = int(env.get("STEPS", c.pop("STEPS", 2000)))
    c.pop("STEPS", None)
    m = intbirth.MoeBirth(tables, init, c, windows_bytes=win)
    assert list(m.param_order) == contract["param_order"], \
        f"{arm}: param_order DISAGREE - refusing"
    m.run(steps)
    sha = m.mark()
    good = sha == env["final_sha"]
    ok &= good
    print(f"{arm}: steps {steps} loss {m.loss} nz {m.nz:.3f} "
          f"{'PASS' if good else 'FAIL want ' + env['final_sha'][:16]}")
print("GRAVMOE PASS" if ok else "GRAVMOE FAIL")
sys.exit(0 if ok else 1)
```

NOTE: the exact pins.json / contract.json field names above are a best guess at the house schema; when the artifacts land, adapt the FIELD ACCESS ONLY (never the engine) to the shipped schema. If the shipped contract disagrees with the engine's param_order or shapes, that is a refuse-if-disagree relay item, not something to paper over in the script.

- [ ] **Step 2: Dry-run the refusal path**

Without artifacts present, run: `python verify_gravmoe.py ../../build-rel/bindings /nonexistent` — expect a clean FileNotFoundError (no partial runs). If a local copy of the artifacts exists (Artin's transfer), run for real and iterate on rounding-placement flips (the two flagged in Task 3) ONLY if a pin fails, one flip per run, recording each attempt.

- [ ] **Step 3: Commit**

```bash
git add tools/int_adamw/verify_gravmoe.py
git commit -m "feat: gravmoe acceptance script - verify-the-knob bounds printout, refuse-if-disagree, pins reproduction"
```

---

### Task 6: Reproduce the pins (BLOCKED on artifact transfer)

**Files:**
- None new; runs Task 5's script.

- [ ] **Step 1:** When Artin lands `detbwd_gmoe_ref/` locally, run `python verify_gravmoe.py ../../build-rel/bindings <ref_dir>`.
- [ ] **Step 2:** Target: RB1 (`c6766da2...`) and RB1-S16 (`14981553...`) PASS, plus the saturated-contract (COND=0 QK=0) arms in pins.json. RB3/G-RB1/S1 print SKIP.
- [ ] **Step 3:** On PASS, write the verdict relay note in `docs/relay/` (following the format of `2026-08-01-4-multiblock-verdict.md`) and commit.
