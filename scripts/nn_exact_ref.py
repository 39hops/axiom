"""FX-V1 exact-mode reference + acceptance gate (rung 2b; spec:
docs/specs/2026-07-27-axnn-exact.md).

Three roles in one file:
  1. TABLE GENERATION (the once-at-export artifacts): fx.act / fx.exp
     / fx.rsqrt / fx.rope.* — fp64 + round-half-even, shipped as f32
     tensors carrying exact integers.
  2. INDEPENDENT INTEGER REFERENCE: a numpy-int64 implementation of
     the FX-V1 forward, written from the spec (floor shifts, floor
     division, declared sats/tables). Shares no code with the C++.
  3. ACCEPTANCE: builds a random-weight crystal-shape container, runs
     axiom-nn-exact and the reference on 100 prompts, and requires
     BIT-IDENTICAL per-prompt and battery hashes. Two independent
     implementations agreeing bit-exact is the local stand-in for the
     Mac / 3080 arms, which run the same container against this spec.

Run: python scripts/nn_exact_ref.py [tool] [workdir]
"""
import json
import math
import os
import random
import struct
import subprocess
import sys

import numpy as np

TOOL = sys.argv[1] if len(sys.argv) > 1 else "build-rel/axiom-nn-exact.exe"
WORK = sys.argv[2] if len(sys.argv) > 2 else "data/qual"
N_PROMPTS = 100

FRAC = 16
ACT_SAT = 2048 << 16
CENT_SAT = 512 << 16
WEIGHT_SAT = 128 << 16
EPS_Q32 = 42950


# ------------------------------------------------ declared primitives
def rne(x):
    return int(round(x))  # python round = round-half-even


def f32_to_q16(f):
    """Bit-exact fp32 -> Q.16 RNE (mirrors the C++ integer algorithm)."""
    (b,) = struct.unpack("<I", struct.pack("<f", f))
    neg = b >> 31
    exp = (b >> 23) & 0xFF
    frac = b & 0x7FFFFF
    if exp == 0xFF:
        raise ValueError("nan/inf")
    mant = frac if exp == 0 else (frac | (1 << 23))
    if mant == 0:
        return 0
    shift = exp - 150 + 16
    if shift >= 0:
        mag = mant << shift
    else:
        r = -shift
        if r > 63:
            return 0
        keep = mant >> r
        rem = mant & ((1 << r) - 1)
        half = 1 << (r - 1)
        mag = keep + (1 if (rem > half or (rem == half and keep & 1)) else 0)
    return -mag if neg else mag


def to_fx_weights(a32, sat=WEIGHT_SAT):
    flat = [max(-sat, min(sat, f32_to_q16(float(v)))) for v in a32.flat]
    return np.array(flat, dtype=np.int64).reshape(a32.shape)


def satv(a, bound):
    return np.clip(a, -bound, bound)


def floor_div(a, b):
    return a // b  # python/numpy floor semantics == spec rule 3


def lerp(table, u, fbits):
    # C++ rule: idx >= len-1 returns table[-1] (exp hits this at d2==0
    # for every softmax max element — the boundary must match exactly)
    idx = u >> fbits
    frac = u & ((1 << fbits) - 1)
    over = idx >= len(table) - 1
    idx2 = np.minimum(idx, len(table) - 2)
    t0, t1 = table[idx2], table[idx2 + 1]
    val = t0 + (((t1 - t0) * frac) >> fbits)
    return np.where(over, table[-1], val)


def rsqrt_fx(v, table):
    v = int(v)
    k = v.bit_length() - 1
    s = k - 31
    if s & 1:
        s += 1
    m = v >> s if s >= 0 else v << -s
    u = m - (1 << 30)
    idx = min(u >> 23, len(table) - 2)
    frac = u & ((1 << 23) - 1)
    r = table[idx] + (((table[idx + 1] - table[idx]) * frac) >> 23)
    sh = s // 2 - 1
    return int(r) >> sh if sh >= 0 else int(r) << -sh


# ------------------------------------------------------------ tables
def gen_tables(cfg):
    t = {}
    act = cfg["act"]
    if act != "relu":
        def gelu(x):
            return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))

        def gelu_tanh(x):
            c = math.sqrt(2 / math.pi)
            return 0.5 * x * (1 + math.tanh(c * (x + 0.044715 * x ** 3)))

        def silu(x):
            return x / (1 + math.exp(-x))

        f = {"gelu": gelu, "gelu_tanh": gelu_tanh, "silu": silu}[act]
        t["fx.act.table"] = [rne(f(-32 + i / 32) * 65536)
                             for i in range(2049)]
    t["fx.exp.table"] = [rne(math.exp(-16 + i / 128) * 65536)
                         for i in range(2049)]
    t["fx.rsqrt.table"] = [rne(65536 / math.sqrt(1 + i / 128))
                           for i in range(385)]
    if cfg["pos"] == "rope":
        dh = cfg["d_model"] // cfg["n_heads"]
        half = dh // 2
        cos, sin = [], []
        for pos in range(cfg["max_seq"]):
            for p in range(half):
                ang = pos * cfg["rope_theta"] ** (-2.0 * p / dh)
                cos.append(rne(math.cos(ang) * 65536))
                sin.append(rne(math.sin(ang) * 65536))
        t["fx.rope.cos"] = cos
        t["fx.rope.sin"] = sin
    return t


# ------------------------------------------------- integer reference
class FxRef:
    def __init__(self, cfg, weights_f32, tables):
        self.cfg = cfg
        self.w = {k: to_fx_weights(v) for k, v in weights_f32.items()}
        self.tab = {k: np.array(v, dtype=np.int64) for k, v in tables.items()}

    def _linear(self, x, wname, bname):
        w = self.w[wname]
        acc = x @ w.T  # int64 exact (headroom-proved)
        if bname in self.w:
            acc = acc + (self.w[bname] << 16)
        return satv(acc >> 16, ACT_SAT)

    def _norm(self, x, prefix):
        D = self.cfg["d_model"]
        g = self.w[prefix + ".weight"]
        b = self.w.get(prefix + ".bias")
        out = np.zeros_like(x)
        for t in range(x.shape[0]):
            row = x[t]
            mean = floor_div(int(row.sum()), D) \
                if self.cfg["norm"] == "layernorm" else 0
            c = satv(row - mean, CENT_SAT)
            var = floor_div(int((c * c).sum()), D)
            inv = rsqrt_fx(var + EPS_Q32, self.tab["fx.rsqrt.table"])
            n1 = (c * inv) >> 16
            n2 = (n1 * g) >> 16
            out[t] = satv(n2 + (b if b is not None else 0), ACT_SAT)
        return out

    def _rope(self, q, T, H, dh):
        half = dh // 2
        tc = self.tab["fx.rope.cos"].reshape(-1, half)
        ts = self.tab["fx.rope.sin"].reshape(-1, half)
        q = q.reshape(T, H, dh).copy()
        if self.cfg["rope_style"] == "half":
            i0 = np.arange(half)
            i1 = i0 + half
        else:
            i0 = 2 * np.arange(half)
            i1 = i0 + 1
        for t in range(T):
            c, s = tc[t], ts[t]
            a, b = q[t, :, i0].T.copy(), q[t, :, i1].T.copy()
            q[t, :, i0] = satv((a * c - b * s) >> 16, ACT_SAT).T
            q[t, :, i1] = satv((a * s + b * c) >> 16, ACT_SAT).T
        return q.reshape(T, H * dh)

    def _act(self, v):
        if self.cfg["act"] == "relu":
            return np.maximum(v, 0)
        at = self.tab["fx.act.table"]
        lo = -(32 << 16)
        hi = (32 << 16) - 1
        out = v.copy()
        mid = (v >= lo) & (v <= hi)
        out[v < lo] = 0
        out[mid] = lerp(at, v[mid] - lo, 11)
        return out  # v > hi: identity tail

    def logits_q32(self, toks):
        cfg = self.cfg
        D, H = cfg["d_model"], cfg["n_heads"]
        dh = D // H
        T = len(toks)
        x = self.w["tok_emb.weight"][toks].copy()
        if cfg["pos"] == "learned":
            x = satv(x + self.w["pos_emb.weight"][:T], ACT_SAT)
        else:
            x = satv(x, ACT_SAT)
        et = self.tab["fx.exp.table"]
        scale = rsqrt_fx(dh << 32, self.tab["fx.rsqrt.table"])
        for i in range(cfg["n_layers"]):
            L = f"layers.{i}."
            h = self._norm(x, L + "ln1")
            q = self._linear(h, L + "attn.q.weight", L + "attn.q.bias")
            k = self._linear(h, L + "attn.k.weight", L + "attn.k.bias")
            v = self._linear(h, L + "attn.v.weight", L + "attn.v.bias")
            if cfg["pos"] == "rope":
                q = self._rope(q, T, H, dh)
                k = self._rope(k, T, H, dh)
            qh = q.reshape(T, H, dh)
            kh = k.reshape(T, H, dh)
            vh = v.reshape(T, H, dh)
            attn = np.zeros((T, H, dh), dtype=np.int64)
            for hh in range(H):
                for t in range(T):
                    dot = (qh[t, hh] * kh[: t + 1, hh]).sum(axis=1)
                    score = ((dot >> 16) * scale) >> 16
                    mx = int(score.max())
                    d2 = np.maximum(score - mx, -(16 << 16))
                    ew = lerp(et, d2 + (16 << 16), 9)
                    Z = int(ew.sum()) or 1
                    w2 = floor_div(ew << 16, Z)
                    attn[t, hh] = (w2[:, None] * vh[: t + 1, hh]).sum(axis=0)
            attn = satv(attn.reshape(T, D) >> 16, ACT_SAT)
            proj = self._linear(attn, L + "attn.o.weight", L + "attn.o.bias")
            x = satv(x + proj, ACT_SAT)
            h = self._norm(x, L + "ln2")
            f1 = self._act(self._linear(h, L + "ffn.fc1.weight",
                                        L + "ffn.fc1.bias"))
            f2 = self._linear(f1, L + "ffn.fc2.weight", L + "ffn.fc2.bias")
            x = satv(x + f2, ACT_SAT)
        xf = self._norm(x, "ln_f")
        head = self.w.get("head.weight", self.w["tok_emb.weight"])
        return xf[-1] @ head.T  # Q.32, never rescaled

    def hash(self, toks):
        h = 14695981039346656037
        for v in self.logits_q32(toks):
            u = int(v) & 0xFFFFFFFFFFFFFFFF
            for i in range(8):
                h ^= (u >> (8 * i)) & 0xFF
                h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h


# ---------------------------------------------------------- container
def write_axnn(path, cfg, weights_f32, tables):
    with open(path, "wb") as f:
        f.write(b"AXNN")
        f.write(struct.pack("<I", 1))
        cj = json.dumps(cfg).encode()
        f.write(struct.pack("<I", len(cj)))
        f.write(cj)

        def tensor(name, arr):
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", len(arr.shape)))
            for d in arr.shape:
                f.write(struct.pack("<Q", d))
            f.write(arr.astype("<f4").tobytes())

        for name, a in weights_f32.items():
            tensor(name, a)
        for name, vals in tables.items():
            tensor(name, np.array(vals, dtype=np.float64))


def random_weights(cfg, seed):
    rng = np.random.default_rng(seed)
    D, F, V = cfg["d_model"], cfg["d_ff"], cfg["vocab"]
    w = {"tok_emb.weight": rng.normal(0, 0.05, (V, D)).astype(np.float32)}
    if cfg["pos"] == "learned":
        w["pos_emb.weight"] = rng.normal(0, 0.05, (cfg["max_seq"], D)) \
            .astype(np.float32)
    for i in range(cfg["n_layers"]):
        L = f"layers.{i}."
        for nm in ("q", "k", "v", "o"):
            w[L + f"attn.{nm}.weight"] = rng.normal(0, 0.05, (D, D)) \
                .astype(np.float32)
            w[L + f"attn.{nm}.bias"] = rng.normal(0, 0.05, D) \
                .astype(np.float32)
        w[L + "ffn.fc1.weight"] = rng.normal(0, 0.05, (F, D)) \
            .astype(np.float32)
        w[L + "ffn.fc1.bias"] = rng.normal(0, 0.05, F).astype(np.float32)
        w[L + "ffn.fc2.weight"] = rng.normal(0, 0.05, (D, F)) \
            .astype(np.float32)
        w[L + "ffn.fc2.bias"] = rng.normal(0, 0.05, D).astype(np.float32)
        for ln in ("ln1", "ln2"):
            w[L + ln + ".weight"] = rng.normal(1, 0.02, D) \
                .astype(np.float32)
            if cfg["norm"] == "layernorm":
                w[L + ln + ".bias"] = rng.normal(0, 0.02, D) \
                    .astype(np.float32)
    w["ln_f.weight"] = rng.normal(1, 0.02, D).astype(np.float32)
    if cfg["norm"] == "layernorm":
        w["ln_f.bias"] = rng.normal(0, 0.02, D).astype(np.float32)
    if not cfg.get("tied_head", True):
        w["head.weight"] = rng.normal(0, 0.05, (V, D)).astype(np.float32)
    return w


def run_variant(tag, cfg, seed):
    weights = random_weights(cfg, seed)
    tables = gen_tables(cfg)
    model_path = os.path.join(WORK, f"nn_exact_{tag}.axnn")
    prompts_path = os.path.join(WORK, f"nn_exact_{tag}_prompts.txt")
    write_axnn(model_path, cfg, weights, tables)
    rng = random.Random(seed + 1)
    prompts = [[rng.randrange(cfg["vocab"])
                for _ in range(rng.randint(4, 48))]
               for _ in range(N_PROMPTS)]
    with open(prompts_path, "w", newline="\n") as f:
        for p in prompts:
            f.write(" ".join(map(str, p)) + "\n")
    out = subprocess.run([TOOL, model_path, prompts_path],
                         capture_output=True, text=True, check=True)
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    got = [ln.split()[0] for ln in lines[:-1]]
    battery_cpp = lines[-1].split()[1]
    ref = FxRef(cfg, weights, tables)
    mismatches = 0
    battery = 14695981039346656037
    for p, g in zip(prompts, got):
        h = ref.hash(p)
        for i in range(8):
            battery ^= (h >> (8 * i)) & 0xFF
            battery = (battery * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        if f"{h:016x}" != g:
            mismatches += 1
    ok = mismatches == 0 and f"{battery:016x}" == battery_cpp
    print(f"{'PASS' if ok else 'FAIL'} {tag}: {mismatches}/{N_PROMPTS} "
          f"hash mismatches, battery cpp={battery_cpp} "
          f"ref={battery:016x}")
    return ok


def main():
    crystal = {"d_model": 256, "n_layers": 8, "n_heads": 4, "d_ff": 1024,
               "vocab": 47, "max_seq": 512, "eps": 1e-5,
               "rope_theta": 10000.0, "rope_style": "half"}
    a = dict(crystal, norm="layernorm", act="gelu", pos="learned",
             tied_head=True)
    b = dict(crystal, norm="rmsnorm", act="silu", pos="rope",
             tied_head=False)
    ok = run_variant("a_ln_gelu_learned", a, 20260727)
    ok &= run_variant("b_rms_silu_rope", b, 31415926)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
