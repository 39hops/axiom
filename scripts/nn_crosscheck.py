"""AXNN acceptance gate: logits vs torch fp32 within 1e-4 on 100
prompts (relay 2026-07-27-2 ask 2).

Builds a torch reference model matching the AXNN spec exactly, fills
it with seeded random weights at the crystal shape (d256/8L/ffn1024/
h4, vocab 47), exports the AXNN container, runs axiom-nn-logits on
100 random prompts, and compares last-position logits. Two config
variants are gated so the acceptance covers the convention switches,
not one lucky default:
  A: layernorm + gelu(exact) + learned positions + tied head
  B: rmsnorm + silu + rope(half) + separate head

Run: python scripts/nn_crosscheck.py [tool-path] [workdir]
Exit 0 iff both variants pass.
"""
import json
import math
import os
import struct
import subprocess
import sys

import torch

TOOL = sys.argv[1] if len(sys.argv) > 1 else "build-rel/axiom-nn-logits.exe"
WORK = sys.argv[2] if len(sys.argv) > 2 else \
    "data/qual"
N_PROMPTS = 100
TOL = 1e-4


# ------------------------------------------------------------- export
def write_axnn(path, cfg, tensors):
    with open(path, "wb") as f:
        f.write(b"AXNN")
        f.write(struct.pack("<I", 1))
        cj = json.dumps(cfg).encode()
        f.write(struct.pack("<I", len(cj)))
        f.write(cj)
        for name, t in tensors.items():
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", len(t.shape)))
            for d in t.shape:
                f.write(struct.pack("<Q", d))
            f.write(t.detach().to(torch.float32).contiguous()
                    .numpy().tobytes())


# ---------------------------------------------------- torch reference
class Ref(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        D, F, V = cfg["d_model"], cfg["d_ff"], cfg["vocab"]
        self.tok = torch.nn.Embedding(V, D)
        if cfg["pos"] == "learned":
            self.pos = torch.nn.Embedding(cfg["max_seq"], D)
        self.layers = torch.nn.ModuleList()
        for _ in range(cfg["n_layers"]):
            blk = torch.nn.ModuleDict({
                "q": torch.nn.Linear(D, D), "k": torch.nn.Linear(D, D),
                "v": torch.nn.Linear(D, D), "o": torch.nn.Linear(D, D),
            })
            if cfg.get("ffn", "fc") == "swiglu":
                blk["gate"] = torch.nn.Linear(D, F)
                blk["up"] = torch.nn.Linear(D, F)
                blk["down"] = torch.nn.Linear(F, D)
            else:
                blk["fc1"] = torch.nn.Linear(D, F)
                blk["fc2"] = torch.nn.Linear(F, D)
            blk["ln1"] = self._norm(D)
            blk["ln2"] = self._norm(D)
            self.layers.append(blk)
        self.ln_f = self._norm(D)
        if not cfg["tied_head"]:
            self.head = torch.nn.Linear(D, V, bias=False)

    def _norm(self, D):
        if self.cfg["norm"] == "layernorm":
            return torch.nn.LayerNorm(D, eps=self.cfg["eps"])
        return torch.nn.RMSNorm(D, eps=self.cfg["eps"])

    def _act(self, x):
        a = self.cfg["act"]
        if a == "gelu":
            return torch.nn.functional.gelu(x, approximate="none")
        if a == "gelu_tanh":
            return torch.nn.functional.gelu(x, approximate="tanh")
        if a == "silu":
            return torch.nn.functional.silu(x)
        return torch.nn.functional.relu(x)

    def _rope(self, x):  # x: [T,H,dh], half style
        T, H, dh = x.shape
        p = torch.arange(dh // 2, dtype=torch.float32)
        freq = self.cfg["rope_theta"] ** (-2.0 * p / dh)
        t = torch.arange(T, dtype=torch.float32)[:, None] * freq[None, :]
        c, s = torch.cos(t)[:, None, :], torch.sin(t)[:, None, :]
        a, b = x[..., :dh // 2], x[..., dh // 2:]
        return torch.cat([a * c - b * s, a * s + b * c], dim=-1)

    def forward(self, toks):
        cfg = self.cfg
        D, H = cfg["d_model"], cfg["n_heads"]
        dh = D // H
        T = toks.shape[0]
        x = self.tok(toks)
        if cfg["pos"] == "learned":
            x = x + self.pos(torch.arange(T))
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        for blk in self.layers:
            h = blk["ln1"](x)
            q = blk["q"](h).view(T, H, dh)
            k = blk["k"](h).view(T, H, dh)
            v = blk["v"](h).view(T, H, dh)
            if cfg["pos"] == "rope":
                q, k = self._rope(q), self._rope(k)
            att = torch.einsum("thd,uhd->htu", q, k) / math.sqrt(dh)
            att = torch.softmax(att + mask, dim=-1)
            out = torch.einsum("htu,uhd->thd", att, v).reshape(T, D)
            x = x + blk["o"](out)
            h = blk["ln2"](x)
            if cfg.get("ffn", "fc") == "swiglu":
                x = x + blk["down"](self._act(blk["gate"](h)) *
                                    blk["up"](h))
            else:
                x = x + blk["fc2"](self._act(blk["fc1"](h)))
        x = self.ln_f(x)
        w = self.tok.weight if cfg["tied_head"] else self.head.weight
        return x @ w.T


def export_tensors(ref, cfg):
    t = {"tok_emb.weight": ref.tok.weight}
    if cfg["pos"] == "learned":
        t["pos_emb.weight"] = ref.pos.weight
    for i, blk in enumerate(ref.layers):
        L = f"layers.{i}."
        if cfg.get("attn_fused", 0):
            # v1.1 fused layout: rows stacked q|k|v
            t[L + "attn.qkv.weight"] = torch.cat(
                [blk[n].weight for n in ("q", "k", "v")], dim=0)
            t[L + "attn.qkv.bias"] = torch.cat(
                [blk[n].bias for n in ("q", "k", "v")], dim=0)
            t[L + "attn.o.weight"] = blk["o"].weight
            t[L + "attn.o.bias"] = blk["o"].bias
        else:
            for nm in ("q", "k", "v", "o"):
                t[L + f"attn.{nm}.weight"] = blk[nm].weight
                t[L + f"attn.{nm}.bias"] = blk[nm].bias
        if cfg.get("ffn", "fc") == "swiglu":
            for nm in ("gate", "up", "down"):
                t[L + f"ffn.{nm}.weight"] = blk[nm].weight
                t[L + f"ffn.{nm}.bias"] = blk[nm].bias
        else:
            t[L + "ffn.fc1.weight"] = blk["fc1"].weight
            t[L + "ffn.fc1.bias"] = blk["fc1"].bias
            t[L + "ffn.fc2.weight"] = blk["fc2"].weight
            t[L + "ffn.fc2.bias"] = blk["fc2"].bias
        for ln, tag in (("ln1", L + "ln1"), ("ln2", L + "ln2")):
            t[tag + ".weight"] = blk[ln].weight
            if cfg["norm"] == "layernorm":
                t[tag + ".bias"] = blk[ln].bias
    t["ln_f.weight"] = ref.ln_f.weight
    if cfg["norm"] == "layernorm":
        t["ln_f.bias"] = ref.ln_f.bias
    if not cfg["tied_head"]:
        t["head.weight"] = ref.head.weight
    return t


def run_variant(tag, cfg, seed):
    torch.manual_seed(seed)
    ref = Ref(cfg)
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0.0, 0.05)
    model_path = os.path.join(WORK, f"nn_cross_{tag}.axnn")
    prompts_path = os.path.join(WORK, f"nn_cross_{tag}_prompts.txt")
    write_axnn(model_path, cfg, export_tensors(ref, cfg))
    gen = torch.Generator().manual_seed(seed + 1)
    prompts = []
    for _ in range(N_PROMPTS):
        n = int(torch.randint(4, 48, (1,), generator=gen))
        prompts.append(torch.randint(0, cfg["vocab"], (n,), generator=gen))
    with open(prompts_path, "w", newline="\n") as f:
        for p in prompts:
            f.write(" ".join(str(int(x)) for x in p) + "\n")
    out = subprocess.run([TOOL, model_path, prompts_path],
                         capture_output=True, text=True, check=True)
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    assert len(lines) == N_PROMPTS, f"{tag}: line count {len(lines)}"
    worst = 0.0
    with torch.no_grad():
        for p, ln in zip(prompts, lines):
            want = ref(p)[-1].float()
            got = torch.tensor([float(v) for v in ln.split()])
            worst = max(worst, float((want - got).abs().max()))
    ok = worst <= TOL
    print(f"{'PASS' if ok else 'FAIL'} {tag}: max|Δlogit| = {worst:.3e} "
          f"(tol {TOL}) over {N_PROMPTS} prompts")
    return ok


def main():
    crystal = {"d_model": 256, "n_layers": 8, "n_heads": 4, "d_ff": 1024,
               "vocab": 47, "max_seq": 512, "eps": 1e-5,
               "rope_theta": 10000.0, "rope_style": "half"}
    a = dict(crystal, norm="layernorm", act="gelu", pos="learned",
             tied_head=True)
    b = dict(crystal, norm="rmsnorm", act="silu", pos="rope",
             tied_head=False)
    # c = the S2 winner's exact conventions (v1.1, relay 2026-07-28-4):
    # rmsnorm eps 1e-6 / silu / rope-half / tied head / fused qkv /
    # swiglu ffn — d_ff at the swiglu crystal shape
    c = dict(crystal, norm="rmsnorm", act="silu", pos="rope",
             tied_head=True, eps=1e-6, ffn="swiglu", attn_fused=1,
             head="tied", axnn_minor=1)
    ok = run_variant("a_ln_gelu_learned_tied", a, 20260727)
    ok &= run_variant("b_rms_silu_rope_head", b, 31415926)
    ok &= run_variant("c_s2_swiglu_fused_tied", c, 27182818)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
