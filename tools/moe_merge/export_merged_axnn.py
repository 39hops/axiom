"""Merged-crystal export + torch reference (relay 2026-07-31 ask).

Takes a scaffold-MoE checkpoint (umoe_gravmoe_*), averages the four
experts into ONE dense SwiGLU per block (merge_experts semantics),
keeps the router, and:
  1. writes an AXNN v1.2 container (axnn_minor 2, ffn_gate
     "switch_top1", n_experts 4; house tensor dialect + the router as
     blocks.{i}.moe.router.weight);
  2. runs the torch merged forward (fp32, full-prefix, explicit
     causal attention) greedily on a prompt-id file and writes the
     reference streams — the token-identical acceptance target for
     axiom-nn-moe-greedy.

Run from the llmopt repo root (torch + the checkpoint live there):
  python export_merged_axnn.py <ckpt.pt> <out.axnn> <prompts.txt> \
      <ref_streams.txt> <n_new>
"""
import hashlib
import json
import math
import struct
import sys

import torch
import torch.nn.functional as F

D, LAYERS, HEADS, NE = 64, 8, 8, 4


def merged_sd(ckpt_path):
    d = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = d["sd"]
    out = {k: v for k, v in sd.items() if ".moe." not in k}
    for li in range(LAYERS):
        p = f"blocks.{li}.moe"
        for src, dst in (("g", "gate"), ("u", "up"), ("d", "down")):
            out[f"blocks.{li}.{dst}.weight"] = torch.stack(
                [sd[f"{p}.exp.{e}.{src}.weight"] for e in range(NE)]
            ).mean(0)
        out[f"blocks.{li}.moe.router.weight"] = sd[f"{p}.router.weight"]
    return out


def write_axnn(sd, path):
    V = sd["emb.weight"].shape[0]
    Fd = sd["blocks.0.gate.weight"].shape[0]
    cfg = {"d_model": D, "n_layers": LAYERS, "n_heads": HEADS,
           "d_ff": Fd, "vocab": V, "max_seq": 512, "norm": "rmsnorm",
           "act": "silu", "pos": "rope", "rope_style": "half",
           "eps": 1e-6, "rope_theta": 10000.0, "ffn": "swiglu",
           "attn_fused": "qkv", "head": "separate", "axnn_minor": 2,
           "ffn_gate": "switch_top1", "n_experts": NE}
    cfg_b = json.dumps(cfg).encode()
    with open(path, "wb") as f:
        f.write(b"AXNN" + struct.pack("<I", 1))
        f.write(struct.pack("<I", len(cfg_b)) + cfg_b)
        for name in sorted(sd):
            t = sd[name].float().contiguous()
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)) + nb)
            f.write(struct.pack("<I", t.dim()))
            for d_ in t.shape:
                f.write(struct.pack("<Q", d_))
            f.write(t.numpy().tobytes())
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(f"{path}: {len(sd)} tensors sha256 {sha}")


def rmsnorm(x, g):
    return g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def rope(v, pos0=0):
    T, H, hd = v.shape[1], v.shape[0], v.shape[2]
    half = hd // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
    ang = (torch.arange(pos0, pos0 + T)[:, None] * freq[None, :])
    cos, sin = ang.cos(), ang.sin()
    v1, v2 = v[..., :half], v[..., half:]
    return torch.cat([v1 * cos - v2 * sin, v1 * sin + v2 * cos], -1)


@torch.no_grad()
def logits_last(sd, ids):
    x = sd["emb.weight"][ids]                            # [T, D]
    T = x.shape[0]
    hd = D // HEADS
    mask = torch.full((T, T), float("-inf")).triu(1)
    for li in range(LAYERS):
        p = f"blocks.{li}"
        h = rmsnorm(x, sd[f"{p}.n1.g"])
        qkv = h @ sd[f"{p}.qkv.weight"].T
        q, k, v = qkv.chunk(3, -1)
        q = rope(q.view(T, HEADS, hd).transpose(0, 1))
        k = rope(k.view(T, HEADS, hd).transpose(0, 1))
        v = v.view(T, HEADS, hd).transpose(0, 1)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(hd) + mask
        a = (F.softmax(s, -1) @ v).transpose(0, 1).reshape(T, D)
        x = x + a @ sd[f"{p}.o.weight"].T
        h = rmsnorm(x, sd[f"{p}.n2.g"])
        g = h @ sd[f"{p}.gate.weight"].T
        u = h @ sd[f"{p}.up.weight"].T
        ff = (F.silu(g) * u) @ sd[f"{p}.down.weight"].T
        top_p = F.softmax(h @ sd[f"{p}.moe.router.weight"].T,
                          -1).max(-1).values
        x = x + top_p.unsqueeze(-1) * ff
    x = rmsnorm(x, sd["norm.g"])
    return x[-1] @ sd["head.weight"].T


def main():
    ckpt, out_axnn, prompts_f, ref_f, n_new = sys.argv[1:6]
    sd = merged_sd(ckpt)
    write_axnn(sd, out_axnn)
    with open(prompts_f) as f:
        rows = [[int(t) for t in ln.split()] for ln in f if ln.strip()]
    with open(ref_f, "w") as f:
        for ids in rows:
            ids = list(ids)
            gen = []
            for _ in range(int(n_new)):
                nxt = int(logits_last(sd, torch.tensor(ids)).argmax())
                gen.append(nxt)
                ids.append(nxt)
            f.write(" ".join(map(str, gen)) + "\n")
    print(f"reference streams -> {ref_f} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
