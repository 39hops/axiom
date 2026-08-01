"""FX-V3 table build: integer twin of the MERGED Hebbian-MoE crystal.

Reuses the promoted P3 recipe (llmopt/decoding/deterministic.py
make_tables: sigma-law codes, sigma/8 on emb/head interface tensors,
shipped rope/exp/silu tables) on the merged state dict — the router
[E, D] is a 2D tensor and quantizes under the same sigma/2 law —
and adds ONE new table, the router-softmax exp table in the house
detbwd_r1b construction (Q=512, domain [-8, 0] in Q units, values
round(exp(x)*Q)): at equal Q the table sha must match house's
exactly, a free cross-lab check before any forward runs.

Run from the llmopt repo root:
  python make_tables.py <umoe_ckpt.pt> <out_tables.pt>
"""
import hashlib
import sys

sys.path.insert(0, ".")
import torch  # noqa: E402

from llmopt.decoding.deterministic import make_tables  # noqa: E402

Q = 512
TSE = 8 * Q
D, HEADS, LAYERS, NE = 64, 8, 8, 4


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
        out[f"blocks.{li}.router.weight"] = sd[f"{p}.router.weight"]
    return out


def main(ckpt, out):
    sd = merged_sd(ckpt)
    t = make_tables(sd, D, HEADS, out)
    xs = torch.arange(-TSE, 1, dtype=torch.float64) / Q
    t["rexp.tab"] = torch.round(torch.exp(xs) * Q).to(torch.int64)
    h = hashlib.sha256(t["rexp.tab"].numpy().tobytes()).hexdigest()
    print(f"rexp table sha {h[:16]} (house detbwd_r1b Q={Q})")
    torch.save(t, out)
    fsha = hashlib.sha256(open(out, "rb").read()).hexdigest()
    print(f"{out}: {len(t)} tensors, file sha256 {fsha}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
