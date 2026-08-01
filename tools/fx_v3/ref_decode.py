"""FX-V3 Python integer reference: the P3 DeterministicLM with the
switch_top1 router gate inserted (spec in the FX-V3 relay):

  rl  = gemm(h2, router)                  # a-scale, sigma-law codes
  s   = rdiv(rl * Q, A)                   # Q units
  d   = clamp(s - max(s), min=-TSE-1); e = d < -TSE ? 0 : rexp[d+TSE]
  p   = rdiv(e_win * Q, sum(e))           # e_win = rexp[-1] = Q
  x   = clamp(x + rdiv(down_out * p, Q), +-ACT_CLAMP)

Everything else is the promoted twin verbatim. Two-runtime
self-check for the C++ implementation; house's reference decides.

Run from the llmopt repo root:
  python ref_decode.py <tables.pt> <prompts.txt> <n_new>
Prints one stream per line + the streams sha256 (Python repr
convention, as FX-V2).
"""
import hashlib
import sys

sys.path.insert(0, ".")
import torch  # noqa: E402

from llmopt.decoding.deterministic import (  # noqa: E402
    A, ACT_CLAMP, DeterministicLM, _rdiv)

Q = 512
TSE = 8 * Q
D, LAYERS, HEADS = 64, 8, 8


class GatedLM(DeterministicLM):
    def step(self, tok_id, past, pos):
        q_e = int(self.t["emb.weight.q"])
        x = _rdiv(self.t["emb.weight.codes"][tok_id] * A,
                  q_e).view(1, 1, self.d)
        new_past = []
        for li in range(self.layers):
            p = f"blocks.{li}"
            h = self._rmsnorm(x, f"{p}.n1")
            qkv = self._gemm(h, f"{p}.qkv")
            q, k, v = qkv.split(self.d, dim=-1)
            q = q.view(1, 1, self.heads, self.hd).transpose(1, 2)
            k = k.view(1, 1, self.heads, self.hd).transpose(1, 2)
            v = v.view(1, 1, self.heads, self.hd).transpose(1, 2)
            q, k = self._rope(q, pos), self._rope(k, pos)
            if past is not None:
                k = torch.cat([past[li][0], k], 2)
                v = torch.cat([past[li][1], v], 2)
            new_past.append((k, v))
            a = self._attn(q, k, v).transpose(1, 2).reshape(1, 1, self.d)
            x = torch.clamp(x + self._gemm(a, f"{p}.o"),
                            -ACT_CLAMP, ACT_CLAMP)
            h = self._rmsnorm(x, f"{p}.n2")
            g = self._gemm(h, f"{p}.gate")
            u = self._gemm(h, f"{p}.up")
            gi = torch.clamp(g, -(1 << 15), (1 << 15))
            ff = _rdiv(self.t["silu.tab"][gi + (1 << 15)] * u, A)
            ff = torch.clamp(ff, -(1 << 15), (1 << 15))
            dn = self._gemm(ff, f"{p}.down")
            # switch_top1 gate: scale by the winning router probability
            rl = self._gemm(h, f"{p}.router")            # a-scale
            s = _rdiv(rl * Q, A)                          # Q units
            d = torch.clamp(s - s.max(), min=-TSE - 1)
            e = torch.where(d < -TSE, torch.zeros_like(d),
                            self.t["rexp.tab"][d + TSE])
            top_p = _rdiv(self.t["rexp.tab"][-1] * Q, e.sum())
            x = torch.clamp(x + _rdiv(dn * top_p, Q),
                            -ACT_CLAMP, ACT_CLAMP)
        x = self._rmsnorm(x, "norm")
        return self._gemm(x, "head").squeeze(), new_past


def main(tables, prompts_f, n_new):
    m = GatedLM(tables, D, LAYERS, 0, HEADS, "cpu")
    with open(prompts_f) as f:
        rows = [[int(t) for t in ln.split()] for ln in f if ln.strip()]
    streams = [m.greedy(ids, int(n_new)) for ids in rows]
    for s in streams:
        print(" ".join(map(str, s)))
    print("ref streams sha:",
          hashlib.sha256(repr(streams).encode()).hexdigest(),
          file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:4])
