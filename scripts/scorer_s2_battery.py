"""Pin the S2 scorer 20-prompt battery + expected logits (relay -5).

Battery construction (deterministic from pinned inputs, reproducible
house-side): walk data/llmopt/scorer_battery_v1.jsonl in file order,
frame each (state, children[0].child) pair through the prompt spec's
frame, tokenize with greedy-longest-match over the spec's 40-token
vocab, and keep the first 20 frames that tokenize cleanly AND fit
max_seq. Round-trip decode is asserted per prompt.

Outputs (data/qual/):
  scorer_s2_battery20.txt      token-id lines (axiom-nn-logits input)
  scorer_s2_battery20_meta.jsonl  {idx, row, text} per prompt
  scorer_s2_expected_logits.txt   one line of 40 logits per prompt
Prints sha256 of all three + the model file (the substrate fence).

Run: python scripts/scorer_s2_battery.py [tool] [model] [spec] [rows]
"""
import hashlib
import json
import subprocess
import sys

TOOL = sys.argv[1] if len(sys.argv) > 1 else "build-rel\\axiom-nn-logits.exe"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "data/llmopt/scorer_s2_dist.axnn"
SPEC = sys.argv[3] if len(sys.argv) > 3 else \
    "data/llmopt/scorer_s2_prompt_spec.json"
ROWS = sys.argv[4] if len(sys.argv) > 4 else \
    "data/llmopt/scorer_battery_v1.jsonl"
N = 20
MAX_SEQ = 512


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def tokenize(text, vocab):
    """Greedy longest match; None if any position has no match."""
    by_len = sorted(range(len(vocab)), key=lambda i: -len(vocab[i]))
    ids = []
    pos = 0
    while pos < len(text):
        for i in by_len:
            piece = vocab[i]
            if piece and text.startswith(piece, pos):
                ids.append(i)
                pos += len(piece)
                break
        else:
            return None
    return ids


def main():
    spec = json.load(open(SPEC, encoding="utf-8"))
    vocab = spec["vocab"]
    prompts, meta = [], []
    with open(ROWS, encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            if len(prompts) >= N:
                break
            row = json.loads(line)
            child = row["children"][0]["child"]
            text = spec["frame"].format(cur=row["state"], child=child)
            ids = tokenize(text, vocab)
            if ids is None or len(ids) > MAX_SEQ:
                continue
            assert "".join(vocab[i] for i in ids) == text, "round-trip"
            prompts.append(ids)
            meta.append({"idx": len(prompts) - 1, "row": row_idx,
                         "text": text})
    assert len(prompts) == N, f"only {len(prompts)} clean prompts"

    battery = "data/qual/scorer_s2_battery20.txt"
    meta_path = "data/qual/scorer_s2_battery20_meta.jsonl"
    out_path = "data/qual/scorer_s2_expected_logits.txt"
    with open(battery, "w", newline="\n") as f:
        for p in prompts:
            f.write(" ".join(str(i) for i in p) + "\n")
    with open(meta_path, "w", newline="\n") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")

    res = subprocess.run([TOOL, MODEL, battery], capture_output=True,
                         text=True, check=True)
    lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
    assert len(lines) == N, f"tool returned {len(lines)} lines"
    with open(out_path, "w", newline="\n") as f:
        for ln in lines:
            f.write(ln.strip() + "\n")

    for p in (MODEL, battery, meta_path, out_path):
        print(f"sha256 {sha(p)}  {p}")


if __name__ == "__main__":
    main()
