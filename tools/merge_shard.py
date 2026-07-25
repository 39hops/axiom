#!/usr/bin/env python3
"""Farm-shard merge with delivery gate (llmopt audit 2026-07-24).

Merges source jsonl files into one shard, strictly from closed files:
- unparseable lines are DROPPED and reported (a killed farm process can
  leave a truncated tail; raw cat splices the next row onto it — the
  line-41 interleave defect)
- dedupe on (cur, nxt) per the round-1 convention
- full JSON parse pass over the output before it may be delivered

Usage: python tools/merge_shard.py <out.jsonl> <src...>
Exit 1 if any source line was dropped (inspect before delivering).
"""
import json
import sys


def main() -> int:
    out_path, srcs = sys.argv[1], sys.argv[2:]
    seen, rows, dropped = set(), [], []
    for path in srcs:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    dropped.append((path, line[:90]))
                    continue
                key = (row["cur"], row["nxt"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(line)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    with open(out_path, encoding="utf-8") as fh:  # delivery gate
        for line in fh:
            json.loads(line)
    print(f"rows: {len(rows)}  dropped: {len(dropped)}  gate: PASS")
    for path, frag in dropped:
        print(f"  DROPPED {path} :: {frag}", file=sys.stderr)
    return 1 if dropped else 0


if __name__ == "__main__":
    sys.exit(main())
