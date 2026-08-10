"""Classify macOS `sample` call-tree self-time into anchor-v2 probe
buckets: GCD/normalize vs SEAM (floor/decimal round-trip) vs RING
(bigint mul/add/div outside gcd) vs OTHER. Bucket = nearest matching
ancestor; self time = node count minus direct children counts."""
import re
import sys

PAT = re.compile(r"^( +)(\d+) (.+?)(?:  \(in .*)?$")

def bucket_of(sym):
    if "ax::gcd" in sym or "normalize" in sym:
        return "gcd"
    if ("floor_big" in sym or "to_string" in sym or "stoll" in sym
            or "operator long long" in sym or "from_chars" in sym):
        return "seam"
    if ("ax::bigint" in sym or "ax::operator" in sym
            or "ax::rational" in sym or "karatsuba" in sym
            or "ntt" in sym):
        return "ring"
    return None

def main(path):
    nodes = []  # (depth, count, bucket, children_sum_ref)
    stack = []  # (depth, idx)
    totals = {"gcd": 0, "seam": 0, "ring": 0, "other": 0}
    lines = open(path).read().splitlines()
    # keep only the call-graph section
    try:
        start = next(i for i, l in enumerate(lines)
                     if l.startswith("Call graph"))
        end = next(i for i, l in enumerate(lines)
                   if l.startswith("Total number in stack"))
    except StopIteration:
        start, end = 0, len(lines)
    kids = {}
    rows = []
    for ln in lines[start:end]:
        m = PAT.match(ln)
        if not m:
            continue
        depth = len(m.group(1))
        cnt = int(m.group(2))
        sym = m.group(3)
        while stack and stack[-1][0] >= depth:
            stack.pop()
        parent = stack[-1][1] if stack else None
        b = bucket_of(sym) or (rows[parent][2] if parent is not None
                               else "other")
        idx = len(rows)
        rows.append((depth, cnt, b))
        kids[idx] = 0
        if parent is not None:
            kids[parent] += cnt
        stack.append((depth, idx))
    for idx, (depth, cnt, b) in enumerate(rows):
        self_t = cnt - kids[idx]
        if self_t > 0:
            totals[b] += self_t
    tot = sum(totals.values()) or 1
    for k, v in sorted(totals.items(), key=lambda x: -x[1]):
        print(f"{k:6s} {v:8d}  {100*v/tot:5.1f}%")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(f"--- {p.split('/')[-1]}")
        main(p)
