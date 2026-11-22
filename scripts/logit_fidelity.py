#!/usr/bin/env python3
"""logit_fidelity.py DIR VARIANT...: how far the logits of each VARIANT are from the reference run's, row by row, over the whole vocabulary.

DIR holds `ref-rankR.json` (the reference run's chunk0 description of rank R: `vocab_base`) and, per variant, `VARIANT/rankR-chunkC.stats`, the files
`scripts/logit_stats.c` writes for each rank's shard (10 f64 a row; see there). The four shards combine by log-sum-exp:
  LSE_A = M + log(sum_s ZA_s e^(mA_s - M)),  KL(A||B) = sum_s e^(mA_s - LSE_A) SA_s - LSE_A + LSE_B.
Prints, per variant: rows, KL(reference || variant) mean / median / p99 / max, top-1 agreement over all rows, over rows whose reference margin
(top-1 minus top-2 logit) is at least 0.5, and the number of disagreements by margin. Stdlib only (the nodes and the Mac have no numpy).
"""
import glob
import json
import math
import os
import struct
import sys

RANKS = 4


def rows_of(path):
    b = open(path, "rb").read()
    n = len(b) // 80
    return [struct.unpack_from("<10d", b, 80 * i) for i in range(n)]


def lse(parts):  # [(m, z)] -> log sum z e^m
    M = max(m for m, _ in parts)
    return M + math.log(sum(z * math.exp(m - M) for m, z in parts))


def variant(d, name, base):
    per_rank = []
    for r in range(RANKS):
        rows = []
        for f in sorted(glob.glob(os.path.join(d, name, f"rank{r}-chunk*.stats"))):
            rows += rows_of(f)
        per_rank.append(rows)
    n = len(per_rank[0])
    assert n and all(len(x) == n for x in per_rank), f"{name}: rows per rank {[len(x) for x in per_rank]}"
    kl, agree, margins, bad = [], 0, [], []
    for i in range(n):
        s = [per_rank[r][i] for r in range(RANKS)]
        la = lse([(x[0], x[1]) for x in s])
        lb = lse([(x[5], x[6]) for x in s])
        cross = sum(math.exp(x[0] - la) * x[2] for x in s)
        kl.append(max(cross - la + lb, 0.0))
        ia = max(range(RANKS), key=lambda r: s[r][0])  # the rank holding A's argmax
        ib = max(range(RANKS), key=lambda r: s[r][5])
        top_a = base[ia] + int(s[ia][3])
        top_b = base[ib] + int(s[ib][7])
        seconds = sorted([x[4] for x in s] + [x[0] for r, x in enumerate(s) if r != ia], reverse=True)
        margin = s[ia][0] - seconds[0]
        margins.append(margin)
        if top_a == top_b:
            agree += 1
        else:
            bad.append(margin)
    kls = sorted(kl)
    pick = lambda q: kls[min(n - 1, int(q * n))]
    wide = [m for m in margins if m >= 0.5]
    wide_bad = [m for m in bad if m >= 0.5]
    print(f"{name}: {n} rows")
    print(f"  KL(ref||variant)   mean {sum(kl)/n:.3e}   median {pick(0.5):.3e}   p99 {pick(0.99):.3e}   max {kls[-1]:.3e}")
    print(f"  top-1 agreement    {agree}/{n} = {agree/n:.4f}   ({len(wide) - len(wide_bad)}/{len(wide)} = {(len(wide)-len(wide_bad))/max(len(wide),1):.4f} of rows with margin >= 0.5)")
    print("  disagreements by reference margin:  " + "  ".join(f"<{t}: {sum(1 for m in bad if m < t)}" for t in (0.05, 0.1, 0.25, 0.5, 1.0)) + f"  >=1.0: {sum(1 for m in bad if m >= 1.0)}")


def main():
    d, names = sys.argv[1], sys.argv[2:]
    base = [json.load(open(os.path.join(d, f"ref-rank{r}.json")))["vocab_base"] for r in range(RANKS)]
    for n in names:
        variant(d, n, base)


if __name__ == "__main__":
    main()
