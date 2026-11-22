#!/usr/bin/env python3
"""Per-phase attribution of an nsys sqlite export (the idea of the gb10-cuda-kernel-perf skill's pprof.py, time-sliced; stdlib only).

  nsys_phases.py RANK.sqlite NAME:START_S:END_S [NAME:START_S:END_S ...]

START_S and END_S are seconds from the rank's first kernel. For each phase it prints the GPU span, the sum and the union of the
kernel durations (idle = span minus union), the kernels by category (nccl, moe, linear, attention, drafter, ...) and by name, the
host time in the CUDA runtime API (launches, stream syncs) and the sync and memcpy counts. Collective kernels (`ncclDevKernel_*`)
include the time they wait for their peers. A profile is made with `PROFILE_NSYS=... ./inkling2.sh start`, `profile_driver.py`
and `nsys export -t sqlite` (docs/exec-plans/completed/autoresearch-segment-a.md, "Profile").
"""
import collections
import sqlite3
import sys

path, phases = sys.argv[1], sys.argv[2:]  # phases: name:start_s:end_s (relative to the first kernel)
c = sqlite3.connect(path)
S = dict(c.execute("select id, value from StringIds"))
ks = c.execute("select start, end, shortName, gridX, gridY, gridZ, streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
t0 = ks[0][0]
rt = c.execute("select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start").fetchall()
sy = c.execute("select start, end, syncType from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION order by start").fetchall()
mc = c.execute("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY order by start").fetchall()


def union(iv):
    iv = sorted(iv)
    tot, cur_s, cur_e = 0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return tot + (cur_e - cur_s if cur_e is not None else 0)


def cat(n):
    n = n.lower()
    if "nccl" in n:
        return "nccl"
    for key, name in (("moe", "moe"), ("expert", "moe"), ("route", "moe"), ("attn", "attention"), ("attention", "attention"),
                      ("exl3", "linear_exl3"), ("linear", "linear"), ("gemv", "linear"), ("gemm", "linear"),
                      ("markov", "drafter"), ("dspark", "drafter"), ("topk", "drafter"), ("norm", "norm"),
                      ("mix", "attnres_mix"), ("conv", "short_conv"), ("silu", "act"), ("sum", "reduce")):
        if key in n:
            return name
    return "other"


for ph in phases:
    name, a, b = ph.split(":")
    lo, hi = t0 + int(float(a) * 1e9), t0 + int(float(b) * 1e9)
    k = [x for x in ks if lo <= x[0] < hi]
    span = (hi - lo) / 1e6
    un = union([(x[0], x[1]) for x in k]) / 1e6
    tot = sum(x[1] - x[0] for x in k) / 1e6
    r = [x for x in rt if lo <= x[0] < hi]
    rname = collections.defaultdict(lambda: [0, 0])
    for s, e, nid in r:
        rname[S[nid]][0] += e - s
        rname[S[nid]][1] += 1
    s_ = [x for x in sy if lo <= x[0] < hi]
    print(f"\n=== {path} {name} [{a}s,{b}s): span {span:.0f} ms, kernels {len(k)}, sum {tot:.0f} ms, union {un:.0f} ms "
          f"(gpu idle {100 * (1 - un / span):.1f}%), streams {sorted({x[6] for x in k})}")
    bycat = collections.defaultdict(lambda: [0, 0])
    byname = collections.defaultdict(lambda: [0, 0])
    for s, e, nm, gx, gy, gz, st in k:
        n = S[nm]
        bycat[cat(n)][0] += e - s
        bycat[cat(n)][1] += 1
        byname[n][0] += e - s
        byname[n][1] += 1
    print("  by category (sum of kernel durations, overlap counted twice):")
    for n, (t, cnt) in sorted(bycat.items(), key=lambda kv: -kv[1][0]):
        print(f"    {t / 1e6:9.1f} ms {100 * t / (tot * 1e6):5.1f}% n={cnt:7d} avg {t / cnt / 1e3:8.1f} us  {n}")
    print("  top kernels:")
    for n, (t, cnt) in sorted(byname.items(), key=lambda kv: -kv[1][0])[:16]:
        print(f"    {t / 1e6:9.1f} ms n={cnt:7d} avg {t / cnt / 1e3:8.1f} us  {n[:78]}")
    print("  runtime API (host time, top):")
    for n, (t, cnt) in sorted(rname.items(), key=lambda kv: -kv[1][0])[:8]:
        print(f"    {t / 1e6:9.1f} ms n={cnt:7d} avg {t / cnt / 1e3:8.1f} us  {n}")
    st = sum(x[1] - x[0] for x in s_) / 1e6
    print(f"  sync events {len(s_)} total {st:.0f} ms;  memcpy n={sum(1 for x in mc if lo <= x[0] < hi)}")
