#!/usr/bin/env python3
"""Unit check of the harness's decisions, no server (stdlib only): the score is zero unless every required gate has
its exact `PASS <id>` line and is capped by the lane count; a reply is held to its pinned reference byte for byte
(a one-id difference, a different finish reason, a different prompt and an unpinned name all fail), and a concurrent
reply is held to the solo reply of its prompt.

    python3 tests/test_bench.py
"""

from __future__ import annotations

import math
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench", "tools"))
import bench  # noqa: E402
import score  # noqa: E402


def check(cond, what):
    if not cond:
        raise AssertionError(what)


def test_score() -> None:
    m = {"decode_c1_tok_s": 70.0, "prefill_32k_tok_s": 4000.0, "lanes_at_max_context": 4.0}
    req = ["pin", "spec", "capacity"]
    allpass = {g: "PASS" for g in req}
    raw, got, bad = score.score(m, allpass, req)
    check(math.isclose(raw, 1.0) and math.isclose(got, 1.0) and not bad, f"targets met scores 1: {raw} {got} {bad}")
    raw, got, bad = score.score({**m, "decode_c1_tok_s": 35.0, "prefill_32k_tok_s": 4000.0}, allpass, req)
    check(math.isclose(got, math.sqrt(0.5)), f"half the decode rate: sqrt(0.5), got {got}")
    raw, got, bad = score.score({**m, "lanes_at_max_context": 2.0}, allpass, req)
    check(math.isclose(got, 0.5), f"2 of 4 lanes halves the score, got {got}")
    raw, got, bad = score.score({**m, "lanes_at_max_context": 8.0}, allpass, req)
    check(math.isclose(got, 1.0), "more lanes than the target do not raise it")
    raw, got, bad = score.score(m, {**allpass, "spec": "FAIL"}, req)
    check(got == 0.0 and raw == 1.0 and bad == ["spec (FAIL)"], f"a failed gate zeroes the score, keeps raw: {got} {raw} {bad}")
    raw, got, bad = score.score(m, {"pin": "PASS", "spec": "PASS"}, req)
    check(got == 0.0 and bad == ["capacity (missing)"], f"a gate that never ran zeroes it: {got} {bad}")
    raw, got, bad = score.score({"decode_c1_tok_s": 70.0}, allpass, req)
    check(raw == 0.0 and got == 0.0, "a missing metric scores 0, nothing is assumed")
    with tempfile.TemporaryDirectory() as tmp:
        g = os.path.join(tmp, "gates.txt")
        with open(g, "w") as f:
            f.write("PASS pin\nFAIL spec\nPASS spec\nPASS pin2 extra\nPASS  capacity\n")
        parsed = score.read_gates(g)
        check(parsed == {"pin": "PASS", "spec": "PASS"}, f"gate lines are exact; a later line wins: {parsed}")
        mf = os.path.join(tmp, "metrics.txt")
        with open(mf, "w") as f:
            f.write("noise\nMETRIC decode_c1_tok_s=12.5\nMETRIC bad=x\nMETRIC prefill_32k_tok_s=3000\n")
        check(score.read_metrics(mf) == {"decode_c1_tok_s": 12.5, "prefill_32k_tok_s": 3000.0}, "metric lines parsed")
    print("ok   target_score: targets met = 1, ratios, lane cap, zero on a failed or missing gate or metric")


def rep(ids, finish="length", sha="a" * 64) -> dict:
    return {"prompt_sha256": sha, "generated_ids": ids, "finish_reason": finish}


def test_pin() -> None:
    want = rep([5, 6, 7, 8])
    check(bench.first_diff("x", rep([5, 6, 7, 8]), want) is None, "equal reply passes")
    d = bench.first_diff("x", rep([5, 6, 9, 8]), want)
    check(d and "differ at 2" in d, f"one id differs at index 2: {d}")
    d = bench.first_diff("x", rep([5, 6, 7]), want)
    check(d and "differ at 3" in d and "lengths 3/4" in d, f"a shorter reply differs where it stops: {d}")
    d = bench.first_diff("x", rep([5, 6, 7, 8], finish="stop"), want)
    check(d and "finish stop/length" in d, f"a different finish reason fails: {d}")
    d = bench.first_diff("x", rep([5, 6, 7, 8], sha="b" * 64), want)
    check(d and "prompt differs" in d, f"a different prompt fails: {d}")

    ref = {"decode/json": rep([1, 2]), "prefill/32k": rep([3])}
    check(bench.pinned_name("conc8/json#1") == "decode/json" and bench.pinned_name("prefill/32k") == "prefill/32k",
          "a concurrent reply is held to its solo prompt's entry")
    replies = {"decode/json": {**rep([1, 2]), "x": 1}, "conc4/json#0": rep([1, 2]), "conc8/json#1": rep([1, 9])}
    errs = bench.check_pinned(replies, ref)
    check(len(errs) == 1 and errs[0].startswith("conc8/json#1: generated ids differ at 1"), f"only the bad one: {errs}")
    errs = bench.check_pinned({"prefill/128k": rep([4])}, ref)
    check(errs == ["prefill/128k: prefill/128k is not in the pinned reference"], f"an unpinned reply fails: {errs}")
    check(bench.check_pinned({}, ref) == [], "a screen that ran nothing compares nothing")
    print("ok   pin: byte-exact ids, finish reason, prompt, unpinned name; concurrent held to the solo entry")


def test_hist() -> None:
    m = {"rexl3_request_ttft_seconds_sum": 2.5, "rexl3_request_ttft_seconds_count": 4.0,
         'rexl3_request_ttft_seconds_bucket{le="1"}': 3.0}
    check(bench.hist(m, bench.TTFT) == (2.5, 4.0), "histogram sum and count, buckets ignored")
    try:
        bench.hist({"rexl3_other": 1.0}, bench.TTFT)
    except SystemExit as e:
        check("no rexl3_request_ttft_seconds_count" in str(e), str(e))
    else:
        raise AssertionError("a missing histogram must stop the run, not read as zero")
    print("ok   server histogram read: sum/count, missing series stops the run")


def memwatch(mins, guard="running") -> str:
    return "\n".join(
        f"[r{r}] memwatch rank={r} guard={guard} window=none window_min_mib=n/a loading_min_mib={m} ready_min_mib={m} "
        f"min_mib={m} min_at=2026-11-18T05:49:02Z now_mib={m + 500} updated=2026-11-18T05:59:35Z" for r, m in enumerate(mins))


def plan(prefix: str) -> str:
    return ("[r0] rexl3 event=inkling2_plan slots=4 max_context=1048576 kv_tokens=4194304 lanes_at_max_context=4 "
            f"free_after_gib=41.00 prefix_entries=16 prefix_gib={prefix} prefix_reach_tokens=1223936 spare_gib=8.03\n")


def test_memcheck() -> None:
    import memcheck
    p, m = memcheck.check(memwatch([9875, 11417, 11101, 11967]), plan("6.03"))
    check(not p and m["mem_min_gib_norm"] == m["mem_min_gib"] == 9.64, f"the recorded boot passes unchanged: {p} {m}")
    p, m = memcheck.check(memwatch([11040, 12760, 12360, 12670]), plan("4.52"))
    check(not p and m["prefix_gib_recorded_gap"] == 1.51 and m["mem_min_gib"] == 10.78 and m["mem_min_gib_norm"] == 9.27,
          f"a boot with a smaller cache is normalized down to the recorded configuration: {p} {m}")
    p, m = memcheck.check(memwatch([8300, 12760, 12360, 12670]), plan("4.52"))
    check(len(p) == 1 and p[0].startswith("rank 0: MemAvailable minimum 8300 MiB") and m["mem_min_gib"] == 8.11,
          f"8.1 GiB raw is under the floor once normalized to the 6.03 GiB cache, so the verdict does not depend on the boot: {p}")
    p, _ = memcheck.check(memwatch([8300, 12760, 12360, 12670]), plan("6.03"))
    check(not p, f"the same raw minimum passes on a boot that had the recorded cache: {p}")
    p, _ = memcheck.check(memwatch([9875, 11417, 11101, 11967]), plan("7.50"))
    check(not p, f"a cache larger than the recorded one is not credited: {p}")
    p, _ = memcheck.check(memwatch([9875, 11417, 11101, 11967], guard="stopped"), plan("6.03"))
    check(len(p) == 4 and all("not running" in x for x in p), f"a guard that is not running fails every rank: {p}")
    p, _ = memcheck.check(memwatch([9875, 11417, 11101]), plan("6.03"))
    check(p == ["rank 3: not in the memwatch report"], f"a missing rank fails: {p}")
    p, _ = memcheck.check(memwatch([9875, 11417, 11101, 11967]), "no plan here\n")
    check(len(p) == 1 and "prefix_gib" in p[0], f"no capacity line fails: {p}")
    print("ok   memory gate: floor 8 GiB on the minimum normalized to the recorded 6.03 GiB prefix cache; guards, ranks, plan line required")


def main() -> int:
    test_score()
    test_pin()
    test_hist()
    test_memcheck()
    return 0


if __name__ == "__main__":
    sys.exit(main())
