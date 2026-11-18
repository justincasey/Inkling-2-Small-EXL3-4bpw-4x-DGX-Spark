#!/usr/bin/env python3
"""The harness score from a run's `METRIC name=value` lines and its `PASS id` / `FAIL id` gate lines (stdlib only).

  score.py --metrics FILE --gates FILE --required "id id ..."

    target_score_raw = sqrt(decode_c1_tok_s / 70 * prefill_32k_tok_s / 4000) * min(1, lanes_at_max_context / 4)
    target_score     = target_score_raw when every required gate has the line `PASS <id>` (exact, whole line) in the
                       gates file, else 0

A missing metric makes the raw score 0 as well (nothing is assumed). Prints `METRIC target_score_raw=`,
`METRIC target_score=`, `METRIC gates_passed=` and `METRIC gates_required=`; each gate that is not passed is named on
stderr as `NOT PASSED <id> (missing|FAIL)`. Exit 0 always: the caller decides what a zero means.
"""

from __future__ import annotations

import argparse
import math
import sys

DECODE_TARGET, PREFILL_TARGET, LANES_TARGET = 70.0, 4000.0, 4.0


def read_metrics(path: str) -> dict[str, float]:
    out = {}
    with open(path) as f:
        for line in f:
            if line.startswith("METRIC ") and "=" in line:
                k, _, v = line[7:].strip().partition("=")
                try:
                    out[k] = float(v)
                except ValueError:
                    pass
    return out


def read_gates(path: str) -> dict[str, str]:
    """{id: "PASS" | "FAIL"}; a later line for an id wins (a gate run again)."""
    out = {}
    with open(path) as f:
        for line in f.read().splitlines():
            head, _, gid = line.partition(" ")
            if head in ("PASS", "FAIL") and gid and " " not in gid:
                out[gid] = head
    return out


def raw_score(m: dict[str, float]) -> float:
    need = ("decode_c1_tok_s", "prefill_32k_tok_s", "lanes_at_max_context")
    if any(k not in m for k in need) or m["decode_c1_tok_s"] <= 0 or m["prefill_32k_tok_s"] <= 0:
        return 0.0
    return math.sqrt(m["decode_c1_tok_s"] / DECODE_TARGET * m["prefill_32k_tok_s"] / PREFILL_TARGET) \
        * min(1.0, m["lanes_at_max_context"] / LANES_TARGET)


def score(metrics: dict[str, float], gates: dict[str, str], required: list[str]) -> tuple[float, float, list[str]]:
    """(raw, scored, problems): scored is 0 unless every required gate passed."""
    raw = raw_score(metrics)
    problems = [f"{g} ({gates.get(g, 'missing')})" for g in required if gates.get(g) != "PASS"]
    return raw, (0.0 if problems else raw), problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--gates", required=True)
    ap.add_argument("--required", default="", help="space-separated gate ids")
    ap.add_argument("--screen", action="store_true",
                    help="a partial run (a screen or a pin run): target_score is 0, only the raw score is meaningful")
    a = ap.parse_args(argv)
    required = a.required.split()
    raw, scored, problems = score(read_metrics(a.metrics), read_gates(a.gates), required)
    if a.screen:
        scored, problems = 0.0, []
    for p in problems:
        print(f"NOT PASSED {p}", file=sys.stderr)
    print(f"METRIC target_score_raw={raw:.4f}")
    print(f"METRIC target_score={scored:.4f}")
    print(f"METRIC gates_passed={len(required) - len(problems)}")
    print(f"METRIC gates_required={len(required)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
