#!/usr/bin/env python3
"""The `memory` gate from `./inkling2.sh memwatch report` and the same boot's capacity line (stdlib only).

  memcheck.py --report memwatch.txt --boot-log boot-a.log

Every rank's MemAvailable minimum since its guard armed (`min_mib`, the boot's loading phase included) must stay at or
above the engine's floor (8 GiB), and its guard must still be running. The minimum is first normalized to the recorded
configuration: the capacity plan reads the node's free memory at boot (39 to 41 GiB between boots of the same engine,
with whatever the kernel and driver still hold) and gives the prefix cache what is left (4.52 to 6.03 GiB). A boot with
a smaller cache than the recorded 6.03 GiB measured more headroom than the recorded configuration has, so that
difference is subtracted before comparing with the floor: the gate then gives the same verdict on any boot.

Prints `METRIC mem_min_gib_rank<N>=` (raw), `METRIC mem_min_gib=` (raw, lowest rank), `METRIC mem_min_gib_norm=`
(normalized, lowest rank) and `METRIC prefix_gib_recorded_gap=`; each problem on stderr. Exit 1 on any problem.
"""

from __future__ import annotations

import argparse
import re
import sys

FLOOR_MIB = 8192
RECORDED_PREFIX_GIB = 6.03


def parse_report(text: str) -> dict[int, dict[str, str]]:
    """{rank: {key: value}} from lines like `[r0] memwatch rank=0 guard=running ... min_mib=9875 ...`."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"\[r(\d)\] memwatch (.*)$", line.strip())
        if m:
            out[int(m.group(1))] = dict(kv.split("=", 1) for kv in m.group(2).split() if "=" in kv)
    return out


def prefix_gib(boot_log: str) -> float | None:
    m = re.search(r"event=inkling2_plan\b.* prefix_gib=([0-9.]+)", boot_log)
    return float(m.group(1)) if m else None


def check(report: str, boot_log: str) -> tuple[list[str], dict[str, float]]:
    """(problems, metrics)."""
    problems: list[str] = []
    metrics: dict[str, float] = {}
    prefix = prefix_gib(boot_log)
    if prefix is None:
        problems.append("no inkling2_plan line with prefix_gib in the boot log")
        prefix = RECORDED_PREFIX_GIB
    gap_mib = max(0.0, RECORDED_PREFIX_GIB - prefix) * 1024
    metrics["prefix_gib_recorded_gap"] = round(gap_mib / 1024, 2)
    ranks = parse_report(report)
    raw, norm = [], []
    for r in range(4):
        f = ranks.get(r)
        if f is None:
            problems.append(f"rank {r}: not in the memwatch report")
            continue
        try:
            low = int(f["min_mib"])
        except (KeyError, ValueError):
            problems.append(f"rank {r}: no min_mib in the memwatch report")
            continue
        if f.get("guard") != "running":
            problems.append(f"rank {r}: its memory guard is {f.get('guard')!r}, not running")
        metrics[f"mem_min_gib_rank{r}"] = round(low / 1024, 2)
        raw.append(low)
        norm.append(low - gap_mib)
        if low - gap_mib < FLOOR_MIB:
            problems.append(f"rank {r}: MemAvailable minimum {low} MiB, {low - gap_mib:.0f} MiB at the recorded {RECORDED_PREFIX_GIB} "
                            f"GiB prefix cache, is under the {FLOOR_MIB} MiB floor")
    if raw:
        metrics["mem_min_gib"] = round(min(raw) / 1024, 2)
        metrics["mem_min_gib_norm"] = round(min(norm) / 1024, 2)
    return problems, metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", required=True, help="./inkling2.sh memwatch report output")
    ap.add_argument("--boot-log", required=True, help="./inkling2.sh start output of the same boot")
    a = ap.parse_args(argv)
    with open(a.report) as f:
        report = f.read()
    with open(a.boot_log) as f:
        boot_log = f.read()
    problems, metrics = check(report, boot_log)
    for k, v in metrics.items():
        print(f"METRIC {k}={v}")
    for p in problems:
        print(f"FAIL {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
