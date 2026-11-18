#!/usr/bin/env python3
"""Engine greedy vs oracle fixtures, token by token (stdlib only).

For every fixture of an oracle manifest (oracle/README.md "Fixtures": FIXTURES/manifest.json + NAME.safetensors), runs
the engine greedy (temperature 0, token-id prompt) from the fixture's prompt for exactly len(generated) tokens,
continuing past EOS 200006 by re-requesting with prompt + everything so far, and compares with the oracle's tokens:
first divergence, the oracle's top-1 - top-2 logit margin there (from the fixture's logits) and whether that is an
excused near-tie (< 0.05, compare.py's tie_margin). Exit 0 when every fixture is equal or excused.

    greedy.py --fixtures ~/inkling2-oracle/fixtures [--only NAME ...] [--teacher-force] --out greedy.json

--teacher-force: after each fixture, one more request with prompt = fixture prompt + engine tokens[:-1] (max_tokens 1),
so a boot with DUMP_DIR set holds the prefill rows merge_dump.py needs for compare.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling_api as ia  # noqa: E402
import stlite  # noqa: E402

TIE_MARGIN = 0.05  # oracle/compare.py GATES_ENGINE["tie_margin"] (the correctness gate's)


def margin_at(path: str, row: int) -> tuple[float, int, int] | None:
    """(top1 - top2, top1 id, top2 id) of the fixture's logits row, or None without logits."""
    t, _ = stlite.read_header(path)
    if "logits" not in t or row >= t["logits"]["shape"][0]:
        return None
    r = stlite.read_f32_row(path, "logits", row)
    i1 = max(range(len(r)), key=r.__getitem__)
    v1 = r[i1]
    r[i1] = float("-inf")
    i2 = max(range(len(r)), key=r.__getitem__)
    return v1 - r[i2], i1, i2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ia.add_api_args(ap)
    ap.add_argument("--fixtures", required=True, help="directory with manifest.json and NAME.safetensors")
    ap.add_argument("--only", nargs="*", help="fixture names (default: all in the manifest)")
    ap.add_argument("--teacher-force", action="store_true", help="also prefill prompt + engine tokens[:-1] (for dumps)")
    ap.add_argument("--out", help="write the JSON report here (also printed)")
    a = ap.parse_args(argv)

    with open(os.path.join(a.fixtures, "manifest.json")) as f:
        manifest = json.load(f)
    names = a.only or sorted(manifest["fixtures"])
    api = ia.api_from_args(a)
    rep = {"url": api.url, "model": api.model, "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "fixtures": {}}
    ok = True
    for name in names:
        ent = manifest["fixtures"].get(name)
        if ent is None:
            raise SystemExit(f"{name}: not in {a.fixtures}/manifest.json")
        path = os.path.join(a.fixtures, name + ".safetensors")
        _, meta = stlite.read_header(path)
        plen = int(meta["prompt_len"])
        if plen != int(ent["prompt_len"]):
            raise SystemExit(f"{name}: prompt_len {plen} in the fixture, {ent['prompt_len']} in the manifest")
        prompt = stlite.read_ints(path, "tokens")[:plen]
        oracle = [int(t) for t in ent["generated"]]
        engine, requests, secs = ia.greedy(api, prompt, len(oracle))
        diff = [i for i, (x, y) in enumerate(zip(oracle, engine)) if x != y]
        r = {"prompt_len": plen, "generated": len(oracle), "equal": not diff, "first_divergence": None,
             "mismatch_positions": diff, "requests": requests, "seconds": round(secs, 3),
             "engine_generated": engine, "oracle_generated": oracle}
        if diff:
            i = diff[0]
            m = margin_at(path, plen - 1 + i)
            r.update(first_divergence=i, oracle_token=oracle[i], engine_token=engine[i],
                     oracle_margin=None if m is None else m[0],
                     oracle_top2=None if m is None else [m[1], m[2]],
                     excused_tie=m is not None and m[0] < TIE_MARGIN)
        r["pass"] = r["equal"] or bool(r.get("excused_tie"))
        ok &= r["pass"]
        if a.teacher_force:
            _, dt = api.complete(prompt + engine[:-1], 1)
            r["teacher_forced_rows"] = plen + len(engine) - 1
            r["teacher_force_seconds"] = round(dt, 3)
        rep["fixtures"][name] = r
        print(f"{name}: {'equal' if r['equal'] else 'DIVERGES at %d' % r['first_divergence']}"
              f"{' (excused tie, margin %.4g)' % r['oracle_margin'] if r.get('excused_tie') else ''}"
              f" ({len(oracle)} tokens, {requests} requests, {secs:.1f}s)", file=sys.stderr)
    rep["pass"] = ok
    txt = json.dumps(rep, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt + "\n")
    print(txt)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
