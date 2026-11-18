#!/usr/bin/env python3
"""Byte-exactness checks against rank 0's API (stdlib only).

  exactness.py concurrent [--k 5] [--max-tokens 64]
      concurrent == solo: four different prompts (about 30, 400, 1500 and 6000 tokens) each alone, then all four at
      once; the greedy ids (temperature 0) and the prompt_logprobs top-k (ids and values, compared as the floats' repr)
      must be identical. Generation and prompt_logprobs are separate requests (the server's prompt_logprobs samples
      nothing), each phase sent four at a time. Exit 1 on any difference.

  exactness.py record OUT.json [--label TEXT]
      every reply (generated ids, finish reason, prompt_logprobs) of the same fixed prompt set, solo, one at a time,
      into OUT.json: the `replies` part depends on nothing but the engine's output.

  exactness.py diff A.json B.json
      byte-exact comparison of two records' `replies` (two boots; two --prefill-chunk widths); names the first
      difference. Exit 1 when they differ.

The prompts are fixed texts tokenized by the server (/tokenize), so every run and boot sees the same ids.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling_api as ia  # noqa: E402

TEXTS = {
    "short": "Write a haiku about a lighthouse keeper who collects sea glass.",
    "medium": "Summarize the following log in one paragraph.\n\n" + " ".join(ia.filler_sentences(20, 11)),
    "long": "Here are the inventory notes.\n\n" + " ".join(ia.filler_sentences(80, 12)) +
            "\n\nWhich items were weighed, and where?",
    "xlong": "Archive transcript follows.\n\n" + " ".join(ia.filler_sentences(330, 13)) +
             "\n\nList three places mentioned most often.",
}
LONG_MIN = 5000  # the xlong prompt must exceed this many tokens


def prompts(api: ia.Api) -> dict[str, list[int]]:
    out = {k: api.tokenize(v) for k, v in TEXTS.items()}
    if len(out["xlong"]) <= LONG_MIN:
        raise SystemExit(f"the xlong prompt is {len(out['xlong'])} tokens, not > {LONG_MIN}: lengthen TEXTS['xlong']")
    return out


def run_one(api: ia.Api, ids: list[int], what: str, k: int, max_tokens: int):
    if what == "gen":
        resp, dt = api.complete(ids, max_tokens)
        gen, finish = ia.generated_ids(resp)
        return {"generated_ids": gen, "finish_reason": finish}, dt
    try:
        plp, dt = api.prompt_logprobs(ids, k)
    except ia.ApiError as e:
        # an engine without the full-vocabulary score path (Inkling 2 today) refuses prompt_logprobs by name: the check then
        # rests on the generated ids alone, and the report says so
        if e.status == 400 and "not supported" in str(e):
            return {"prompt_logprobs": "unsupported"}, 0.0
        raise
    return {"prompt_logprobs": plp}, dt


def phase(api, ps, what, k, max_tokens, concurrent: bool):
    names = list(ps)
    if concurrent:
        with ThreadPoolExecutor(len(names)) as ex:
            futs = {n: ex.submit(run_one, api, ps[n], what, k, max_tokens) for n in names}
            return {n: futs[n].result() for n in names}
    return {n: run_one(api, ps[n], what, k, max_tokens) for n in names}


def canon(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))  # floats as repr: exact round trip


def first_diff(a: dict, b: dict) -> str | None:
    if canon(a) == canon(b):
        return None
    if "generated_ids" in a:
        ga, gb = a["generated_ids"], b["generated_ids"]
        i = next((j for j, (x, y) in enumerate(zip(ga, gb)) if x != y), min(len(ga), len(gb)))
        return f"generated ids differ at {i} ({ga[i:i + 3]} vs {gb[i:i + 3]}), finish {a['finish_reason']}/{b['finish_reason']}"
    pa, pb = a["prompt_logprobs"], b["prompt_logprobs"]
    if isinstance(pa, str) or isinstance(pb, str):
        return f"prompt_logprobs {pa if isinstance(pa, str) else 'present'} vs {pb if isinstance(pb, str) else 'present'}"
    for j, (x, y) in enumerate(zip(pa, pb)):
        if canon(x) != canon(y):
            return f"prompt_logprobs differ at position {j}: {canon(x)[:200]} vs {canon(y)[:200]}"
    return f"prompt_logprobs lengths {len(pa)} vs {len(pb)}"


def cmd_concurrent(a) -> int:
    api = ia.api_from_args(a)
    ps = prompts(api)
    rep = {"url": api.url, "model": api.model, "prompt_tokens": {n: len(v) for n, v in ps.items()}, "k": a.k,
           "max_tokens": a.max_tokens, "checks": {}}
    ok = True
    for what in ("gen", "logprobs"):
        solo = phase(api, ps, what, a.k, a.max_tokens, False)
        conc = phase(api, ps, what, a.k, a.max_tokens, True)
        for n in ps:
            d = first_diff(solo[n][0], conc[n][0])
            rep["checks"][f"{what}/{n}"] = {"equal": d is None, "difference": d, "solo_s": round(solo[n][1], 3),
                                            "concurrent_s": round(conc[n][1], 3)}
            ok &= d is None
            print(f"{what:8s} {n:6s} ({len(ps[n])} tokens): {'equal' if d is None else 'DIFFERS: ' + d}", file=sys.stderr)
    rep["pass"] = ok
    print(json.dumps(rep, indent=1))
    return 0 if ok else 1


def cmd_record(a) -> int:
    api = ia.api_from_args(a)
    ps = prompts(api)
    replies, timing = [], {}
    for n, ids in ps.items():
        g, dt_g = run_one(api, ids, "gen", a.k, a.max_tokens)
        p, dt_p = run_one(api, ids, "logprobs", a.k, a.max_tokens)
        replies.append({"name": n, "prompt_ids": ids, **g, **p})
        timing[n] = {"gen_s": round(dt_g, 3), "prompt_logprobs_s": round(dt_p, 3)}
    status, models, _ = api.call("GET", "/v1/models")
    rec = {"meta": {"url": api.url, "model": api.model, "label": a.label, "k": a.k, "max_tokens": a.max_tokens,
                    "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "models": models, "timing": timing},
           "replies": replies}
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    sizes = ", ".join(f"{r['name']} {len(r['prompt_ids'])}" for r in replies)
    print(f"{a.out}: {len(replies)} replies (prompt tokens: {sizes})")
    return 0


def cmd_diff(a) -> int:
    with open(a.a) as f:
        ra = json.load(f)
    with open(a.b) as f:
        rb = json.load(f)
    if canon(ra["replies"]) == canon(rb["replies"]):
        print(f"identical: {len(ra['replies'])} replies ({ra['meta'].get('label')} vs {rb['meta'].get('label')})")
        return 0
    by_b = {r["name"]: r for r in rb["replies"]}
    for r in ra["replies"]:
        o = by_b.get(r["name"])
        if o is None:
            print(f"{r['name']}: only in {a.a}")
            continue
        if r["prompt_ids"] != o["prompt_ids"]:
            print(f"{r['name']}: different prompt ids (tokenizer or prompt set changed)")
            continue
        for part in ({"generated_ids": r["generated_ids"], "finish_reason": r["finish_reason"]},
                     {"prompt_logprobs": r["prompt_logprobs"]}):
            other = {k: o.get(k) for k in part}
            d = first_diff(part, other) if all(v is not None for v in other.values()) else "missing in B"
            if d:
                print(f"{r['name']}: {d}")
    for n in sorted(set(by_b) - {r["name"] for r in ra["replies"]}):
        print(f"{n}: only in {a.b}")
    print("DIFFERENT")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("concurrent", "record"):
        p = sub.add_parser(name)
        ia.add_api_args(p)
        p.add_argument("--k", type=int, default=5, help="prompt_logprobs top-k (server max 20)")
        p.add_argument("--max-tokens", type=int, default=64)
        if name == "record":
            p.add_argument("out")
            p.add_argument("--label", default="", help="free text stored in meta (boot, prefill chunk, ...)")
    p = sub.add_parser("diff")
    p.add_argument("a")
    p.add_argument("b")
    a = ap.parse_args(argv)
    return {"concurrent": cmd_concurrent, "record": cmd_record, "diff": cmd_diff}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
