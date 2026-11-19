#!/usr/bin/env python3
"""The fixed performance workload against rank 0's API (stdlib only), printed as `METRIC name=value` lines.

Every request is greedy (temperature 0) on /v1/completions with token-id prompts and `"prefix_cache": false`, so each
one is cold and nothing is read from or stored in the prefix cache (a reply that reports cached tokens fails the run).
Token counts come from `usage.completion_tokens`, never from stream chunks. Time to first token comes from the
server's own histogram `rexl3_request_ttft_seconds` (/metrics sum and count read around a solo request): the span
from the HTTP middleware's entry to the first engine token batch, i.e. the prefill. Decode speed of a request is
(completion_tokens - 1) / (e2e - ttft), e2e being the server's `rexl3_request_e2e_seconds` (entry to the end of
generation) read the same way; the client's wall clock is only recorded.

  --phases warm,decode,conc,prefill-32k,prefill,prefill-1m     (default: decode,conc,prefill; autoresearch.sh passes its own)
  warm         discarded: the decode prompts at 128 tokens, four at once, and one cold 32K prefill
  decode       c = 1 over four prompt types (prose, code, structured, json; 512 tokens each), DSpark as booted.
               decode_c1_tok_s is the mean of the four per-type rates; decode_acceptance and decode_tokens_per_round
               come from usage.spec
  conc         the same prompts at c = 4 (each once) and c = 8 (each twice): aggregate completion tokens over the wall
               span. The server runs 4 slots, so c = 8 is two waves of 4. Every reply must equal its solo reply
  prefill-32k  one cold 32,768-token prompt (128 generated tokens): the quick prefill screen
  prefill      cold 32,768-, 131,072- and 262,144-token prompts, one at a time (128 generated tokens each):
               prefill_<n>k_tok_s = prompt_tokens / ttft, and the decode speed at that context (decode_ctx<n>k_tok_s).
               262,144 (256K) is the largest context of a routine run
  prefill-1m   one cold 1,000,000-token prompt (128 generated tokens): prefill_1m_tok_s and decode_ctx1m_tok_s. About 25 minutes:
               never part of a routine run, a benchmark of the largest context (PHASES=prefill-1m ./autoresearch.sh)

Pinned replies. `--pin PATH` pins the replies of this run (generated ids and finish reason per named request): it writes PATH,
or, when PATH exists, extends it: every entry already in it must be reproduced byte for byte (a difference fails the run and
leaves PATH unchanged) and entries not in it yet are added, recorded under meta.extended. Pin only on the unmodified engine.
Every other run compares each reply it produced with `--ref PATH` byte for byte: `PASS pin` when all equal, otherwise
`FAIL pin ...` lines and exit 1. A screen (a subset of phases) compares what it ran. A pin run also requires every concurrent
reply to equal its solo reply.

Prompts are fixed texts (DECODE_TEXTS) and a long filler stream built from inkling_api.filler_sentences blocks (a seed
each): ids for the 32K, 128K and 256K prompts (tokenized block by block through the server's /tokenize), text for the 1M one
(the server refuses a body of more than 300,000 JSON nodes, so a million ids cannot be sent; its real size is
usage.prompt_tokens). A pinned reply records the sha256 of the prompt as defined (its ids, or its text).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling_api as ia  # noqa: E402

PROMPT_VERSION = "1"
TTFT = "rexl3_request_ttft_seconds"
E2E = "rexl3_request_e2e_seconds"
DECODE_TOKENS = 512
LONG_TOKENS = 128
SIZES = {"32k": 32_768, "128k": 131_072, "256k": 262_144, "1m": 1_000_000}
BLOCK_SENTENCES = 800
SUFFIX = "\n\nQuestion: Which three people appear most often in the notes above?\nAnswer:"
PHASES = ("warm", "decode", "conc", "prefill-32k", "prefill", "prefill-1m")
DEFAULT_PHASES = "decode,conc,prefill"

DECODE_TEXTS = {
    "prose": (
        "The following is the first chapter of a long literary novel. It is set in a small fishing village on a cold "
        "northern coast, and it follows an aging lighthouse keeper named Marta Ilves as the winter storms arrive.\n\n"
        "Chapter One: The Last Light\n\n"
        "By the time the first snow reached the harbor of Vellmar, Marta had already counted the boats twice. "),
    "code": (
        '"""A small, dependency-free event scheduler with priorities and cancellation."""\n\n'
        "from __future__ import annotations\n\n"
        "import heapq\nimport itertools\nimport time\nfrom dataclasses import dataclass, field\n"
        "from typing import Any, Callable\n\n\n"
        "@dataclass(order=True)\nclass ScheduledEvent:\n"
        "    when: float\n    priority: int\n    sequence: int\n"
        "    callback: Callable[..., Any] = field(compare=False)\n"
        "    args: tuple = field(compare=False, default=())\n"
        "    cancelled: bool = field(compare=False, default=False)\n\n\n"
        "class Scheduler:\n"
        '    """Runs callbacks at or after their due time, lowest priority number first on ties."""\n\n'
        "    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:\n"
        "        self._clock = clock\n"
        "        self._queue: list[ScheduledEvent] = []\n"
        "        self._counter = itertools.count()\n\n"
        "    def schedule("),
    "structured": (
        "# Deployment runbook: payments-gateway\n\n## 1. Service overview\n\n"
        "| Field | Value |\n|---|---|\n| Owner | platform-payments |\n| Tier | 1 |\n| Regions | eu-west-1, us-east-2 |\n\n"
        "## 2. Configuration\n\n```yaml\nservice:\n  name: payments-gateway\n  replicas: 6\n  resources:\n"
        "    requests:\n      cpu: \"500m\"\n      memory: 512Mi\n  healthcheck:\n    path: /healthz\n"
        "    interval_seconds: 10\n```\n\n## 3. Rollout procedure\n\n"
        "1. Confirm the dashboards are green and no incident is open.\n2."),
    "json": (
        "The following JSON array lists the open support tickets exported from the helpdesk on 2026-03-14.\n\n[\n"
        '  {"id": "T-1001", "customer": "Northwind Traders", "priority": "high", "status": "open", '
        '"assignee": "a.okafor", "tags": ["billing", "refund"], "opened": "2026-03-11T09:14:00Z", '
        '"summary": "Customer was charged twice for the March invoice."},\n'
        '  {"id": "T-1002", "customer": "Contoso Ltd", "priority": "low", "status": "pending", '
        '"assignee": "m.larsen", "tags": ["feature-request"], "opened": "2026-03-11T13:40:00Z", '
        '"summary": "Asks whether the export can include archived projects."},\n'
        '  {"id": "T-1003",'),
}

FAILS: list[str] = []


def say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def fail(what: str) -> None:
    FAILS.append(what)
    say(f"FAIL {what}")


def digest(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


# ── prompts ──
def block_text(i: int) -> str:
    """Block i of the filler stream: BLOCK_SENTENCES varied sentences (seed 7000 + i), deterministic."""
    return " ".join(ia.filler_sentences(BLOCK_SENTENCES, 7000 + i))


def long_ids(api: ia.Api, n: int) -> list[int]:
    """Exactly n ids: the blocks tokenized one by one (each is small enough for any body limit) and concatenated, cut so
    that the closing question (SUFFIX) ends the prompt. Sent as ids: the server's tokenizer is not in the time to first
    token. The server refuses more than 300,000 JSON nodes in a body, so this is for prompts up to 128K."""
    tail = api.tokenize(SUFFIX)
    ids, i = [], 0
    while len(ids) < n - len(tail):
        ids += api.tokenize(block_text(i))
        i += 1
    return ids[: n - len(tail)] + tail


def long_text(api: ia.Api, n: int) -> str:
    """The same stream as text of about n tokens (blocks counted one by one, joined by spaces, then SUFFIX): a 1M-id
    body is over the server's JSON node limit, a text body is not. The real count is usage.prompt_tokens."""
    tail = len(api.tokenize(SUFFIX))
    parts, total, i = [], 0, 0
    while total < n - tail:
        parts.append(block_text(i))
        total += len(api.tokenize(parts[-1]))
        i += 1
    return " ".join(parts) + SUFFIX


# ── requests ──
def hist(metrics: dict[str, float], name: str) -> tuple[float, float]:
    """(sum, count) of histogram `name`, summed over its label sets."""
    if not any(k.split("{")[0] == name + "_count" for k in metrics):
        near = sorted({k.split("{")[0] for k in metrics if k.startswith("rexl3_request_")})
        raise SystemExit(f"/metrics has no {name}_count (request series present: {near})")
    return (sum(v for k, v in metrics.items() if k.split("{")[0] == name + "_sum"),
            sum(v for k, v in metrics.items() if k.split("{")[0] == name + "_count"))


def prompt_digest(prompt) -> str:
    """sha256 of the prompt as defined: the JSON of its ids, or the UTF-8 bytes of its text."""
    return digest(prompt) if isinstance(prompt, list) else hashlib.sha256(prompt.encode()).hexdigest()


def reply(prompt, resp: dict, wall: float) -> dict:
    g, finish = ia.generated_ids(resp)
    u = resp.get("usage") or {}
    return {"prompt_tokens": int(u["prompt_tokens"]), "prompt_sha256": prompt_digest(prompt), "generated_ids": g,
            "finish_reason": finish, "completion_tokens": int(u["completion_tokens"]), "wall_s": round(wall, 4),
            "spec": ia.spec_counters(u), "cached_tokens": ia.cached_tokens(u)}


def solo(api: ia.Api, prompt, max_tokens: int) -> dict:
    """One request alone: the reply, the server's time to first token (prefill) and the decode speed after it, both
    from the server's histograms (middleware entry to first token batch, and to the end of generation)."""
    m0 = api.metrics()
    t0, e0 = hist(m0, TTFT), hist(m0, E2E)
    resp, wall = api.complete(prompt, max_tokens, prefix_cache=False)
    m1 = api.metrics()
    t1, e1 = hist(m1, TTFT), hist(m1, E2E)
    if round(t1[1] - t0[1]) != 1 or round(e1[1] - e0[1]) != 1:
        raise SystemExit(f"the server's request histograms moved by {t1[1] - t0[1]} / {e1[1] - e0[1]} requests around a "
                         f"solo request: it is not otherwise idle")
    r = reply(prompt, resp, wall)
    ttft, e2e = t1[0] - t0[0], e1[0] - e0[0]
    r["ttft_s"] = round(ttft, 4)
    r["decode_s"] = round(e2e - ttft, 4)
    r["decode_tok_s"] = round((r["completion_tokens"] - 1) / (e2e - ttft), 3) if e2e > ttft else None
    if r["cached_tokens"]:
        fail(f"a cold request ({r['prompt_tokens']} tokens) was served {r['cached_tokens']} tokens from the prefix cache")
    return r


def together(api: ia.Api, jobs: list[tuple[str, list[int], int]]) -> tuple[list[tuple[str, dict]], float]:
    """All jobs at once (a barrier); (key, reply) per job and the wall span from the first start to the last end."""
    barrier = threading.Barrier(len(jobs))
    stamps: list[tuple[float, float]] = []
    lock = threading.Lock()

    def one(job):
        key, ids, mt = job
        barrier.wait()
        t0 = time.monotonic()
        resp, wall = api.complete(ids, mt, prefix_cache=False)
        t1 = time.monotonic()
        with lock:
            stamps.append((t0, t1))
        return key, reply(ids, resp, wall)

    with ThreadPoolExecutor(len(jobs)) as pool:
        out = list(pool.map(one, jobs))
    for _, r in out:
        if r["cached_tokens"]:
            fail(f"a cold concurrent request was served {r['cached_tokens']} tokens from the prefix cache")
    return out, max(t1 for _, t1 in stamps) - min(t0 for t0, _ in stamps)


# ── pin ──
def reply_key(r: dict) -> dict:
    return {"prompt_sha256": r["prompt_sha256"], "generated_ids": r["generated_ids"], "finish_reason": r["finish_reason"]}


def first_diff(name: str, got: dict, want: dict) -> str | None:
    """None when the reply equals the pinned one byte for byte, else a one-line account of the first difference."""
    if got["prompt_sha256"] != want["prompt_sha256"]:
        return f"{name}: the prompt differs from the pinned one (prompt set or tokenizer changed)"
    a, b = got["generated_ids"], want["generated_ids"]
    if a == b and got["finish_reason"] == want["finish_reason"]:
        return None
    i = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return (f"{name}: generated ids differ at {i} ({a[i:i + 3]} vs pinned {b[i:i + 3]}), lengths {len(a)}/{len(b)}, "
            f"finish {got['finish_reason']}/{want['finish_reason']}")


def pinned_name(key: str) -> str:
    """The reference entry a reply is held to: concurrent replies (`conc4/prose#0`) answer their solo prompt."""
    return "decode/" + key.split("/", 1)[1].split("#")[0] if key.startswith("conc") else key


def check_pinned(replies: dict[str, dict], ref: dict[str, dict]) -> list[str]:
    out = []
    for key, r in replies.items():
        name = pinned_name(key)
        if name not in ref:
            out.append(f"{key}: {name} is not in the pinned reference")
            continue
        d = first_diff(key, reply_key(r), ref[name])
        if d:
            out.append(d)
    return out


def extend_reference(ref: dict[str, dict], replies: dict[str, dict]) -> tuple[dict[str, dict], list[str]]:
    """The reference after a pin run: every entry already pinned must be reproduced byte for byte (else a difference is
    returned and the caller must not write), entries not pinned yet are added. Concurrent replies are never pinned: each was
    already held to the solo reply of its prompt."""
    merged, errs = dict(ref), []
    for key, r in replies.items():
        if key.startswith("conc"):
            continue
        if key in ref:
            d = first_diff(key, reply_key(r), ref[key])
            if d:
                errs.append(d)
        else:
            merged[key] = {**reply_key(r), "prompt_tokens": r["prompt_tokens"]}
    return merged, errs


# ── phases ──
def phase_warm(api: ia.Api, prompts: dict[str, list[int]]) -> None:
    jobs = [(n, ids, 128) for n, ids in prompts.items()]
    together(api, jobs)
    api.complete(long_ids(api, SIZES["32k"]), 16, prefix_cache=False)
    say("warm: done (discarded)")


def phase_decode(api, prompts, replies, metrics, raw) -> None:
    rates, spec = [], []
    for name, ids in prompts.items():
        r = solo(api, ids, DECODE_TOKENS)
        replies[f"decode/{name}"] = r
        raw[f"decode/{name}"] = r
        rates.append(r["decode_tok_s"])
        spec.append(r)
        metrics[f"decode_{name}_tok_s"] = r["decode_tok_s"]
        say(f"decode {name}: {r['completion_tokens']} tokens, ttft {r['ttft_s']} s, {r['decode_tok_s']} tok/s, "
            f"spec {r['spec']}")
    metrics["decode_c1_tok_s"] = round(statistics.fmean(rates), 3)
    d = sum((r["spec"] or {}).get("drafted", 0) for r in spec)
    acc = sum((r["spec"] or {}).get("accepted", 0) for r in spec)
    rounds = sum((r["spec"] or {}).get("rounds", 0) for r in spec)
    n = sum(r["completion_tokens"] for r in spec)
    if d:
        metrics["decode_acceptance"] = round(acc / d, 4)
    if rounds:
        metrics["decode_tokens_per_round"] = round(n / rounds, 4)
    metrics["decode_tokens"] = n


def phase_conc(api, prompts, replies, metrics, raw, pin_mode: bool) -> None:
    for c in (4, 8):
        jobs = [(n, ids, DECODE_TOKENS) for _ in range(c // len(prompts)) for n, ids in prompts.items()]
        out, span = together(api, jobs)
        n_tok = sum(r["completion_tokens"] for _, r in out)
        metrics[f"decode_c{c}_tok_s"] = round(n_tok / span, 3)
        raw[f"conc{c}"] = {"span_s": round(span, 3), "replies": {f"{k}#{i}": r for i, (k, r) in enumerate(out)}}
        say(f"conc c={c}: {n_tok} tokens in {span:.2f} s = {n_tok / span:.2f} tok/s")
        for i, (name, r) in enumerate(out):
            solo_r = replies.get(f"decode/{name}")
            if solo_r is not None:
                d = first_diff(f"conc{c}/{name}#{i} vs solo", reply_key(r), reply_key(solo_r))
                if d:
                    fail(f"concurrent != solo: {d}")
            elif pin_mode:
                fail("a pin run needs the decode phase before conc (concurrent == solo)")
            replies.setdefault(f"conc{c}/{name}#{i}", r)


def phase_long(api, label: str, replies, metrics, raw) -> None:
    n = SIZES[label]
    r = solo(api, long_text(api, n) if label == "1m" else long_ids(api, n), LONG_TOKENS)
    replies[f"prefill/{label}"] = r
    raw[f"prefill/{label}"] = r
    metrics[f"prefill_{label}_s"] = r["ttft_s"]
    metrics[f"prefill_{label}_tok_s"] = round(r["prompt_tokens"] / r["ttft_s"], 1)
    metrics[f"decode_ctx{label}_tok_s"] = r["decode_tok_s"]
    say(f"prefill {label}: {r['prompt_tokens']} tokens in {r['ttft_s']} s = {r['prompt_tokens'] / r['ttft_s']:.0f} tok/s; "
        f"then {r['completion_tokens']} tokens at {r['decode_tok_s']} tok/s")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ia.add_api_args(ap)
    ap.add_argument("--phases", default=DEFAULT_PHASES)
    ap.add_argument("--out", help="write every reply and timing here (JSON)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ref", help="compare every reply with this pinned reference")
    g.add_argument("--pin", help="pin the replies of this run: write the reference, or extend an existing one (see above)")
    ap.add_argument("--engine", default="", help="engine commit, stored in a pin's meta")
    a = ap.parse_args(argv)
    phases = [p for p in a.phases.split(",") if p]
    bad = [p for p in phases if p not in PHASES]
    if bad:
        raise SystemExit(f"unknown phase(s) {bad}; known: {', '.join(PHASES)}")
    ref, existing, old_meta = None, {}, None
    if a.ref:
        with open(a.ref) as f:
            ref = json.load(f)["replies"]
    elif os.path.exists(a.pin):
        with open(a.pin) as f:
            old = json.load(f)
        existing, old_meta = old["replies"], old["meta"]

    api = ia.api_from_args(a)
    prompts = {n: api.tokenize(t) for n, t in DECODE_TEXTS.items()}
    replies: dict[str, dict] = {}
    metrics: dict[str, float] = {}
    raw: dict = {}
    if "warm" in phases:
        phase_warm(api, prompts)
    if "decode" in phases:
        phase_decode(api, prompts, replies, metrics, raw)
    if "conc" in phases:
        phase_conc(api, prompts, replies, metrics, raw, bool(a.pin))
    for label in ("32k", "128k", "256k"):
        if "prefill" in phases or (label == "32k" and "prefill-32k" in phases):
            phase_long(api, label, replies, metrics, raw)
    if "prefill-1m" in phases:
        phase_long(api, "1m", replies, metrics, raw)

    if a.pin:
        # the reference holds the solo and long replies; a concurrent reply was already required to equal its solo one.
        # A pinned entry must be reproduced byte for byte; an entry not pinned yet is added (a new phase's first pin).
        before = len(existing)
        reproduced = sum(1 for k in replies if k in existing)
        pinned, errs = extend_reference(existing, replies)
        for e in errs:
            fail(f"pin run: {e}")
        added = sorted(set(pinned) - set(existing))
        if not FAILS:
            meta = old_meta or {"engine": a.engine, "decode_tokens": DECODE_TOKENS, "long_tokens": LONG_TOKENS,
                                "prompt_version": PROMPT_VERSION, "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            if added and before:
                meta.setdefault("extended", []).append({"engine": a.engine, "added": added,
                                                       "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            with open(a.pin, "w") as f:
                json.dump({"meta": meta, "replies": pinned}, f, indent=1)
                f.write("\n")
            say(f"{a.pin}: {len(pinned)} replies pinned, added {added or 'none'}; {reproduced} already pinned ones reproduced")
    else:
        # a concurrent reply answers the same prompt as its solo one, so it is held to that pinned reply
        for e in check_pinned(replies, ref):
            fail(e)
    for k in sorted(metrics):
        print(f"METRIC {k}={metrics[k]}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"phases": phases, "metrics": metrics, "raw": raw, "failures": FAILS}, f, indent=1)
            f.write("\n")
    if not a.pin:
        print("PASS pin" if not FAILS else "FAIL pin")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
