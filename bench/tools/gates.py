#!/usr/bin/env python3
"""Correctness gates against rank 0's API (stdlib only): drafted == serial, cached == uncached, vision records.

Every generation is greedy (temperature 0) on /v1/completions with token-id prompts (`return_token_ids`), compared by
generated token ids (the EOS of a `stop` answer included). The prompt set is exactness.py's (short ~30, medium ~400,
long > 1024, xlong > 5000 tokens) plus `70k`, more than 64K tokens of filler (inkling_api.filler_sentences, sized on the
server's /tokenize as needle.py does). Request fields (crates/server/src/api.rs): `"draft": false` is the serial
reference path (no speculation), `"prefix_cache": false` opts a request out of the prefix cache (nothing looked up,
nothing stored). Evidence read from each reply's `usage`: `spec` {drafted, accepted, rounds} and
`prompt_tokens_details.cached_tokens`; /metrics `rexl3_prefix_cache_*` counters around each cache request.

  gates.py spec OUT.json [--max-tokens 128] [--serial-boot]
      per prompt: the serial reference ("draft": false, "prefix_cache": false) and the drafted reply ("prefix_cache":
      false, drafting on); then short/medium/long/xlong drafted all four at once vs their solo serial references. Pass:
      identical ids for every pair, every serial reply drafted 0 tokens, and some drafted reply drafted tokens (with
      --serial-boot, for a --spec none boot: none did). Records usage.spec per reply, acceptance (accepted / drafted) and
      tokens per round. Exit 1 on any failure.

  gates.py cache OUT.json [--max-tokens 128] [--min-cut 512] [--nonce TEXT]
      per prompt (each opened with a per-run nonce line, so nothing of it is cached yet): the uncached reference
      ("prefix_cache": false), then the same prompt twice with the cache (cold, warm); then two pairs of prompts sharing
      a prefix (> 1024 tokens and > 64K tokens) with different tails: both uncached, then both cached. Pass: identical
      ids to the reference for every cached reply; 0 cached tokens on every uncached and cold reply; the warm reply and
      the second shared-prefix reply served from the cache (cached_tokens > 0, at most the shared length) whenever the
      prompt reaches the cache's smallest cut (--min-cut, the engine's CutPolicy default 512). Exit 1 on any failure.

  gates.py vision OUT.json [--max-tokens 64] [--no-images]
      the synthetic images (synth_images.py: red square, blue circle, green|yellow halves; chat, reasoning_effort
      "none") and exactness.py's text prompts: replies, token ids, usage. --no-images on a VISION=off boot records the
      text part alone. Exit 1 when a request fails.

  gates.py diff A.json B.json
      two vision records (two boots) byte-exact: the text replies always, the image replies when both records hold
      them (a VISION=off record has none: then text alone, said so). Exit 1 on any difference.

  gates.py spec-diff A.json B.json
      the drafted replies (solo and concurrent) of two spec records, e.g. a --spec dspark boot vs a --spec none boot.
      Exit 1 on any difference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exactness as ex  # noqa: E402
import inkling_api as ia  # noqa: E402
import synth_images as si  # noqa: E402

BIG_MIN = 65536             # the 70k prompt (and the long shared prefix) must exceed this many tokens
OVER_1K = 1024              # the long prompt (and the short shared prefix) must exceed this many tokens
CONCURRENT = ("short", "medium", "long", "xlong")
PREFIX_COUNTERS = ("rexl3_prefix_cache_lookups_total", "rexl3_prefix_cache_hits_total",
                   "rexl3_prefix_cache_tokens_saved_total", "rexl3_prefix_cache_inserts_total")


# ── prompts ──
def big_text(api: ia.Api, tokens: int, seed: int, intro: str, question: str) -> str:
    """Filler of more than `tokens` tokens (server-tokenized) between `intro` and `question`."""
    per = len(api.tokenize(" ".join(ia.filler_sentences(400, seed + 99)))) / 400
    n = int((tokens - len(api.tokenize(intro + question))) / per * 1.02) + 1
    while True:
        text = intro + " ".join(ia.filler_sentences(n, seed)) + question
        got = len(api.tokenize(text))
        if got > tokens:
            return text
        n += int((tokens - got) / per) + 16


def texts(api: ia.Api, big_tokens: int, header: str = "") -> dict[str, str]:
    out = {k: header + v for k, v in ex.TEXTS.items()}
    out["70k"] = header + big_text(api, big_tokens, 14, "Archive transcript follows.\n\n",
                                   "\n\nWhich three people appear most often?")
    return out


def tokenize_all(api: ia.Api, ts: dict[str, str]) -> dict[str, list[int]]:
    ps = {k: api.tokenize(v) for k, v in ts.items()}
    for name, least in (("long", OVER_1K), ("xlong", ex.LONG_MIN), ("70k", BIG_MIN)):
        if len(ps[name]) <= least:
            raise SystemExit(f"the {name} prompt is {len(ps[name])} tokens, not > {least}")
    return ps


def digest(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def gen(api: ia.Api, ids: list[int], max_tokens: int, **fields) -> dict:
    resp, dt = api.complete(ids, max_tokens, **fields)
    g, finish = ia.generated_ids(resp)
    u = resp.get("usage") or {}
    return {"generated_ids": g, "finish_reason": finish, "seconds": round(dt, 3), "usage": u,
            "spec": ia.spec_counters(u), "cached_tokens": ia.cached_tokens(u), "fields": fields}


def first_diff(a: dict, b: dict) -> str | None:
    ga, gb = a["generated_ids"], b["generated_ids"]
    if ga == gb and a["finish_reason"] == b["finish_reason"]:
        return None
    i = next((j for j, (x, y) in enumerate(zip(ga, gb)) if x != y), min(len(ga), len(gb)))
    return f"ids differ at {i} ({ga[i:i + 3]} vs {gb[i:i + 3]}), lengths {len(ga)}/{len(gb)}, " \
           f"finish {a['finish_reason']}/{b['finish_reason']}"


def meta(api: ia.Api, a, kind: str, **more) -> dict:
    status, models, _ = api.call("GET", "/v1/models", timeout=30)
    return {"kind": kind, "url": api.url, "model": api.model, "label": a.label, "max_tokens": a.max_tokens,
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "models": models, **more}


def write(path: str, rec: dict) -> None:
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")


def load(path: str, kind: str) -> dict:
    with open(path) as f:
        rec = json.load(f)
    if rec.get("meta", {}).get("kind") != kind:
        raise SystemExit(f"{path}: not a gates.py {kind} record (kind {rec.get('meta', {}).get('kind')!r})")
    return rec


def say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def totals(replies: list[dict]) -> dict:
    d = sum((r["spec"] or {}).get("drafted", 0) for r in replies)
    acc = sum((r["spec"] or {}).get("accepted", 0) for r in replies)
    rounds = sum((r["spec"] or {}).get("rounds", 0) for r in replies)
    n = sum(int(r["usage"].get("completion_tokens", 0)) for r in replies)
    return {"replies": len(replies), "drafted": d, "accepted": acc, "rounds": rounds, "completion_tokens": n,
            "acceptance": round(acc / d, 4) if d else None, "tokens_per_round": round(n / rounds, 4) if rounds else None}


# ── spec ──
def cmd_spec(a) -> int:
    api = ia.api_from_args(a)
    ps = tokenize_all(api, texts(api, a.big_tokens))
    serial_f, drafted_f = {"draft": False, "prefix_cache": False}, {"draft": True, "prefix_cache": False}
    replies, errs = [], []
    for name, ids in ps.items():
        s = gen(api, ids, a.max_tokens, **serial_f)
        d = gen(api, ids, a.max_tokens, **drafted_f)
        diff = first_diff(s, d)
        replies.append({"name": name, "prompt_tokens": len(ids), "prompt_sha256": digest(ids), "serial": s,
                        "drafted": d, "equal": diff is None, "difference": diff})
        sp = d["spec"] or {}
        say(f"{name:7s} ({len(ids)} tokens): {'equal' if diff is None else 'DIFFERS: ' + diff}; drafted "
            f"{sp.get('drafted')} accepted {sp.get('accepted')} rounds {sp.get('rounds')} acceptance {sp.get('acceptance')} "
            f"tokens/round {sp.get('tokens_per_round')}")
        if diff:
            errs.append(f"{name}: drafted != serial: {diff}")
    ref = {r["name"]: r["serial"] for r in replies}
    barrier = threading.Barrier(len(CONCURRENT))

    def one(name):
        barrier.wait()
        return gen(api, ps[name], a.max_tokens, **drafted_f)

    with ThreadPoolExecutor(len(CONCURRENT)) as pool:
        futs = {n: pool.submit(one, n) for n in CONCURRENT}
        conc = {n: futs[n].result() for n in CONCURRENT}
    concurrent = []
    for n in CONCURRENT:
        diff = first_diff(ref[n], conc[n])
        concurrent.append({"name": n, "prompt_tokens": len(ps[n]), "prompt_sha256": digest(ps[n]), "drafted": conc[n],
                           "equal": diff is None, "difference": diff})
        say(f"concurrent {n:7s}: {'equal' if diff is None else 'DIFFERS: ' + diff}")
        if diff:
            errs.append(f"concurrent {n}: drafted (4 at once) != solo serial: {diff}")
    for r in replies:
        if r["serial"]["spec"] is None:
            errs.append(f"{r['name']}: no usage.spec in the serial reply (the server cannot prove its path)")
        elif r["serial"]["spec"]["drafted"] != 0:
            errs.append(f"{r['name']}: the serial reply (\"draft\": false) drafted {r['serial']['spec']['drafted']} tokens")
    drafted = [r["drafted"] for r in replies] + [c["drafted"] for c in concurrent]
    t = totals(drafted)
    if a.serial_boot and t["drafted"]:
        errs.append(f"--serial-boot, but the drafted replies drafted {t['drafted']} tokens")
    if not a.serial_boot and not t["drafted"]:
        errs.append("no drafted reply drafted a token: drafting is off (a --spec none boot needs --serial-boot)")
    rec = {"meta": meta(api, a, "spec", serial_boot=a.serial_boot, big_tokens=a.big_tokens,
                        prompt_tokens={n: len(v) for n, v in ps.items()}),
           "replies": replies, "concurrent": concurrent,
           "acceptance": {"drafted_solo": totals([r["drafted"] for r in replies]),
                          "drafted_concurrent": totals([c["drafted"] for c in concurrent]), "drafted_all": t,
                          "serial": totals([r["serial"] for r in replies])},
           "errors": errs, "pass": not errs}
    write(a.out, rec)
    for e in errs:
        say(f"FAIL {e}")
    say(f"spec: {'pass' if not errs else 'FAIL'}; drafted {t['drafted']} accepted {t['accepted']} rounds {t['rounds']} "
        f"acceptance {t['acceptance']} tokens/round {t['tokens_per_round']} -> {a.out}")
    return 0 if not errs else 1


def cmd_spec_diff(a) -> int:
    ra, rb = load(a.a, "spec"), load(a.b, "spec")
    bad = 0
    for part in ("replies", "concurrent"):
        by_b = {r["name"]: r for r in rb[part]}
        for r in ra[part]:
            o = by_b.get(r["name"])
            if o is None:
                say(f"{part}/{r['name']}: only in {a.a}")
                bad += 1
            elif r["prompt_sha256"] != o["prompt_sha256"]:
                say(f"{part}/{r['name']}: different prompts (tokenizer or prompt set changed)")
                bad += 1
            else:
                d = first_diff(r["drafted"], o["drafted"])
                say(f"{part}/{r['name']}: {'identical' if d is None else 'DIFFERS: ' + d}")
                bad += d is not None
        for n in sorted(set(by_b) - {r["name"] for r in ra[part]}):
            say(f"{part}/{n}: only in {a.b}")
            bad += 1
    for path, rec in ((a.a, ra), (a.b, rb)):
        t = rec["acceptance"]["drafted_all"]
        say(f"{path}: label {rec['meta'].get('label')!r}, drafted {t['drafted']} accepted {t['accepted']} "
            f"acceptance {t['acceptance']} tokens/round {t['tokens_per_round']}")
    print("identical" if not bad else f"DIFFERENT ({bad})")
    return 0 if not bad else 1


# ── cache ──
class Probe:
    """The /metrics prefix counters around one request (None when /metrics cannot be read)."""

    def __init__(self, api: ia.Api):
        self.api = api

    def snap(self):
        try:
            m = self.api.metrics()
        except (ia.ApiError, OSError):
            return None
        return {k: m.get(k) for k in PREFIX_COUNTERS + ("rexl3_prefix_cache_enabled",)}

    def run(self, ids, max_tokens, **fields):
        before = self.snap()
        r = gen(self.api, ids, max_tokens, **fields)
        after = self.snap()
        if before and after:
            r["metrics_delta"] = {k.replace("rexl3_prefix_cache_", ""): (after[k] - before[k])
                                  if after[k] is not None and before[k] is not None else None for k in PREFIX_COUNTERS}
            r["cache_enabled"] = after.get("rexl3_prefix_cache_enabled")
        else:
            r["metrics_delta"] = None
        return r


def common_prefix(x: list[int], y: list[int]) -> int:
    return next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))


def cmd_cache(a) -> int:
    api = ia.api_from_args(a)
    probe = Probe(api)
    nonce = a.nonce or f"{time.time_ns():x}"
    header = f"Session {nonce}.\n\n"
    ps = tokenize_all(api, texts(api, a.big_tokens, header))
    errs, entries = [], []
    uncached, cached = {"prefix_cache": False}, {"prefix_cache": True}

    def zero(name, what, r):
        if r["cached_tokens"] is None:
            errs.append(f"{name} {what}: the reply reports no usage.prompt_tokens_details.cached_tokens")
        elif r["cached_tokens"] != 0:
            errs.append(f"{name} {what}: {r['cached_tokens']} cached tokens (expected a full prefill)")

    def hit(name, what, r, n_prompt, most):
        if n_prompt < a.min_cut:
            r["hit_expected"] = False   # below the smallest cut nothing is stored
            return
        r["hit_expected"] = True
        c = r["cached_tokens"]
        if c is None:
            errs.append(f"{name} {what}: the reply reports no cached_tokens")
        elif not 0 < c <= most:
            errs.append(f"{name} {what}: {c} cached tokens (expected 1..{most}: served from the cache)")

    def same(name, what, ref, r):
        d = first_diff(ref, r)
        if d:
            errs.append(f"{name} {what} != uncached: {d}")
        return d is None

    for name, ids in ps.items():
        ref = probe.run(ids, a.max_tokens, **uncached)
        cold = probe.run(ids, a.max_tokens, **cached)
        warm = probe.run(ids, a.max_tokens, **cached)
        zero(name, "uncached", ref)
        zero(name, "cold", cold)
        hit(name, "warm", warm, len(ids), len(ids))
        eq = same(name, "cold", ref, cold) & same(name, "warm", ref, warm)
        entries.append({"name": name, "prompt_tokens": len(ids), "prompt_sha256": digest(ids), "uncached": ref,
                        "cold": cold, "warm": warm, "equal": eq})
        say(f"{name:7s} ({len(ids)} tokens): {'equal' if eq else 'DIFFERS'}; cached tokens uncached "
            f"{ref['cached_tokens']} cold {cold['cached_tokens']} warm {warm['cached_tokens']}")

    shared = []
    for label, tokens, seed in (("shared-1k", 2000, 21), ("shared-70k", a.big_tokens, 22)):
        prefix = header + big_text(api, tokens, seed, "Shared archive follows.\n\n", "\n\n")
        tails = ["Which three items were weighed most often?", "Name two people who worked near the river gate."]
        ids = [api.tokenize(prefix + t) for t in tails]
        n_shared = common_prefix(ids[0], ids[1])
        least = OVER_1K if label == "shared-1k" else BIG_MIN
        if n_shared <= least:
            raise SystemExit(f"{label}: the shared prefix is {n_shared} tokens, not > {least}")
        refs = [probe.run(x, a.max_tokens, **uncached) for x in ids]
        first = probe.run(ids[0], a.max_tokens, **cached)
        second = probe.run(ids[1], a.max_tokens, **cached)
        for i, r in enumerate(refs):
            zero(label, f"tail {i} uncached", r)
        zero(label, "tail 0 cold", first)
        hit(label, "tail 1 (shared prefix cached)", second, n_shared, n_shared)
        eq = same(label, "tail 0 cached", refs[0], first) & same(label, "tail 1 cached", refs[1], second)
        shared.append({"name": label, "shared_tokens": n_shared, "prompt_tokens": [len(x) for x in ids],
                       "prompt_sha256": [digest(x) for x in ids], "uncached": refs, "cached": [first, second],
                       "equal": eq})
        say(f"{label} (shared {n_shared} of {len(ids[0])}/{len(ids[1])} tokens): {'equal' if eq else 'DIFFERS'}; "
            f"cached tokens tail 0 {first['cached_tokens']} tail 1 {second['cached_tokens']}")
    status, models, _ = api.call("GET", "/v1/models", timeout=30)
    reach = models["data"][0].get("prefix_cache_reach_tokens") if status == 200 and isinstance(models, dict) else None
    rec = {"meta": meta(api, a, "cache", nonce=nonce, min_cut=a.min_cut, big_tokens=a.big_tokens,
                        prefix_cache_reach_tokens=reach, prompt_tokens={n: len(v) for n, v in ps.items()}),
           "prompts": entries, "shared": shared, "errors": errs, "pass": not errs}
    write(a.out, rec)
    for e in errs:
        say(f"FAIL {e}")
    say(f"cache: {'pass' if not errs else 'FAIL'} (prefix_cache_reach_tokens {reach}) -> {a.out}")
    return 0 if not errs else 1


# ── vision ──
def image_reply(api: ia.Api, name: str, max_tokens: int) -> dict:
    _, question, words = si.IMAGES[name]
    png = si.image_png(name)
    msgs = [{"role": "user", "content": [{"type": "text", "text": question},
                                         {"type": "image_url", "image_url": {"url": si.data_url(png)}}]}]
    resp, dt = api.chat(msgs, max_tokens, reasoning_effort="none")
    ch = resp["choices"][0]
    msg = ch.get("message", {})
    content, reasoning = msg.get("content") or "", msg.get("reasoning_content") or ""
    if ch.get("token_ids") is None:
        raise ia.ApiError(200, resp, "/v1/chat/completions (no token_ids)")
    return {"name": name, "png_sha256": hashlib.sha256(png).hexdigest(), "question": question,
            "token_ids": [int(t) for t in ch["token_ids"]], "finish_reason": ch.get("finish_reason"),
            "content": content, "reasoning": reasoning, "usage": resp.get("usage"), "seconds": round(dt, 3),
            "names_colour": all(w in (content + " " + reasoning).lower() for w in words)}


def cmd_vision(a) -> int:
    api = ia.api_from_args(a)
    images = []
    if not a.no_images:
        for name in si.IMAGES:
            r = image_reply(api, name, a.max_tokens)
            images.append(r)
            say(f"image {name}: {len(r['token_ids'])} tokens, finish {r['finish_reason']}, names its colour: "
                f"{r['names_colour']}; content {r['content'][:120]!r}")
    text = []
    for name, prompt in ex.TEXTS.items():
        ids = api.tokenize(prompt)
        g = gen(api, ids, a.max_tokens)
        text.append({"name": name, "prompt_tokens": len(ids), "prompt_sha256": digest(ids),
                     "generated_ids": g["generated_ids"], "finish_reason": g["finish_reason"], "usage": g["usage"]})
        say(f"text {name} ({len(ids)} tokens): {len(g['generated_ids'])} ids, finish {g['finish_reason']}")
    rec = {"meta": meta(api, a, "vision", images=not a.no_images), "images": images, "text": text}
    write(a.out, rec)
    say(f"vision: recorded {len(images)} image and {len(text)} text replies -> {a.out}")
    return 0


def image_key(r: dict) -> dict:
    return {k: r[k] for k in ("png_sha256", "question", "token_ids", "finish_reason", "content", "reasoning")}


def text_key(r: dict) -> dict:
    return {k: r[k] for k in ("prompt_sha256", "generated_ids", "finish_reason")}


def compare(part: str, xs: list[dict], ys: list[dict], key, a_name: str, b_name: str) -> int:
    """Byte-exact comparison of two lists of replies by name; prints each verdict, returns the number of differences."""
    by_y, bad = {r["name"]: r for r in ys}, 0
    for r in xs:
        o = by_y.get(r["name"])
        if o is None:
            say(f"{part}/{r['name']}: only in {a_name}")
            bad += 1
            continue
        ka, kb = key(r), key(o)
        if ex.canon(ka) == ex.canon(kb):
            say(f"{part}/{r['name']}: identical")
            continue
        bad += 1
        diffs = [k for k in ka if ex.canon(ka[k]) != ex.canon(kb[k])]
        ida, idb = ka.get("token_ids", ka.get("generated_ids")), kb.get("token_ids", kb.get("generated_ids"))
        at = common_prefix(ida, idb)
        say(f"{part}/{r['name']}: DIFFERS in {diffs}; ids first differ at {at} ({ida[at:at + 3]} vs {idb[at:at + 3]})")
    for n in sorted(set(by_y) - {r["name"] for r in xs}):
        say(f"{part}/{n}: only in {b_name}")
        bad += 1
    return bad


def cmd_diff(a) -> int:
    ra, rb = load(a.a, "vision"), load(a.b, "vision")
    bad = compare("text", ra["text"], rb["text"], text_key, a.a, a.b)
    ia_, ib_ = ra["meta"].get("images"), rb["meta"].get("images")
    if ia_ and ib_:
        bad += compare("image", ra["images"], rb["images"], image_key, a.a, a.b)
    else:
        say(f"images: not compared ({a.a} {'has' if ia_ else 'has no'} images, {a.b} {'has' if ib_ else 'has no'} "
            f"images: a --no-images (VISION=off) record); text compared alone")
    print("identical" if not bad else f"DIFFERENT ({bad})")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, mt in (("spec", 128), ("cache", 128), ("vision", 64)):
        p = sub.add_parser(name)
        ia.add_api_args(p)
        p.add_argument("out")
        p.add_argument("--max-tokens", type=int, default=mt, help="per reply (default: %(default)s)")
        p.add_argument("--label", default="", help="free text stored in meta (boot, SPEC, VISION, ...)")
        if name != "vision":
            p.add_argument("--big-tokens", type=int, default=70_000,
                           help=f"size of the 70k prompt and the long shared prefix (> {BIG_MIN}; default: %(default)s)")
    sub.choices["spec"].add_argument("--serial-boot", action="store_true",
                                     help="the boot runs --spec none: require that no drafted reply drafted a token")
    sub.choices["cache"].add_argument("--min-cut", type=int, default=512,
                                      help="the cache's smallest cut: a shorter prompt is never a hit (default: %(default)s)")
    sub.choices["cache"].add_argument("--nonce", help="the per-run line opening every prompt (default: the time)")
    sub.choices["vision"].add_argument("--no-images", action="store_true", help="text only (a VISION=off boot)")
    for name in ("diff", "spec-diff"):
        p = sub.add_parser(name)
        p.add_argument("a")
        p.add_argument("b")
    a = ap.parse_args(argv)
    if getattr(a, "big_tokens", BIG_MIN + 1) <= BIG_MIN:
        raise SystemExit(f"--big-tokens must exceed {BIG_MIN}")
    return {"spec": cmd_spec, "cache": cmd_cache, "vision": cmd_vision, "diff": cmd_diff,
            "spec-diff": cmd_spec_diff}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
