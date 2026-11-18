#!/usr/bin/env python3
"""Needle in a haystack against rank 0's chat API (stdlib only; the `tokenizers` package is used if --tokenizer is
given and it is installed).

Builds a long prompt of varied filler sentences (deterministic per seed) with one pass-code sentence at a given depth,
sized to --tokens prompt tokens (must stay below --max-model-len minus the answer), asks for the pass-code via
/v1/chat/completions (streamed, temperature 0) and reports whether the answer holds it, the prompt tokens and the
prefill seconds (time to the first streamed token). Exit 0 when every lane found its needle.

    needle.py --tokens 200000 --depth 0.5
    needle.py --tokens 1000000 --depth 0.25
    needle.py --tokens 1000000 --lanes 4          # four concurrent long requests, needles at depths 0.2 .. 0.8:
                                                  # reports for each lane admitted / answered / refused (and how)

Sizing: tokens per filler sentence are calibrated on a sample (the pack tokenizer from --tokenizer, else the server's
/tokenize), and the exact rendered size comes from /v1/chat/completions/render when the server accepts the body; the
prompt is trimmed until it fits. A huge body may need the server's --max-body-mib / --max-rendered-prompt-mib raised
(EXTRA_FLAGS); a refusal is reported with the server's message.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling_api as ia  # noqa: E402

QUESTION = "What is the secret pass-code stated in the archive above? Answer with the pass-code only."
INTRO = "Below is a long archive of notes. Exactly one note states a secret pass-code.\n\n"
WORDS = ["AMBER", "COBALT", "FALCON", "JUNIPER", "MAGENTA", "NEBULA", "OTTER", "QUARTZ", "SAFFRON", "TUNDRA", "VIOLET"]


class Counter:
    def __init__(self, api: ia.Api, tokenizer_path: str | None):
        self.api, self.tok = api, None
        if tokenizer_path:
            from tokenizers import Tokenizer  # optional dependency, only with --tokenizer
            self.tok = Tokenizer.from_file(tokenizer_path)

    def count(self, text: str) -> int:
        if self.tok is not None:
            return len(self.tok.encode(text, add_special_tokens=False).ids)
        return len(self.api.tokenize(text))


def build(lane: int, depth: float, n_sent: int, seed: int) -> tuple[str, str]:
    rng = random.Random(seed * 1000 + lane)
    code = f"{rng.choice(WORDS)}-{rng.randint(1000, 9999)}-{rng.choice(WORDS)}"
    sents = ia.filler_sentences(n_sent, seed * 1000 + lane)
    at = min(len(sents), max(0, round(depth * len(sents))))
    sents.insert(at, f"Important notice: the secret pass-code is {code}. Keep it safe.")
    return INTRO + " ".join(sents) + "\n\n" + QUESTION, code


def messages(system: str, user: str) -> list[dict]:
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]


def rendered_count(api: ia.Api, system: str, user: str) -> int | None:
    status, out, _ = api.call("POST", "/v1/chat/completions/render", {"model": api.model,
                                                                      "messages": messages(system, user)})
    return int(out["count"]) if status == 200 and isinstance(out, dict) and "count" in out else None


def size_lane(api, counter, a, lane: int, depth: float, per_sent: float) -> dict:
    limit = a.max_model_len - a.max_tokens - 16
    target = min(a.tokens, limit)
    overhead = counter.count(INTRO + QUESTION) + 64
    n_sent = max(1, int((target - overhead) / per_sent))
    for _ in range(4):
        user, code = build(lane, depth, n_sent, a.seed)
        n = rendered_count(api, a.system, user)
        if n is None:  # the server refused the render (size limits): the calibrated estimate
            return {"user": user, "code": code, "sentences": n_sent, "prompt_tokens_est": int(n_sent * per_sent + overhead),
                    "sizing": "estimate"}
        if n <= limit:
            return {"user": user, "code": code, "sentences": n_sent, "prompt_tokens_rendered": n, "sizing": "render"}
        n_sent = int(n_sent - (n - limit) / per_sent * 1.02) - 1
    raise SystemExit(f"lane {lane}: could not fit the prompt under {limit} tokens")


def ask(api: ia.Api, a, lane: dict, start: threading.Barrier | None) -> None:
    body = {"model": api.model, "messages": messages(a.system, lane.pop("user")), "temperature": 0,
            "max_tokens": a.max_tokens, "stream_options": {"include_usage": True}}
    if start is not None:
        start.wait()
    t0 = time.monotonic()
    content, reasoning, usage, first, finish = [], [], None, None, None
    try:
        for t, ev in api.stream("/v1/chat/completions", body):
            lane.setdefault("admitted_s", round(t, 3))
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                c, r = d.get("content") or "", d.get("reasoning_content") or d.get("reasoning") or ""
                if (c or r) and first is None:
                    first = t
                content.append(c)
                reasoning.append(r)
                finish = ch.get("finish_reason") or finish
            if "error" in ev:
                lane["error"] = ev["error"]
    except ia.ApiError as e:
        lane.update(status="refused", http_status=e.status, error=e.body)
        return
    except Exception as e:  # network failure, timeout
        lane.update(status="error", error=f"{type(e).__name__}: {e}")
        return
    answer, think = "".join(content), "".join(reasoning)
    lane.update(status="answered" if answer.strip() and "error" not in lane else "empty",
                http_status=200, admitted=True, prefill_s=None if first is None else round(first, 3),
                total_s=round(time.monotonic() - t0, 3), finish_reason=finish, usage=usage,
                prompt_tokens=(usage or {}).get("prompt_tokens"), answer=answer, reasoning_tail=think[-300:],
                found=lane["code"] in answer, found_in_reasoning=lane["code"] in think)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ia.add_api_args(ap)
    ap.add_argument("--tokens", type=int, default=200_000, help="prompt tokens per lane (default: %(default)s)")
    ap.add_argument("--depth", type=float, help="needle depth 0..1 (default 0.5; with --lanes N: spread 0.2..0.8)")
    ap.add_argument("--lanes", type=int, default=1, help="concurrent requests (default: %(default)s)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=1_048_576, help="the server's --max-model-len")
    ap.add_argument("--system", default="Thinking effort level: 0", help="system message ('' for none)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tokenizer", help="the pack's tokenizer.json (needs the tokenizers package); default /tokenize")
    ap.add_argument("--out", help="write the JSON report here (also printed)")
    a = ap.parse_args(argv)
    if a.lanes < 1:
        raise SystemExit("--lanes must be >= 1")
    api = ia.api_from_args(a)
    counter = Counter(api, a.tokenizer)
    sample = " ".join(ia.filler_sentences(2000, a.seed + 99))
    per_sent = counter.count(sample) / 2000
    if a.depth is not None:
        depths = [a.depth] * a.lanes
    elif a.lanes == 1:
        depths = [0.5]
    else:
        depths = [0.2 + 0.6 * i / (a.lanes - 1) for i in range(a.lanes)]
    lanes = []
    for i, d in enumerate(depths):
        lane = {"lane": i, "depth": round(d, 4), "target_tokens": a.tokens}
        lane.update(size_lane(api, counter, a, i, d, per_sent))
        lanes.append(lane)
        print(f"lane {i}: {lane['sentences']} sentences, "
              f"{lane.get('prompt_tokens_rendered') or '~%d' % lane['prompt_tokens_est']} prompt tokens, depth {d:.2f}",
              file=sys.stderr)
    t0 = time.monotonic()
    if a.lanes == 1:
        ask(api, a, lanes[0], None)
    else:
        start = threading.Barrier(a.lanes)
        threads = [threading.Thread(target=ask, args=(api, a, lane, start)) for lane in lanes]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    rep = {"url": api.url, "model": api.model, "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "tokens_per_sentence": round(per_sent, 4), "lanes": lanes, "wall_s": round(time.monotonic() - t0, 3),
           "all_admitted": all(l.get("admitted") for l in lanes),
           "all_answered": all(l.get("status") == "answered" for l in lanes),
           "pass": all(l.get("found") for l in lanes)}
    for l in lanes:
        print(f"lane {l['lane']}: {l.get('status')} found={l.get('found')} prompt_tokens={l.get('prompt_tokens')} "
              f"prefill_s={l.get('prefill_s')} total_s={l.get('total_s')}"
              f"{' error=' + str(l.get('error'))[:300] if l.get('error') else ''}", file=sys.stderr)
    txt = json.dumps(rep, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt + "\n")
    print(txt)
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
