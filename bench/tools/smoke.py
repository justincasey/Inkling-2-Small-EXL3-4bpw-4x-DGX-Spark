#!/usr/bin/env python3
"""Smoke test of rank 0's API (stdlib only): one chat completion and one raw completion, temperature 0.

Fails (exit 1) on a non-200 answer or an empty one. Prints each reply, its token ids and timing. The chat uses the
oracle fixture chat-capital-effort0's system line (oracle/prompts.jsonl), so its reply is comparable to the fixture.

    smoke.py [--url http://192.168.1.121:8000] [--model NAME]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling_api as ia  # noqa: E402

QUESTION = "What is the capital of France? Answer in one word."
SYSTEM = "Thinking effort level: 0"
RAW = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ia.add_api_args(ap)
    ap.add_argument("--max-tokens", type=int, default=32)
    a = ap.parse_args(argv)
    api = ia.api_from_args(a)
    ok = True

    status, out, dt = api.call("POST", "/v1/chat/completions", {
        "model": api.model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}],
        "temperature": 0, "max_tokens": a.max_tokens, "return_token_ids": True})
    if status != 200 or not isinstance(out, dict):
        print(f"FAIL chat: HTTP {status}: {str(out)[:500]}")
        ok = False
    else:
        ch = out["choices"][0]
        msg = ch.get("message", {})
        content, reasoning = msg.get("content") or "", msg.get("reasoning_content") or ""
        usage = out.get("usage", {})
        good = bool(content.strip() or reasoning.strip()) and usage.get("completion_tokens", 1) > 0
        print(f"{'ok  ' if good else 'FAIL'} chat ({dt:.2f}s, finish {ch.get('finish_reason')}, usage {json.dumps(usage)})")
        print(f"     content:   {content!r}")
        if reasoning:
            print(f"     reasoning: {reasoning!r}")
        print(f"     token_ids: {ch.get('token_ids')}")
        ok &= good

    status, out, dt = api.call("POST", "/v1/completions", {
        "model": api.model, "prompt": RAW, "max_tokens": 16, "temperature": 0, "return_token_ids": True})
    if status != 200 or not isinstance(out, dict):
        print(f"FAIL completion: HTTP {status}: {str(out)[:500]}")
        ok = False
    else:
        ch = out["choices"][0]
        text, ids = ch.get("text") or "", ch.get("token_ids")
        good = bool(text.strip() or ids)
        print(f"{'ok  ' if good else 'FAIL'} completion ({dt:.2f}s, finish {ch.get('finish_reason')}, "
              f"usage {json.dumps(out.get('usage', {}))})")
        print(f"     prompt:    {RAW!r}")
        print(f"     text:      {text!r}")
        print(f"     token_ids: {ids}")
        ok &= good
    print("smoke: pass" if ok else "smoke: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
