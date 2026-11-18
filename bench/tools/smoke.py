#!/usr/bin/env python3
"""Smoke test of rank 0's API (stdlib only), every request greedy (temperature 0).

Checks, each printed with its pass/fail, timing, finish reason and usage (usage.spec drafted/accepted/rounds when the
server reports it); exit 1 when any fails:

  chat         the oracle fixture chat-capital-effort0's system line (oracle/prompts.jsonl) and question: a 200 with a
               non-empty reply
  completion   a raw /v1/completions continuation: a 200 with text or token ids
  tool         one `tools` function (get_weather(city)) and a user turn that needs it: choices[0].message.tool_calls[0]
               names get_weather and its arguments parse as JSON with a non-empty `city`
  think-on     reasoning_effort "high": a non-empty reasoning_content, and neither content nor reasoning holds a framing
               marker (<|content_thinking|>, <|end_message|>, ...) nor the content the reasoning text
  think-off    reasoning_effort "none": a reply, no framing marker in content or reasoning, no reasoning text inside the
               content. Inkling 2 has no thinking switch (the effort is a template level) and reasons at level 0 too, so
               reasoning is allowed here; reasoning_tokens is recorded for both cases
  image        (not with --skip-image: a VISION=off boot refuses image parts) a synthetic 240x240 PNG, a red square on
               white, sent as a data URL image_url part: the reply (content or reasoning) names the colour red

    smoke.py [--url http://192.168.1.121:8000] [--model NAME] [--skip-image]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling_api as ia  # noqa: E402
import synth_images as si  # noqa: E402

QUESTION = "What is the capital of France? Answer in one word."
SYSTEM = "Thinking effort level: 0"
RAW = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,"
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city's name"}},
                   "required": ["city"]}}}]
TOOL_QUESTION = "What is the weather like in Paris right now? Use the get_weather tool."
THINK_QUESTION = "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?"
IMAGE = "red-square"


def framing_in(text: str) -> list[str]:
    return [m for m in ia.FRAMING if m in text]


def usage_line(out: dict) -> str:
    u = out.get("usage", {})
    s = ia.spec_counters(u)
    spec = "" if s is None else (f"; spec drafted={s['drafted']} accepted={s['accepted']} rounds={s['rounds']}"
                                 f" acceptance={s['acceptance']} tokens/round={s['tokens_per_round']}")
    return f"usage {json.dumps(u)}{spec}"


def report(name: str, good: bool, dt: float, out: dict, lines: list[str], why: str = "") -> bool:
    ch = out["choices"][0]
    print(f"{'ok  ' if good else 'FAIL'} {name} ({dt:.2f}s, finish {ch.get('finish_reason')}, {usage_line(out)})"
          f"{'' if good else ': ' + why}")
    for line in lines:
        print(f"     {line}")
    return good


def ask(api: ia.Api, name: str, path: str, body: dict):
    """(reply, seconds) or None after printing the failure."""
    status, out, dt = api.call("POST", path, dict(body, model=api.model))
    if status != 200 or not isinstance(out, dict) or not out.get("choices"):
        print(f"FAIL {name}: HTTP {status}: {str(out)[:500]}")
        return None
    return out, dt


def message(out: dict) -> tuple[dict, str, str]:
    msg = out["choices"][0].get("message", {})
    return msg, msg.get("content") or "", msg.get("reasoning_content") or ""


def check_chat(api, a) -> bool:
    r = ask(api, "chat", "/v1/chat/completions", {
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}],
        "temperature": 0, "max_tokens": a.max_tokens, "return_token_ids": True})
    if r is None:
        return False
    out, dt = r
    _, content, reasoning = message(out)
    good = bool(content.strip() or reasoning.strip()) and out.get("usage", {}).get("completion_tokens", 1) > 0
    lines = [f"content:   {content!r}"] + ([f"reasoning: {reasoning!r}"] if reasoning else [])
    return report("chat", good, dt, out, lines + [f"token_ids: {out['choices'][0].get('token_ids')}"], "empty reply")


def check_completion(api, a) -> bool:
    r = ask(api, "completion", "/v1/completions",
            {"prompt": RAW, "max_tokens": 16, "temperature": 0, "return_token_ids": True})
    if r is None:
        return False
    out, dt = r
    ch = out["choices"][0]
    text, ids = ch.get("text") or "", ch.get("token_ids")
    return report("completion", bool(text.strip() or ids), dt, out,
                  [f"prompt:    {RAW!r}", f"text:      {text!r}", f"token_ids: {ids}"], "empty reply")


def check_tool(api, a) -> bool:
    r = ask(api, "tool", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": TOOL_QUESTION}], "tools": TOOLS, "reasoning_effort": "none",
        "temperature": 0, "max_tokens": a.tool_max_tokens, "return_token_ids": True})
    if r is None:
        return False
    out, dt = r
    msg, content, reasoning = message(out)
    calls = msg.get("tool_calls") or []
    why, args = "", None
    if not calls:
        why = "no tool_calls"
    else:
        fn = calls[0].get("function", {})
        raw = fn.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError:
            why = f"arguments are not JSON: {raw!r}"
        if not why and fn.get("name") != "get_weather":
            why = f"called {fn.get('name')!r}, not get_weather"
        elif not why and not (isinstance(args, dict) and isinstance(args.get("city"), str) and args["city"].strip()):
            why = f"arguments {raw!r} hold no city"
    lines = [f"tool_calls: {json.dumps(calls)}", f"content:    {content!r}"] + ([f"reasoning:  {reasoning[-300:]!r}"]
                                                                             if reasoning else [])
    return report("tool", not why, dt, out, lines, why)


def think(api, a, effort: str):
    r = ask(api, f"think-{'on' if effort != 'none' else 'off'}", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": THINK_QUESTION}], "reasoning_effort": effort, "temperature": 0,
        "max_tokens": a.think_max_tokens, "return_token_ids": True})
    if r is None:
        return None
    out, dt = r
    _, content, reasoning = message(out)
    rt = ((out.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
    why = []
    if framing_in(content):
        why.append(f"content holds {framing_in(content)}")
    if framing_in(reasoning):
        why.append(f"reasoning holds {framing_in(reasoning)}")
    # a reasoning of a few characters ("9") may well recur in the answer; a leak is the reasoning text itself
    if len(reasoning.strip()) >= 16 and reasoning.strip() in content:
        why.append("the reasoning text is inside the content")
    lines = [f"reasoning_effort {effort!r}: reasoning_tokens={rt}, reasoning {len(reasoning)} chars, content "
             f"{len(content)} chars", f"content:   {content[:300]!r}", f"reasoning: {reasoning[:300]!r}"]
    return out, dt, content, reasoning, why, lines


def check_think_on(api, a) -> bool:
    r = think(api, a, "high")
    if r is None:
        return False
    out, dt, content, reasoning, why, lines = r
    if not reasoning.strip():
        why.insert(0, "empty reasoning_content")
    return report("think-on", not why, dt, out, lines, "; ".join(why))


def check_think_off(api, a) -> bool:
    r = think(api, a, "none")
    if r is None:
        return False
    out, dt, content, reasoning, why, lines = r
    if not (content.strip() or reasoning.strip()):
        why.insert(0, "empty reply")
    return report("think-off", not why, dt, out, lines, "; ".join(why))


def check_image(api, a) -> bool:
    _, question, words = si.IMAGES[IMAGE]
    url = si.data_url(si.image_png(IMAGE))
    r = ask(api, "image", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": [{"type": "text", "text": question},
                                                  {"type": "image_url", "image_url": {"url": url}}]}],
        "reasoning_effort": "none", "temperature": 0, "max_tokens": a.image_max_tokens, "return_token_ids": True})
    if r is None:
        return False
    out, dt = r
    _, content, reasoning = message(out)
    missing = [w for w in words if w not in (content + " " + reasoning).lower()]
    where = "content" if all(w in content.lower() for w in words) else "reasoning" if not missing else "neither"
    lines = [f"image {IMAGE} ({si.SIZE}x{si.SIZE} PNG, {len(url)} bytes as a data URL), colour words {words} in: {where}",
             f"content:   {content!r}"] + ([f"reasoning: {reasoning[:300]!r}"] if reasoning else [])
    return report("image", not missing, dt, out, lines, f"the reply does not name {missing}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ia.add_api_args(ap)
    ap.add_argument("--max-tokens", type=int, default=32, help="the chat check's max_tokens (default: %(default)s)")
    ap.add_argument("--tool-max-tokens", type=int, default=512)
    ap.add_argument("--think-max-tokens", type=int, default=1024)
    ap.add_argument("--image-max-tokens", type=int, default=256)
    ap.add_argument("--skip-image", action="store_true", help="no image check (a VISION=off boot)")
    a = ap.parse_args(argv)
    api = ia.api_from_args(a)
    checks = [check_chat, check_completion, check_tool, check_think_on, check_think_off]
    if not a.skip_image:
        checks.append(check_image)
    results = {c.__name__[6:].replace("_", "-"): c(api, a) for c in checks}
    if a.skip_image:
        print("skip image (--skip-image: VISION=off)")
    ok = all(results.values())
    summary = ", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in results.items())
    print(f"smoke: {'pass' if ok else 'FAIL'} ({summary})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
