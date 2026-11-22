#!/usr/bin/env python3
"""loop_probe.py run OUT.json | compare A.json B.json: how often greedy decoding falls into a repetition loop, and how far two engines' replies drift apart.

`run`: 32 prompts of many kinds (the harness's four decode prompts first, then prose, code, markup, data, chat-style and question prompts), each decoded greedily for
512 tokens at one request at a time against the running engine (token-id prompts through the server's tokenizer); stores every reply's ids and its speed.
`compare`: per prompt the first position where the replies differ, each side's distinct 4-gram ratio (1.0: no repeated 4-gram, near 0: a loop) and whether it
looped (ratio below 0.5); then the loop counts, the mean ratio, the mean tokens/s and the median first difference. A bit-identical engine pair shows no
difference; a bits-changing change (Segment B: int8 dense core) changes the greedy path of most prompts, and a path can fall into a loop the other does not.

    python3 scripts/loop_probe.py run .work/loop/bf16.json          # against http://127.0.0.1:8000 or INKLING2_URL
    python3 scripts/loop_probe.py compare .work/loop/bf16.json .work/loop/int8.json
"""

from __future__ import annotations

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench", "tools"))
import bench  # noqa: E402
import inkling_api as ia  # noqa: E402

N_TOKENS = 512
LOOP_BELOW = 0.5

EXTRA = [
    "Write a short story about a clockmaker who discovers that one of her clocks runs backwards.\n\nOnce upon a time, in a narrow street of old Prague,",
    "The history of the printing press begins in the fifteenth century. Johannes Gutenberg, a goldsmith from Mainz,",
    "Explain how a transformer language model computes attention, step by step, for a reader who knows linear algebra but no machine learning.\n\nStep 1:",
    "def fibonacci(n: int) -> int:\n    \"\"\"Return the n-th Fibonacci number using fast doubling.\"\"\"\n",
    "fn parse_header(input: &[u8]) -> Result<Header, ParseError> {\n    let mut cursor = Cursor::new(input);\n",
    "async function retry(fn, { attempts = 5, baseDelayMs = 100 } = {}) {\n  let lastError;\n  for (let i = 0; i < attempts; i++) {\n",
    "SELECT c.name, COUNT(o.id) AS orders, SUM(o.total) AS revenue\nFROM customers c\nJOIN orders o ON o.customer_id = c.id\nWHERE o.created_at >= DATE '2026-01-01'\n",
    "# Meeting notes: Q3 planning\n\nAttendees: Priya, Tomas, Wen, Imani\n\n## Agenda\n\n1. Roadmap review\n2.",
    "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"utf-8\">\n  <title>Order confirmation</title>\n</head>\n<body>\n  <h1>Thank you for your order</h1>\n  <table>\n",
    "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: search-indexer\n  labels:\n    app: search-indexer\nspec:\n  replicas: 3\n  template:\n",
    "timestamp,host,level,message\n2026-03-14T10:00:01Z,web-01,INFO,request completed in 42ms\n2026-03-14T10:00:02Z,web-02,WARN,slow query: 1.8s\n2026-03-14T10:00:03Z,web-01,",
    "{\n  \"name\": \"inventory-service\",\n  \"version\": \"2.4.1\",\n  \"dependencies\": {\n    \"express\": \"^4.18.2\",\n",
    "Subject: Follow-up on the supplier contract\n\nHi Elena,\n\nThank you for the call yesterday. As discussed, I am writing to confirm",
    "Q: What are the main differences between TCP and UDP?\nA:",
    "Q: Why does the moon always show the same face to the Earth?\nA:",
    "Translate to French:\n\nThe meeting has been moved to Thursday afternoon because the conference room is not available on Wednesday.\n\nFrench:",
    "Summarize the following paragraph in one sentence.\n\nParagraph: The city council voted on Tuesday to approve a new bike lane network covering forty kilometers, funded in part by a regional transport grant. Opponents argued that the plan would reduce parking, while supporters pointed to a decline in cycling injuries in comparable cities.\n\nSummary:",
    "Here is a list of ten tips for learning a new language:\n\n1.",
    "Chapter 3\n\nThe expedition left base camp at dawn. Snow had fallen overnight, and the porters moved slowly along the ridge,",
    "# README\n\n## Installation\n\n```bash\npip install fastqueue\n```\n\n## Usage\n\n```python\nfrom fastqueue import Queue\n",
    "Recipe: slow-cooked lentil soup\n\nIngredients:\n- 300 g brown lentils\n- 1 onion\n- 2 carrots\n\nMethod:\n1.",
    "Dear customer,\n\nWe are writing to inform you of a change to our terms of service, effective 1 June 2026. The main changes are as follows:\n\n1.",
    "class LRUCache:\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n        self.data = {}\n",
    "# Troubleshooting guide: database replication lag\n\n## Symptoms\n\n- Replicas report seconds_behind_master above 30\n- Read-your-writes failures after deploys\n\n## Checks\n\n1.",
    "Name,Department,Start date,Salary\nAda Lovelace,Research,2021-03-01,98000\nGrace Hopper,Engineering,2019-07-15,112000\nAlan Turing,",
    "The following is a conversation between a user and a helpful assistant.\n\nUser: I need help planning a three-day trip to Lisbon.\nAssistant: Happy to help! Here is a suggested plan.\n\nDay 1:",
    "Abstract: We study the effect of weight quantization on the long-context behaviour of large language models. We find that",
    "```python\nimport asyncio\n\nasync def worker(name, queue):\n    while True:\n        item = await queue.get()\n",
    "In mathematics, a group is a set equipped with an operation that combines any two elements to form a third element. To be a group, the operation must satisfy",
    "Release notes v3.2.0\n\n### Features\n\n- Added streaming exports\n- Added role-based access control\n\n### Fixes\n\n-",
]


def prompts(api: ia.Api) -> list[tuple[str, list[int]]]:
    texts = list(bench.DECODE_TEXTS.items()) + [(f"extra{i:02d}", t) for i, t in enumerate(EXTRA)]
    return [(name, api.tokenize(t)) for name, t in texts]


def distinct4(ids: list[int]) -> float:
    g = [tuple(ids[i : i + 4]) for i in range(len(ids) - 3)]
    return len(set(g)) / max(len(g), 1)


def run(out: str) -> None:
    api = ia.Api(ia.default_url())
    api.complete(bench.long_ids(api, 64), 8)  # warm
    res = {}
    for name, ids in prompts(api):
        gen, requests, wall = ia.greedy(api, ids, N_TOKENS)
        res[name] = {"prompt_len": len(ids), "generated": gen, "requests": requests, "seconds": round(wall, 3), "tok_s": round(len(gen) / wall, 2)}
        print(f"{name:12} {len(gen)} tokens  {len(gen) / wall:6.1f} tok/s  distinct-4gram {distinct4(gen):.3f}", flush=True)
    json.dump({"n_tokens": N_TOKENS, "replies": res}, open(out, "w"))


def compare(pa: str, pb: str) -> None:
    a, b = json.load(open(pa))["replies"], json.load(open(pb))["replies"]
    rows = []
    for name in a:
        ga, gb = a[name]["generated"], b[name]["generated"]
        first = next((i for i, (x, y) in enumerate(zip(ga, gb)) if x != y), None)
        rows.append((name, first, distinct4(ga), distinct4(gb), a[name]["tok_s"], b[name]["tok_s"]))
        print(f"{name:12} first difference {first if first is not None else 'none':>5}   distinct-4gram {rows[-1][2]:.3f} -> {rows[-1][3]:.3f}"
              f"{'   LOOP' if rows[-1][2] < LOOP_BELOW else '       '}{' -> LOOP' if rows[-1][3] < LOOP_BELOW else ''}   {rows[-1][4]:6.1f} -> {rows[-1][5]:6.1f} tok/s")
    n = len(rows)
    la, lb = sum(r[2] < LOOP_BELOW for r in rows), sum(r[3] < LOOP_BELOW for r in rows)
    same = sum(r[1] is None for r in rows)
    firsts = sorted(r[1] for r in rows if r[1] is not None)
    print(f"\n{n} prompts: identical replies {same}; median first difference {statistics.median(firsts) if firsts else 'n/a'}; loops (distinct-4gram < {LOOP_BELOW}): "
          f"{la} -> {lb}; mean distinct-4gram {statistics.mean(r[2] for r in rows):.3f} -> {statistics.mean(r[3] for r in rows):.3f}; "
          f"mean tok/s {statistics.mean(r[4] for r in rows):.1f} -> {statistics.mean(r[5] for r in rows):.1f}")
    nl = [r for r in rows if r[2] >= LOOP_BELOW and r[3] >= LOOP_BELOW]
    if nl:
        print(f"on the {len(nl)} prompts that loop in neither: mean tok/s {statistics.mean(r[4] for r in nl):.1f} -> {statistics.mean(r[5] for r in nl):.1f} "
              f"(x{statistics.mean(r[5] for r in nl) / statistics.mean(r[4] for r in nl):.3f})")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "run":
        os.makedirs(os.path.dirname(sys.argv[2]) or ".", exist_ok=True)
        run(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
