#!/usr/bin/env python3
"""Drive a fixed request sequence inside an nsys window (stdlib only; a lab tool for PROFILE_NSYS profiles).

  profile_driver.py T0 OPEN_AT [PREFILL_TOKENS]

T0 is the epoch second taken just before `PROFILE_NSYS="--delay=OPEN_AT ..." ./inkling2.sh start`; the window opens OPEN_AT seconds
after it. After a warm pass (a 64-token decode and a 4K prefill, outside the window) it waits until the window is open, then sends,
one at a time and cold: the prose prompt (128 tokens), the code prompt (128 tokens) and a 32,768-token prefill (16 tokens).
With PREFILL_TOKENS (e.g. 262144) it sends only that prefill (16 tokens): give nsys a --duration long enough for it.
"""
import sys
import time

sys.path.insert(0, "bench/tools")
import bench  # noqa: E402
import inkling_api as ia  # noqa: E402

t0, open_at = float(sys.argv[1]), float(sys.argv[2])
only = int(sys.argv[3]) if len(sys.argv) > 3 else None
api = ia.Api(ia.default_url())
prompts = {n: api.tokenize(t) for n, t in bench.DECODE_TEXTS.items()}
big = bench.long_ids(api, only or 32768)
# warm pass outside the window: decode and a short prefill
api.complete(prompts["code"], 64, prefix_cache=False)
api.complete(bench.long_ids(api, 4096), 8, prefix_cache=False)
wait = t0 + open_at + 3 - time.time()
print(f"warm done; waiting {wait:.1f} s for the window", flush=True)
if wait > 0:
    time.sleep(wait)
t = time.time()
for name, n in (("prose", 128), ("code", 128)) if only is None else ():
    r, wall = api.complete(prompts[name], n, prefix_cache=False)
    print(f"{name}: {r['usage']['completion_tokens']} tokens in {wall:.2f} s, spec {r['usage'].get('spec')}", flush=True)
r, wall = api.complete(big, 16, prefix_cache=False)
print(f"prefill: {r['usage']['prompt_tokens']} tokens, request {wall:.2f} s", flush=True)
print(f"window used {time.time() - t:.1f} s; offset from T0 {t - t0:.1f} s", flush=True)
