#!/usr/bin/env python3
"""plp_req.py N: one cold N-id prefill, 1 token (the engine's dump writes every chunk's logits)."""
import sys
sys.path.insert(0, "bench/tools")
import bench, inkling_api as ia
api = ia.Api(ia.default_url())
ids = bench.long_ids(api, int(sys.argv[1]))
r, wall = api.complete(ids, 1, prefix_cache=False)
print(f"{r['usage']['prompt_tokens']} prompt tokens in {wall:.1f} s", flush=True)
