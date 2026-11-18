"""Generate oracle fixtures over a pack folder holding the whole model (every routed expert).

    python oracle/gen_fixtures.py --pack DIR --prompts oracle/prompts.jsonl --out DIR [--workers N] [--threads N]
                                  [--tp 4] [--only NAME ...] [--verify-recompute]

Prompts (JSON lines): {"name": ..., "text": ...} (tokenized with the pack's tokenizer.json, special-token strings
recognised, nothing added) or {"name": ..., "ids": [...]}, optional "new_tokens" (default --new-tokens).
Per prompt the oracle prefills, then greedy-decodes from its cache, and writes OUT/NAME.safetensors:
  tokens I64 [N] (prompt + generated), logits F32 [N-1, 200058] (row t predicts token t+1), h_final BF16 [N-1, H],
  hidden BF16 [L+1, N-1, H] (P entering each layer, then P after the last), router_idx I32 / router_w F32
  [L, N-1, k], router_margin F32 [L, N-1]; metadata: prompt_len, tp, pack config sha256, oracle source sha256.
plus OUT/manifest.json (sha256 of every fixture, decoded continuations, timings).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch

import st
from inkling2_ref import Inkling2Config, Inkling2Ref, generate
from pack import PackExperts, PackIndex, PackWeights

HERE = os.path.dirname(os.path.abspath(__file__))
NUMERICS_VERSION = "1"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def oracle_source_sha() -> str:
    h = hashlib.sha256()
    for fn in sorted(os.listdir(HERE)):
        if fn.endswith(".py"):
            with open(os.path.join(HERE, fn), "rb") as f:
                h.update(fn.encode() + b"\0" + f.read())
    return h.hexdigest()


def load_prompts(path: str, only: list[str] | None) -> list[dict]:
    with open(path) as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    names = [p["name"] for p in prompts]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate prompt names")
    if only:
        missing = set(only) - set(names)
        if missing:
            raise ValueError(f"no such prompts: {sorted(missing)}")
        prompts = [p for p in prompts if p["name"] in only]
    return prompts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--prompts", default=os.path.join(HERE, "prompts.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0, help="expert worker processes (0 = in process)")
    ap.add_argument("--threads", type=int, default=0, help="torch intra-op threads (0 = torch default)")
    ap.add_argument("--new-tokens", type=int, default=16)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--verify-recompute", action="store_true",
                    help="also run every sequence uncached and require identical logits (doubles the cost)")
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    with open(os.path.join(a.pack, "config.json")) as f:
        cfg = Inkling2Config.from_hf(json.load(f))
    prompts = load_prompts(a.prompts, a.only)
    tok = None
    if any("text" in p for p in prompts):
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(os.path.join(a.pack, "tokenizer.json"))
    ix = PackIndex(a.pack)
    experts = PackExperts(ix, workers=a.workers)
    model = Inkling2Ref(cfg, PackWeights(ix), experts, tp=a.tp)
    os.makedirs(a.out, exist_ok=True)
    meta_common = {
        "tp": str(a.tp),
        "pack_config_sha256": sha256_file(os.path.join(a.pack, "config.json")),
        "oracle_source_sha256": oracle_source_sha(),
        "numerics_version": NUMERICS_VERSION,
    }
    manifest_path = os.path.join(a.out, "manifest.json")
    manifest = {"meta": meta_common, "fixtures": {}}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest["fixtures"] = json.load(f).get("fixtures", {})
    try:
        for p in prompts:
            ids = p["ids"] if "ids" in p else tok.encode(p["text"], add_special_tokens=False).ids
            n_new = int(p.get("new_tokens", a.new_tokens))
            t0 = time.time()
            res = generate(model, ids, n_new)
            secs = time.time() - t0
            if a.verify_recompute:
                full = model.forward(res["tokens"][:-1])
                if not torch.equal(full.logits, res["logits"]):
                    raise AssertionError(f"{p['name']}: cached decode differs from the full recompute")
            tensors = {k: res[k] for k in ("tokens", "logits", "h_final", "hidden", "router_idx", "router_w",
                                           "router_margin")}
            path = os.path.join(a.out, f"{p['name']}.safetensors")
            st.write(path, tensors, dict(meta_common, name=p["name"], prompt_len=str(res["prompt_len"])))
            gen = res["tokens"][res["prompt_len"]:].tolist()
            manifest["fixtures"][p["name"]] = {
                "sha256": sha256_file(path), "prompt_len": res["prompt_len"], "generated": gen,
                "generated_text": tok.decode(gen, skip_special_tokens=False) if tok else None,
                "seconds": round(secs, 1), "verified_recompute": a.verify_recompute,
            }
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=1)
            print(f"{p['name']}: {res['prompt_len']} + {len(gen)} tokens in {secs:.0f}s -> {path}", flush=True)
    finally:
        experts.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
