#!/usr/bin/env python3
"""Recipe intake: manifest, tensor census, quant overlay and a fit estimate.

Evidence-gathering for a new DGX Spark serving recipe. Reads HF metadata and
safetensors headers only (small non-LFS files are downloaded and hashed); it
never downloads weights.

  python3 scripts/recipe_intake.py \
      --repo vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-4.26bpw \
      --revision 64f1d0a1516a63609b5b43346e9a8e3c0d58f210 \
      --tp 2 --image vllm/vllm-openai:minimax-m31-flash \
      --out recipes/MiniMax-M3.1-Flash-EXL3-2x-DGX-Spark/intake

Outputs (all deterministic):
  manifest.tsv     <sha256>\\t<bytes>\\t<path> for every file at the revision
  census.json      per-module dtypes/shapes/bytes; routed experts collapsed
  overlay.json     quantization_config for EXL3/mixed packs that match the
                   MiniMax pattern (null + reason otherwise)
  intake.json      summary, per-TP footprint, KV headroom estimate, image digest

The footprint is an estimate: it shards 2-D linear weights and the draft by TP,
keeps 1-D/norm tensors and a vocab-non-aligned EXL3 LM head replicated, and
splits embeddings by vocab. Re-measure on the first boot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import urllib.request
from collections import defaultdict

ROUTED = ".block_sparse_moe.experts."
SUFFIXES = (".weight_scale_inv", ".weight_scale", ".weight", ".trellis", ".suh", ".svh", ".mul1", ".bias")


def http_json(url: str, timeout: int = 60) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def http_bytes(url: str, timeout: int = 120) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def header_http(url: str, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        n = struct.unpack("<Q", r.read(8))[0]
    req = urllib.request.Request(url, headers={"Range": f"bytes=8-{8 + n - 1}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        header = json.loads(r.read())
    header.pop("__metadata__", None)
    return header


def module_of(name: str) -> str:
    for suffix in SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def suffix_of(name: str) -> str:
    for suffix in SUFFIXES:
        if name.endswith(suffix):
            return suffix.lstrip(".")
    return "tensor"


def build_census(entries, repo: str, revision: str) -> dict:
    modules = {}
    routed = defaultdict(int)
    dtypes = defaultdict(int)
    files = defaultdict(int)
    for filename, name, meta in entries:
        size = meta["data_offsets"][1] - meta["data_offsets"][0]
        dtypes[meta["dtype"]] += size
        files[filename] += size
        suffix = suffix_of(name)
        if ROUTED in name:
            root, rest = name.split(ROUTED, 1)
            projection = rest.split(".", 2)[1] if "." in rest else "*"
            routed[f"{root}{ROUTED}*.{projection}.{suffix}"] += size
            continue
        mod = module_of(name)
        slot = modules.setdefault(mod, {"bytes": 0, "tensors": {}})
        slot["tensors"][suffix] = {"dtype": meta["dtype"], "shape": list(meta["shape"]), "bytes": size}
        slot["bytes"] += size
    return {
        "repo": repo,
        "revision": revision,
        "total_bytes": sum(files.values()),
        "files": {k: files[k] for k in sorted(files)},
        "dtypes": {k: dtypes[k] for k in sorted(dtypes)},
        "modules": {k: modules[k] for k in sorted(modules)},
        "routed_tensors": {k: routed[k] for k in sorted(routed)},
    }


def derive_overlay(census: dict) -> tuple[dict | None, list[str]]:
    warnings = []
    modules = census["modules"]
    mxfp8, blocks = [], set()
    for name, mod in modules.items():
        weight = mod["tensors"].get("weight")
        scale = mod["tensors"].get("weight_scale_inv")
        if weight and weight["dtype"] == "F8_E4M3" and scale:
            mxfp8.append(name)
            rows = weight["shape"][0] // scale["shape"][0]
            cols = weight["shape"][1] // scale["shape"][1]
            blocks.add((rows, cols))
    if not mxfp8:
        return None, ["no MXFP8 non-routed linears: this is not the MiniMax mixed-format pattern"]
    if len(blocks) != 1:
        return None, [f"multiple MXFP8 block geometries: {sorted(blocks)}"]
    exl3 = [n for n, m in modules.items() if "trellis" in m["tensors"]]
    if len(exl3) != 1:
        return None, [f"expected one non-routed EXL3 module, found {exl3}"]
    head = modules[exl3[0]]["tensors"]
    trellis = head["trellis"]
    bits = trellis["bytes"] * 8 / (trellis["shape"][0] * 16 * trellis["shape"][1] * 16)
    if abs(bits - round(bits)) > 0.01:
        return None, [f"non-routed EXL3 head is not an integer bit width: {bits:.3f}"]
    ignored = sorted(
        n
        for n, m in modules.items()
        if "weight" in m["tensors"]
        and m["tensors"]["weight"]["dtype"] in ("F32", "BF16", "F16")
        and len(m["tensors"]["weight"]["shape"]) == 2
    )
    # Normalize a VL wrapper prefix if present.
    ignored = sorted({n[len("language_model."):] if n.startswith("language_model.") else n for n in ignored} | {"lm_head"})
    overlay = {
        "quantization_config": {
            "quant_method": "exl3",
            "version": "0.0.2",
            "non_routed_exl3": {
                "modules": [exl3[0].split(".")[-1]],
                "bits": int(round(bits)),
                "codebook": "mul1" if "mul1" in head else "mcg",
            },
            "non_routed_quantization": {
                "quant_method": "mxfp8",
                "activation_scheme": "dynamic",
                "weight_block_size": list(next(iter(blocks))),
                "ignored_layers": ignored,
            },
        }
    }
    if len(mxfp8) < 12:
        warnings.append(f"only {len(mxfp8)} MXFP8 modules; verify the classification")
    return overlay, warnings


def footprint(census: dict, tp: int, node_mem_gib: float, guard_gib: float,
              activation_gib: float, kv_kib_per_token: float, draft_bytes: int = 0) -> dict:
    modules = census["modules"]
    sharded = replicated = 0
    replicated_names = []
    for name, mod in modules.items():
        size = mod["bytes"]
        weight = mod["tensors"].get("weight")
        is_linear = bool(weight and len(weight["shape"]) == 2)
        is_embed = "embed_tokens" in name
        is_head = name.endswith("lm_head") or "lm_head" in name
        head_rows = weight["shape"][0] if is_head and weight else 0
        # An EXL3 vocab head whose rows do not divide into 128-aligned TP chunks
        # is loaded whole by every rank (the MiniMax LM-head rule).
        head_aligned = bool(head_rows and (head_rows // tp) % 128 == 0)
        if is_linear and not (is_head and not head_aligned):
            sharded += size
        elif is_embed and tp > 1:
            sharded += size
        else:
            replicated += size
            replicated_names.append(name)
    routed = sum(census["routed_tensors"].values())
    # Routed experts shard by TP in this family; keep them separate in the report.
    per_rank_bytes = routed / tp + sharded / tp + replicated + draft_bytes / tp
    headroom = node_mem_gib - per_rank_bytes / 2**30 - guard_gib - activation_gib
    kv_tokens = int(headroom * 1024 * 1024 / kv_kib_per_token) if headroom > 0 else 0
    return {
        "tp": tp,
        "routed_experts_gib": round(routed / 2**30, 2),
        "sharded_non_routed_gib": round(sharded / 2**30, 2),
        "replicated_gib": round(replicated / 2**30, 2),
        "draft_gib": round(draft_bytes / 2**30, 2),
        "weights_per_rank_gib": round(per_rank_bytes / 2**30, 2),
        "node_mem_gib": node_mem_gib,
        "guard_gib": guard_gib,
        "activation_and_graphs_gib": activation_gib,
        "headroom_gib": round(headroom, 2),
        "kv_tokens": kv_tokens,
        "kv_kib_per_token": kv_kib_per_token,
        "replicated_modules_sample": replicated_names[:12],
        "confidence": "estimate: re-measure MemAvailable and the engine KV line on the first boot",
    }


def image_digest(image: str) -> dict | None:
    """Docker Hub tag -> manifest-list digest and per-platform digests."""
    if ":" not in image or "/" not in image:
        return None
    repo, tag = image.rsplit(":", 1)
    if repo.startswith("docker.io/"):
        repo = repo[len("docker.io/"):]
    try:
        token = http_json(
            f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
        )["token"]
        req = urllib.request.Request(
            f"https://registry-1.docker.io/v2/{repo}/manifests/{tag}",
            headers={
                "Authorization": "Bearer " + token,
                "Accept": "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            digest = r.headers.get("Docker-Content-Digest")
            body = json.loads(r.read())
        platforms = [
            {"platform": m.get("platform"), "digest": m.get("digest")}
            for m in body.get("manifests", [])
        ]
        return {"image": image, "index_digest": digest, "platforms": platforms}
    except Exception as exc:  # noqa: BLE001
        return {"image": image, "error": str(exc)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--node-mem-gib", type=float, default=121.7)
    ap.add_argument("--guard-gib", type=float, default=1.0)
    ap.add_argument("--activation-gib", type=float, default=4.0)
    ap.add_argument("--kv-kib-per-token", type=float, default=27.5)
    ap.add_argument("--image", help="container image tag to resolve to a digest")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files_dir = os.path.join(args.out, "files")
    os.makedirs(files_dir, exist_ok=True)

    info = http_json(f"https://huggingface.co/api/models/{args.repo}?blobs=true")
    if info.get("sha") and info["sha"] != args.revision:
        print(f"WARNING: current head {info['sha']} differs from --revision {args.revision}", file=sys.stderr)
    gated = bool(info.get("gated"))
    siblings = info.get("siblings") or []
    base = f"https://huggingface.co/{args.repo}/resolve/{args.revision}/"

    rows = []
    for s in siblings:
        path = s["rfilename"]
        lfs = s.get("lfs")
        if lfs:
            rows.append((lfs["sha256"], lfs.get("size", s.get("size")), path))
        else:
            dst = os.path.join(files_dir, path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                data = http_bytes(base + path)
                with open(dst, "wb") as f:
                    f.write(data)
            rows.append((hashlib.sha256(open(dst, "rb").read()).hexdigest(), os.path.getsize(dst), path))
    rows.sort(key=lambda r: r[2])
    with open(os.path.join(args.out, "manifest.tsv"), "w") as f:
        for digest, size, path in rows:
            f.write(f"{digest}\t{size}\t{path}\n")

    index = http_json(base + "model.safetensors.index.json")
    shards = sorted(set(index["weight_map"].values()))
    entries = []
    for i, shard in enumerate(shards, 1):
        header = header_http(base + shard)
        for name, meta in header.items():
            entries.append((shard, name, meta))
        print(f"[{i}/{len(shards)}] {shard}", file=sys.stderr)
    census = build_census(entries, args.repo, args.revision)
    with open(os.path.join(args.out, "census.json"), "w") as f:
        json.dump(census, f, indent=1, sort_keys=True)

    overlay, warnings = derive_overlay(census)
    if overlay:
        with open(os.path.join(args.out, "overlay.json"), "w") as f:
            json.dump(overlay, f, indent=2, sort_keys=True)
    # Auxiliary weights (draft/MTP/EAGLE heads, wherever the pack puts them):
    # any safetensors file that the main index does not reference.
    indexed = set(index["weight_map"].values())
    aux = defaultdict(int)
    for _, size, path in rows:
        if path.endswith(".safetensors") and path not in indexed:
            aux[os.path.dirname(path) or "."] += size
    draft_bytes = sum(aux.values())
    fit = footprint(census, args.tp, args.node_mem_gib, args.guard_gib, args.activation_gib, args.kv_kib_per_token, draft_bytes)
    summary = {
        "repo": args.repo,
        "revision": args.revision,
        "gated": gated,
        "license": (info.get("cardData") or {}).get("license"),
        "files": len(rows),
        "bytes": sum(r[1] for r in rows),
        "gib": round(sum(r[1] for r in rows) / 2**30, 3),
        "dtypes": {k: round(v / 2**30, 3) for k, v in census["dtypes"].items()},
        "modules": len(census["modules"]),
        "routed_rows": len(census["routed_tensors"]),
        "overlay": bool(overlay),
        "auxiliary_weights_gib": {k: round(v / 2**30, 3) for k, v in sorted(aux.items())},
        "fit": fit,
        "image": image_digest(args.image) if args.image else None,
        "warnings": warnings,
    }
    with open(os.path.join(args.out, "intake.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)

    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
