#!/usr/bin/env python3
"""Assign every pack file to the ranks that hold it, from the files' own safetensors headers.

The pack stripes each layer's routed experts over four files: `model-experts-LL-SS` holds layer LL's experts
E with E mod 4 == SS. Rank r owns exactly those experts (the engine's expert split is E mod 4 == r), so it needs
only the 48 files with SS == r. Every other file (non-expert shards, the drafter, config, tokenizer) is on every rank.

The script reads each expert file's header over HTTP Range requests at the pinned revision and refuses a file whose
tensors break the stripe; non-expert files and the drafter must hold no routed expert. Output: config/pack-ranks.tsv
with `rank<TAB>bytes<TAB>path` (rank `*` = every rank), in manifest order.
"""
import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recipe_intake import header_http  # noqa: E402

RANKS = 4
EXPERT_FILE = re.compile(r"^model-experts-(\d+)-(\d+)\.safetensors$")
EXPERT_TENSOR = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts\.(\d+)\.")
N_EXPERTS = 384


def check(url: str, path: str) -> str | None:
    """None if the file's tensors obey its stripe (or hold no expert when not an expert file), else the reason."""
    names = header_http(url)
    m = EXPERT_FILE.match(path)
    hits = [(int(t.group(1)), int(t.group(2))) for t in map(EXPERT_TENSOR.search, names) if t]
    if not m:
        return f"{path}: {len(hits)} routed-expert tensors in a shared file" if hits else None
    layer, stripe = int(m.group(1)), int(m.group(2))
    if stripe >= RANKS:
        return f"{path}: stripe {stripe} >= {RANKS}"
    if len(hits) != len(names):
        return f"{path}: {len(names) - len(hits)} tensors that are not routed experts"
    bad = sorted({(l, e) for l, e in hits if l != layer or e % RANKS != stripe})
    if bad:
        return f"{path}: experts off the stripe, e.g. layer {bad[0][0]} expert {bad[0][1]}"
    experts = {e for _, e in hits}
    if experts != set(range(stripe, N_EXPERTS, RANKS)):
        return f"{path}: holds {len(experts)} experts, expected the {N_EXPERTS // RANKS} with e mod {RANKS} == {stripe}"
    return None


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(root / "intake/manifest.tsv"))
    ap.add_argument("--out", default=str(root / "config/pack-ranks.tsv"))
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    a = ap.parse_args()

    rows = [line.rstrip("\n").split("\t") for line in open(a.manifest) if line.strip()]
    base = f"https://huggingface.co/{a.repo}/resolve/{a.revision}"
    tensors = [p for _, _, p in rows if p.endswith(".safetensors")]
    with ThreadPoolExecutor(16) as pool:
        errors = [e for e in pool.map(lambda p: check(f"{base}/{p}", p), tensors) if e]
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    per_rank = [0] * RANKS
    with open(a.out, "w") as out:
        for _, size, path in rows:
            m = EXPERT_FILE.match(path)
            rank = str(int(m.group(2))) if m else "*"
            out.write(f"{rank}\t{size}\t{path}\n")
            for r in range(RANKS):
                if rank in ("*", str(r)):
                    per_rank[r] += int(size)
    for r, b in enumerate(per_rank):
        print(f"rank {r}: {b / 1e9:.2f} GB")
    print(f"wrote {a.out}: {len(rows)} files, {len(tensors)} headers checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
