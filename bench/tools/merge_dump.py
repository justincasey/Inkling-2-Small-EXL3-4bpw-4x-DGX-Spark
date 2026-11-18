#!/usr/bin/env python3
"""Assemble oracle/compare.py's engine dump from a REXL3 Inkling 2 TP dump (stdlib only).

A boot with DUMP_DIR set (REXL3_INKLING2_DUMP on every rank) writes, per prefill chunk, on rank r:
  rank<r>/chunk<NNNNN>.json        {"slot", "first", "rows", "tokens", "vocab_base": SHARD*r, "vocab_cols": SHARD}
  rank<r>/chunk<NNNNN>.logits.f32  f32 [rows, SHARD]: soft-capped logits of rank r's vocab shard (global column =
                                   vocab_base + col; global columns >= VOCAB are -inf padding)
and on rank 0 also chunk<NNNNN>.hidden.bf16 [L+1, rows, H], .hfinal.bf16 [rows, H], .router_idx.i32 [L, rows, K],
.router_w.f32 [L, rows, K]. Chunk numbers are per rank in call order; a prompt is consecutive chunks of one slot with
`first` running from 0.

Copy the four rank directories under one DIR, then:

    merge_dump.py --dump DIR --fixture FIX.safetensors --greedy greedy.json --out ENGINE.safetensors

`tokens` of the output is the fixture's prompt + the ENGINE's greedy tokens (from greedy.py's report, entry --name,
default the fixture's file name), so compare.py's token gate compares two real decodes. The logits/hidden rows come
from the prefill whose chunks cover tokens[:N-1]: run greedy.py --teacher-force against the dumping boot, which sends
prompt + engine tokens[:-1] as one prompt. The output layout is the fixture layout (oracle/README.md): tokens I64 [N],
logits F32 [N-1, VOCAB], hidden BF16 [L+1, N-1, H], h_final BF16 [N-1, H], router_idx I32 / router_w F32 [L, N-1, K]
(router_margin is not in a dump). Shapes are flags so the unit test can use tiny ones.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stlite  # noqa: E402

NEG_INF = struct.pack("<f", float("-inf"))


def chunk_index(rank_dir: str) -> dict[int, dict]:
    out = {}
    for name in os.listdir(rank_dir):
        m = re.fullmatch(r"chunk(\d+)\.json", name)
        if m:
            with open(os.path.join(rank_dir, name)) as f:
                out[int(m.group(1))] = json.load(f)
    return out


def chunk_path(rank_dir: str, num: int, suffix: str) -> str:
    """The file of chunk `num` (the JSON's own zero padding, found by number)."""
    for width in (5, 6, 7, 8, 4, 3, 2, 1):
        p = os.path.join(rank_dir, f"chunk{num:0{width}d}.{suffix}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{rank_dir}: no chunk{num:05d}.{suffix}")


def sequences(index: dict[int, dict]) -> list[list[int]]:
    """Chunk-number lists, one per prompt: a chunk with first == 0 opens its slot's prompt, a chunk whose first is the
    slot's running end extends it; anything else closes the slot's prompt."""
    done, open_ = [], {}
    for num in sorted(index):
        c = index[num]
        slot, first = c["slot"], c["first"]
        cur = open_.get(slot)
        if first == 0:
            if cur:
                done.append(cur)
            open_[slot] = [num]
        elif cur and index[cur[-1]]["first"] + index[cur[-1]]["rows"] == first:
            cur.append(num)
        else:
            if cur:
                done.append(cur)
            open_.pop(slot, None)
    done.extend(v for v in open_.values() if v)
    done.sort(key=lambda s: s[0])
    return done


def find_prefill(index: dict[int, dict], want: list[int]) -> list[tuple[int, int]]:
    """[(chunk, rows taken)] of the LAST prompt whose tokens start with `want` (len(want) rows)."""
    best = None
    for seq in sequences(index):
        toks: list[int] = []
        for num in seq:
            toks += index[num]["tokens"]
            if len(toks) >= len(want):
                break
        if toks[:len(want)] == want:
            best = seq
    if best is None:
        raise SystemExit(f"no prompt in the dump covers the {len(want)} input rows (fixture prompt + engine tokens "
                         "minus the last): run greedy.py --teacher-force against the dumping boot")
    out, need = [], len(want)
    for num in best:
        take = min(index[num]["rows"], need)
        out.append((num, take))
        need -= take
        if need == 0:
            break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="directory holding rank0 .. rank<R-1>")
    ap.add_argument("--fixture", required=True, help="oracle fixture .safetensors (prompt tokens, prompt_len)")
    ap.add_argument("--greedy", required=True, help="greedy.py report (engine_generated of entry --name)")
    ap.add_argument("--name", help="fixture name in the greedy report (default: the fixture file name)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--shard-cols", type=int, default=50256)
    ap.add_argument("--vocab", type=int, default=200058)
    ap.add_argument("--hidden", type=int, default=4608)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--topk", type=int, default=8)
    a = ap.parse_args(argv)

    name = a.name or os.path.basename(a.fixture).removesuffix(".safetensors")
    _, meta = stlite.read_header(a.fixture)
    plen = int(meta["prompt_len"])
    prompt = stlite.read_ints(a.fixture, "tokens")[:plen]
    with open(a.greedy) as f:
        rep = json.load(f)
    if name not in rep.get("fixtures", {}):
        raise SystemExit(f"{a.greedy}: no fixture {name!r}")
    gen = [int(t) for t in rep["fixtures"][name]["engine_generated"]]
    tokens = prompt + gen
    n = len(tokens)
    rows = n - 1
    if a.ranks * a.shard_cols < a.vocab:
        raise SystemExit("ranks * shard-cols < vocab")

    dirs = [os.path.join(a.dump, f"rank{r}") for r in range(a.ranks)]
    idx = [chunk_index(d) for d in dirs]
    plan = find_prefill(idx[0], tokens[:rows])
    for r in range(a.ranks):
        for num, _ in plan:
            c, c0 = idx[r].get(num), idx[0][num]
            if c is None:
                raise SystemExit(f"rank {r}: chunk {num} missing (chunk numbers must agree across ranks)")
            for k in ("slot", "first", "rows", "tokens"):
                if c[k] != c0[k]:
                    raise SystemExit(f"rank {r}: chunk {num} {k} differs from rank 0's")
            if c["vocab_base"] != a.shard_cols * r or c["vocab_cols"] != a.shard_cols:
                raise SystemExit(f"rank {r}: chunk {num} vocab_base/vocab_cols {c['vocab_base']}/{c['vocab_cols']}, "
                                 f"expected {a.shard_cols * r}/{a.shard_cols}")
            want = c["rows"] * a.shard_cols * 4
            got = os.path.getsize(chunk_path(dirs[r], num, "logits.f32"))
            if got != want:
                raise SystemExit(f"rank {r}: chunk {num} logits are {got} bytes, expected {want}")

    def logits():
        pad_from = a.vocab - a.shard_cols * (a.ranks - 1)  # columns of the last shard that are real
        files = [None] * a.ranks
        try:
            for num, take in plan:
                files = [open(chunk_path(dirs[r], num, "logits.f32"), "rb") for r in range(a.ranks)]
                for _ in range(take):
                    parts = [f.read(a.shard_cols * 4) for f in files]
                    tail = parts[-1][pad_from * 4:]
                    if tail != NEG_INF * (a.shard_cols - pad_from):
                        raise SystemExit(f"chunk {num}: the padding columns >= {a.vocab} are not -inf "
                                         "(vocab shard mapping wrong?)")
                    parts[-1] = parts[-1][:pad_from * 4]
                    yield b"".join(parts)
                for f in files:
                    f.close()
        finally:
            for f in files:
                if f is not None and not f.closed:
                    f.close()

    def layered(suffix: str, planes: int, width: int):
        """[planes, rows, width] pieces of rank 0's per-chunk [planes, chunk_rows, width] files, rows concatenated."""
        def gen():
            fs = {num: open(chunk_path(dirs[0], num, suffix), "rb") for num, _ in plan}
            try:
                for num, _ in plan:
                    want = planes * idx[0][num]["rows"] * width
                    if os.path.getsize(fs[num].name) != want:
                        raise SystemExit(f"chunk {num}: {suffix} is {os.path.getsize(fs[num].name)} bytes, expected {want}")
                for p in range(planes):
                    for num, take in plan:
                        fs[num].seek(p * idx[0][num]["rows"] * width)
                        yield fs[num].read(take * width)
            finally:
                for f in fs.values():
                    f.close()
        return gen

    def has(suffix: str) -> bool:
        try:
            for num, _ in plan:
                chunk_path(dirs[0], num, suffix)
            return True
        except FileNotFoundError:
            return False

    H, L, K = a.hidden, a.layers, a.topk
    tensors = [("tokens", "I64", [n], lambda: [struct.pack(f"<{n}q", *tokens)]),
               ("logits", "F32", [rows, a.vocab], logits)]
    if has("hidden.bf16"):
        tensors.append(("hidden", "BF16", [L + 1, rows, H], layered("hidden.bf16", L + 1, H * 2)))
    if has("hfinal.bf16"):
        tensors.append(("h_final", "BF16", [rows, H], layered("hfinal.bf16", 1, H * 2)))
    if has("router_idx.i32"):
        tensors.append(("router_idx", "I32", [L, rows, K], layered("router_idx.i32", L, K * 4)))
    if has("router_w.f32"):
        tensors.append(("router_w", "F32", [L, rows, K], layered("router_w.f32", L, K * 4)))
    stlite.write_streaming(a.out, tensors, {
        "prompt_len": plen, "source": "rexl3 engine dump", "fixture": name,
        "chunks": ",".join(f"{num}:{take}" for num, take in plan), "dump": os.path.abspath(a.dump)})
    print(json.dumps({"out": a.out, "tokens": n, "prompt_len": plen, "rows": rows,
                      "chunks": [[num, take] for num, take in plan], "tensors": [t[0] for t in tensors]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
