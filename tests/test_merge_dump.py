#!/usr/bin/env python3
"""Unit check of bench/tools/merge_dump.py on synthetic dumps (stdlib only).

1. Tiny shapes (4 ranks x 5 vocab columns, vocab 18, hidden 6, 3 layers, top-2): several prompts in the dump (a decoy
   slot, the bare fixture prompt, the teacher-forced prompt split over chunks interleaved with another slot); the
   merged tensors must equal the ground truth byte for byte, and broken dumps must be refused.
2. Real shapes (vocab 200058 over 4 x 50256 columns, hidden 4608, 48 layers, top-8) with 2 rows; with
   --compare-python PY (a Python that has torch, e.g. oracle/.venv/bin/python) oracle/compare.py must pass the merged
   dump against a fixture holding the same numbers.

    python3 tests/test_merge_dump.py [--compare-python oracle/.venv/bin/python]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from array import array

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench", "tools"))
import merge_dump  # noqa: E402
import stlite  # noqa: E402


class Truth:
    """Ground truth of a full sequence: logits [T, ranks*shard] f32 (padding -inf), hidden u16 [L+1, T, H], etc."""

    def __init__(self, T, ranks, shard, vocab, H, L, K):
        self.T, self.ranks, self.shard, self.vocab, self.H, self.L, self.K = T, ranks, shard, vocab, H, L, K
        cols = ranks * shard
        self.logits = [array("f", [((t * 7 + v * 13) % 101) / 8.0 - 6.0 if v < vocab else float("-inf")
                                   for v in range(cols)]) for t in range(T)]
        bf = lambda x: struct.unpack("<I", struct.pack("<f", x))[0] >> 16  # noqa: E731
        self.hidden = [[array("H", [bf(math.sin(l * 1.3 + t * 0.7 + h * 0.011) + 0.01) for h in range(H)])
                        for t in range(T)] for l in range(L + 1)]
        self.hfinal = [array("H", [bf(math.cos(t * 0.5 + h * 0.013)) for h in range(H)]) for t in range(T)]
        self.ridx = [[array("i", [(l * 31 + t * 7 + k * 3) % 384 for k in range(K)]) for t in range(T)] for l in range(L)]
        self.rw = [[array("f", [1.0 / (k + 1 + l + t) for k in range(K)]) for t in range(T)] for l in range(L)]


def write_chunk(d, num, slot, first, tokens, truth: Truth | None, rows_from: int):
    """One chunk on every rank; rows are truth rows [rows_from, rows_from + len(tokens)) (None: garbage rows)."""
    n = len(tokens)
    for r in range(4 if truth is None else truth.ranks):
        rd = os.path.join(d, f"rank{r}")
        os.makedirs(rd, exist_ok=True)
        shard = truth.shard if truth else 5
        meta = {"slot": slot, "first": first, "rows": n, "tokens": tokens, "vocab_base": shard * r, "vocab_cols": shard}
        with open(os.path.join(rd, f"chunk{num:05d}.json"), "w") as f:
            json.dump(meta, f)
        with open(os.path.join(rd, f"chunk{num:05d}.logits.f32"), "wb") as f:
            for i in range(n):
                row = truth.logits[rows_from + i][shard * r:shard * (r + 1)] if truth else array("f", [9.0] * shard)
                f.write(row.tobytes())
        if r == 0 and truth:
            with open(os.path.join(rd, f"chunk{num:05d}.hidden.bf16"), "wb") as f:
                for l in range(truth.L + 1):
                    for i in range(n):
                        f.write(truth.hidden[l][rows_from + i].tobytes())
            with open(os.path.join(rd, f"chunk{num:05d}.hfinal.bf16"), "wb") as f:
                for i in range(n):
                    f.write(truth.hfinal[rows_from + i].tobytes())
            for suffix, src in (("router_idx.i32", truth.ridx), ("router_w.f32", truth.rw)):
                with open(os.path.join(rd, f"chunk{num:05d}.{suffix}"), "wb") as f:
                    for l in range(truth.L):
                        for i in range(n):
                            f.write(src[l][rows_from + i].tobytes())


def write_fixture(path, tokens, plen, truth: Truth | None = None):
    tensors = [("tokens", "I64", [len(tokens)], lambda: [struct.pack(f"<{len(tokens)}q", *tokens)])]
    if truth:  # a fixture with the same numbers: compare.py must pass
        T, V = len(tokens) - 1, truth.vocab
        tensors += [
            ("logits", "F32", [T, V], lambda: (truth.logits[t][:V].tobytes() for t in range(T))),
            ("hidden", "BF16", [truth.L + 1, T, truth.H],
             lambda: (truth.hidden[l][t].tobytes() for l in range(truth.L + 1) for t in range(T))),
            ("h_final", "BF16", [T, truth.H], lambda: (truth.hfinal[t].tobytes() for t in range(T))),
            ("router_idx", "I32", [truth.L, T, truth.K], lambda: (truth.ridx[l][t].tobytes() for l in range(truth.L) for t in range(T))),
            ("router_w", "F32", [truth.L, T, truth.K], lambda: (truth.rw[l][t].tobytes() for l in range(truth.L) for t in range(T))),
            ("router_margin", "F32", [truth.L, T], lambda: [array("f", [0.5] * (truth.L * T)).tobytes()]),
        ]
    stlite.write_streaming(path, tensors, {"prompt_len": plen, "tp": 4})


def write_greedy(path, name, gen):
    with open(path, "w") as f:
        json.dump({"fixtures": {name: {"engine_generated": gen}}}, f)


def check_output(out, truth: Truth, seq):
    t, meta = stlite.read_header(out)
    T, V = len(seq) - 1, truth.vocab
    assert stlite.read_ints(out, "tokens") == seq, "tokens"
    assert t["logits"]["shape"] == [T, V], t["logits"]["shape"]
    want = b"".join(truth.logits[i][:V].tobytes() for i in range(T))
    assert stlite.read_bytes(out, t["logits"]) == want, "logits bytes"
    want = b"".join(truth.hidden[l][i].tobytes() for l in range(truth.L + 1) for i in range(T))
    assert t["hidden"]["shape"] == [truth.L + 1, T, truth.H] and stlite.read_bytes(out, t["hidden"]) == want, "hidden"
    assert stlite.read_bytes(out, t["h_final"]) == b"".join(truth.hfinal[i].tobytes() for i in range(T)), "h_final"
    for name, src in (("router_idx", truth.ridx), ("router_w", truth.rw)):
        want = b"".join(src[l][i].tobytes() for l in range(truth.L) for i in range(T))
        assert t[name]["shape"] == [truth.L, T, truth.K] and stlite.read_bytes(out, t[name]) == want, name
    return meta


def expect_refusal(argv, what):
    try:
        merge_dump.main(argv)
    except SystemExit as e:
        if e.code not in (0, None):
            print(f"ok   refuses {what}: {e.code}")
            return
    raise AssertionError(f"merge_dump accepted {what}")


def tiny(tmp):
    shape = ["--ranks", "4", "--shard-cols", "5", "--vocab", "18", "--hidden", "6", "--layers", "3", "--topk", "2"]
    prompt, oracle_gen, engine_gen = [11, 12, 13, 14, 15], [7, 8, 9], [7, 3, 9]
    seq = prompt + engine_gen          # N = 8, rows 7
    truth = Truth(len(seq), 4, 5, 18, 6, 3, 2)
    dump = os.path.join(tmp, "tiny")
    write_chunk(dump, 0, 1, 0, [1, 2, 3], None, 0)                       # decoy prompt, slot 1
    write_chunk(dump, 1, 0, 0, prompt[:3], truth, 0)                     # greedy's own prefill of the bare prompt
    write_chunk(dump, 2, 0, 3, prompt[3:], truth, 3)
    write_chunk(dump, 3, 2, 0, seq[:3], truth, 0)                        # teacher-forced prompt, slot 2 ...
    write_chunk(dump, 4, 1, 3, [4, 5], None, 0)                          # ... interleaved with slot 1
    write_chunk(dump, 5, 2, 3, seq[3:6], truth, 3)
    write_chunk(dump, 6, 2, 6, seq[6:7], truth, 6)
    fix = os.path.join(tmp, "fix.safetensors")
    write_fixture(fix, prompt + oracle_gen, len(prompt))
    gr = os.path.join(tmp, "greedy.json")
    write_greedy(gr, "fix", engine_gen)
    out = os.path.join(tmp, "tiny.safetensors")
    merge_dump.main(["--dump", dump, "--fixture", fix, "--greedy", gr, "--out", out] + shape)
    meta = check_output(out, truth, seq)
    assert meta["prompt_len"] == "5" and meta["chunks"] == "3:3,5:3,6:1", meta
    print("ok   tiny dump: tokens, logits (padding cut), hidden, h_final, router merged byte-exact from chunks 3,5,6")

    # refusals: no covering prefill, wrong padding, ranks that disagree
    write_greedy(gr, "fix", [7, 4, 9])
    expect_refusal(["--dump", dump, "--fixture", fix, "--greedy", gr, "--out", out] + shape, "a dump without the rows")
    write_greedy(gr, "fix", engine_gen)
    bad = os.path.join(tmp, "badpad")
    t2 = Truth(len(seq), 4, 5, 18, 6, 3, 2)
    t2.logits[1][19] = 0.0
    write_chunk(bad, 0, 0, 0, seq[:7], t2, 0)
    expect_refusal(["--dump", bad, "--fixture", fix, "--greedy", gr, "--out", out] + shape, "non -inf padding")
    mism = os.path.join(tmp, "mismatch")
    write_chunk(mism, 0, 0, 0, seq[:7], truth, 0)
    with open(os.path.join(mism, "rank2", "chunk00000.json")) as f:
        m = json.load(f)
    m["vocab_base"] = 0
    with open(os.path.join(mism, "rank2", "chunk00000.json"), "w") as f:
        json.dump(m, f)
    expect_refusal(["--dump", mism, "--fixture", fix, "--greedy", gr, "--out", out] + shape, "a wrong vocab_base")


def real(tmp, compare_python):
    prompt, gen = [200000, 1234], [5678]
    seq = prompt + gen                 # N = 3, rows 2
    truth = Truth(len(seq), 4, 50256, 200058, 4608, 48, 8)
    dump = os.path.join(tmp, "real")
    write_chunk(dump, 0, 0, 0, seq[:2], truth, 0)
    fix = os.path.join(tmp, "real-fix.safetensors")
    write_fixture(fix, prompt + gen, len(prompt), truth)
    gr = os.path.join(tmp, "real-greedy.json")
    write_greedy(gr, "real-fix", gen)
    out = os.path.join(tmp, "real-engine.safetensors")
    merge_dump.main(["--dump", dump, "--fixture", fix, "--greedy", gr, "--out", out])
    check_output(out, truth, seq)
    print("ok   real shapes (2 rows): logits [2, 200058], hidden [49, 2, 4608], router [48, 2, 8] byte-exact")
    if compare_python:
        rep = os.path.join(tmp, "compare.json")
        p = subprocess.run([os.path.abspath(compare_python), os.path.join(ROOT, "oracle", "compare.py"), fix, out, "--json", rep],
                           cwd=os.path.join(ROOT, "oracle"), capture_output=True, text=True)
        if p.returncode != 0:
            raise AssertionError(f"compare.py failed on a merged dump of the fixture's own numbers:\n{p.stdout}\n{p.stderr}")
        with open(rep) as f:
            checks = json.load(f)["checks"]
        print(f"ok   oracle/compare.py passes the merged dump ({', '.join(sorted(checks))})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--compare-python", help="a Python with torch: also run oracle/compare.py on the real-shape dump")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        tiny(tmp)
        real(tmp, a.compare_python)
    return 0


if __name__ == "__main__":
    sys.exit(main())
