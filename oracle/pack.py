"""The EXL3 pack as oracle weight/expert sources.

`PackWeights` reads plain tensors (BF16/F32) from a pack folder by positional reads, through
`model.safetensors.index.json` when present (else every *.safetensors header), so only what a step asks for is resident.
`PackExperts` decodes routed experts (`model.llm.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj` EXL3 groups:
trellis/suh/svh + codebook marker) with `oracle/exl3.py` and runs their FFNs, optionally in worker processes.
A folder that lacks some experts (a rank's split staging) raises KeyError when a missing expert is selected.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Sequence

import numpy as np
import torch

import exl3
import st
from inkling2_ref import Rounding, expert_ffn

PROJS = ("gate_proj", "up_proj", "down_proj")


class PackIndex:
    def __init__(self, root: str):
        self.root = root
        idx = os.path.join(root, "model.safetensors.index.json")
        self._entries: dict[str, st.Entry] = {}
        self._headers: dict[str, dict[str, st.Entry]] = {}
        if os.path.exists(idx):
            with open(idx) as f:
                self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        else:
            self.weight_map = {}
            for fn in sorted(os.listdir(root)):
                if fn.endswith(".safetensors"):
                    for name in self._header(fn):
                        if name in self.weight_map:
                            raise ValueError(f"{name} appears in {self.weight_map[name]} and {fn}")
                        self.weight_map[name] = fn

    def _header(self, fn: str) -> dict[str, st.Entry]:
        if fn not in self._headers:
            self._headers[fn] = st.read_header(os.path.join(self.root, fn))
        return self._headers[fn]

    def entry(self, name: str) -> st.Entry:
        if name not in self._entries:
            fn = self.weight_map.get(name)
            if fn is None:
                raise KeyError(f"{name} is not in the pack index")
            path = os.path.join(self.root, fn)
            if not os.path.exists(path):
                raise KeyError(f"{name} lives in {fn}, which this folder does not hold")
            hdr = self._header(fn)
            if name not in hdr:
                raise KeyError(f"the index places {name} in {fn}, whose header does not list it")
            self._entries[name] = hdr[name]
        return self._entries[name]

    def __contains__(self, name: str) -> bool:
        fn = self.weight_map.get(name)
        return fn is not None and os.path.exists(os.path.join(self.root, fn))


class PackWeights:
    def __init__(self, index: PackIndex):
        self.ix = index

    def tensor(self, name: str) -> torch.Tensor:
        return st.read_tensor(self.ix.entry(name))

    def rows(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        e = self.ix.entry(name)
        uniq = sorted(set(int(i) for i in ids.tolist()))
        got = {i: st.read_rows(e, i, i + 1)[0] for i in uniq}
        return torch.stack([got[int(i)] for i in ids.tolist()])

    def row_range(self, name: str, start: int, stop: int) -> torch.Tensor:
        return st.read_rows(self.ix.entry(name), start, stop)


def load_exl3(ix: PackIndex, prefix: str) -> exl3.Exl3Linear:
    """One EXL3 group by prefix: trellis int16, suh/svh fp16 (the pack's `out_scales: always`), codebook marker."""
    trellis = st.read_numpy(ix.entry(prefix + ".trellis"))
    suh = st.read_numpy(ix.entry(prefix + ".suh")).astype(np.float64)
    svh = st.read_numpy(ix.entry(prefix + ".svh")).astype(np.float64)
    markers = [m for m in ("mcg", "mul1") if f"{prefix}.{m}" in ix.weight_map]
    if len(markers) > 1:
        raise ValueError(f"{prefix}: both mcg and mul1 markers")
    if markers:
        raw = st.read_numpy(ix.entry(f"{prefix}.{markers[0]}"))
        cb = exl3.codebook_from_marker(markers[0], int(raw.reshape(-1).view(np.uint32)[0]))
    else:
        cb = "3inst"
    return exl3.Exl3Linear(trellis, cb, suh, svh)


def expert_prefix(layer: int, e: int, proj: str) -> str:
    return f"model.llm.layers.{layer}.mlp.experts.{e}.{proj}"


_worker_ix: PackIndex | None = None


def exit_with_parent(poll_s: float = 1.0) -> None:
    """For a pool worker's initializer: exit as soon as the process that started it is gone. A spawn worker whose
    parent is killed would otherwise wait on its call queue forever (reparented to init), holding its memory."""
    parent = os.getppid()

    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(poll_s)
        os._exit(1)

    threading.Thread(target=watch, name="exit-with-parent", daemon=True).start()


def _worker_init(root: str) -> None:
    global _worker_ix
    exit_with_parent()
    _worker_ix = PackIndex(root)


def _worker_ffn(layer: int, e: int, x: np.ndarray, gate_cap: float, up_cap: float, exact: bool,
                fp16_in: bool = False) -> np.ndarray:
    ix = _worker_ix
    lins = [load_exl3(ix, expert_prefix(layer, e, p)) for p in PROJS]
    return expert_ffn(*lins, x, gate_cap, up_cap, Rounding.exact() if exact else Rounding(), fp16_in)


class PackExperts:
    def __init__(self, index: PackIndex, workers: int = 0):
        self.ix = index
        self.pool = None
        if workers > 0:
            self.pool = ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=_worker_init,
                                            initargs=(index.root,))

    def ffn_many(self, layer: int, jobs: Sequence[tuple[int, np.ndarray]], gate_cap: float, up_cap: float,
                 rounding: Rounding, fp16_in: bool = False) -> list[np.ndarray]:
        if self.pool is None:
            out = []
            for e, x in jobs:
                lins = [load_exl3(self.ix, expert_prefix(layer, e, p)) for p in PROJS]
                out.append(expert_ffn(*lins, x, gate_cap, up_cap, rounding, fp16_in))
            return out
        futs = [self.pool.submit(_worker_ffn, layer, e, x, gate_cap, up_cap, rounding.is_exact, fp16_in)
                for e, x in jobs]
        return [f.result() for f in futs]

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()
