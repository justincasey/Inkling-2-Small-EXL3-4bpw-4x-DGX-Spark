"""Tiny safetensors reader/writer for the bench tools (stdlib only, the layout of oracle/st.py): header parsing,
positional reads, and a streaming writer whose tensors are produced piece by piece, so a [N, 200058] logits tensor is
never held in memory."""

from __future__ import annotations

import json
import math
import struct
from typing import Callable, Iterable

SIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I16": 2, "I32": 4, "I64": 8, "U16": 2, "U8": 1}
MAX_HEADER_BYTES = 100_000_000


def read_header(path: str) -> tuple[dict, dict]:
    """(tensors {name: {"dtype", "shape", "start", "nbytes"}} with absolute offsets, __metadata__)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        if n > MAX_HEADER_BYTES:
            raise ValueError(f"{path}: header length {n} exceeds the cap")
        h = json.loads(f.read(n))
    meta = h.pop("__metadata__", {})
    out = {}
    for name, v in h.items():
        a, b = v["data_offsets"]
        shape = [int(d) for d in v["shape"]]
        if b - a != math.prod(shape) * SIZE[v["dtype"]]:
            raise ValueError(f"{path}: {name} {v['dtype']} {shape} spans {b - a} bytes")
        out[name] = {"dtype": v["dtype"], "shape": shape, "start": 8 + n + a, "nbytes": b - a}
    return out, meta


def read_bytes(path: str, entry: dict, offset: int = 0, nbytes: int | None = None) -> bytes:
    n = entry["nbytes"] - offset if nbytes is None else nbytes
    with open(path, "rb") as f:
        f.seek(entry["start"] + offset)
        raw = f.read(n)
    if len(raw) != n:
        raise OSError(f"{path}: short read")
    return raw


def read_ints(path: str, name: str) -> list[int]:
    """A whole I64/I32 tensor as a flat list."""
    t, _ = read_header(path)
    e = t[name]
    fmt = {"I64": "q", "I32": "i"}[e["dtype"]]
    raw = read_bytes(path, e)
    return list(struct.unpack(f"<{len(raw) // SIZE[e['dtype']]}{fmt}", raw))


def read_f32_row(path: str, name: str, row: int) -> list[float]:
    t, _ = read_header(path)
    e = t[name]
    if e["dtype"] != "F32":
        raise ValueError(f"{path}: {name} is {e['dtype']}, not F32")
    cols = math.prod(e["shape"][1:])
    raw = read_bytes(path, e, row * cols * 4, cols * 4)
    return list(struct.unpack(f"<{cols}f", raw))


def write_streaming(path: str, tensors: list[tuple[str, str, list[int], Callable[[], Iterable[bytes]]]],
                    metadata: dict[str, str] | None = None) -> None:
    """Write [(name, dtype, shape, producer)] in order; each producer yields the tensor's bytes in row-major order and
    must yield exactly prod(shape) * itemsize bytes (checked)."""
    header: dict = {}
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    off = 0
    for name, dt, shape, _ in tensors:
        n = math.prod(shape) * SIZE[dt]
        header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for name, dt, shape, produce in tensors:
            want, got = math.prod(shape) * SIZE[dt], 0
            for piece in produce():
                f.write(piece)
                got += len(piece)
            if got != want:
                raise ValueError(f"{name}: produced {got} bytes, the header says {want}")
