"""Minimal safetensors I/O for the oracle: header parsing, positional reads (no whole-file mmap, so resident memory stays
at what a caller asks for) and a writer. Dtypes: BF16, F16, F32, F64, I16, I32, I64, U16, U8."""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass

import numpy as np
import torch

MAX_HEADER_BYTES = 100_000_000

_NP = {"F16": np.float16, "F32": np.float32, "F64": np.float64, "I16": np.int16, "I32": np.int32, "I64": np.int64,
       "U16": np.uint16, "U8": np.uint8}
_SIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I16": 2, "I32": 4, "I64": 8, "U16": 2, "U8": 1}
_TORCH_TO_ST = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.float64: "F64",
                torch.int16: "I16", torch.int32: "I32", torch.int64: "I64", torch.uint8: "U8"}


@dataclass(frozen=True)
class Entry:
    path: str
    dtype: str
    shape: tuple[int, ...]
    start: int  # absolute byte offset of the data in the file
    nbytes: int

    @property
    def row_bytes(self) -> int:
        n = _SIZE[self.dtype]
        for d in self.shape[1:]:
            n *= d
        return n


def parse_header(raw: bytes, path: str, header_len: int) -> dict[str, Entry]:
    meta = json.loads(raw)
    out = {}
    base = 8 + header_len
    for name, v in meta.items():
        if name == "__metadata__":
            continue
        a, b = v["data_offsets"]
        shape = tuple(int(d) for d in v["shape"])
        dt = v["dtype"]
        if dt in _SIZE and (b - a) != int(np.prod(shape, dtype=np.int64)) * _SIZE[dt]:
            raise ValueError(f"{path}: {name} {dt} {shape} spans {b - a} bytes")
        out[name] = Entry(path, dt, shape, base + a, b - a)
    return out


def read_header(path: str) -> dict[str, Entry]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        if n > MAX_HEADER_BYTES:
            raise ValueError(f"{path}: header length {n} exceeds the cap")
        return parse_header(f.read(n), path, n)


def _pread(path: str, start: int, n: int) -> bytes:
    fd = os.open(path, os.O_RDONLY)
    try:
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            chunk = os.preadv(fd, [view[got:]], start + got)
            if chunk == 0:
                raise OSError(f"{path}: short read at {start + got}")
            got += chunk
        return bytes(buf)
    finally:
        os.close(fd)


def decode(raw: bytes, dtype: str, shape: tuple[int, ...]) -> torch.Tensor:
    """Bytes -> torch tensor of the stored dtype (BF16 stays bfloat16)."""
    if dtype == "BF16":
        a = np.frombuffer(raw, dtype=np.uint16).copy()
        return torch.from_numpy(a.view(np.int16)).view(torch.bfloat16).reshape(shape)
    if dtype == "U16":
        return torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).astype(np.int32)).reshape(shape)
    return torch.from_numpy(np.frombuffer(raw, dtype=_NP[dtype]).copy()).reshape(shape)


def read_tensor(e: Entry) -> torch.Tensor:
    return decode(_pread(e.path, e.start, e.nbytes), e.dtype, e.shape)


def read_rows(e: Entry, start: int, stop: int) -> torch.Tensor:
    """Rows [start, stop) of a tensor (first dimension)."""
    rb = e.row_bytes
    return decode(_pread(e.path, e.start + start * rb, (stop - start) * rb), e.dtype, (stop - start, *e.shape[1:]))


def read_numpy(e: Entry) -> np.ndarray:
    """Raw numpy view for non-BF16 dtypes (EXL3 trellis int16, scales fp16, markers int32)."""
    if e.dtype == "BF16":
        raise ValueError("BF16 has no numpy dtype; use read_tensor")
    return np.frombuffer(_pread(e.path, e.start, e.nbytes), dtype=_NP[e.dtype]).reshape(e.shape)


def encode(t) -> tuple[str, tuple[int, ...], bytes]:
    if isinstance(t, np.ndarray):
        inv = {v: k for k, v in _NP.items()}
        dt = inv[np.dtype(t.dtype).type]
        return dt, tuple(t.shape), np.ascontiguousarray(t).tobytes()
    t = t.detach().contiguous().cpu()
    dt = _TORCH_TO_ST[t.dtype]
    if t.dtype == torch.bfloat16:
        raw = t.view(torch.int16).numpy().tobytes()
    else:
        raw = t.numpy().tobytes()
    return dt, tuple(t.shape), raw


def write(path: str, tensors: dict, metadata: dict[str, str] | None = None) -> None:
    """Write a safetensors file from torch tensors / numpy arrays (insertion order kept)."""
    header: dict = {}
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    blobs = []
    off = 0
    for name, t in tensors.items():
        dt, shape, raw = encode(t)
        header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def load_all(path: str) -> dict[str, torch.Tensor]:
    return {k: read_tensor(e) for k, e in read_header(path).items()}
