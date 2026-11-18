"""EXL3 (ExLlamaV3 trellis) decoding on the CPU, in numpy.

A port of REXL3's `gb-exl3` CPU reference (`crates/exl3/src/lib.rs` at upstream/main 06181b1), which is itself a port of
TensorFold's `cuda/exl3/format.py` (MIT, ExLlamaV3 format by turboderp). `unpack` reproduces `gb_exl3::unpack` bit for bit
(W_q as fp16 bits), `decode_weight` reproduces `gb_exl3::decode_weight` bit for bit (same butterflies, same order), and
`forward` is the float64 layer `y = ((((x * suh) @ H_K) @ W_q) @ H_N) * svh` with fast Walsh-Hadamard rotations, which
equals `gb_exl3::forward` to float64 rounding. `oracle/exl3dq` (Rust, depends on gb-exl3 by path) is the cross-check.

Shapes: trellis int16 [K/16, N/16, 16 * bits]; suh [K]; svh [N]; W_q / W [K, N] (input dim K, output dim N).
"""

from __future__ import annotations

import numpy as np

HAD = 128
_INST3_MUL = 89226354
_INST3_ADD = 64248484
_MCG_MUL = 0xCBAC1FED
_MUL1_MUL = 0x83DCD12D
_MASK = 0x8FFF8FFF
_FLIP = 0x3B603B60
_MUL1_SCALE = 0x1EEE
_MUL1_BIAS = 0xC931

MARKER_MCG = 0xCBAC1FED
MARKER_MUL1 = 0x83DCD12D
CODEBOOKS = ("3inst", "mcg", "mul1")
BITS_TWICE = (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16)

_tables: dict[str, np.ndarray] = {}


def _f16(bits: np.ndarray) -> np.ndarray:
    return bits.astype(np.uint16).view(np.float16).astype(np.float64)


def codebook(cb: str) -> np.ndarray:
    """fp16 bit pattern (uint16) of every 16-bit trellis state, for one codebook."""
    if cb in _tables:
        return _tables[cb]
    s = np.arange(65536, dtype=np.uint64)
    if cb == "mul1":
        x = (s * np.uint64(_MUL1_MUL)) & np.uint64(0xFFFFFFFF)
        h = 1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
        scale = _f16(np.array([_MUL1_SCALE]))[0]
        bias = _f16(np.array([_MUL1_BIAS]))[0]
        val = h.astype(np.float64) * scale + bias  # exact in float64: one rounding, to fp16
    elif cb in ("mcg", "3inst"):
        if cb == "mcg":
            x = (s * np.uint64(_MCG_MUL)) & np.uint64(0xFFFFFFFF)
        else:
            x = (s * np.uint64(_INST3_MUL) + np.uint64(_INST3_ADD)) & np.uint64(0xFFFFFFFF)
        x = (x & np.uint64(_MASK)) ^ np.uint64(_FLIP)
        val = _f16(x & np.uint64(0xFFFF)) + _f16(x >> np.uint64(16))
    else:
        raise ValueError(f"unknown codebook {cb!r}")
    # numpy's float64 -> float16 is one round-to-nearest-even, as gb_exl3::f16::f64_to_f16
    table = val.astype(np.float16).view(np.uint16)
    table.setflags(write=False)
    _tables[cb] = table
    return table


def bits_twice_from_trellis_last(last: int) -> int:
    if last % 8 != 0 or last // 8 not in BITS_TWICE:
        raise ValueError(f"unsupported trellis width {last}")
    return last // 8


def stream_ends(twice: int) -> np.ndarray:
    """E(p), p = 0..255: where value p's 16-bit window ends in the tile's bitstream (exclusive)."""
    p1 = np.arange(1, 257, dtype=np.int64)
    if twice % 2 == 0:
        return p1 * (twice // 2)
    return (p1 * twice - (p1 % 2)) // 2


def tile_positions() -> tuple[np.ndarray, np.ndarray]:
    """(row, column) inside its 16x16 tile of each of a tile's 256 values, in stream order."""
    p = np.arange(256)
    lane, j = p // 8, p % 8
    r = 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1)
    c = lane // 4 + 8 * (j >> 2)
    return r, c


def states(trellis: np.ndarray, twice: int) -> np.ndarray:
    """16-bit state of every value: trellis int16 [..., 16 * bits] -> uint64 [..., 256]."""
    nw16 = 8 * twice
    if trellis.shape[-1] != nw16:
        raise ValueError(f"tile must hold {nw16} words, got {trellis.shape[-1]}")
    lead = trellis.shape[:-1]
    t = np.ascontiguousarray(trellis).astype(np.int16, copy=False).view(np.uint16).reshape(-1, nw16 // 2, 2)
    words = t[..., 0].astype(np.uint64) | (t[..., 1].astype(np.uint64) << np.uint64(16))
    nw = nw16 // 2
    first = stream_ends(twice) + 32 * nw - 16
    i0 = (first // 32) % nw
    off = (first % 32).astype(np.uint64)
    i1 = (i0 + 1) % nw
    pair = (words[:, i0] << np.uint64(32)) | words[:, i1]
    st = (pair >> (np.uint64(48) - off)) & np.uint64(0xFFFF)
    return st.reshape(*lead, 256)


def unpack(trellis: np.ndarray, codebook_name: str) -> np.ndarray:
    """W_q [K, N] as fp16 bits (uint16), from trellis int16 [K/16, N/16, 16 * bits]."""
    if trellis.ndim != 3:
        raise ValueError(f"trellis must be [K/16, N/16, 16*bits], got {trellis.shape}")
    kt, nt, last = trellis.shape
    twice = bits_twice_from_trellis_last(last)
    if twice % 2 == 1 and codebook_name != "mul1":
        raise ValueError(f"{twice / 2}-bit tiles need the mul1 codebook")
    st = states(trellis, twice)  # [kt, nt, 256]
    vals = codebook(codebook_name)[st.astype(np.int64)]
    r, c = tile_positions()
    tile = np.empty((kt, nt, 256), dtype=np.uint16)
    tile[..., r * 16 + c] = vals
    return tile.reshape(kt, nt, 16, 16).transpose(0, 2, 1, 3).reshape(kt * 16, nt * 16)


def fwht_128(x: np.ndarray, axis: int) -> np.ndarray:
    """H x / sqrt(128) on every block of 128 along `axis`, by the butterflies of gb_exl3's `fwht_128` (same per-element
    operations, same order: the result is bit-identical to the Rust loop)."""
    x = np.moveaxis(np.asarray(x, dtype=np.float64), axis, -1)
    n = x.shape[-1]
    if n % HAD != 0:
        raise ValueError(f"rotated dimension {n} must be a multiple of {HAD}")
    lead = x.shape[:-1]
    y = x.reshape(*lead, n // HAD, HAD).copy()
    h = 1
    while h < HAD:
        v = y.reshape(*lead, n // HAD, HAD // (2 * h), 2, h)
        a = v[..., 0, :].copy()
        b = v[..., 1, :]
        v[..., 0, :] = a + b
        v[..., 1, :] = a - b
        h *= 2
    y *= 1.0 / np.sqrt(np.float64(HAD))
    return np.moveaxis(y.reshape(*lead, n), -1, axis)


def decode_weight(trellis: np.ndarray, codebook_name: str, suh: np.ndarray, svh: np.ndarray) -> np.ndarray:
    """The layer's effective float64 weight [K, N]: W[i, j] = suh[i] * (H_K W_q H_N)[i, j] * svh[j], so forward(x) = x @ W.
    Bit-identical to gb_exl3::decode_weight."""
    w = _f16(unpack(trellis, codebook_name))
    k, n = w.shape
    if k % HAD or n % HAD:
        raise ValueError(f"K={k} and N={n} must be multiples of {HAD}")
    suh = np.asarray(suh, dtype=np.float64)
    svh = np.asarray(svh, dtype=np.float64)
    if suh.shape != (k,) or svh.shape != (n,):
        raise ValueError(f"scales {suh.shape}/{svh.shape} do not match [K={k}], [N={n}]")
    w = fwht_128(w, axis=0)  # H_K W_q, per column
    w = w * suh[:, None]
    w = fwht_128(w, axis=1)  # (.) H_N, per row
    return w * svh[None, :]


def forward(x: np.ndarray, wq: np.ndarray, suh: np.ndarray, svh: np.ndarray) -> np.ndarray:
    """y = ((((x * suh) @ H_K) @ W_q) @ H_N) * svh in float64; x [rows, K], W_q [K, N] float64 values -> [rows, N]."""
    xh = fwht_128(np.asarray(x, dtype=np.float64) * suh[None, :], axis=1)
    y = xh @ wq
    return fwht_128(y, axis=1) * svh[None, :]


class Exl3Linear:
    """One EXL3 linear layer, decoded once (W_q kept as float64 values)."""

    def __init__(self, trellis: np.ndarray, codebook_name: str, suh: np.ndarray, svh: np.ndarray):
        self.wq = _f16(unpack(trellis, codebook_name))
        self.suh = np.asarray(suh, dtype=np.float64)
        self.svh = np.asarray(svh, dtype=np.float64)
        k, n = self.wq.shape
        if self.suh.shape != (k,) or self.svh.shape != (n,):
            raise ValueError(f"scales {self.suh.shape}/{self.svh.shape} do not match W_q {self.wq.shape}")

    @property
    def k(self) -> int:
        return self.wq.shape[0]

    @property
    def n(self) -> int:
        return self.wq.shape[1]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return forward(x, self.wq, self.suh, self.svh)


def codebook_from_marker(name: str, marker: int) -> str:
    """The codebook a group's marker tensor names (`.mcg` / `.mul1`); a wrong value is refused, as gb_exl3's scan does."""
    want = {"mcg": MARKER_MCG, "mul1": MARKER_MUL1}[name]
    if marker & 0xFFFFFFFF != want:
        raise ValueError(f".{name} marker holds {marker & 0xFFFFFFFF:#x}, expected {want:#x}")
    return name
