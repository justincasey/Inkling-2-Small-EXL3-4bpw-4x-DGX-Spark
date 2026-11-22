#!/usr/bin/env python3
"""Independent check of the engine's MXFP8 quantizer (crates/cuda/src/inkling2/mxfp8.rs of the REXL3 engine) on deterministic weights (stdlib only).

The format is written here from its definition, not ported: e4m3 is enumerated (every code's exact value) and a value is encoded as the nearest one,
ties to the even code, saturating at 448; a block of 32 weights gets the scale 2^(floor(log2 amax) - 8), or twice that when amax is past 448 of it and
that scale leaves the smaller squared error (compared in float64), the exponent kept within 27..134 (stored with bias 127). The weights are the
engine test's `golden_weights()` ([16][512] bf16 from an integer LCG with per-block magnitudes, an outlier, a clipped block, tiny blocks, zeros and -0).
The engine's host test hashes the dequantized bf16 bits and the scale bytes (FNV-1a 64); both must equal GOLDEN below. If the quantizer is changed on
purpose, change both constants together.

    python3 tests/test_mxfp8.py
"""

from __future__ import annotations

import math
import struct
import sys

GOLDEN = (13162351915896225148, 15951772806054347177)  # (dequantized bf16 bytes, scale bytes)
N, K = 16, 512
E_MIN, E_MAX = 27, 134


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bf16_bits(x: float) -> int:
    b = struct.unpack("<I", struct.pack("<f", x))[0]
    return ((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFFFFFF) >> 16


def bf16_value(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def fnv(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def golden_weights() -> list[int]:
    s, out = 12345, []
    for i in range(N * K):
        s = (s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        blk = i // 32
        x = ((s >> 40) - (1 << 23)) * 2.0**-23 * 2.0 ** ((blk * 7919) % 14 - 10)
        if (blk % 5, i % 32) == (1, 3):
            x = f32(x * 40.0)
        elif (blk % 5, i % 32) == (2, 0):
            x = 0.0
        elif (blk % 5, i % 32) == (3, 7):
            x = -0.0
        elif blk % 5 == 4 and blk % 10 == 4:
            x *= 2.0**-14
        out.append(bf16_bits(f32(x)))
    return out


# e4m3 from its definition: sign, 4 exponent bits (bias 7), 3 mantissa bits, subnormals at exponent 0, no infinity, 0x7F / 0xFF are NaN.
CODES = {}
for c in range(128):
    e, m = c >> 3, c & 7
    if c == 0x7F:
        continue
    CODES[c] = m * 2.0**-9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
assert max(CODES.values()) == 448.0 and len(CODES) == 127


def encode(x: float) -> int:
    """The code of the e4m3 value nearest to x (ties to the even code), saturating; sign included."""
    sign = 0x80 if math.copysign(1.0, x) < 0 else 0
    a = min(abs(x), 448.0)
    best = None
    for c, v in CODES.items():
        d = abs(a - v)
        if best is None or d < best[0] or (d == best[0] and c % 2 == 0 and best[1] % 2 == 1):
            best = (d, c)
    return sign | best[1]


def decode(c: int) -> float:
    return math.copysign(CODES[c & 0x7F], -1.0 if c & 0x80 else 1.0)


_enc = {}


def encode_cached(x: float) -> int:
    k = (x, math.copysign(1.0, x))
    if k not in _enc:
        _enc[k] = encode(x)
    return _enc[k]


def quantize_block(v: list[float]) -> tuple[int, list[float]]:
    """(stored exponent, dequantized values) of a block of 32."""
    amax = max(abs(x) for x in v)
    if amax == 0.0:
        return 127, [decode(encode_cached(x)) for x in v]
    e0 = min(max(math.floor(math.log2(amax)) + 127 - 8, E_MIN), E_MAX)

    def run(e: int):
        scale = 2.0 ** (e - 127)
        dq = [decode(encode_cached(x / scale)) * scale for x in v]
        return sum((x - d) ** 2 for x, d in zip(v, dq)), dq

    err0, dq0 = run(e0)
    if amax <= 448.0 * 2.0 ** (e0 - 127) or e0 == E_MAX:
        return e0, dq0
    err1, dq1 = run(e0 + 1)
    return (e0 + 1, dq1) if err1 < err0 else (e0, dq0)


def main() -> int:
    w = golden_weights()
    deq, scales = bytearray(), bytearray()
    for r in range(N):
        for b in range(K // 32):
            blk = [bf16_value(w[r * K + b * 32 + j]) for j in range(32)]
            e, dq = quantize_block(blk)
            scales.append(e)
            for d in dq:
                deq += struct.pack("<H", bf16_bits(f32(d)))
    got = (fnv(bytes(deq)), fnv(bytes(scales)))
    assert got == GOLDEN, f"the independent quantizer disagrees with the engine's golden: {got} != {GOLDEN}"
    print("ok   mxfp8: an independent implementation of the format reproduces the engine's quantization of 8,192 golden weights bit for bit")
    # a dequantized weight is exactly representable in bf16 (so the kernels' bf16 arithmetic is exact) and a scale stays in range
    assert all(E_MIN <= e <= E_MAX for e in scales)
    for i in range(0, len(deq), 2):
        v = bf16_value(struct.unpack_from("<H", deq, i)[0])
        assert f32(v) == v
    print("ok   mxfp8: scales within 27..134, every dequantized value exact in bf16")
    # the ends of the exponent range, by the definition alone: a block of 2^-100 clamps up to 27, a zero block stores 127, a block near 2^120 clamps to 134
    assert quantize_block([2.0**-100] + [0.0] * 31)[0] == E_MIN
    assert quantize_block([0.0] * 32)[0] == 127
    e, dq = quantize_block([2.0**120] + [0.0] * 31)
    assert e == E_MAX and dq[0] == 448.0 * 2.0 ** (E_MAX - 127), "the largest scale clips, it does not grow"
    print("ok   mxfp8: the exponent clamps at both ends and a zero block stores scale 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
