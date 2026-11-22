#!/usr/bin/env python3
"""Independent check of the engine's MXINT8 quantizer (crates/cuda/src/inkling2/mxint8.rs of the REXL3 engine) on deterministic weights (stdlib only).

The format is written here from its definition, not ported: a block of 32 weights (bf16) gets the scale 2^(floor(log2 amax) - 6) (amax then lies in 64..128 of
it), the exponent kept within 27..134 (stored with bias 127; a block of zeros stores 127); each weight is divided by the scale, rounded to the nearest integer
(ties to even) and clamped to -127..127; the dequantized value is code x scale, exactly representable in bf16. The weights are the engine test's
`golden_weights()` ([16][512] bf16 from an integer LCG with per-block magnitudes, an outlier, a zero among normals, -0 and tiny blocks). The engine's host test
hashes the dequantized bf16 bits and the scale bytes (FNV-1a 64); both must equal GOLDEN below. If the quantizer is changed on purpose, change both
constants together.

    python3 tests/test_mxint8.py
"""

from __future__ import annotations

import math
import struct
import sys

GOLDEN = (2454325368447749915, 6942978267287460357)  # (dequantized bf16 bytes, scale bytes)
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


def quantize_block(v: list[float]) -> tuple[int, list[float]]:
    """(stored exponent, dequantized values) of a block of 32."""
    amax = max(abs(x) for x in v)
    if amax == 0.0:
        return 127, [0.0] * 32
    e = min(max(math.floor(math.log2(amax)) - 6 + 127, E_MIN), E_MAX)
    scale = 2.0 ** (e - 127)
    # Python's round() is round-half-even on floats
    codes = [max(-127, min(127, round(x / scale))) for x in v]
    return e, [c * scale for c in codes]


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
    print("ok   mxint8: an independent implementation of the format reproduces the engine's quantization of 8,192 golden weights bit for bit")
    # a dequantized weight is exactly representable in bf16 (so the kernels' bf16 arithmetic is exact) and a scale stays in range
    assert all(E_MIN <= e <= E_MAX for e in scales)
    for i in range(0, len(deq), 2):
        v = bf16_value(struct.unpack_from("<H", deq, i)[0])
        assert f32(v) == v
    print("ok   mxint8: scales within 27..134, every dequantized value exact in bf16")
    # the ends of the exponent range, by the definition alone: a block of 2^-100 clamps up to 27, a zero block stores 127, a block near 2^120 clamps to 134
    assert quantize_block([2.0**-100] + [0.0] * 31)[0] == E_MIN
    assert quantize_block([0.0] * 32)[0] == 127
    e, dq = quantize_block([2.0**120] + [0.0] * 31)
    assert e == E_MAX and dq[0] == 127.0 * 2.0 ** (E_MAX - 127), "the largest scale clips, it does not grow"
    # amax in 64..128 of the scale; 127.9 clips to 127; 128 moves to the next binade
    assert quantize_block([100.0] + [0.0] * 31) == (127, [100.0] + [0.0] * 31)
    assert quantize_block([127.9] + [0.0] * 31)[1][0] == 127.0
    assert quantize_block([128.0] + [0.0] * 31) == (128, [128.0] + [0.0] * 31)
    print("ok   mxint8: the exponent clamps at both ends, a zero block stores scale 1, 127.9 clips to 127 and 128 takes the next scale")
    return 0


if __name__ == "__main__":
    sys.exit(main())
