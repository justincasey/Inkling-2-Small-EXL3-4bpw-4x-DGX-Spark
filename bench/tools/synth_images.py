"""Synthetic test images for the vision checks (stdlib only: zlib + struct).

Each image is a known colour and shape on white, 240 x 240 pixels (6 x 6 of Inkling 2's 40-pixel cells, so nothing is
padded), written as an 8-bit RGB PNG and sent as a `data:image/png;base64,...` URL. `decode_png` reads such a PNG back
(every filter type of the PNG spec, 8-bit RGB or RGBA, non-interlaced), for the unit tests.
"""

from __future__ import annotations

import base64
import struct
import zlib

SIZE = 240
WHITE = (255, 255, 255)
RED = (220, 20, 20)
BLUE = (20, 40, 220)
GREEN = (20, 170, 40)
YELLOW = (245, 215, 20)


def png(width: int, height: int, pixel) -> bytes:
    """An 8-bit RGB PNG of `pixel(x, y) -> (r, g, b)`; every row filter 0 (none)."""
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw.extend(pixel(x, y))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")


def decode_png(data: bytes) -> tuple[int, int, list[list[tuple[int, ...]]]]:
    """(width, height, rows of RGB pixel tuples) of an 8-bit RGB/RGBA non-interlaced PNG; checks every chunk's CRC."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    pos, idat, ihdr = 8, b"", None
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos:pos + 4])
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        (crc,) = struct.unpack(">I", data[pos + 8 + n:pos + 12 + n])
        if zlib.crc32(kind + body) & 0xFFFFFFFF != crc:
            raise ValueError(f"{kind!r}: bad CRC")
        if kind == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat += body
        elif kind == b"IEND":
            break
        pos += 12 + n
    if ihdr is None:
        raise ValueError("no IHDR")
    w, h, depth, ctype, _, _, interlace = ihdr
    if depth != 8 or ctype not in (2, 6) or interlace:
        raise ValueError(f"unsupported PNG: depth {depth}, colour type {ctype}, interlace {interlace}")
    bpp = 3 if ctype == 2 else 4
    raw, stride = zlib.decompress(idat), w * bpp
    rows, prev, i = [], bytearray(stride), 0
    for _ in range(h):
        f, line = raw[i], bytearray(raw[i + 1:i + 1 + stride])
        i += 1 + stride
        for x in range(stride):
            a = line[x - bpp] if x >= bpp else 0
            b = prev[x]
            c = prev[x - bpp] if x >= bpp else 0
            if f == 1:
                line[x] = (line[x] + a) & 255
            elif f == 2:
                line[x] = (line[x] + b) & 255
            elif f == 3:
                line[x] = (line[x] + (a + b) // 2) & 255
            elif f == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[x] = (line[x] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
            elif f != 0:
                raise ValueError(f"bad filter {f}")
        rows.append([tuple(line[x:x + 3]) for x in range(0, stride, bpp)])
        prev = line
    return w, h, rows


def data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


def red_square(x: int, y: int):
    return RED if 60 <= x < 180 and 60 <= y < 180 else WHITE


def blue_circle(x: int, y: int):
    return BLUE if (x - 119.5) ** 2 + (y - 119.5) ** 2 <= 80 ** 2 else WHITE


def green_yellow(x: int, y: int):
    return GREEN if x < SIZE // 2 else YELLOW


# name -> (pixel function, the question, the colour words the reply must all contain (case-insensitive))
IMAGES = {
    "red-square": (red_square, "What colour is the shape in this image, and what shape is it? Answer briefly.", ["red"]),
    "blue-circle": (blue_circle, "What colour is the shape in this image, and what shape is it? Answer briefly.", ["blue"]),
    "green-yellow": (green_yellow, "This image has a left half and a right half. What colour is each half? Answer briefly.",
                     ["green", "yellow"]),
}


def image_png(name: str) -> bytes:
    return png(SIZE, SIZE, IMAGES[name][0])
