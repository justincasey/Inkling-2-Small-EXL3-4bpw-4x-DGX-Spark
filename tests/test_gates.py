#!/usr/bin/env python3
"""Unit check of the pure parts of bench/tools/gates.py, synth_images.py and the inkling_api helpers (stdlib only, no
server): the PNG writer decodes back (also through every PNG row filter), record diffs (vision `diff`, `spec-diff`)
find a one-token difference and pass identical records, usage parsing.

    python3 tests/test_gates.py
"""

from __future__ import annotations

import base64
import contextlib
import copy
import io
import json
import os
import struct
import sys
import tempfile
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench", "tools"))
import gates  # noqa: E402
import inkling_api as ia  # noqa: E402
import synth_images as si  # noqa: E402


def check(cond, what):
    if not cond:
        raise AssertionError(what)


def filtered_png(w, h, pixel) -> bytes:
    """An RGB PNG whose rows cycle through filter types 0..4 (the encoder side of the PNG spec)."""
    rows = [bytes(c for x in range(w) for c in pixel(x, y)) for y in range(h)]
    raw, prev = bytearray(), bytes(w * 3)
    for y, row in enumerate(rows):
        f = y % 5
        out = bytearray([f])
        for i, v in enumerate(row):
            a = row[i - 3] if i >= 3 else 0
            b = prev[i]
            c = prev[i - 3] if i >= 3 else 0
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            pred = [0, a, b, (a + b) // 2, a if pa <= pb and pa <= pc else b if pb <= pc else c][f]
            out.append((v - pred) & 255)
        raw += out
        prev = row

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b""))


def test_png():
    for name, (fn, _, words) in si.IMAGES.items():
        data = si.image_png(name)
        w, h, rows = si.decode_png(data)
        check((w, h) == (si.SIZE, si.SIZE), f"{name}: {w}x{h}")
        check(all(rows[y][x] == fn(x, y) for y in range(h) for x in range(w)), f"{name}: pixels differ after decoding")
        url = si.data_url(data)
        check(url.startswith("data:image/png;base64,") and base64.b64decode(url.split(",", 1)[1]) == data,
              f"{name}: data URL round trip")
        check(words and all(w.islower() for w in words), f"{name}: colour words {words}")
    print(f"ok   synth_images: {len(si.IMAGES)} images write and decode back pixel for pixel ({si.SIZE}x{si.SIZE}, "
          f"{si.SIZE // 40}x{si.SIZE // 40} cells of 40 px)")
    check(si.red_square(0, 0) == si.WHITE and si.red_square(120, 120) == si.RED, "red square geometry")
    check(si.blue_circle(120, 120) == si.BLUE and si.blue_circle(5, 5) == si.WHITE, "blue circle geometry")
    check(si.green_yellow(0, 0) == si.GREEN and si.green_yellow(239, 0) == si.YELLOW, "green/yellow halves")
    pix = lambda x, y: ((x * 37 + y * 11) % 256, (x * y) % 256, (255 - x * 5 - y) % 256)  # noqa: E731
    w, h, rows = si.decode_png(filtered_png(23, 17, pix))
    check((w, h) == (23, 17) and all(rows[y][x] == pix(x, y) for y in range(h) for x in range(w)),
          "decode_png: rows with filters 1..4")
    print("ok   decode_png: every row filter (0..4) decodes to the encoded pixels")
    bad = bytearray(si.image_png("blue-circle"))
    bad[40] ^= 0xFF
    try:
        si.decode_png(bytes(bad))
    except (ValueError, zlib.error):
        print("ok   decode_png: a corrupted chunk is refused")
    else:
        raise AssertionError("decode_png accepted a corrupted PNG")


def run(argv) -> tuple[int, str]:
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()) as out:
        rc = gates.main(argv)
    return rc, err.getvalue() + out.getvalue()


def write(path, rec):
    with open(path, "w") as f:
        json.dump(rec, f)
    return path


def vision_record(images=True):
    img = [{"name": n, "png_sha256": "s" + n, "question": "q", "token_ids": [1, 2, 3, 200006], "finish_reason": "stop",
            "content": "Red.", "reasoning": "", "usage": {}, "seconds": 1.0, "names_colour": True} for n in si.IMAGES]
    text = [{"name": n, "prompt_tokens": 9, "prompt_sha256": "p" + n, "generated_ids": [5, 6, 7], "finish_reason": "length",
             "usage": {}} for n in ("short", "medium", "long", "xlong")]
    return {"meta": {"kind": "vision", "images": images, "label": "x"}, "images": img if images else [], "text": text}


def spec_reply(ids, drafted=0):
    u = {"completion_tokens": len(ids), "spec": {"drafted": drafted, "accepted": drafted // 2, "rounds": len(ids)}}
    return {"generated_ids": ids, "finish_reason": "length", "usage": u, "spec": ia.spec_counters(u)}


def spec_record(drafted):
    names = ("short", "medium", "long", "xlong", "70k")
    replies = [{"name": n, "prompt_sha256": "p" + n, "serial": spec_reply([1, 2, 3]), "drafted": spec_reply([1, 2, 3], drafted)}
               for n in names]
    conc = [{"name": n, "prompt_sha256": "p" + n, "drafted": spec_reply([1, 2, 3], drafted)} for n in names[:4]]
    return {"meta": {"kind": "spec", "label": f"drafted {drafted}"}, "replies": replies, "concurrent": conc,
            "acceptance": {"drafted_all": gates.totals([r["drafted"] for r in replies] + [c["drafted"] for c in conc])}}


def test_diffs(tmp):
    a = write(os.path.join(tmp, "a.json"), vision_record())
    b = write(os.path.join(tmp, "b.json"), vision_record())
    rc, out = run(["diff", a, b])
    check(rc == 0 and "identical" in out, f"identical vision records: rc {rc}\n{out}")
    r = vision_record()
    r["images"][1]["token_ids"][2] = 4
    rc, out = run(["diff", a, write(os.path.join(tmp, "c.json"), r)])
    check(rc == 1 and "image/blue-circle: DIFFERS in ['token_ids']; ids first differ at 2" in out,
          f"one image token: rc {rc}\n{out}")
    r = vision_record()
    r["text"][3]["generated_ids"] = [5, 6, 8]
    rc, out = run(["diff", a, write(os.path.join(tmp, "d.json"), r)])
    check(rc == 1 and "text/xlong: DIFFERS" in out, f"one text token: rc {rc}\n{out}")
    r = vision_record()
    r["images"][0]["content"] = "Red!"
    rc, out = run(["diff", a, write(os.path.join(tmp, "e.json"), r)])
    check(rc == 1 and "DIFFERS in ['content']" in out, f"content bytes: rc {rc}\n{out}")
    off = write(os.path.join(tmp, "off.json"), vision_record(images=False))
    rc, out = run(["diff", a, off])
    check(rc == 0 and "text compared alone" in out, f"vision on vs off, same text: rc {rc}\n{out}")
    r = vision_record(images=False)
    r["text"].pop()
    rc, out = run(["diff", a, write(os.path.join(tmp, "f.json"), r)])
    check(rc == 1 and "text/xlong: only in" in out, f"a missing text reply: rc {rc}\n{out}")
    print("ok   gates.py diff: identical records pass; one image token, one text token, content bytes, a missing reply "
          "fail; a VISION=off record compares text alone")

    sa = write(os.path.join(tmp, "sa.json"), spec_record(12))
    sb = write(os.path.join(tmp, "sb.json"), spec_record(0))
    rc, out = run(["spec-diff", sa, sb])
    check(rc == 0 and "identical" in out, f"spec-diff identical: rc {rc}\n{out}")
    r = spec_record(0)
    r["concurrent"][2]["drafted"]["generated_ids"] = [1, 9, 3]
    rc, out = run(["spec-diff", sa, write(os.path.join(tmp, "sc.json"), r)])
    check(rc == 1 and "concurrent/long: DIFFERS: ids differ at 1" in out, f"spec-diff one token: rc {rc}\n{out}")
    r = spec_record(0)
    r["replies"][4]["prompt_sha256"] = "other"
    rc, out = run(["spec-diff", sa, write(os.path.join(tmp, "sd.json"), r)])
    check(rc == 1 and "different prompts" in out, f"spec-diff prompts: rc {rc}\n{out}")
    try:
        run(["spec-diff", sa, a])
    except SystemExit as e:
        check("not a gates.py spec record" in str(e), str(e))
    else:
        raise AssertionError("spec-diff accepted a vision record")
    print("ok   gates.py spec-diff: identical drafted replies pass; one token, other prompts, a wrong record kind fail")


def test_helpers():
    u = {"completion_tokens": 128, "prompt_tokens_details": {"cached_tokens": 4096},
         "spec": {"drafted": 300, "accepted": 120, "rounds": 64}}
    s = ia.spec_counters(u)
    check(s == {"drafted": 300, "accepted": 120, "rounds": 64, "acceptance": 0.4, "tokens_per_round": 2.0}, str(s))
    check(ia.spec_counters({"completion_tokens": 3}) is None and ia.cached_tokens(u) == 4096
          and ia.cached_tokens({}) is None, "spec/cached parsing")
    t = gates.totals([{"spec": s, "usage": u}, {"spec": None, "usage": {"completion_tokens": 2}}])
    check(t["drafted"] == 300 and t["completion_tokens"] == 130 and t["acceptance"] == 0.4, str(t))
    x = {"generated_ids": [1, 2, 3], "finish_reason": "stop"}
    check(gates.first_diff(x, copy.deepcopy(x)) is None, "first_diff equal")
    check("ids differ at 3" in gates.first_diff(x, {"generated_ids": [1, 2, 3, 4], "finish_reason": "length"}), "prefix")
    check(gates.common_prefix([1, 2, 3], [1, 2, 4]) == 2 and gates.common_prefix([1], [1, 2]) == 1, "common_prefix")
    m = ia.parse_metrics("# HELP rexl3_prefix_cache_hits_total x\n# TYPE rexl3_prefix_cache_hits_total counter\n"
                         "rexl3_prefix_cache_hits_total 3\nrexl3_x{a=\"b c\"} 1.5\n")
    check(m == {"rexl3_prefix_cache_hits_total": 3.0, 'rexl3_x{a="b c"}': 1.5}, str(m))
    check(len(gates.digest([1, 2])) == 64 and gates.digest([1, 2]) != gates.digest([2, 1]), "digest")
    print("ok   usage.spec acceptance and tokens/round, cached_tokens, totals, first_diff, /metrics parsing")


def main() -> int:
    test_png()
    with tempfile.TemporaryDirectory() as tmp:
        test_diffs(tmp)
    test_helpers()
    return 0


if __name__ == "__main__":
    sys.exit(main())
