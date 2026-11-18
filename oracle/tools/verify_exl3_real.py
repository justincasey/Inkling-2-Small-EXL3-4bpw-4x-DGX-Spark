"""Fetch a few real EXL3 expert groups from the pinned pack (HTTP range reads from the Hugging Face mirror, no full
download), check the numpy decode against REXL3's CPU decoder (exl3dq) bit for bit, and record per-expert weight and
FFN output statistics as a sanity check of the 4-bit decode.

    oracle/.venv/bin/python oracle/tools/verify_exl3_real.py [--layers 0 47] [--experts 0 1 2 3 383]

Writes oracle/.cache/real-groups/{groups.safetensors,groups.json} (read by tests/test_exl3.py) and
oracle/records/exl3-real-verify.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import struct
import subprocess
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
sys.path.insert(0, ORACLE)

import exl3  # noqa: E402
import st  # noqa: E402
from inkling2_ref import expert_ffn, np_bf16, Rounding  # noqa: E402

REPO = "0xSero/Inkling-2-Small-EXL3-4bpw"
REV = "996dc404c1feb3abd5330af8a3c52270ffe0008d"
BASE = f"https://huggingface.co/{REPO}/resolve/{REV}/"
OUT = os.path.join(ORACLE, ".cache", "real-groups")


def _token() -> str | None:
    p = os.path.expanduser("~/.cache/huggingface/token")
    return open(p).read().strip() if os.path.exists(p) else None


def get(fn: str, start: int | None = None, stop: int | None = None) -> bytes:
    req = urllib.request.Request(BASE + fn)
    if start is not None:
        req.add_header("Range", f"bytes={start}-{stop - 1}")
    tok = _token()
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if start is not None and len(data) != stop - start:
                raise OSError(f"{fn}: got {len(data)} bytes for a {stop - start}-byte range")
            return data
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError


def remote_header(fn: str) -> dict[str, st.Entry]:
    (n,) = struct.unpack("<Q", get(fn, 0, 8))
    return st.parse_header(get(fn, 8, 8 + n), fn, n)


def fetch(names: list[str], weight_map: dict[str, str]) -> dict[str, np.ndarray | object]:
    by_file: dict[str, list[str]] = {}
    for n in names:
        by_file.setdefault(weight_map[n], []).append(n)
    out = {}
    for fn, ns in sorted(by_file.items()):
        hdr = remote_header(fn)
        for n in ns:
            e = hdr[n]
            out[n] = st.decode(get(fn, e.start, e.start + e.nbytes), e.dtype, e.shape)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 47])
    ap.add_argument("--experts", type=int, nargs="+", default=[0, 1, 2, 3, 383])
    ap.add_argument("--tokens", type=int, default=16, help="random normed inputs per expert for the FFN statistics")
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    weight_map = json.loads(get("model.safetensors.index.json"))["weight_map"]
    prefixes = [f"model.llm.layers.{lay}.mlp.experts.{e}.{p}" for lay in a.layers for e in a.experts
                for p in ("gate_proj", "up_proj", "down_proj")]
    names = [f"{p}.{part}" for p in prefixes for part in ("trellis", "suh", "svh", "mcg")]
    norms = [f"model.llm.layers.{lay}.mlp_norm.weight" for lay in a.layers]
    t0 = time.time()
    got = fetch(names + norms, weight_map)
    fetched_s = time.time() - t0
    tensors = {n: (v.numpy() if hasattr(v, "numpy") and n in names else v) for n, v in got.items() if n in names}
    st.write(os.path.join(OUT, "groups.safetensors"), tensors)
    with open(os.path.join(OUT, "groups.json"), "w") as f:
        json.dump({"repo": REPO, "revision": REV, "prefixes": prefixes}, f, indent=1)

    crate = os.path.join(ORACLE, "exl3dq")
    subprocess.run(["cargo", "build", "--release", "--quiet"], cwd=crate, check=True)
    exe = os.path.join(crate, "target", "release", "exl3dq")
    from pack import PackIndex, load_exl3

    ix = PackIndex(OUT)
    rec: dict = {"repo": REPO, "revision": REV, "fetched_seconds": round(fetched_s, 1), "groups": {}, "experts": {}}
    tmp = os.path.join(OUT, "tmp")
    os.makedirs(tmp, exist_ok=True)
    for p in prefixes:
        lin = load_exl3(ix, p)
        marker = int(np.asarray(tensors[p + ".mcg"]).reshape(-1).view(np.uint32)[0])
        subprocess.run([exe, "wq", OUT, p, os.path.join(tmp, "wq.npy")], check=True)
        subprocess.run([exe, "dense", OUT, p, os.path.join(tmp, "w.npy")], check=True)
        wq_equal = bool(np.array_equal(np.load(os.path.join(tmp, "wq.npy")), lin.wq.astype(np.float16).view(np.uint16)))
        w_rust = np.load(os.path.join(tmp, "w.npy"))
        w_py = exl3.decode_weight(np.asarray(tensors[p + ".trellis"]), "mcg", lin.suh, lin.svh)
        dense_equal = bool(np.array_equal(w_py.view(np.uint64), w_rust.view(np.uint64)))
        rec["groups"][p] = {
            "k": lin.k, "n": lin.n, "marker": hex(marker), "wq_bits_equal": wq_equal, "dense_bits_equal": dense_equal,
            "w_rms": float(np.sqrt((w_py ** 2).mean())), "w_absmax": float(np.abs(w_py).max()),
            "wq_rms": float(np.sqrt((lin.wq ** 2).mean())),
        }
        print(p, rec["groups"][p], flush=True)
    rng = np.random.default_rng(0)
    for lay in a.layers:
        gamma = got[f"model.llm.layers.{lay}.mlp_norm.weight"].float().numpy().astype(np.float64)
        z = rng.standard_normal((a.tokens, gamma.shape[0]))
        x = np_bf16(z / np.sqrt((z ** 2).mean(-1, keepdims=True)) * gamma)  # what mlp_norm hands the experts
        for e in a.experts:
            pre = f"model.llm.layers.{lay}.mlp.experts.{e}."
            t1 = time.time()
            g, u, d = (load_exl3(ix, pre + q) for q in ("gate_proj", "up_proj", "down_proj"))
            gv, uv = g(x), u(x)
            y = expert_ffn(g, u, d, x, 4.0, 25.0, Rounding())
            secs = time.time() - t1
            rms = lambda v: float(np.sqrt((v ** 2).mean()))  # noqa: E731
            rec["experts"][f"{lay}.{e}"] = {"x_rms": rms(x), "gate_rms": rms(gv), "up_rms": rms(uv),
                                             "y_rms": rms(y), "finite": bool(np.isfinite(y).all()),
                                             "decode_and_ffn_seconds": round(secs, 2)}
            print(lay, e, rec["experts"][f"{lay}.{e}"], flush=True)
    ys = [v["y_rms"] for v in rec["experts"].values()]
    rec["summary"] = {
        "groups": len(prefixes),
        "all_wq_bits_equal": all(v["wq_bits_equal"] for v in rec["groups"].values()),
        "all_dense_bits_equal": all(v["dense_bits_equal"] for v in rec["groups"].values()),
        "all_markers_mcg": all(v["marker"] == hex(exl3.MARKER_MCG) for v in rec["groups"].values()),
        "y_rms_min": min(ys), "y_rms_max": max(ys),
        "all_finite": all(v["finite"] for v in rec["experts"].values()),
        "max_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20 if sys.platform == "darwin" else 1 << 10)),
        "host": platform.platform(),
    }
    os.makedirs(os.path.join(ORACLE, "records"), exist_ok=True)
    with open(os.path.join(ORACLE, "records", "exl3-real-verify.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec["summary"], indent=1))
    ok = rec["summary"]["all_wq_bits_equal"] and rec["summary"]["all_dense_bits_equal"] and rec["summary"]["all_finite"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
