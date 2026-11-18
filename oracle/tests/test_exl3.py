"""EXL3 numpy port vs REXL3's CPU reference (gb-exl3 via the exl3dq CLI): codebooks, W_q bits and the dense decode bit for
bit, the layer forward to float64 rounding. Groups under oracle/.cache/real-groups (fetched by tools/verify_exl3_real.py
from the pack) are checked too when present."""

import json
import os
import shutil
import subprocess

import numpy as np
import pytest

import exl3
import st
from synthetic import random_exl3_group

ORACLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRATE = os.path.join(ORACLE, "exl3dq")
REAL = os.path.join(ORACLE, ".cache", "real-groups")


@pytest.fixture(scope="session")
def exl3dq():
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not installed: the Rust cross-check needs it")
    subprocess.run(["cargo", "build", "--release", "--quiet"], cwd=CRATE, check=True)
    return os.path.join(CRATE, "target", "release", "exl3dq")


def run(bin_, *args):
    subprocess.run([bin_, *map(str, args)], check=True)


def write_groups(d, groups):
    tensors = {}
    for prefix, g in groups.items():
        for part, arr in g.items():
            tensors[f"{prefix}.{part}"] = arr
    st.write(os.path.join(d, "groups.safetensors"), tensors)


CASES = [  # (prefix, K, N, bits * 2, codebook)
    ("mcg4", 256, 384, 8, "mcg"),
    ("mcg4t", 384, 256, 8, "mcg"),
    ("mul1_25", 256, 128, 5, "mul1"),
    ("inst3_3", 128, 256, 6, "3inst"),
    ("mcg2", 128, 128, 4, "mcg"),
]


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("exl3")
    rng = np.random.default_rng(42)
    groups = {p: random_exl3_group(rng, k, n, twice, cb) for p, k, n, twice, cb in CASES}
    write_groups(d, groups)
    return d, groups


def test_codebooks_match_rust(exl3dq, tmp_path):
    for cb in exl3.CODEBOOKS:
        out = tmp_path / f"{cb}.npy"
        run(exl3dq, "table", cb, out)
        np.testing.assert_array_equal(np.load(out), exl3.codebook(cb))


def test_groups_accepted_by_gb_exl3(exl3dq, synthetic_dir):
    d, _ = synthetic_dir
    out = subprocess.run([exl3dq, "groups", d], check=True, capture_output=True, text=True).stdout
    got = {j["prefix"]: j for j in map(json.loads, out.splitlines())}
    for p, k, n, twice, cb in CASES:
        assert got[p]["codebook"] == cb and got[p]["k"] == k and got[p]["n"] == n and got[p]["bits"] == twice / 2


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_unpack_and_dense_bit_exact_vs_rust(exl3dq, synthetic_dir, tmp_path, case):
    d, groups = synthetic_dir
    prefix, k, n, _, cb = case
    g = groups[prefix]
    run(exl3dq, "wq", d, prefix, tmp_path / "wq.npy")
    np.testing.assert_array_equal(np.load(tmp_path / "wq.npy"), exl3.unpack(g["trellis"], cb))
    run(exl3dq, "dense", d, prefix, tmp_path / "w.npy")
    w_rust = np.load(tmp_path / "w.npy")
    w_py = exl3.decode_weight(g["trellis"], cb, g["suh"].astype(np.float64), g["svh"].astype(np.float64))
    assert w_py.shape == (k, n)
    np.testing.assert_array_equal(w_py.view(np.uint64), w_rust.view(np.uint64))


@pytest.mark.parametrize("case", CASES[:3], ids=[c[0] for c in CASES[:3]])
def test_forward_matches_rust(exl3dq, synthetic_dir, tmp_path, case):
    d, groups = synthetic_dir
    prefix, k, n, _, cb = case
    g = groups[prefix]
    x = np.random.default_rng(1).standard_normal((5, k))
    np.save(tmp_path / "x.npy", x)
    run(exl3dq, "forward", d, prefix, tmp_path / "x.npy", tmp_path / "y.npy")
    lin = exl3.Exl3Linear(g["trellis"], cb, g["suh"].astype(np.float64), g["svh"].astype(np.float64))
    np.testing.assert_allclose(lin(x), np.load(tmp_path / "y.npy"), rtol=1e-12, atol=1e-12 * np.abs(lin(x)).max())


def test_dense_decode_is_forward_on_identity():
    g = random_exl3_group(np.random.default_rng(3), 256, 384)
    suh, svh = g["suh"].astype(np.float64), g["svh"].astype(np.float64)
    w = exl3.decode_weight(g["trellis"], "mcg", suh, svh)
    lin = exl3.Exl3Linear(g["trellis"], "mcg", suh, svh)
    np.testing.assert_allclose(lin(np.eye(256)), w, rtol=1e-13, atol=1e-13 * np.abs(w).max())


def test_tile_positions_are_a_permutation():
    r, c = exl3.tile_positions()
    assert sorted((r * 16 + c).tolist()) == list(range(256))


def test_wrong_marker_refused(exl3dq, tmp_path):
    g = random_exl3_group(np.random.default_rng(5), 128, 128)
    g["mcg"] = np.array(0x12345678, dtype=np.int32).reshape(())
    write_groups(tmp_path, {"bad": g})
    r = subprocess.run([exl3dq, "wq", tmp_path, "bad", tmp_path / "o.npy"], capture_output=True, text=True)
    assert r.returncode != 0 and "marker" in r.stderr
    with pytest.raises(ValueError, match="marker"):
        exl3.codebook_from_marker("mcg", 0x12345678)


def real_prefixes():
    path = os.path.join(REAL, "groups.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)["prefixes"]


@pytest.mark.skipif(not real_prefixes(), reason="no real pack groups fetched (run tools/verify_exl3_real.py)")
@pytest.mark.parametrize("prefix", real_prefixes())
def test_real_pack_groups_bit_exact_vs_rust(exl3dq, tmp_path, prefix):
    from pack import PackIndex, load_exl3

    ix = PackIndex(REAL)
    lin = load_exl3(ix, prefix)
    run(exl3dq, "wq", REAL, prefix, tmp_path / "wq.npy")
    np.testing.assert_array_equal(np.load(tmp_path / "wq.npy"), lin.wq.astype(np.float16).view(np.uint16))
    run(exl3dq, "dense", REAL, prefix, tmp_path / "w.npy")
    trellis = st.read_numpy(ix.entry(prefix + ".trellis"))
    w_py = exl3.decode_weight(trellis, "mcg", lin.suh, lin.svh)
    np.testing.assert_array_equal(w_py.view(np.uint64), np.load(tmp_path / "w.npy").view(np.uint64))
