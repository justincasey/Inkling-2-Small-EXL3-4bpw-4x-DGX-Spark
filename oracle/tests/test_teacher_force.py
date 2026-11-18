"""teacher_force.py on a dump the oracle itself wrote (synthetic EXL3 pack): every op agrees exactly, the router picks
are identical, the residual is exact, and an altered engine plane shows up in its own op only."""

import json
import os

import numpy as np
import pytest
import torch

import st
import teacher_force
from inkling2_ref import Inkling2Config, Inkling2Ref
from pack import PackExperts, PackIndex, PackWeights
from synthetic import random_weights, tiny_config, write_pack


@pytest.fixture(scope="module")
def oracle_dump(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("pack"))
    cfg = tiny_config()
    write_pack(root, cfg, random_weights(cfg, seed=2, dense_experts=False), seed=2)
    with open(os.path.join(root, "config.json")) as f:
        cfg = Inkling2Config.from_hf(json.load(f))
    ix = PackIndex(root)
    m = Inkling2Ref(cfg, PackWeights(ix), PackExperts(ix), unembed_chunk=100)
    m.taps = {}
    ids = torch.from_numpy(np.random.default_rng(3).integers(0, 300, 13))
    out = m.forward(ids, record=True)
    q = cfg.num_heads * cfg.head_dim // 4
    planes = lambda k: torch.stack([m.taps[i][k] for i in range(cfg.num_layers)])  # noqa: E731
    dump = {"tokens": ids, "hidden": out.hidden, "router_idx": out.router_idx, "router_w": out.router_w,
            "attn_core": planes("core")[:, :, :q].to(torch.bfloat16), "attn": planes("attn").to(torch.bfloat16),
            "mlp_in": planes("mlp_in").to(torch.bfloat16), "router_logits": planes("router_logits").float(),
            "moe": planes("moe").float()}
    return root, cfg, dump


def run(tmp_path, root, dump, layers="0-11"):
    eng = str(tmp_path / "eng.safetensors")
    st.write(eng, dump, {"prompt_len": "13"})
    rep = str(tmp_path / "rep.json")
    assert teacher_force.main(["--pack", root, "--engine", eng, "--out", rep, "--layers", layers]) == 0
    with open(rep) as f:
        return json.load(f)["layers"]


def test_oracle_dump_agrees_exactly(oracle_dump, tmp_path):
    root, cfg, dump = oracle_dump
    lays = run(tmp_path, root, dump)
    for lay in lays:
        assert lay["attn"]["none"]["core"]["rel_max"] == 0 and lay["attn"]["none"]["a"]["rel_max"] == 0
        assert lay["mix"]["rel_max"] == 0 and lay["moe"]["none"]["rel_max"] == 0
        assert lay["router"]["picks_identical"] == 1.0 and lay["router"]["logit_err_max"] == 0
        assert lay["router"]["engine_select_consistent"] and lay["residual_mismatch"] == 0
        assert lay["attn_noise"]["a"]["rel_max"] < 2e-2 and lay["moe_noise"]["rel_max"] < 2e-2
    glob = [lay for lay in lays if lay["global"]]
    assert glob and all(lay["attn"]["kv8"]["core"]["rel_max"] > 0 for lay in glob)


def test_altered_plane_is_named(oracle_dump, tmp_path):
    root, cfg, dump = oracle_dump
    bad = dict(dump)
    moe = dump["moe"].clone()
    moe[3] *= 1.01
    bad["moe"] = moe
    lays = run(tmp_path, root, bad, layers="2-4")
    by = {lay["layer"]: lay for lay in lays}
    assert by[3]["moe"]["none"]["rel_med"] > 5e-3 and by[3]["residual_mismatch"] > 0
    assert by[2]["moe"]["none"]["rel_max"] == 0 and by[4]["moe"]["none"]["rel_max"] == 0
