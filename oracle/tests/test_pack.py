"""The real-pack code path on a synthetic pack folder (tiny config, EXL3 mcg 4-bit experts striped E mod 4, an index):
loader, expert decode, worker pool, fixture generation and the compare gates."""

import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

import compare
import gen_fixtures
import pack
import st
from inkling2_ref import Inkling2Config, Inkling2Ref
from pack import PackExperts, PackIndex, PackWeights, load_exl3
from synthetic import random_weights, tiny_config, write_pack


@pytest.fixture(scope="module")
def pack_dir(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("pack"))
    cfg = tiny_config()
    write_pack(root, cfg, random_weights(cfg, seed=2, dense_experts=False), seed=2)
    return root, cfg


def model_for(root, workers=0, tp=4):
    with open(os.path.join(root, "config.json")) as f:
        cfg = Inkling2Config.from_hf(json.load(f))
    ix = PackIndex(root)
    return Inkling2Ref(cfg, PackWeights(ix), PackExperts(ix, workers=workers), tp=tp, unembed_chunk=100)


def test_config_round_trip(pack_dir):
    root, cfg = pack_dir
    with open(os.path.join(root, "config.json")) as f:
        assert Inkling2Config.from_hf(json.load(f)) == cfg


def test_expert_groups_sane(pack_dir):
    root, cfg = pack_dir
    ix = PackIndex(root)
    gate = load_exl3(ix, "model.llm.layers.3.mlp.experts.6.gate_proj")
    down = load_exl3(ix, "model.llm.layers.3.mlp.experts.6.down_proj")
    assert (gate.k, gate.n) == (cfg.hidden_size, cfg.moe_intermediate)
    assert (down.k, down.n) == (cfg.moe_intermediate, cfg.hidden_size)
    assert ix.weight_map["model.llm.layers.3.mlp.experts.6.gate_proj.trellis"] == "model-experts-03-02.safetensors"


def test_pack_prefill_decode_equals_recompute(pack_dir):
    root, _ = pack_dir
    m = model_for(root)
    x = torch.from_numpy(np.random.default_rng(11).integers(0, 300, 14))
    full = m.forward(x, record=True)
    assert torch.isfinite(full.logits).all()
    cache = m.new_cache()
    parts = [m.forward(x[:6], cache, record=True)] + [m.forward(x[j:j + 1], cache, record=True) for j in range(6, 14)]
    assert torch.equal(torch.cat([p.logits for p in parts]), full.logits)
    assert torch.equal(torch.cat([p.hidden for p in parts], 1), full.hidden)


def test_worker_pool_matches_in_process(pack_dir):
    root, _ = pack_dir
    x = torch.from_numpy(np.random.default_rng(12).integers(0, 300, 7))
    a = model_for(root).forward(x)
    mp = model_for(root, workers=2)
    try:
        b = mp.forward(x)
    finally:
        mp.experts.close()
    assert torch.equal(a.logits, b.logits)


def test_workers_exit_when_their_parent_is_killed(pack_dir, tmp_path):
    """A killed generator must not leave its spawn workers behind (they held memory on the oracle host)."""
    root, _ = pack_dir
    here = os.path.dirname(os.path.abspath(pack.__file__))
    script = tmp_path / "parent.py"
    script.write_text(
        "import sys, time\n"
        f"sys.path.insert(0, {here!r})\n"
        "from pack import PackExperts, PackIndex\n"
        "if __name__ == '__main__':\n"
        f"    ex = PackExperts(PackIndex({root!r}), workers=2)\n"
        "    ex.pool.submit(int, 1).result()\n"
        "    print(*ex.pool._processes, flush=True)\n"
        "    time.sleep(600)\n")
    p = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    pids = [int(v) for v in p.stdout.readline().split()]
    assert pids
    p.kill()
    p.wait()

    def alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False

    deadline = time.time() + 20
    while any(alive(q) for q in pids) and time.time() < deadline:
        time.sleep(0.2)
    assert not any(alive(q) for q in pids)


def test_missing_expert_file_is_an_error(pack_dir, tmp_path):
    root, _ = pack_dir
    for fn in os.listdir(root):
        if not fn.startswith("model-experts-") or fn.endswith("-00.safetensors"):
            os.symlink(os.path.join(root, fn), tmp_path / fn)
    m = model_for(str(tmp_path))  # rank 0's split: experts E mod 4 == 0 only
    with pytest.raises(KeyError, match="does not hold"):
        m.forward([1, 2, 3, 4, 5, 6])


def test_gen_fixtures_and_compare(pack_dir, tmp_path):
    root, _ = pack_dir
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"name": "p1", "ids": [5, 17, 3, 99, 250, 7], "new_tokens": 4}) + "\n")
    out = tmp_path / "fx"
    assert gen_fixtures.main(["--pack", root, "--prompts", str(prompts), "--out", str(out),
                              "--verify-recompute"]) == 0
    path = out / "p1.safetensors"
    o, meta = compare.load(str(path))
    assert meta["prompt_len"] == "6" and o["tokens"].shape[0] == 10 and o["logits"].shape == (9, 300)
    assert o["hidden"].shape == (13, 9, 256) and o["hidden"].dtype == torch.bfloat16
    man = json.loads((out / "manifest.json").read_text())
    assert man["fixtures"]["p1"]["generated"] == o["tokens"][6:].tolist()
    # identical engine dump: passes; a small perturbation: passes; a broken layer: fails
    assert compare.compare(o, o, 6)["pass"]
    noisy = dict(o)
    g = torch.Generator().manual_seed(0)
    noisy["hidden"] = (o["hidden"].double() * (1 + 1e-3 * torch.randn(o["hidden"].shape, generator=g, dtype=torch.float64))).to(torch.bfloat16)
    noisy["logits"] = o["logits"] + 1e-3 * torch.randn(o["logits"].shape, generator=g)
    assert compare.compare(o, noisy, 6)["pass"]
    broken = dict(o)
    h = o["hidden"].clone()
    h[7] = torch.randn(h[7].shape, generator=g).to(torch.bfloat16)
    broken["hidden"] = h
    rep = compare.compare(o, broken, 6)
    assert not rep["pass"] and rep["checks"]["hidden"]["worst_layer"] == 7
    eng = tmp_path / "engine.safetensors"
    st.write(str(eng), noisy)
    assert compare.main([str(path), str(eng)]) == 0


def test_compare_token_divergence_rules():
    t = torch.tensor([1, 2, 3, 4, 5])
    logits = torch.zeros(4, 10)
    logits[:, 0] = 5.0
    logits[2, 1] = 4.99  # near-tie at the row predicting token 3
    o = {"tokens": t, "logits": logits}
    e = {"tokens": torch.tensor([1, 2, 3, 9, 9]), "logits": logits.clone()}
    rep = compare.compare(o, e, 2)
    assert rep["checks"]["tokens"]["first_divergence"] == 3 and rep["checks"]["tokens"]["excused_tie"]
    assert rep["checks"]["logits"]["positions"] == 3
    e2 = {"tokens": torch.tensor([1, 2, 9, 9, 9]), "logits": logits.clone()}
    assert not compare.compare(o, e2, 2)["pass"]


def test_gate_set_follows_the_fixture_emulation():
    assert compare.gates_for("engine") is compare.GATES_ENGINE and compare.gates_for("none") is compare.GATES_SPEC
    with pytest.raises(ValueError, match="no gate set"):
        compare.gates_for("kv8,moe_fp16")
    # the correctness gate is never looser than the drift gate
    e, s = compare.GATES_ENGINE, compare.GATES_SPEC
    assert e["kl_mean_max"] < s["kl_mean_max"] and e["top1_agree_min"] > s["top1_agree_min"]
