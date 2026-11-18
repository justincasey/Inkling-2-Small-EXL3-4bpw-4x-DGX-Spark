"""DSpark drafter reference (`dspark_ref.py`) on a tiny synthetic drafter: TP and chunking invariance, rollback, the
local window and the global layer, the non-causal block, f_rnn, the op structure against an independent transcription of
the vLLM module flow, the fold, the argmax rule and the RoPE table."""

import numpy as np
import pytest
import torch

from dspark_ref import DSparkConfig, DSparkRef, rope_table
from inkling2_ref import DictWeights, Rounding, greedy
from synthetic import random_dspark_weights, random_features, tiny_dspark_config
from vllm_dspark_transcription import VDSpark

N = 24  # committed rows available to the tests
ANCHOR = 17


@pytest.fixture(scope="module")
def drafter():
    cfg = tiny_dspark_config()
    w = random_dspark_weights(cfg, seed=3)
    feats = random_features(cfg, N + 1, seed=4, scales=(0.5, 1, 1.5, 2, 3, 4, 6, 1))
    src = DictWeights(w)

    def make(tp=4, rounding=None):
        return DSparkRef(cfg, src, tp=tp, rounding=rounding, unembed_chunk=40)

    return cfg, w, feats, make


def run(ref, feats, p, anchor=ANCHOR, chunks=None, force_prev=None):
    st = ref.new_state()
    cuts = [0] + list(chunks or []) + [p]
    for a, b in zip(cuts, cuts[1:]):
        ref.commit(st, feats[a:b], a)
    return st, ref.propose(st, anchor, force_prev=force_prev, trace=True)


def same(a, b):
    assert a.drafts == b.drafts
    for k in a.trace:
        assert torch.equal(a.trace[k], b.trace[k]), k


# ----------------------------------------------------------------------------------------------------------------------


def test_exact_tp1_equals_tp4(drafter):
    cfg, w, feats, make = drafter
    _, a = run(make(1, Rounding.exact()), feats, 15)
    _, b = run(make(4, Rounding.exact()), feats, 15)
    assert a.drafts == b.drafts
    for k in a.trace:
        assert torch.allclose(a.trace[k], b.trace[k], rtol=0, atol=1e-10), k


def test_rounded_tp1_differs_from_tp4_only_by_rounding(drafter):
    cfg, w, feats, make = drafter
    _, a = run(make(1), feats, 15)
    _, b = run(make(4), feats, 15)
    _, x = run(make(1, Rounding.exact()), feats, 15)
    for t in (a, b):
        assert (t.trace["hd"].to(torch.float64) - x.trace["hd"]).abs().max() < 0.1
    assert not torch.equal(a.trace["hd"], b.trace["hd"])  # the rank partials round differently


@pytest.mark.parametrize("chunks", [[1], [5, 6, 13], list(range(1, 20)), [19]])
def test_incremental_commits_equal_one_commit(drafter, chunks):
    cfg, w, feats, make = drafter
    ref = make()
    s1, one = run(ref, feats, 20)
    s2, inc = run(ref, feats, 20, chunks=chunks)
    same(one, inc)
    assert torch.equal(s1.ctx_kv(), s2.ctx_kv())


def test_rollback_equals_fresh_replay(drafter):
    cfg, w, feats, make = drafter
    ref = make()
    other = random_features(cfg, N, seed=9)
    st = ref.new_state()
    ref.commit(st, other[:18], 0)  # a branch that diverges from position 9 on
    ref.commit(st, feats[9:14], 9)  # rollback to 9, new rows 9..13: rows 14..17 become stale
    ref.commit(st, feats[14:16], 14)
    got = ref.propose(st, ANCHOR, trace=True)
    mixed = torch.cat([other[:9], feats[9:16]])
    fresh_state, fresh = run(ref, mixed, 16)
    same(got, fresh)
    assert torch.equal(st.ctx_kv(), fresh_state.ctx_kv())


def test_local_window_and_global_layer(drafter):
    cfg, w, feats, make = drafter
    ref = make()
    p = 20
    wdw = cfg.sliding_window
    st, base = run(ref, feats, p)

    def perturbed(layer, pos):
        s, _ = run(ref, feats, p)
        s.k[layer][pos] += 0.5
        s.v[layer][pos] -= 0.5
        return ref.propose(s, ANCHOR, trace=True).trace["hd"]

    for layer in cfg.local_layer_ids:
        assert torch.equal(perturbed(layer, p - wdw - 1), base.trace["hd"]), layer
        assert not torch.equal(perturbed(layer, p - wdw), base.trace["hd"]), layer
    glob = [i for i in range(cfg.num_layers) if not cfg.is_local(i)]
    assert glob == [2]
    assert not torch.equal(perturbed(2, 0), base.trace["hd"])
    # through the features: row 0 is outside every local window, so only the global layer can see it
    f2 = feats.clone()
    f2[0] = random_features(cfg, 1, seed=11)[0]
    _, moved = run(ref, f2, p)
    assert not torch.equal(moved.trace["hd"], base.trace["hd"])


def test_block_is_non_causal_and_anchor_ctx_row_is_never_read(drafter):
    cfg, w, feats, make = drafter
    p = 15
    _, base = run(make(), feats, p)
    w2 = dict(w)
    w2["mask_embed"] = random_dspark_weights(cfg, seed=12)["mask_embed"]
    _, moved = run(DSparkRef(cfg, DictWeights(w2), tp=4, unembed_chunk=40), feats, p)
    assert not torch.equal(moved.trace["hd"][0], base.trace["hd"][0])  # the anchor row attends to the mask rows
    # a committed-then-rolled-back row at p (the anchor's own position) stays stale and unread
    ref = make()
    st = ref.new_state()
    ref.commit(st, feats[:p + 1], 0)
    ref.commit(st, feats[p - 1:p], p - 1)
    same(ref.propose(st, ANCHOR, trace=True), base)


def test_frnn_is_last_committed_row(drafter):
    cfg, w, feats, make = drafter
    ref = make()
    st, got = run(ref, feats, 12)
    want = ref.mix(feats[11:12].to(torch.float64), cfg.num_layers)[0]
    assert torch.equal(got.trace["f_rnn"].to(torch.float64), want)
    ref.commit(st, feats[20:21], 11)  # overwrite position 11
    want2 = ref.mix(feats[20:21].to(torch.float64), cfg.num_layers)[0]
    assert torch.equal(ref.propose(st, ANCHOR, trace=True).trace["f_rnn"].to(torch.float64), want2)
    assert not torch.equal(want, want2)


@pytest.mark.parametrize("tp", [1, 4])
def test_exact_matches_vllm_transcription(drafter, tp):
    """Commit, propose, accept k, commit (overwriting the block's slots in V's cache), propose again, plus a rollback;
    V's flow keeps the stale block K/V and an extra ctx row at the anchor position in its cache."""
    cfg, w, feats, make = drafter
    ref = make(tp, Rounding.exact())
    v = VDSpark(cfg, w, max_pos=N + 1)
    f64 = feats.to(torch.float64)
    st = ref.new_state()

    def commit(a, b):
        ref.commit(st, f64[a:b], a)
        v.precompute_and_store_context_kv(f64[a:b].reshape(b - a, -1), torch.arange(a, b))
        v.prepare_context_anchor(f64[b - 1])

    def check(p, anchor):
        got = ref.propose(st, anchor, trace=True)
        drafts, hd, logits = v.draft(anchor, p)
        assert got.drafts == drafts
        assert torch.allclose(got.trace["hd"], hd, rtol=0, atol=1e-9)
        assert torch.allclose(got.trace["logits"], logits, rtol=0, atol=1e-9)
        return drafts

    commit(0, 9)
    d = check(9, ANCHOR)
    commit(9, 12)  # accepted anchor + d1 + d2 -> next anchor at 12
    # V holds a ctx row at the next anchor's position (written by an earlier step); the block overwrites it
    v.precompute_and_store_context_kv(f64[12:13].reshape(1, -1), torch.arange(12, 13))
    check(12, d[2])
    commit(12, 20)
    check(20, 5)
    commit(14, 16)  # rollback
    check(16, 7)


def test_fold_exact(drafter):
    cfg, w, feats, make = drafter
    ref = make(4, Rounding.exact())
    _, got = run(ref, feats, 10)
    u = w["model.llm.unembed.weight"].to(torch.float64)[: cfg.valid_vocab]
    wout = w["markov_head.hidden_out.weight"].to(torch.float64)
    bout = w["markov_head.hidden_out.bias"].to(torch.float64)
    for latent in got.trace["latent"]:
        want = u @ (wout @ latent + bout)
        assert torch.allclose(ref.markov_bias(latent), want, rtol=0, atol=1e-10)
    rounded = make()
    f = rounded.fold()
    assert torch.equal(f.weight, (u @ wout).to(torch.float32).to(torch.bfloat16).to(torch.float64))
    assert f.weight.shape == (cfg.valid_vocab, cfg.markov_rank)


def test_argmax_first_max_and_nan(drafter):
    x = torch.tensor([[1.0, 3.0, 3.0, 2.0], [float("nan"), 0.5, 0.5, -1.0], [float("nan")] * 3 + [-5.0]])
    assert greedy(x).tolist() == [1, 1, 3]
    cfg, w, feats, make = drafter
    ref = make()
    _, got = run(ref, feats, 10)
    lg = got.trace["logits"].to(torch.float64)
    for i, d in enumerate(got.drafts):
        m = lg[i].max()
        assert lg[i, d] == m and (lg[i, :d] < m).all()


def test_teacher_forcing(drafter):
    cfg, w, feats, make = drafter
    ref = make()
    _, free = run(ref, feats, 10)
    _, forced_own = run(ref, feats, 10, force_prev=free.drafts)
    same(free, forced_own)
    other = [(d + 1) % cfg.valid_vocab for d in free.drafts]
    _, forced = run(ref, feats, 10, force_prev=other)
    assert torch.equal(forced.trace["logits"][0], free.trace["logits"][0])  # step 1 reads only the anchor
    assert not torch.equal(forced.trace["logits"][1], free.trace["logits"][1])


def test_rope_table_fp32_formula():
    d, theta = 128, 1e6
    pos = np.array([0, 1, 2, 7, 1023, 1024, 1031, 65535, 1048575])
    i = np.arange(d // 2)
    inv = (np.float32(1.0) / np.float32(theta ** (2.0 * i / d))).astype(np.float32)
    ang = pos.astype(np.float32)[:, None] * inv[None, :]  # one fp32 multiply
    assert ang.dtype == np.float32
    want_c = np.cos(ang.astype(np.float64)).astype(np.float32)
    want_s = np.sin(ang.astype(np.float64)).astype(np.float32)
    c, s = rope_table(torch.from_numpy(pos), d, theta, Rounding())
    assert np.array_equal(c.numpy().astype(np.float32), want_c)
    assert np.array_equal(s.numpy().astype(np.float32), want_s)
    assert np.array_equal(c.numpy(), want_c.astype(np.float64))  # fp32 values, held in float64
    # V's get_rope builds the table with fp32 powf / cosf / sinf, which are not correctly rounded (torch's CPU powf is
    # one ulp off at i = 37 for theta 1e6): the spec's correctly rounded table agrees to a few fp32 ulps of the angle
    tf = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    tc = (torch.from_numpy(pos).float()[:, None] * tf[None]).cos()
    assert np.abs(tc.numpy().astype(np.float64) - c.numpy()).max() < 1e-4


def test_config_from_hf():
    conf = {
        "text_config": {"hidden_size": 4608, "num_hidden_layers": 48, "vocab_size": 201024,
                        "unpadded_vocab_size": 200058, "final_logit_softcapping": 30.0},
        "mtp_config": {"hidden_size": 4608, "num_hidden_layers": 5, "num_attention_heads": 48,
                       "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 18432, "q_bias": False,
                       "o_bias": False, "rms_norm_eps": 1e-06, "rope_theta": 1000000.0, "local_layer_ids": [0, 1, 3, 4],
                       "sliding_window_size": 1024, "block_size": 8, "markov_rank": 256,
                       "markov_head_type": "rnn_hidden", "target_layer_ids": [5, 11, 17, 23, 29, 35, 41, 48]},
    }
    assert DSparkConfig.from_hf(conf) == DSparkConfig()
    conf["mtp_config"]["markov_head_type"] = "mlp"
    with pytest.raises(ValueError):
        DSparkConfig.from_hf(conf)


def test_tp_must_divide(drafter):
    cfg, w, feats, make = drafter
    with pytest.raises(ValueError):
        make(3)


def test_fixture_generator_contract(tmp_path, drafter):
    """gen_dspark_fixtures.py over a tiny pack (mtp.safetensors + target tensors + config.json) and tiny target
    fixtures writes every contract file with the contract's shapes, and its drafts are the reference's."""
    import json

    import gen_dspark_fixtures as gen
    import st

    cfg, w, _, make = drafter
    pack, fx, out = tmp_path / "pack", tmp_path / "fx", tmp_path / "out"
    pack.mkdir()
    fx.mkdir()
    target = {k: v for k, v in w.items() if k.startswith("model.llm.")}
    st.write(str(pack / "model-nonexpert-00.safetensors"), target)
    st.write(str(pack / "mtp.safetensors"), {k: v for k, v in w.items() if k not in target})
    mtp = {"hidden_size": cfg.hidden_size, "num_hidden_layers": cfg.num_layers, "num_attention_heads": cfg.num_heads,
           "num_key_value_heads": cfg.num_kv_heads, "head_dim": cfg.head_dim,
           "intermediate_size": cfg.intermediate_size, "rms_norm_eps": cfg.eps, "rope_theta": cfg.rope_theta,
           "local_layer_ids": list(cfg.local_layer_ids), "sliding_window_size": cfg.sliding_window,
           "block_size": cfg.block_size, "markov_rank": cfg.markov_rank, "markov_head_type": "rnn_hidden",
           "q_bias": False, "o_bias": False, "target_layer_ids": list(cfg.target_layer_ids)}
    text = {"hidden_size": cfg.hidden_size, "num_hidden_layers": cfg.target_layer_ids[-1],
            "vocab_size": cfg.vocab_size, "unpadded_vocab_size": cfg.valid_vocab, "final_logit_softcapping": 30.0}
    (pack / "config.json").write_text(json.dumps({"text_config": text, "mtp_config": mtp}))
    g = torch.Generator().manual_seed(5)
    for name, t in (("raw-counting", 50), ("chat-sky-effort0.9", 49)):
        st.write(str(fx / f"{name}.safetensors"), {
            "tokens": torch.randint(0, cfg.valid_vocab, (t + 1,), generator=g),
            "hidden": torch.randn(cfg.target_layer_ids[-1] + 1, t, cfg.hidden_size, generator=g).to(torch.bfloat16),
            "h_final": torch.randn(t, cfg.hidden_size, generator=g).to(torch.bfloat16)})
    assert gen.main(["--pack", str(pack), "--fixtures", str(fx), "--out", str(out), "--threads", "1"]) == 0
    h, rk, nkv, d = cfg.hidden_size, cfg.markov_rank, cfg.num_kv_heads, cfg.head_dim
    for name, case in gen.CASES.items():
        meta = json.loads((out / f"{name}_meta.json").read_text())
        assert set(meta) == {"tp", "anchors", "n_positions", "feature_source", "mtp_sha256", "oracle_source_sha256",
                             "seconds"}
        n, p = meta["n_positions"], len(case["anchors"])
        want = {"feats": (np.uint16, (n, 8, h)), "tokens": (np.int64, (n + 1,)), "anchors": (np.int64, (p,)),
                "drafts": (np.int64, (p, 7)), "hd": (np.uint16, (p, 8, h)), "frnn": (np.uint16, (p, h)),
                "state": (np.uint16, (p, 7, rk)), "latent": (np.uint16, (p, 7, rk)),
                "logits": (np.float32, (p, 7, cfg.valid_vocab)), "ctxkv": (np.uint16, (5, n, 2, nkv, d))}
        arr = {k: np.load(out / f"{name}_{k}.npy") for k in want}
        for k, (dt, shape) in want.items():
            assert arr[k].dtype == dt and arr[k].shape == shape, (name, k)
        # replay the last anchor directly from the stored features
        ref = make(case["tp"])
        feats = torch.from_numpy(arr["feats"].view(np.int16)).view(torch.bfloat16)
        st_ = ref.new_state()
        last = case["anchors"][-1]
        ref.commit(st_, feats[:last], 0)
        prop = ref.propose(st_, int(arr["tokens"][last]), trace=True)
        assert prop.drafts == arr["drafts"][-1].tolist()
        assert np.array_equal(prop.trace["logits"].numpy(), arr["logits"][-1])
