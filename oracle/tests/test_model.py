"""Whole-model tests on the tiny synthetic config (12 layers = 2 AttnRes blocks, global layers 5 and 11, window 8)."""

import numpy as np
import pytest
import torch

import hf_transcription as hf
import inkling2_ref as ref
from inkling2_ref import Rounding


def ids(n, seed=0, vocab=300):
    return torch.from_numpy(np.random.default_rng(seed).integers(0, vocab, size=n))


def cosine(a, b):
    a, b = a.to(torch.float64).flatten(), b.to(torch.float64).flatten()
    return float(a @ b / (a.norm() * b.norm()))


@pytest.mark.parametrize("tp", [1, 4])
def test_unrounded_oracle_equals_hf_transcription(tiny, tp):
    cfg, w, make = tiny
    x = ids(21, seed=3)  # longer than the window (8) and the global rel extent (8), past the log-scaling floor (6)
    out = make(tp=tp, rounding=Rounding.exact()).forward(x, record=True)
    want_logits, want_hidden = hf.forward(cfg, w, x)
    torch.testing.assert_close(out.hidden, want_hidden, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(out.logits, want_logits, rtol=1e-9, atol=1e-9)


def fixed_routing(make, w, tp, rounding=None):
    """The tiny model with router biases that pin every token to the same experts (the synthetic router's margins are
    small enough that a bf16 ulp flips routes, and a flip is a discrete jump, not rounding drift)."""
    m = make(tp=tp, rounding=rounding)
    fixed = dict(w)
    for i in range(m.cfg.num_layers):
        fixed[f"model.llm.layers.{i}.mlp.gate.bias"] = torch.tensor(
            [3.0 * ((e * 5 + i) % m.cfg.n_experts) for e in range(m.cfg.n_experts)], dtype=torch.float32)
    m.w = m.experts.w = ref.DictWeights(fixed)
    return m


def each_layer(m, hidden, n):
    """Every layer run alone from the reference state: P entering it and the completed blocks before it."""
    pos, step = torch.arange(n), m.cfg.layers_per_block
    outs = []
    for i in range(m.cfg.num_layers):
        blocks = [m.r.bf16(hidden[j].double()) for j in range(0, i, step)]
        outs.append(m.layer(i, m.r.bf16(hidden[i].double()), blocks, pos, m.new_cache().layers[i]))
    return outs


def rel_max(a, b):
    """Worst token's relative L2 error of a against b."""
    a, b = a.double().reshape(a.shape[0], -1), b.double().reshape(b.shape[0], -1)
    return float(((a - b).norm(dim=-1) / b.norm(dim=-1)).max())


BF16_STEP = 2.0 ** -8  # one bf16 rounding moves a value by at most half of this, relative


@pytest.mark.parametrize("layer", [0, 1, 5, 6, 11])
def test_each_op_costs_bf16_level_error(tiny, layer):
    """From identical float64 inputs, each rounded op lands within a few bf16 steps of the unrounded op: one step for
    a single-store op (mix + norm, conv), a few for the chains (attention: ~8 stores; MoE: router, g/u, act, partials)."""
    cfg, w, make = tiny
    ex, rd = fixed_routing(make, w, 1, Rounding.exact()), fixed_routing(make, w, 4)
    hidden = ex.forward(ids(21, seed=4), record=True).hidden
    p, blocks, pos = hidden[layer], [hidden[j] for j in range(0, layer, 6)], torch.arange(21)
    ln = f"model.llm.layers.{layer}."
    if cfg.is_global(layer):
        norms = [ex.w64(ln + f"{n}_input_norm.weight") for n in "qkv"]
    else:
        norms = [ex.w64(ln + "attn_norm.weight")]
    q = ex.query(f"model.llm.attn_res.attn_queries.{layer - 1}") if layer else None
    mix = lambda m: m.mix(blocks, p, q, norms) if layer else [ref.rmsnorm_w(p, norms[0], cfg.eps, m.r)]  # noqa: E731
    xs = mix(ex)
    for got, want in zip(mix(rd), xs):
        assert rel_max(got, want) < BF16_STEP
    xq, xk, xv = (xs * 3)[:3] if len(xs) == 1 else xs
    a = ex.attention(layer, xq, xk, xv, pos, ex.new_cache().layers[layer])
    assert rel_max(rd.attention(layer, xq, xk, xv, pos, rd.new_cache().layers[layer]), a) < 4 * BF16_STEP
    cw, z = ex.w64(ln + "attn_sconv.weight")[:, 0, :], torch.zeros(3, cfg.hidden_size, dtype=torch.float64)
    assert rel_max(ref.static_conv(a, cw, z, rd.r)[0], ref.static_conv(a, cw, z, ex.r)[0]) < BF16_STEP
    assert rel_max(rd.moe(layer, xs[0], None), ex.moe(layer, xs[0], None)) < 2 * BF16_STEP


@pytest.mark.parametrize("tp", [1, 4])
def test_each_layer_rounded_tracks_unrounded(tiny, tp):
    """A whole rounded layer, from the unrounded model's state, stays within a few percent of the unrounded layer
    (the per-op errors above, through the layer's gain)."""
    _, w, make = tiny
    x = ids(21, seed=4)
    exact = fixed_routing(make, w, 1, Rounding.exact())
    hidden = exact.forward(x, record=True).hidden
    want = each_layer(exact, hidden, 21)
    got = each_layer(fixed_routing(make, w, tp), hidden, 21)
    for i in range(len(want)):
        assert rel_max(got[i], want[i]) < 12 * BF16_STEP, f"layer {i}"


def test_output_dtypes(tiny):
    _, _, make = tiny
    out = make().forward(ids(3), record=True)
    assert out.logits.dtype == torch.float32 and out.hidden.dtype == torch.bfloat16
    assert out.h_final.dtype == torch.bfloat16 and out.router_w.dtype == torch.float32


def test_attnres_schedule(tiny):
    cfg, _, make = tiny
    m = make()
    calls = []
    orig_query, orig_mix = m.query, m.mix

    def query(name):
        calls.append(["q", name])
        return orig_query(name)

    def mix(blocks, partial, q, norms):
        calls[-1] += [len(blocks), q.shape[0], len(norms)]
        return orig_mix(blocks, partial, q, norms)

    m.query, m.mix = query, mix
    m.forward(ids(5))
    want = []
    for i in range(cfg.num_layers):
        if i > 0:
            nq = 3 if cfg.is_global(i) else 1
            want.append(["q", f"model.llm.attn_res.attn_queries.{i - 1}", (i - 1) // 6 + 1, nq, nq])
        want.append(["q", f"model.llm.attn_res.mlp_queries.{i}", i // 6 + 1, 1, 1])
    want.append(["q", "model.llm.attn_res.output_query", cfg.num_layers // 6, 1, 1])
    assert calls == want


@pytest.mark.parametrize("splits", [[5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [7, 3, 11], [1] * 21, [20, 1]])
def test_prefill_then_decode_equals_recompute(tiny, splits):
    _, _, make = tiny
    m = make(tp=4)
    x = ids(21, seed=5)
    full = m.forward(x, record=True)
    cache = m.new_cache()
    parts, at = [], 0
    for n in splits:
        parts.append(m.forward(x[at:at + n], cache, record=True))
        at += n
    assert cache.pos == 21
    assert torch.equal(torch.cat([p.logits for p in parts]), full.logits)
    assert torch.equal(torch.cat([p.hidden for p in parts], 1), full.hidden)
    assert torch.equal(torch.cat([p.router_idx for p in parts], 1), full.router_idx)
    # local layers keep a ring of window - 1 positions, global layers everything
    for i, lc in enumerate(cache.layers):
        assert lc.k.shape[0] == (21 if m.cfg.is_global(i) else m.cfg.sliding_window - 1)


def test_cached_equals_uncached_continuation(tiny):
    _, _, make = tiny
    m = make(tp=4)
    x = ids(16, seed=6)
    cache = m.new_cache()
    m.forward(x[:9], cache)
    snap = cache.clone()
    a = m.forward(x[9:], cache)
    b = m.forward(x[9:], snap)
    assert torch.equal(a.logits, b.logits)
    assert torch.equal(a.logits, m.forward(x).logits[9:])


def test_tp_changes_only_rounding(tiny):
    """Without rounding, the tp=4 split (wo column slices, experts by E mod 4, rank-order sum) is the tp=1 model."""
    _, _, make = tiny
    x = ids(12, seed=7)
    ea = make(tp=1, rounding=Rounding.exact()).forward(x)
    eb = make(tp=4, rounding=Rounding.exact()).forward(x)
    torch.testing.assert_close(ea.logits, eb.logits, rtol=1e-10, atol=1e-10)


def test_expert_ownership_is_e_mod_tp(tiny, monkeypatch):
    """Rank r's MoE partial holds exactly the experts E with E mod tp == r (never contiguous ranges)."""
    cfg, _, make = tiny
    m = make(tp=4)

    class OneHot:
        def ffn_many(self, layer, jobs, gate_cap, up_cap, rounding):
            out = []
            for e, x in jobs:
                y = np.zeros((x.shape[0], cfg.hidden_size))
                y[:, e] = 1.0
                out.append(y)
            return out

    m.experts = OneHot()
    seen = []
    real = ref.allreduce

    def spy(partials, r):
        if len(partials) == 4:
            seen.append([p.clone() for p in partials])
        return real(partials, r)

    monkeypatch.setattr(ref, "allreduce", spy)
    x = torch.from_numpy(np.random.default_rng(8).standard_normal((5, cfg.hidden_size)))
    x = ref.bf16(x)
    rec = {"idx": [], "w": [], "margin": []}
    m.moe(0, x, rec)
    parts = seen[-1]
    idx = rec["idx"][0]
    for r in range(4):
        nz = set(torch.nonzero(parts[r].abs().sum(0)).flatten().tolist())
        want = {int(e) for e in idx.flatten() if int(e) % 4 == r}
        assert nz == want


def test_drafter_features(tiny):
    cfg, _, make = tiny
    m = make()
    out = m.forward(ids(4), record=True)
    f = m.drafter_features(out)
    assert f.shape == (4, len(cfg.drafter_target_layers), cfg.hidden_size)
    assert torch.equal(f[:, 0], out.hidden[5]) and torch.equal(f[:, -1], out.h_final)


def test_generate_is_greedy_over_cache(tiny):
    _, _, make = tiny
    m = make()
    res = ref.generate(m, ids(6, seed=9).tolist(), 5)
    toks = res["tokens"]
    assert toks.shape[0] == 11 and res["logits"].shape[0] == 10
    full = m.forward(toks)
    assert torch.equal(full.logits[:10], res["logits"])
    assert torch.equal(ref.greedy(full.logits[5:10]), toks[6:])


def test_image_placeholder_refused(tiny):
    cfg, _, make = tiny
    m = make()
    with pytest.raises(ValueError, match="vision"):
        m.embed(torch.tensor([ref.IMAGE_PAD_ID]))


def test_from_hf_reads_pack_config():
    import json
    import os

    path = os.path.join(os.path.dirname(__file__), "..", "..", "intake", "files", "config.json")
    with open(path) as f:
        cfg = ref.Inkling2Config.from_hf(json.load(f))
    assert cfg == ref.Inkling2Config()
    assert [i for i in range(48) if cfg.is_global(i)] == [5, 11, 17, 23, 29, 35, 41, 47]
