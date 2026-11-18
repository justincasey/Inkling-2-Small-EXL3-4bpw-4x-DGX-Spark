"""The engine-numerics emulation (inkling2_ref.Emulation, docs/ORACLE-GAP.md): the int8 KV store of the global layers,
the attention core's tiled online softmax with bf16 P, and the fp16 EXL3 GEMM inputs of the experts."""

import json
import os

import numpy as np
import pytest
import torch

import exl3
import inkling2_ref as ref
from inkling2_ref import F64, Emulation, Inkling2Config, Inkling2Ref, Rounding
from pack import PackExperts, PackIndex, PackWeights, load_exl3
from synthetic import random_weights, tiny_config, write_pack

# tiny-model chunk/tile sizes: several chunks and tiles inside the window (8) and past it
TINY = dict(tile=2, chunk_global=4, chunk_local=4)


def bf16_rows(shape, seed, spread=0.0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape)
    if spread:
        x = x * 10.0 ** rng.uniform(-spread, spread, shape[:-1] + (1,))
    return ref.bf16(torch.from_numpy(x))


# ---------------------------------------------------------------------------------------------------------------- kv8


def test_kv8_matches_its_definition():
    """e = the smallest exponent with amax / 2^e <= 127, clamped to [-24, 8]; q = rint(x / 2^e) saturated to +-127."""
    x = bf16_rows((4000, 128), seed=1, spread=7.0)
    got = ref.kv8_roundtrip(x)
    amax = x.abs().amax(-1, keepdim=True)
    e = torch.ceil(torch.log2(amax / 127)).clamp(-24, 8)  # exact: amax / 127 is a power of two only at 127 * 2^k
    want = torch.round(x / torch.exp2(e)).clamp(-127, 127) * torch.exp2(e)
    assert torch.equal(got, want)
    assert torch.equal(ref.bf16(got), got)  # 7 significant bits: exact in bf16
    unsat = (amax <= 127 * 256.0).squeeze(-1)  # above that e is clamped at 8 and the row saturates
    assert unsat.sum() > 1000 and ((got - x).abs() <= torch.exp2(e) / 2)[unsat].all()


def test_kv8_boundaries():
    t = lambda *v: torch.tensor([list(v)], dtype=F64)  # noqa: E731
    # 1.984375 = 127 * 2^-6 keeps e = -6 and q = 127; one bf16 step above it needs e = -5
    assert torch.equal(ref.kv8_roundtrip(t(1.984375, 0.5)), t(1.984375, 0.5))
    assert torch.equal(ref.kv8_roundtrip(t(1.9921875, 0.5)), t(2.0, 0.5))
    assert torch.equal(ref.kv8_roundtrip(t(1.984375, 2.5 / 64)), t(1.984375, 2.0 / 64))  # 2.5 rounds to even
    assert torch.equal(ref.kv8_roundtrip(t(0.0, 0.0)), t(0.0, 0.0))
    assert torch.equal(ref.kv8_roundtrip(t(2.0 ** 20, 1.0)), t(127 * 256.0, 0.0))  # e clamped at 8, saturated
    assert torch.equal(ref.kv8_roundtrip(t(2.0 ** -40, -(2.0 ** -41))), t(0.0, 0.0))  # e clamped at -24


# ---------------------------------------------------------------------------------------------- tiled attention


def attn_case(t, s, k_start, hq=8, hkv=4, d=16, seed=0):
    rng = np.random.default_rng(seed)
    bf = lambda *sh: ref.bf16(torch.from_numpy(rng.standard_normal(sh)))  # noqa: E731
    q, k, v = bf(t, hq, d) * 2, bf(s, hkv, d) * 2, bf(s, hkv, d)
    bias, gate, sink = bf(hq, t, s), bf(t, hq, d), bf(t, hq)
    k_pos = torch.arange(k_start, k_start + s)
    q_pos = k_pos[-t:]
    return q, k, v, bias, gate, sink, q_pos, k_pos


@pytest.mark.parametrize("window,k_start", [(None, 0), (8, 0), (8, 5)])
def test_tiled_attention_without_rounding_is_attention(window, k_start):
    args = attn_case(t=7, s=15, k_start=k_start)
    want = ref.attention_core(*args, window, Rounding.exact())
    for chunk, tile in ((4, 2), (8, 4), (64, 64), (3, 1)):
        got = ref.attention_core_tiled(*args, window, Rounding.exact(), chunk=chunk, tile=tile)
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


def test_tiled_attention_bf16_p_is_bf16_level_and_depends_on_tiles():
    args = attn_case(t=7, s=15, k_start=3, seed=4)
    exact = ref.attention_core(*args, 8, Rounding())
    a = ref.attention_core_tiled(*args, 8, Rounding(), chunk=4, tile=2)
    b = ref.attention_core_tiled(*args, 8, Rounding(), chunk=8, tile=4)
    rel = lambda x, y: float((x - y).norm() / y.norm())  # noqa: E731
    assert 0 < rel(a, exact) < 4e-3 and 0 < rel(b, exact) < 4e-3
    assert not torch.equal(a, b)  # P is rounded against the running maximum of its own tile sequence


# ---------------------------------------------------------------------------------------------------- whole model


def emulated(make, **flags):
    m = make()
    m.em = Emulation(**flags, **TINY)
    return m


@pytest.mark.parametrize("splits", [[5], [1, 1, 3, 9], [10]])
def test_emulated_prefill_then_decode_equals_recompute(tiny, splits):
    cfg, _, make = tiny
    m = emulated(make, kv8=True, attn_bf16_p=True)
    x = torch.from_numpy(np.random.default_rng(7).integers(0, 300, 19))
    full = m.forward(x, record=True)
    cache, at, parts = m.new_cache(), 0, []
    for n in splits + [len(x) - sum(splits)]:
        parts.append(m.forward(x[at:at + n], cache, record=True))
        at += n
    assert torch.equal(torch.cat([p.logits for p in parts]), full.logits)
    assert torch.equal(torch.cat([p.hidden for p in parts], 1), full.hidden)


def test_kv8_touches_global_layers_only(tiny):
    cfg, _, make = tiny
    x = torch.from_numpy(np.random.default_rng(9).integers(0, 300, 13))
    base, kv8 = make(), emulated(make, kv8=True)
    for m in (base, kv8):
        m.taps = {}
        m.forward(x)
    first_global = min(i for i in range(cfg.num_layers) if cfg.is_global(i))
    for i in range(first_global):  # identical inputs, local attention: identical bits
        assert torch.equal(base.taps[i]["core"], kv8.taps[i]["core"])
    assert not torch.equal(base.taps[first_global]["core"], kv8.taps[first_global]["core"])


def test_emulation_names():
    assert Emulation.parse("none") == Emulation() and Emulation().tag == "none"
    assert Emulation.parse("engine") == Emulation.engine() and Emulation.engine().tag == "engine"
    assert Emulation.parse("kv8, moe_fp16") == Emulation(kv8=True, moe_fp16=True)
    assert Emulation(kv8=True, moe_fp16=True).tag == "kv8,moe_fp16"
    with pytest.raises(ValueError):
        Emulation.parse("fp8")


def test_dense_experts_refuse_fp16_inputs(tiny):
    _, _, make = tiny
    m = emulated(make, moe_fp16=True)
    with pytest.raises(ValueError, match="EXL3"):
        m.forward(torch.tensor([1, 2, 3]))


# ------------------------------------------------------------------------------------------------- fp16 experts


@pytest.fixture(scope="module")
def pack_dir(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("pack"))
    cfg = tiny_config()
    write_pack(root, cfg, random_weights(cfg, seed=2, dense_experts=False), seed=2)
    return root


def test_exl3_fp16_input_is_the_rotated_input_rounded(pack_dir):
    lin = load_exl3(PackIndex(pack_dir), "model.llm.layers.2.mlp.experts.5.gate_proj")
    x = bf16_rows((9, lin.k), seed=3).numpy()
    xh = exl3.fwht_128(x * lin.suh[None, :], axis=1)
    want = exl3.fwht_128(ref.np_f16(xh) @ lin.wq, axis=1) * lin.svh[None, :]
    np.testing.assert_array_equal(lin(x, in_round=ref.np_f16), want)
    plain = lin(x)
    rel = np.linalg.norm(lin(x, in_round=ref.np_f16) - plain) / np.linalg.norm(plain)
    assert 0 < rel < 2e-3


def test_fp16_experts_worker_pool_matches_in_process(pack_dir):
    with open(os.path.join(pack_dir, "config.json")) as f:
        cfg = Inkling2Config.from_hf(json.load(f))
    ix = PackIndex(pack_dir)
    x = torch.from_numpy(np.random.default_rng(5).integers(0, 300, 9))
    outs = []
    for workers in (0, 2):
        experts = PackExperts(ix, workers=workers)
        try:
            m = Inkling2Ref(cfg, PackWeights(ix), experts, unembed_chunk=100, emulate=Emulation(moe_fp16=True))
            outs.append(m.forward(x, record=True))
        finally:
            experts.close()
    assert torch.equal(outs[0].logits, outs[1].logits) and torch.equal(outs[0].hidden, outs[1].hidden)
    plain = Inkling2Ref(cfg, PackWeights(ix), PackExperts(ix), unembed_chunk=100).forward(x, record=True)
    assert not torch.equal(plain.hidden, outs[0].hidden)
