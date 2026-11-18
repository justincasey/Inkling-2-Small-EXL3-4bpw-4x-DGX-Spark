"""Op-level tests: convs and their state, relative bias distances, window masks, sinks and gate, router, softcapped
activation, log scaling, rounding helpers."""

import math

import numpy as np
import pytest
import torch

import inkling2_ref as ref
from inkling2_ref import Rounding

R = Rounding()
X = Rounding.exact()
F64 = torch.float64


def rnd(*shape, seed=0, std=1.0):
    g = torch.Generator().manual_seed(seed)
    return ref.bf16(torch.randn(*shape, generator=g, dtype=F64) * std)


def run_chunks(fn, x, state, splits):
    outs, at = [], 0
    for n in splits:
        y, state = fn(x[at:at + n], state, at, n)
        outs.append(y)
        at += n
    return torch.cat(outs), state


@pytest.mark.parametrize("splits", [[17], [1] * 17, [3, 5, 9], [16, 1]])
def test_static_conv_streaming_equals_batch(splits):
    x, w = rnd(17, 24, seed=1), rnd(24, 4, seed=2, std=0.4)
    s0 = torch.zeros(3, 24, dtype=F64)
    full, fstate = ref.static_conv(x, w, s0, R)
    got, gstate = run_chunks(lambda xc, s, a, n: ref.static_conv(xc, w, s, R), x, s0, splits)
    assert torch.equal(got, full) and torch.equal(gstate, fstate)
    assert torch.equal(fstate, x[-3:])  # the state is the last W-1 raw inputs


@pytest.mark.parametrize("splits", [[17], [1] * 17, [3, 5, 9]])
def test_dynamic_conv_streaming_equals_batch(splits):
    heads, d, h = 2, 8, 32
    x, xd = rnd(17, heads * d, seed=3), rnd(17, h, seed=4)
    sw, wp = rnd(4, heads * d, seed=5, std=0.4), rnd(heads * 4, h, seed=6, std=0.1)
    s0 = torch.zeros(3, heads * d, dtype=F64)
    full, fstate = ref.dynamic_conv(x, xd, sw, wp, heads, s0, R)
    got, gstate = run_chunks(lambda xc, s, a, n: ref.dynamic_conv(xc, xd[a:a + n], sw, wp, heads, s, R), x, s0, splits)
    assert torch.equal(got, full) and torch.equal(gstate, fstate)


def test_static_conv_tap_orientation():
    """w[:, W-1] weights the current token; w[:, 0] the token 3 back; scale (W+1)^-1/2."""
    x = torch.arange(1.0, 7.0, dtype=F64)[:, None]
    s0 = torch.zeros(3, 1, dtype=F64)
    for k, lag in ((3, 0), (0, 3), (2, 1)):
        w = torch.zeros(1, 4, dtype=F64)
        w[0, k] = 1.0
        y, _ = ref.static_conv(x, w, s0, R)
        prev = torch.cat([torch.zeros(lag, 1, dtype=F64), x[:len(x) - lag]])
        torch.testing.assert_close(y, ref.bf16((prev + x) * ref.fp32_const(5 ** -0.5)), rtol=0, atol=0)


def test_dynamic_conv_lag_orientation():
    """static_w[l] weights the token l back; weight_proj row head*W + l adds to lag l of that head only."""
    heads, d, h = 2, 2, 3
    x = torch.arange(1.0, 25.0, dtype=F64).reshape(6, 4)
    xd = torch.zeros(6, h, dtype=F64)
    xd[:, 0] = 1.0
    s0 = torch.zeros(3, heads * d, dtype=F64)
    sw = torch.zeros(4, heads * d, dtype=F64)
    sw[2] = 1.0  # lag 2 for every channel
    wp = torch.zeros(heads * 4, h, dtype=F64)
    wp[1 * 4 + 1, 0] = 0.5  # head 1, lag 1
    y, _ = ref.dynamic_conv(x, xd, sw, wp, heads, s0, R)
    lag = lambda n: torch.cat([torch.zeros(n, 4, dtype=F64), x[:6 - n]])  # noqa: E731
    want = lag(2) + x
    want[:, 2:] += 0.5 * lag(1)[:, 2:]
    torch.testing.assert_close(y, ref.bf16(want * ref.fp32_const(9 ** -0.5)), rtol=0, atol=0)


def test_relative_bias_distances():
    extent, hq, dr = 6, 2, 3
    rel = torch.zeros(dr, extent, dtype=F64)
    rel[1] = torch.arange(1.0, extent + 1)
    r_hat = torch.zeros(4, hq, dr, dtype=F64)
    r_hat[:, :, 1] = 1.0
    q_pos = torch.arange(10, 14)
    k_pos = torch.arange(0, 14)
    b = ref.relative_bias(r_hat, rel, q_pos, k_pos, R)
    dist = q_pos[:, None] - k_pos[None, :]
    want = torch.where((dist >= 0) & (dist < extent), (dist + 1).to(F64), torch.zeros((), dtype=F64))
    assert torch.equal(b[0], want) and torch.equal(b[1], want)


def visible_sets(window, t=12):
    """Attention with zero scores and one-hot values: o_t is the uniform mean over the visible keys."""
    hq = 1
    pos = torch.arange(t)
    q = torch.zeros(t, hq, t, dtype=F64)
    v = torch.eye(t, dtype=F64)[:, None, :]
    big = torch.full((t, hq, t), 60.0, dtype=F64)
    sink = torch.full((t, hq), -1e4, dtype=F64)
    out = ref.attention_core(q, v.clone(), v, torch.zeros(hq, t, t, dtype=F64), big, sink, pos, pos, window, X)
    return [set(torch.nonzero(out[i, 0] > 1e-12).flatten().tolist()) for i in range(t)]


def test_window_masks():
    w = 4
    for i, s in enumerate(visible_sets(w)):
        assert s == set(range(max(0, i - w + 1), i + 1))  # window keys including self
    for i, s in enumerate(visible_sets(None)):
        assert s == set(range(i + 1))


def test_sink_and_gate():
    t, s, hq, hkv, d = 3, 5, 4, 2, 8
    q, k, v = rnd(t, hq, d, seed=10), rnd(s, hkv, d, seed=11), rnd(s, hkv, d, seed=12)
    bias, gate, sink = rnd(hq, t, s, seed=13), rnd(t, hq, d, seed=14), rnd(t, hq, seed=15)
    q_pos, k_pos = torch.arange(2, 5), torch.arange(0, 5)
    got = ref.attention_core(q, k, v, bias, gate, sink, q_pos, k_pos, None, X)
    grp = hq // hkv
    for ti in range(t):
        for h in range(hq):
            vis = [j for j in range(s) if k_pos[j] <= q_pos[ti]]
            logit = torch.stack([q[ti, h] @ k[j, h // grp] / math.sqrt(d) + bias[h, ti, j] for j in vis])
            p = torch.softmax(torch.cat([logit, sink[ti, h:h + 1]]), 0)[:-1]  # extra zero-value logit
            o = sum(p[n] * v[j, h // grp] for n, j in enumerate(vis))
            torch.testing.assert_close(got[ti, h], o * torch.sigmoid(gate[ti, h]), rtol=1e-12, atol=1e-12)


def test_router_selection_ties_bias_and_weights():
    logits = torch.tensor([[0.0, 1.0, -1.0, 1.0, 0.5, 2.0]], dtype=F64)
    bias = torch.zeros(6, dtype=F64)
    idx, w, margin = ref.route(logits, bias, 3, 8.0, R)
    assert idx.tolist() == [[5, 1, 3]]  # tie between 1 and 3 at rank 2/3: both in, lower index first
    idx, _, margin = ref.route(logits, bias, 2, 8.0, R)
    assert idx.tolist() == [[5, 1]] and float(margin[0]) == 0.0  # the tie at the cut goes to the lower index
    bias = torch.tensor([0, 0, 5.0, 0, 0, 0], dtype=F64)  # bias changes the selection ...
    idx, w, _ = ref.route(logits, bias, 2, 8.0, R)
    assert idx.tolist() == [[2, 5]]
    sg = torch.sigmoid(logits[0, [2, 5]])  # ... but not the weights, which use the raw sigmoid
    torch.testing.assert_close(w[0], 8.0 * sg / sg.sum(), rtol=1e-7, atol=0)
    assert abs(float(w.sum()) - 8.0) < 1e-5


def test_softcapped_activation():
    g = np.linspace(-40, 40, 161)[:, None] * np.ones((1, 3))
    u = np.linspace(-200, 200, 161)[:, None] * np.ones((1, 3))
    ident = lambda x: x  # noqa: E731
    a = ref.expert_ffn(lambda x: g, lambda x: u, ident, np.zeros((161, 3)), 4.0, 25.0, X)
    want = 4 * np.tanh(g / 4) / (1 + np.exp(-g)) * 25 * np.tanh(u / 25)  # sigmoid on the raw gate
    np.testing.assert_allclose(a, want, rtol=1e-13, atol=1e-13)
    assert np.abs(a).max() < 100
    rounded = ref.expert_ffn(lambda x: g, lambda x: u, ident, np.zeros((161, 3)), 4.0, 25.0, R)
    gr, ur = ref.np_f32(g), ref.np_f32(u)  # g and u are rounded to fp32, the activation to bf16
    want_r = 4 * np.tanh(gr / 4) / (1 + np.exp(-gr)) * 25 * np.tanh(ur / 25)
    np.testing.assert_array_equal(ref.np_bf16(rounded), rounded)
    np.testing.assert_allclose(rounded, want_r, rtol=2.0 ** -8, atol=0)


def test_log_tau():
    pos = torch.tensor([0, 126998, 127998, 127999, 128000, 255999, 1048575])
    tau = ref.log_tau(pos, 128000.0, 0.1, R)
    assert tau[:4].tolist() == [1.0, 1.0, 1.0, 1.0]  # (pos + 1) / floor <= 1
    want = [1 + 0.1 * math.log((p + 1) / 128000) for p in pos.tolist()[4:]]
    np.testing.assert_allclose(tau[4:].numpy(), np.float32(want), rtol=0, atol=0)


def test_numpy_and_torch_bf16_agree_including_ties():
    rng = np.random.default_rng(0)
    vals = rng.standard_normal(100000) * 10.0 ** rng.integers(-20, 20, 100000)
    base = rng.integers(0, 2 ** 16, 4096).astype(np.uint32) << 16
    halfway = (base | 0x8000).view(np.float32)  # exact ties between two bf16 values
    vals = np.concatenate([vals, halfway[np.isfinite(halfway)].astype(np.float64)])
    np.testing.assert_array_equal(ref.np_bf16(vals), ref.bf16(torch.from_numpy(vals)).numpy())


def test_allreduce_rank_order():
    p = [rnd(3, 5, seed=s) for s in range(4)]
    got = ref.allreduce(p, R)
    acc = p[0].float()
    for x in p[1:]:
        acc = acc + x.float()
    assert torch.equal(got, acc.to(torch.bfloat16).to(F64))


def test_l2clip():
    x = torch.tensor([[3.0, 4.0], [0.3, 0.4]], dtype=F64)
    y = ref.l2clip(x, 1.0)
    torch.testing.assert_close(y[0], x[0] / (5 + 1e-6), rtol=1e-15, atol=0)
    assert torch.equal(y[1], x[1])
