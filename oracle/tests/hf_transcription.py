"""An independent float64 transcription of the HF eager model (transformers fc00b5b3,
src/transformers/models/inkling2/modeling_inkling2.py, Inkling2ForCausalLM.forward) over pack tensor names, with every
bf16 cast removed. It keeps HF's own op forms (F.conv1d with padding, unfold/flip windows for the dynamic conv, the
concatenated-sink softmax, repeat_kv, SDPA for the block residual, index_add MoE) so that comparing it with the oracle
under `Rounding.exact()` checks the oracle's op structure (indexing, orientation, schedule) independently of its
rounding points. Whole sequences only (no cache)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

F64 = torch.float64


def _w(w, name):
    return w[name].to(F64)


def rms_norm(x, weight, eps):  # Inkling2RMSNorm
    return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def unweighted(x, eps=1e-6):  # Inkling2UnweightedRMSNorm
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def l2_clip(x, threshold, scale=None):  # Inkling2L2Clip / Inkling2ScaledL2Clip
    if scale is not None:
        x = x * scale
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x * (threshold / (norm + 1e-6)).clamp(max=1.0)


def short_conv(x, weight):  # Inkling2ShortConvolution, x [T, C], weight [C, 1, K]
    k = weight.shape[-1]
    t, c = x.shape
    out = F.conv1d(x.T[None], weight, padding=k - 1, groups=c)[0, :, :t].T
    return (out + x) / math.sqrt(k + 1)


def dynamic_conv(x, dyn_in, static_w, weight_proj, heads):  # Inkling2DynamicShortConvolution
    t, c = x.shape
    k = static_w.shape[0]
    dynamic = (dyn_in @ weight_proj.T).view(1, t, heads, 1, k)
    weights = static_w.T.reshape(heads, -1, k) + dynamic
    context = x[None].transpose(1, 2)
    context = F.pad(context, (k - 1, 0))[..., -(t + k - 1):]
    windows = context.unfold(-1, k, 1).flip(-1).transpose(1, 2)
    windows = windows.reshape(1, t, heads, -1, k)
    y = (windows * weights).sum(-1).reshape(t, c)
    return (y + x) / math.sqrt(2 * k + 1)


def rel_logits(rel_states, proj, qpos, kpos):  # Inkling2RelativeLogits, rel_states [1, T, H, d]
    extent = proj.shape[1]
    rl = (rel_states @ proj).transpose(1, 2)
    distance = (qpos[:, None] - kpos[None, :])[None, None]
    idx = distance.clamp(0, extent - 1).expand(*rl.shape[:2], -1, -1)
    bias = rl.gather(-1, idx)
    return bias.masked_fill((distance < 0) | (distance >= extent), 0.0)


def attention(cfg, w, i, xq, xk, xv, pos):
    a = f"model.llm.layers.{i}.attn."
    t = xq.shape[0]
    hq, hkv, d = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    glob = cfg.is_global(i)
    q = xq @ _w(w, a + "wq_du.weight").T
    if glob:
        k = dynamic_conv(xk @ _w(w, a + "wk_dv.weight").T, xk, _w(w, a + "k_sconv.static_w"),
                         _w(w, a + "k_sconv.weight_proj"), hkv)
        v = dynamic_conv(xv @ _w(w, a + "wv_dv.weight").T, xv, _w(w, a + "v_sconv.static_w"),
                         _w(w, a + "v_sconv.weight_proj"), hkv)
    else:
        k = short_conv(xk @ _w(w, a + "wk_dv.weight").T, _w(w, a + "k_sconv.weight"))
        v = short_conv(xv @ _w(w, a + "wv_dv.weight").T, _w(w, a + "v_sconv.weight"))
    rel = xq @ _w(w, a + "wr_du.weight").T
    gate = xq @ _w(w, a + "gate_attn.weight").T
    sinks = xq @ _w(w, a + "attn_sink_proj.weight").T
    q = l2_clip(q.view(1, t, hq, d), cfg.qk_clip, _w(w, a + "q_l2_clip.scale"))
    rel = l2_clip(rel.view(1, t, hq, -1), cfg.rel_clip, _w(w, a + "rel_l2_clip.scale"))
    if glob:
        tau = 1.0 + cfg.log_scaling_alpha * torch.log(((pos + 1).to(F64) / cfg.log_scaling_n_floor).clamp(min=1.0))
        q = q * tau[None, :, None, None]
        rel = rel * tau[None, :, None, None]
    q = q.transpose(1, 2)
    k = l2_clip(k.view(1, t, hkv, d), cfg.qk_clip).transpose(1, 2)
    v = unweighted(v.view(1, t, hkv, d), cfg.eps).transpose(1, 2)
    bias = rel_logits(rel, _w(w, a + "rel_logits_proj.proj"), pos, pos)
    grp = hq // hkv
    k = k[:, :, None].expand(1, hkv, grp, t, d).reshape(1, hq, t, d)  # repeat_kv
    v = v[:, :, None].expand(1, hkv, grp, t, d).reshape(1, hq, t, d)
    scores = torch.matmul(q, k.transpose(2, 3)) * d ** -0.5 + bias
    dist = pos[:, None] - pos[None, :]
    allowed = dist >= 0
    if not glob:
        allowed &= dist < cfg.sliding_window
    scores = scores + torch.where(allowed, 0.0, float("-inf")).to(F64)
    sink = sinks.view(1, t, hq).transpose(1, 2).unsqueeze(-1)
    probs = F.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
    out = torch.matmul(probs, v).transpose(1, 2).reshape(t, hq * d)
    out = out * torch.sigmoid(gate)
    return out @ _w(w, a + "wo_ud.weight").T


def moe(cfg, w, i, x):
    pre = f"model.llm.layers.{i}.mlp."
    logits = x @ _w(w, pre + "gate.weight").T
    scores = torch.sigmoid(logits)
    idx = torch.topk(scores + _w(w, pre + "gate.bias"), cfg.top_k, dim=-1).indices
    wts = scores.gather(1, idx)
    wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * cfg.route_scale
    out = torch.zeros_like(x)
    for e in range(cfg.n_experts):
        tok, slot = torch.where(idx == e)
        if tok.numel() == 0:
            continue
        ep = pre + f"experts.{e}."
        g = x[tok] @ _w(w, ep + "gate_proj.weight").T
        u = x[tok] @ _w(w, ep + "up_proj.weight").T
        act = torch.tanh(g / cfg.gate_softcap) * cfg.gate_softcap * torch.sigmoid(g) * \
            torch.tanh(u / cfg.up_softcap) * cfg.up_softcap
        y = act @ _w(w, ep + "down_proj.weight").T
        out.index_add_(0, tok, y * wts[tok, slot, None])
    return out


def attn_res(blocks, partial, query, hidden):  # Inkling2BlockAttentionResidual
    stacked = unweighted(torch.cat([blocks, partial[:, None]], dim=1))  # [T, N, H]
    q = query.to(F64).view(-1, hidden)[None].expand(stacked.shape[0], -1, -1)
    out = F.scaled_dot_product_attention(q, stacked, stacked, scale=hidden ** -0.5)
    return out.unbind(-2)


def forward(cfg, w, ids):
    """Logits [T, unpadded_vocab] float64 and the partial block entering each layer."""
    hdim = cfg.hidden_size
    ids = torch.as_tensor(ids)
    t = ids.shape[0]
    pos = torch.arange(t)
    h = rms_norm(_w(w, "model.llm.embed.weight")[ids], _w(w, "model.llm.embed_norm.weight"), cfg.eps)
    blocks = h.new_zeros(t, 0, hdim)
    hidden = []
    for i in range(cfg.num_layers):
        hidden.append(h)
        ln = f"model.llm.layers.{i}."
        boundary = i % cfg.layers_per_block == 0
        block_residual = h
        if cfg.is_global(i):
            if i == 0:
                qs = ks = vs = h
            else:
                qs, ks, vs = attn_res(blocks, h, w[f"model.llm.attn_res.attn_queries.{i - 1}"], hdim)
            a = attention(cfg, w, i, rms_norm(qs, _w(w, ln + "q_input_norm.weight"), cfg.eps),
                          rms_norm(ks, _w(w, ln + "k_input_norm.weight"), cfg.eps),
                          rms_norm(vs, _w(w, ln + "v_input_norm.weight"), cfg.eps), pos)
        else:
            x = h if i == 0 else attn_res(blocks, h, w[f"model.llm.attn_res.attn_queries.{i - 1}"], hdim)[0]
            xn = rms_norm(x, _w(w, ln + "attn_norm.weight"), cfg.eps)
            a = attention(cfg, w, i, xn, xn, xn, pos)
        a = short_conv(a, _w(w, ln + "attn_sconv.weight"))
        if boundary:
            blocks = torch.cat([blocks, block_residual[:, None]], dim=1)
            residual = a
        else:
            residual = block_residual + a
        m = attn_res(blocks, residual, w[f"model.llm.attn_res.mlp_queries.{i}"], hdim)[0]
        m = moe(cfg, w, i, rms_norm(m, _w(w, ln + "mlp_norm.weight"), cfg.eps))
        m = short_conv(m, _w(w, ln + "mlp_sconv.weight"))
        h = residual + m
    hidden.append(h)
    x = attn_res(blocks, h, w["model.llm.attn_res.output_query"], hdim)[0]
    x = rms_norm(x, _w(w, "model.llm.norm.weight"), cfg.eps)
    logits = x @ _w(w, "model.llm.unembed.weight").T
    logits = torch.tanh(logits / cfg.final_softcap) * cfg.final_softcap
    return logits[:, :cfg.unpadded_vocab_size], torch.stack(hidden)
