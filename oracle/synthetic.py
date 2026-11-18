"""Synthetic Inkling 2 models for tests: a tiny config with the real model's structure (6-layer AttnRes blocks, global
layers at i % 6 == 5, GQA, windowed local attention, 4 short convs per layer, sigmoid top-k router, softcapped experts),
random weights under the pack's tensor names, and random EXL3 expert groups written as a pack folder."""

from __future__ import annotations

import json
import os

import numpy as np
import torch

import exl3
import st
from inkling2_ref import Inkling2Config


def tiny_config(**over) -> Inkling2Config:
    """Clip thresholds sit below the synthetic q/k/r norms, so every clip is active (the real thresholds over a
    16-dim head would leave them idle and the attention far sharper than the real model's)."""
    base = dict(
        hidden_size=256, num_layers=12, vocab_size=320, unpadded_vocab_size=300, num_heads=8, num_kv_heads=4,
        head_dim=16, d_rel=4, rel_extent=8, sliding_window=8, local_layer_ids=tuple(i for i in range(12) if i % 6 != 5),
        conv_kernel=4, layers_per_block=6, n_experts=8, top_k=3, moe_intermediate=128, gate_softcap=4.0,
        up_softcap=25.0, route_scale=3.0, qk_clip=6.0, rel_clip=4.0, log_scaling_n_floor=6.0,
        log_scaling_alpha=0.1, eps=1e-6, final_softcap=30.0, drafter_target_layers=(5, 11, 12),
    )
    base.update(over)
    return Inkling2Config(**base)


def _bf(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16)


def random_weights(cfg: Inkling2Config, seed: int = 0, dense_experts: bool = True) -> dict[str, torch.Tensor]:
    """Plain tensors by pack name (BF16 / F32 as in the pack). Dense experts use nn.Linear layout [out, in]."""
    g = torch.Generator().manual_seed(seed)
    h, hq, hkv, d, dr = cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.d_rel
    rn = lambda *s, std=1.0: torch.randn(*s, generator=g, dtype=torch.float64) * std  # noqa: E731
    near1 = lambda *s: 1.0 + rn(*s, std=0.1)  # noqa: E731
    w: dict[str, torch.Tensor] = {
        "model.llm.embed.weight": _bf(rn(cfg.vocab_size, h)),
        "model.llm.embed_norm.weight": _bf(near1(h)),
        "model.llm.unembed.weight": _bf(rn(cfg.vocab_size, h, std=h ** -0.5 * 4)),
        "model.llm.norm.weight": _bf(near1(h)),
        "model.llm.attn_res.output_query": rn(h, std=2.0).to(torch.float32),
    }
    for i in range(cfg.num_layers):
        pre = f"model.llm.layers.{i}."
        a = pre + "attn."
        glob = cfg.is_global(i)
        if i > 0:
            shape = (3, h) if glob else (h,)
            w[f"model.llm.attn_res.attn_queries.{i - 1}"] = rn(*shape, std=2.0).to(torch.float32)
        w[f"model.llm.attn_res.mlp_queries.{i}"] = rn(h, std=2.0).to(torch.float32)
        if glob:
            for n in "qkv":
                w[pre + f"{n}_input_norm.weight"] = _bf(near1(h))
            for n in "kv":
                w[a + f"{n}_sconv.static_w"] = _bf(rn(cfg.conv_kernel, hkv * d, std=0.4))
                w[a + f"{n}_sconv.weight_proj"] = _bf(rn(hkv * cfg.conv_kernel, h, std=0.4 * h ** -0.5))
        else:
            w[pre + "attn_norm.weight"] = _bf(near1(h))
            for n in "kv":
                w[a + f"{n}_sconv.weight"] = _bf(rn(hkv * d, 1, cfg.conv_kernel, std=0.4))
        w[pre + "mlp_norm.weight"] = _bf(near1(h))
        w[pre + "attn_sconv.weight"] = _bf(rn(h, 1, cfg.conv_kernel, std=0.4))
        w[pre + "mlp_sconv.weight"] = _bf(rn(h, 1, cfg.conv_kernel, std=0.4))
        s = h ** -0.5
        w[a + "wq_du.weight"] = _bf(rn(hq * d, h, std=s * 3))
        w[a + "wr_du.weight"] = _bf(rn(hq * dr, h, std=s * 3))
        w[a + "gate_attn.weight"] = _bf(rn(hq * d, h, std=s))
        w[a + "attn_sink_proj.weight"] = _bf(rn(hq, h, std=s))
        w[a + "wk_dv.weight"] = _bf(rn(hkv * d, h, std=s * 3))
        w[a + "wv_dv.weight"] = _bf(rn(hkv * d, h, std=s))
        w[a + "wo_ud.weight"] = _bf(rn(h, hq * d, std=(hq * d) ** -0.5))
        w[a + "q_l2_clip.scale"] = _bf(near1(hq, d))
        w[a + "rel_l2_clip.scale"] = _bf(near1(hq, dr))
        extent = cfg.rel_extent if glob else cfg.sliding_window
        w[a + "rel_logits_proj.proj"] = _bf(rn(dr, extent, std=0.5))
        w[pre + "mlp.gate.weight"] = _bf(rn(cfg.n_experts, h, std=s))
        w[pre + "mlp.gate.bias"] = rn(cfg.n_experts, std=0.05).to(torch.float32)
        if dense_experts:
            for e in range(cfg.n_experts):
                ep = pre + f"mlp.experts.{e}."
                w[ep + "gate_proj.weight"] = _bf(rn(cfg.moe_intermediate, h, std=s * 2))
                w[ep + "up_proj.weight"] = _bf(rn(cfg.moe_intermediate, h, std=s * 2))
                w[ep + "down_proj.weight"] = _bf(rn(h, cfg.moe_intermediate, std=cfg.moe_intermediate ** -0.5))
    return w


def random_exl3_group(rng: np.random.Generator, k: int, n: int, twice: int = 8, cb: str = "mcg",
                      scale_in: float = 0.03, scale_out: float = 0.03) -> dict[str, np.ndarray]:
    """One random EXL3 group (trellis int16 [K/16, N/16, 8*twice], suh/svh fp16 with random signs, codebook marker)."""
    trellis = rng.integers(-32768, 32768, size=(k // 16, n // 16, 8 * twice), dtype=np.int64).astype(np.int16)
    sgn = lambda m: rng.choice([-1.0, 1.0], size=m)  # noqa: E731
    out = {
        "trellis": trellis,
        "suh": (sgn(k) * scale_in * (1 + 0.2 * rng.random(k))).astype(np.float16),
        "svh": (sgn(n) * scale_out * (1 + 0.2 * rng.random(n))).astype(np.float16),
    }
    if cb != "3inst":
        marker = exl3.MARKER_MCG if cb == "mcg" else exl3.MARKER_MUL1
        out[cb] = np.array(marker, dtype=np.uint32).view(np.int32).reshape(())
    return out


def write_pack(root: str, cfg: Inkling2Config, plain: dict[str, torch.Tensor], seed: int = 0, shards: int = 3,
               expert_files: int = 4) -> None:
    """A pack folder in the real layout: plain tensors over `shards` model-nonexpert-XX files, EXL3 experts striped
    E mod `expert_files` into model-experts-LL-SS files, an index, and a config.json."""
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(seed)
    weight_map: dict[str, str] = {}
    names = sorted(plain)
    for s in range(shards):
        fn = f"model-nonexpert-{s:02d}.safetensors"
        part = {n: plain[n] for n in names[s::shards]}
        st.write(os.path.join(root, fn), part)
        weight_map.update({n: fn for n in part})
    h, inter = cfg.hidden_size, cfg.moe_intermediate
    for layer in range(cfg.num_layers):
        for ss in range(expert_files):
            fn = f"model-experts-{layer:02d}-{ss:02d}.safetensors"
            t: dict[str, np.ndarray] = {}
            for e in range(ss, cfg.n_experts, expert_files):
                for proj, (k, n) in (("gate_proj", (h, inter)), ("up_proj", (h, inter)), ("down_proj", (inter, h))):
                    g = random_exl3_group(rng, k, n, scale_in=k ** -0.25 * 0.3, scale_out=n ** -0.25 * 0.3)
                    pre = f"model.llm.layers.{layer}.mlp.experts.{e}.{proj}"
                    for part, arr in g.items():
                        t[f"{pre}.{part}"] = arr
            st.write(os.path.join(root, fn), t)
            weight_map.update({n: fn for n in t})
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    text = {
        "hidden_size": cfg.hidden_size, "num_hidden_layers": cfg.num_layers, "vocab_size": cfg.vocab_size,
        "unpadded_vocab_size": cfg.unpadded_vocab_size, "num_attention_heads": cfg.num_heads,
        "num_key_value_heads": cfg.num_kv_heads, "head_dim": cfg.head_dim, "d_rel": cfg.d_rel,
        "rel_extent": cfg.rel_extent, "sliding_window_size": cfg.sliding_window,
        "local_layer_ids": list(cfg.local_layer_ids), "sconv_kernel_size": cfg.conv_kernel,
        "n_layers_per_attn_res_block": cfg.layers_per_block, "n_routed_experts": cfg.n_experts,
        "num_experts_per_tok": cfg.top_k, "intermediate_size": cfg.moe_intermediate,
        "mlp_gate_softcap": cfg.gate_softcap, "mlp_up_softcap": cfg.up_softcap, "route_scale": cfg.route_scale,
        "qk_l2_clip_threshold": cfg.qk_clip, "rel_l2_clip_threshold": cfg.rel_clip,
        "log_scaling_n_floor": cfg.log_scaling_n_floor, "log_scaling_alpha": cfg.log_scaling_alpha,
        "rms_norm_eps": cfg.eps, "final_logit_softcapping": cfg.final_softcap, "n_shared_experts": 0,
        "use_gate_bias": True, "use_value_norm_no_gamma": True, "attn_res_normalize_values": True,
        "pre_norm_pre_affine": False, "final_norm_pre_affine": False,
    }
    conf = {"model_type": "inkling2", "text_config": text,
            "mtp_config": {"target_layer_ids": list(cfg.drafter_target_layers)},
            "quantization_config": {"quant_method": "exl3", "bits": 4.0, "codebook": "mcg", "out_scales": "always"}}
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(conf, f, indent=1)
