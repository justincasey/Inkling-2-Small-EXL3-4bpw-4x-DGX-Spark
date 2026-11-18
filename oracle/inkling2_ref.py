"""Inkling 2 Small reference forward ("oracle"): the target text model of docs/PORT-SPEC.md §3, on the CPU.

Numerics (docs/NUMERICS.md): every op between two storage points is evaluated in float64 from the stored values, so the
oracle is the exact-arithmetic value of the spec's formulas up to the roundings the spec names. Values are rounded
  * to bf16 (via fp32: `bf16(fp32(x))`, what a kernel with an fp32 result and a bf16 store gives) at every bf16 storage
    point of PORT-SPEC §3, and
  * to fp32 where the spec names an fp32 value that crosses an op boundary (router logits and weights, the AttnRes P+a
    addend, the rank-order all-reduce sum, unembed output and soft-capped logits, tau).
Because accumulation is float64 (products of bf16/fp16 values are exact there), results do not depend on batch size,
chunking or summation order: prefill + decode reproduces a full recompute bit for bit.

Tensor-parallel rounding (§2) is modelled with `tp` ranks: rank r owns q heads [r*Hq/tp, ...) and the routed experts
E with E mod tp == r; each rank's partial (wo, MoE) is rounded to bf16, the partials are summed in rank order in fp32
and rounded to bf16 once. `tp=1` is the single-device model.

Everything here is torch float64 on the CPU; expert FFNs run in numpy (see `expert_ffn`) so they can run in worker
processes over the EXL3 pack.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np
import torch

F64 = torch.float64
IMAGE_PAD_ID = 200054


# ----------------------------------------------------------------------------------------------------------------------
# rounding


def bf16(x: torch.Tensor) -> torch.Tensor:
    """Round to bf16 via fp32 (both round-to-nearest-even); returned as float64 values."""
    return x.to(torch.float32).to(torch.bfloat16).to(F64)


def f32(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32).to(F64)


def _ident(x):
    return x


def np_f32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32).astype(np.float64)


def np_bf16(x: np.ndarray) -> np.ndarray:
    """numpy twin of `bf16`: fp32 round, then round-to-nearest-even to the top 16 bits (finite values only)."""
    a = np.ascontiguousarray(x, dtype=np.float32)
    if not np.isfinite(a).all():
        raise FloatingPointError("non-finite value at a bf16 rounding point")
    b = a.view(np.uint32).astype(np.uint64)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.astype(np.uint32).view(np.float32).astype(np.float64)


@dataclass(frozen=True)
class Rounding:
    """The rounding points. `exact()` turns them all off and keeps outputs in float64 (used to check the op structure
    against an unrounded float64 transcription of the HF model)."""

    bf16: Callable = bf16
    f32: Callable = f32
    np_bf16: Callable = np_bf16
    np_f32: Callable = np_f32
    is_exact: bool = False

    @staticmethod
    def exact() -> "Rounding":
        return Rounding(_ident, _ident, _ident, _ident, True)

    def const(self, x: float) -> float:
        """A scale constant the kernels hold in fp32 (conv (W+1)^-1/2, AttnRes H^-1/2)."""
        return x if self.is_exact else fp32_const(x)


def fp32_const(x: float) -> float:
    return float(np.float32(x))


def np_f16(x: np.ndarray) -> np.ndarray:
    """fp32 round, then fp16 round-to-nearest-even (the engine's `__floats2half2_rn` of an fp32 value); finite only."""
    h = np.asarray(x, dtype=np.float64).astype(np.float32).astype(np.float16)
    if not np.isfinite(h).all():
        raise FloatingPointError("value outside fp16 range at an emulated fp16 rounding point")
    return h.astype(np.float64)


def kv8_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """REXL3's int8 KV (`nemotron_kv8.cuh`) stored and read back, per row of the last dim (one (token, KV head)): e is
    the smallest exponent with amax / 2^e <= 127 (from amax's fp32 bits), clamped to [-24, 8]; q = rint(x / 2^e)
    (nearest even), saturated to +-127; the value read is q * 2^e (exact in bf16 and fp16). x holds bf16 values."""
    amax = x.abs().amax(-1, keepdim=True).to(torch.float32)
    bits = amax.view(torch.int32).to(torch.int64)
    big_e = ((bits >> 23) & 0xFF) - 127
    e = torch.where((bits & 0x7FFFFF) <= 0x7E0000, big_e - 6, big_e - 5).clamp(-24, 8).to(F64)
    scale = torch.exp2(e)
    return torch.round(x / scale).clamp(-127, 127) * scale


@dataclass(frozen=True)
class Emulation:
    """Deliberately lossy engine choices the oracle can reproduce (docs/ORACLE-GAP.md). All off (the default) is the
    spec's numerics of docs/NUMERICS.md; `engine()` turns on every one the REXL3 port makes:

    * `kv8`: the global layers' K and V are stored as int8 with a power-of-two scale per (token, KV head)
      (`kv8_roundtrip`); every query, the writing step's own included, reads them back from that store.
    * `attn_bf16_p`: the attention core's arithmetic structure (`inkling2_attn.cu`): keys in absolute chunks of
      `chunk_global` / `chunk_local` keys made of absolute `tile`-key tiles, an online softmax per chunk in the exp2
      domain with P rounded to bf16 for P.V (l sums the unrounded p), the chunk states folded left in ascending order.
    * `moe_fp16`: the EXL3 expert GEMM inputs (the rotated `x * suh`, and the rotated `act * suh` of the down
      projection) rounded to fp16 (`moe_experts.cu` / `moe_prefill.cu` `rot_in`, the gate/up epilogue's tail)."""

    kv8: bool = False
    attn_bf16_p: bool = False
    moe_fp16: bool = False
    tile: int = 64
    chunk_global: int = 512
    chunk_local: int = 256

    NAMES = ("kv8", "attn_bf16_p", "moe_fp16")

    @staticmethod
    def engine() -> "Emulation":
        return Emulation(kv8=True, attn_bf16_p=True, moe_fp16=True)

    @staticmethod
    def parse(spec: str) -> "Emulation":
        """'none', 'engine' (all), or a comma list of NAMES."""
        spec = spec.strip()
        if spec in ("", "none"):
            return Emulation()
        if spec == "engine":
            return Emulation.engine()
        names = [s.strip() for s in spec.split(",") if s.strip()]
        bad = [n for n in names if n not in Emulation.NAMES]
        if bad:
            raise ValueError(f"unknown emulation {bad}: choose from none, engine, {', '.join(Emulation.NAMES)}")
        return Emulation(**{n: True for n in names})

    @property
    def tag(self) -> str:
        on = [n for n in self.NAMES if getattr(self, n)]
        return "none" if not on else "engine" if len(on) == len(self.NAMES) else ",".join(on)


# ----------------------------------------------------------------------------------------------------------------------
# config


@dataclass(frozen=True)
class Inkling2Config:
    hidden_size: int = 4608
    num_layers: int = 48
    vocab_size: int = 201024
    unpadded_vocab_size: int = 200058
    num_heads: int = 48
    num_kv_heads: int = 8
    head_dim: int = 128
    d_rel: int = 16
    rel_extent: int = 1024
    sliding_window: int = 1024
    local_layer_ids: tuple[int, ...] = tuple(i for i in range(48) if i % 6 != 5)
    conv_kernel: int = 4
    layers_per_block: int = 6
    n_experts: int = 384
    top_k: int = 8
    moe_intermediate: int = 2304
    gate_softcap: float = 4.0
    up_softcap: float = 25.0
    route_scale: float = 8.0
    qk_clip: float = 22.627416997969522
    rel_clip: float = 8.0
    log_scaling_n_floor: float = 128000.0
    log_scaling_alpha: float = 0.1
    eps: float = 1e-6
    final_softcap: float = 30.0
    drafter_target_layers: tuple[int, ...] = (5, 11, 17, 23, 29, 35, 41, 48)

    def is_global(self, i: int) -> bool:
        return i not in self.local_layer_ids

    @classmethod
    def from_hf(cls, config: dict) -> "Inkling2Config":
        """From the pack's config.json (top level with `text_config`)."""
        t = config.get("text_config", config)
        checks = {
            "pre_norm_pre_affine": False,
            "final_norm_pre_affine": False,
            "use_value_norm_no_gamma": True,
            "attn_res_normalize_values": True,
            "n_shared_experts": 0,
            "use_gate_bias": True,
            "q_bias": False,
            "o_bias": False,
        }
        for k, want in checks.items():
            if t.get(k, want) != want:
                raise ValueError(f"text_config.{k} = {t.get(k)!r}: the oracle implements {want!r} only")
        for a, b in (("swa_head_dim", "head_dim"), ("swa_num_attention_heads", "num_attention_heads"),
                     ("swa_num_key_value_heads", "num_key_value_heads")):
            if a in t and t[a] != t[b]:
                raise ValueError(f"{a} = {t[a]} differs from {b} = {t[b]}; the oracle assumes equal local/global heads")
        drafter = tuple(config.get("mtp_config", {}).get("target_layer_ids", cls.drafter_target_layers))
        return cls(
            hidden_size=t["hidden_size"], num_layers=t["num_hidden_layers"], vocab_size=t["vocab_size"],
            unpadded_vocab_size=t["unpadded_vocab_size"], num_heads=t["num_attention_heads"],
            num_kv_heads=t["num_key_value_heads"], head_dim=t["head_dim"], d_rel=t["d_rel"], rel_extent=t["rel_extent"],
            sliding_window=t["sliding_window_size"], local_layer_ids=tuple(t["local_layer_ids"]),
            conv_kernel=t["sconv_kernel_size"], layers_per_block=t["n_layers_per_attn_res_block"],
            n_experts=t["n_routed_experts"], top_k=t["num_experts_per_tok"], moe_intermediate=t["intermediate_size"],
            gate_softcap=t["mlp_gate_softcap"], up_softcap=t["mlp_up_softcap"], route_scale=t["route_scale"],
            qk_clip=t["qk_l2_clip_threshold"], rel_clip=t["rel_l2_clip_threshold"],
            log_scaling_n_floor=t["log_scaling_n_floor"], log_scaling_alpha=t["log_scaling_alpha"],
            eps=t["rms_norm_eps"], final_softcap=t["final_logit_softcapping"], drafter_target_layers=drafter,
        )


# ----------------------------------------------------------------------------------------------------------------------
# weight sources


class WeightSource(Protocol):
    def tensor(self, name: str) -> torch.Tensor: ...

    def rows(self, name: str, ids: torch.Tensor) -> torch.Tensor: ...

    def row_range(self, name: str, start: int, stop: int) -> torch.Tensor: ...


class DictWeights:
    """Weights held in memory, by checkpoint name, in their stored dtypes."""

    def __init__(self, tensors: dict[str, torch.Tensor]):
        self.t = tensors

    def tensor(self, name: str) -> torch.Tensor:
        return self.t[name]

    def rows(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        return self.t[name][ids]

    def row_range(self, name: str, start: int, stop: int) -> torch.Tensor:
        return self.t[name][start:stop]


Linear = Callable[..., np.ndarray]  # float64 [rows, K] -> float64 [rows, N]; EXL3 layers also take `in_round=`


class ExpertSource(Protocol):
    def ffn_many(self, layer: int, jobs: Sequence[tuple[int, np.ndarray]], gate_cap: float, up_cap: float,
                 rounding: Rounding, fp16_in: bool = False) -> list[np.ndarray]:
        """For each (expert, x [rows, H] float64 holding bf16 values): the expert's down output [rows, H] float64.
        `fp16_in` emulates the engine's fp16 GEMM inputs (`Emulation.moe_fp16`)."""
        ...


def softcap_np(x: np.ndarray, cap: float) -> np.ndarray:
    return cap * np.tanh(x / cap)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """1 / (1 + e^-x) without cancellation for large |x| (e^-|x| never overflows)."""
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def expert_ffn(gate: Linear, up: Linear, down: Linear, x: np.ndarray, gate_cap: float, up_cap: float,
               rounding: Rounding, fp16_in: bool = False) -> np.ndarray:
    """PORT-SPEC §3.14, NUMERICS U2/U11: g, u rounded to fp32; act = softcap(g, 4) * sigmoid(g) * softcap(u, 25) with the
    sigmoid on the raw g; act rounded to bf16 before the down projection; down output float64 (weighted and summed by the
    caller). `fp16_in`: each EXL3 layer's rotated input is rounded to fp16 before the W_q product, as REXL3's kernels
    store it (`Emulation.moe_fp16`); the layers must accept `in_round=`."""
    lin = (lambda f, v: f(v, in_round=np_f16)) if fp16_in else (lambda f, v: f(v))  # noqa: E731
    g = rounding.np_f32(lin(gate, x))
    u = rounding.np_f32(lin(up, x))
    a = softcap_np(g, gate_cap) * sigmoid_np(g) * softcap_np(u, up_cap)
    return lin(down, rounding.np_bf16(a))


class DenseExperts:
    """Experts as dense nn.Linear-layout weights [out, in] (bf16 or any float dtype), by name
    `model.llm.layers.{i}.mlp.experts.{e}.{gate_proj,up_proj,down_proj}.weight`."""

    def __init__(self, weights: WeightSource):
        self.w = weights

    def linear(self, layer: int, e: int, proj: str) -> Linear:
        w = self.w.tensor(f"model.llm.layers.{layer}.mlp.experts.{e}.{proj}.weight").to(F64).numpy()
        return lambda x: x @ w.T

    def ffn_many(self, layer, jobs, gate_cap, up_cap, rounding, fp16_in=False):
        if fp16_in:
            raise ValueError("fp16 GEMM-input emulation is defined on the EXL3 rotated form; dense experts have none")
        return [expert_ffn(self.linear(layer, e, "gate_proj"), self.linear(layer, e, "up_proj"),
                           self.linear(layer, e, "down_proj"), x, gate_cap, up_cap, rounding) for e, x in jobs]


# ----------------------------------------------------------------------------------------------------------------------
# ops


def rmsnorm_w(x: torch.Tensor, w: torch.Tensor, eps: float, r: Rounding) -> torch.Tensor:
    """bf16( x * rsqrt(mean(x^2) + eps) * w ) over the last dim."""
    return r.bf16(x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w)


def l2clip(x: torch.Tensor, c: float) -> torch.Tensor:
    """x * min(c / (||x||_2 + 1e-6), 1) over the last dim."""
    norm = torch.sqrt((x * x).sum(-1, keepdim=True))
    return x * torch.clamp(c / (norm + 1e-6), max=1.0)


def static_conv(x: torch.Tensor, w: torch.Tensor, state: torch.Tensor, r: Rounding) -> tuple[torch.Tensor, torch.Tensor]:
    """§3.4 static short conv. x [T, C] (bf16 values), w [C, W] (tap W-1 = current token), state [W-1, C] = the previous
    W-1 raw inputs (zeros at sequence start). y_t = bf16((sum_k w[:, k] x_{t-W+1+k} + x_t) * fp32((W+1)^-1/2))."""
    t, wk = x.shape[0], w.shape[1]
    full = torch.cat([state, x], 0)
    acc = torch.zeros_like(x)
    for k in range(wk):  # oldest tap first
        acc = acc + w[:, k] * full[k:k + t]
    y = r.bf16((acc + x) * r.const((wk + 1) ** -0.5))
    return y, full[full.shape[0] - (wk - 1):]


def dynamic_conv(x: torch.Tensor, x_dyn: torch.Tensor, static_w: torch.Tensor, weight_proj: torch.Tensor, n_heads: int,
                 state: torch.Tensor, r: Rounding) -> tuple[torch.Tensor, torch.Tensor]:
    """§3.4 dynamic short conv (global-layer K/V). x [T, C] raw K or V; x_dyn [T, H] the layer's k/v input;
    static_w [W, C] (row = backward distance); weight_proj [n_heads * W, H] (row = head * W + lag).
    y_t[c] = bf16((sum_l (static_w[l, c] + dyn_t[head(c), l]) x_{t-l}[c] + x_t[c]) * fp32((2W+1)^-1/2)),
    dyn_t = bf16(x_dyn_t . weight_proj^T)."""
    t, c = x.shape
    wk = static_w.shape[0]
    d = c // n_heads
    dyn = r.bf16(x_dyn @ weight_proj.T).view(t, n_heads, 1, wk)
    taps = (static_w.T.reshape(n_heads, d, wk)[None] + dyn).reshape(t, c, wk)  # [T, C, lag]
    full = torch.cat([state, x], 0)
    acc = torch.zeros_like(x)
    for lag in range(wk):  # newest first
        acc = acc + taps[:, :, lag] * full[wk - 1 - lag:wk - 1 - lag + t]
    y = r.bf16((acc + x) * r.const((2 * wk + 1) ** -0.5))
    return y, full[full.shape[0] - (wk - 1):]


def log_tau(pos: torch.Tensor, n_floor: float, alpha: float, r: Rounding) -> torch.Tensor:
    """§3.10: tau = 1 + alpha * ln(max((pos + 1) / n_floor, 1)), an fp32 value."""
    return r.f32(1.0 + alpha * torch.log(torch.clamp((pos.to(F64) + 1.0) / n_floor, min=1.0)))


def relative_bias(r_hat: torch.Tensor, rel_proj: torch.Tensor, q_pos: torch.Tensor, k_pos: torch.Tensor,
                  r: Rounding) -> torch.Tensor:
    """§3.9: bias[h, t, s] = bf16(r_hat[t, h] . rel_proj[:, q_pos[t] - k_pos[s]]) for 0 <= distance < extent, else 0.
    r_hat [T, Hq, d_rel]; rel_proj [d_rel, extent]. Returns [Hq, T, S] (masking is the caller's)."""
    extent = rel_proj.shape[1]
    table = r.bf16(torch.einsum("thj,je->hte", r_hat, rel_proj))  # [Hq, T, extent]
    dist = q_pos[:, None] - k_pos[None, :]
    inside = (dist >= 0) & (dist < extent)
    idx = dist.clamp(0, extent - 1)[None].expand(table.shape[0], -1, -1)
    return torch.where(inside[None], table.gather(-1, idx), torch.zeros((), dtype=F64))


def attention_core(q_hat: torch.Tensor, k_hat: torch.Tensor, v_hat: torch.Tensor, bias: torch.Tensor,
                   gate: torch.Tensor, sink: torch.Tensor, q_pos: torch.Tensor, k_pos: torch.Tensor,
                   window: int | None, r: Rounding) -> torch.Tensor:
    """§3.11. q_hat [T, Hq, D], k_hat/v_hat [S, Hkv, D], bias [Hq, T, S], gate [T, Hq, D], sink [T, Hq].
    s = q.k * D^-1/2 + bias over visible keys (0 <= dq < window, or causal if window is None);
    out = bf16(softmax(s) V * sigmoid(gate) * sigmoid(lse - sink)), all in float64 (NUMERICS U5)."""
    hq, hkv, d = q_hat.shape[1], k_hat.shape[1], q_hat.shape[2]
    grp = hq // hkv
    k_rep = k_hat.repeat_interleave(grp, dim=1)
    v_rep = v_hat.repeat_interleave(grp, dim=1)
    s = torch.einsum("thd,shd->hts", q_hat, k_rep) * (d ** -0.5) + bias
    dist = q_pos[:, None] - k_pos[None, :]
    vis = dist >= 0
    if window is not None:
        vis = vis & (dist < window)
    s = s.masked_fill(~vis[None], float("-inf"))
    lse = torch.logsumexp(s, dim=-1)  # [Hq, T]
    p = torch.exp(s - lse[..., None])
    o = torch.einsum("hts,shd->thd", p, v_rep)
    out = o * torch.sigmoid(gate) * torch.sigmoid(lse.T - sink)[..., None]
    return r.bf16(out)


_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453


def attention_core_tiled(q_hat: torch.Tensor, k_hat: torch.Tensor, v_hat: torch.Tensor, bias: torch.Tensor,
                         gate: torch.Tensor, sink: torch.Tensor, q_pos: torch.Tensor, k_pos: torch.Tensor,
                         window: int | None, r: Rounding, chunk: int, tile: int) -> torch.Tensor:
    """§3.11 in the arithmetic structure of REXL3's kernel (`inkling2_attn.cu` header; `Emulation.attn_bf16_p`), the
    operations in float64. Keys are cut into absolute chunks of `chunk` keys and those into absolute `tile`-key tiles.
    Per chunk an online softmax over its tiles in ascending order in the exp2 domain: t = (q.k D^-1/2 + bias) log2 e;
    m' = max(m, tile max); a = 2^(m - m'); p = 2^(t - m'); l = l a + sum p (unrounded); o = o a + bf16(p).V. A tile with
    no visible key is a no-op for the row. The chunk states are folded left in ascending order (the empty state is an
    identity), and out = bf16(o / l * sigmoid(gate) * sigmoid(lse - sink)), lse = (m + log2 l) ln 2. With `r` exact, P
    is not rounded and the result equals `attention_core` up to float64 rounding. Same shapes as `attention_core`."""
    hq, hkv, d = q_hat.shape[1], k_hat.shape[1], q_hat.shape[2]
    grp = hq // hkv
    k_rep = k_hat.repeat_interleave(grp, dim=1)
    v_rep = v_hat.repeat_interleave(grp, dim=1)
    s = (torch.einsum("thd,shd->hts", q_hat, k_rep) * (d ** -0.5) + bias) * _LOG2E  # [Hq, T, S]
    dist = q_pos[:, None] - k_pos[None, :]
    vis = dist >= 0
    if window is not None:
        vis = vis & (dist < window)
    ninf = float("-inf")
    s = s.masked_fill(~vis[None], ninf)
    t = q_hat.shape[0]
    big_m = torch.full((hq, t), ninf, dtype=F64)
    big_l = torch.zeros((hq, t), dtype=F64)
    big_o = torch.zeros((hq, t, d), dtype=F64)
    tiles, chunks = k_pos // tile, k_pos // chunk
    for c in torch.unique(chunks).tolist():  # ascending
        m = torch.full((hq, t), ninf, dtype=F64)
        ls = torch.zeros((hq, t), dtype=F64)
        o = torch.zeros((hq, t, d), dtype=F64)
        for tt in torch.unique(tiles[chunks == c]).tolist():
            sel = tiles == tt
            st = s[:, :, sel]
            tm = st.amax(-1)
            live = tm > ninf
            mn = torch.maximum(m, tm)
            a = torch.where(mn == m, torch.ones_like(m), torch.exp2(m - mn))
            p = torch.where(st == ninf, torch.zeros_like(st), torch.exp2(st - mn[..., None]))
            pv = torch.einsum("hts,shd->htd", r.bf16(p), v_rep[sel])
            ls = torch.where(live, ls * a + p.sum(-1), ls)
            o = torch.where(live[..., None], o * a[..., None] + pv, o)
            m = torch.where(live, mn, m)
        take = (big_m == ninf) & (m > ninf)
        both = (big_m > ninf) & (m > ninf)
        mn = torch.maximum(big_m, m)
        sa = torch.where(big_m == mn, torch.ones_like(mn), torch.exp2(big_m - mn))
        sb = torch.where(m == mn, torch.ones_like(mn), torch.exp2(m - mn))
        big_l = torch.where(take, ls, torch.where(both, big_l * sa + ls * sb, big_l))
        big_o = torch.where(take[..., None], o,
                            torch.where(both[..., None], big_o * sa[..., None] + o * sb[..., None], big_o))
        big_m = torch.where(take, m, torch.where(both, mn, big_m))
    lse = (big_m + torch.log2(big_l)) * _LN2  # [Hq, T]
    out = (big_o / big_l[..., None]).permute(1, 0, 2) * torch.sigmoid(gate) * torch.sigmoid(lse.T - sink)[..., None]
    return r.bf16(out)


def route(logits: torch.Tensor, bias: torch.Tensor, k: int, scale: float, r: Rounding):
    """§3.13 on fp32 router logits [T, E]: select by sigmoid(logit) + bias, top-k, ties to the lower expert index;
    weights = scale * sigmoid / sum over the selected sigmoids (fp32). Returns (idx [T, k] in selection order,
    weights [T, k], margin [T] = k-th minus (k+1)-th selection score)."""
    score = torch.sigmoid(logits)
    sel = score + bias
    order = torch.sort(sel, dim=-1, descending=True, stable=True).indices
    idx = order[:, :k]
    s = score.gather(1, idx)
    w = r.f32(s / (s.sum(-1, keepdim=True) + 1e-20) * scale)
    if order.shape[1] > k:
        srt = sel.gather(1, order[:, k - 1:k + 1])
        margin = srt[:, 0] - srt[:, 1]
    else:
        margin = torch.full((sel.shape[0],), float("inf"), dtype=F64)
    return idx, w, margin


def allreduce(partials: Sequence[torch.Tensor], r: Rounding) -> torch.Tensor:
    """§2: bf16 rank partials summed in rank order in fp32, rounded once to bf16."""
    acc = r.f32(partials[0])
    for p in partials[1:]:
        acc = r.f32(acc + p)
    return r.bf16(acc)


# ----------------------------------------------------------------------------------------------------------------------
# cache


@dataclass
class LayerCache:
    k: torch.Tensor  # [S, Hkv, D] bf16 values, positions [k_start, pos)
    v: torch.Tensor
    k_start: int
    conv: dict[str, torch.Tensor]  # 'k', 'v', 'attn', 'mlp' -> [W-1, C] last raw conv inputs


@dataclass
class Cache:
    """Per-sequence decode state (§6): K/V of every layer (local layers keep the last window-1 positions, all a later
    query can see) and the last W-1 raw inputs of the four conv streams of every layer."""

    pos: int
    layers: list[LayerCache]

    def clone(self) -> "Cache":
        return copy.deepcopy(self)


@dataclass
class ForwardOut:
    logits: torch.Tensor  # float32 [T, unpadded_vocab]
    h_final: torch.Tensor  # bf16 [T, H] (drafter feature 48)
    hidden: torch.Tensor | None = None  # bf16 [L+1, T, H]; [i] = partial block P entering layer i, [L] = P after the last
    router_idx: torch.Tensor | None = None  # int32 [L, T, k], ascending selection rank
    router_w: torch.Tensor | None = None  # float32 [L, T, k]
    router_margin: torch.Tensor | None = None  # float32 [L, T]: selection score of the k-th minus the (k+1)-th expert
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------------------------------
# model


class Inkling2Ref:
    def __init__(self, cfg: Inkling2Config, weights: WeightSource, experts: ExpertSource, tp: int = 4,
                 rounding: Rounding | None = None, unembed_chunk: int = 16384, query_chunk: int = 256,
                 emulate: Emulation | None = None):
        if cfg.num_heads % tp or cfg.num_kv_heads % tp:
            raise ValueError(f"tp={tp} does not divide {cfg.num_heads} q / {cfg.num_kv_heads} kv heads")
        self.cfg = cfg
        self.w = weights
        self.experts = experts
        self.tp = tp
        self.r = rounding or Rounding()
        self.em = emulate or Emulation()
        self.unembed_chunk = unembed_chunk
        self.query_chunk = query_chunk
        # stored activations are bf16, fp32-named outputs fp32; the unrounded mode keeps float64 throughout
        self.store_dtype = F64 if self.r.is_exact else torch.bfloat16
        self.out_dtype = F64 if self.r.is_exact else torch.float32
        # when a dict: layer -> {sub-op name -> output} of the last forward (`core`, `attn`, `mlp_in`, `router_logits`,
        # `moe`), for oracle/teacher_force.py
        self.taps: dict | None = None

    def _tap(self, i: int, name: str, x: torch.Tensor) -> None:
        if self.taps is not None:
            self.taps.setdefault(i, {})[name] = x

    # weights ------------------------------------------------------------------------------------------------------
    def w64(self, name: str) -> torch.Tensor:
        return self.w.tensor(name).to(F64)

    def matmul(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """x . W^T of a dense bf16 op (projections, wo partials, router), accumulated in float64 (NUMERICS U3). The one
        hook oracle/teacher_force.py overrides to measure the noise floor of fp32 accumulation."""
        return x @ w.T

    def lin(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """Dense bf16 linear: bf16(x . W^T), float64 accumulate (NUMERICS U3)."""
        return self.r.bf16(self.matmul(x, self.w64(name)))

    def query(self, name: str) -> torch.Tensor:
        """AttnRes pseudo-query pre-scaled once in fp32 by H^-1/2 (NUMERICS U10)."""
        q = self.w.tensor(name).to(torch.float32)
        return self.r.f32(q.to(F64) * self.r.const(self.cfg.hidden_size ** -0.5)).reshape(-1, self.cfg.hidden_size)

    # cache --------------------------------------------------------------------------------------------------------
    def new_cache(self) -> Cache:
        c = self.cfg
        kv = c.num_kv_heads * c.head_dim
        w1 = c.conv_kernel - 1
        z = lambda n: torch.zeros(w1, n, dtype=F64)  # noqa: E731
        layers = [LayerCache(torch.zeros(0, c.num_kv_heads, c.head_dim, dtype=F64),
                             torch.zeros(0, c.num_kv_heads, c.head_dim, dtype=F64), 0,
                             {"k": z(kv), "v": z(kv), "attn": z(c.hidden_size), "mlp": z(c.hidden_size)})
                  for _ in range(c.num_layers)]
        return Cache(0, layers)

    # AttnRes ------------------------------------------------------------------------------------------------------
    def mix(self, blocks: list[torch.Tensor], partial: torch.Tensor, q: torch.Tensor,
            norms: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """§3.2 mix + the following sublayer's RMSNorm. blocks/partial [T, H]; q [Q, H] pre-scaled; one norm per query."""
        x = torch.stack(blocks + [partial])  # [N, T, H]
        rho = torch.rsqrt((x * x).mean(-1) + 1e-6)  # [N, T]
        logits = torch.einsum("nth,qh->qnt", x, q) * rho[None]
        p = torch.softmax(logits, dim=1)
        m = torch.einsum("qnt,nth->qth", p * rho[None], x)
        return [rmsnorm_w(m[j], norms[j], self.cfg.eps, self.r) for j in range(q.shape[0])]

    # attention ----------------------------------------------------------------------------------------------------
    def attention(self, i: int, xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor, pos: torch.Tensor,
                  lc: LayerCache) -> torch.Tensor:
        c, r = self.cfg, self.r
        pre = f"model.llm.layers.{i}.attn."
        t = xq.shape[0]
        hq, hkv, d, dr = c.num_heads, c.num_kv_heads, c.head_dim, c.d_rel
        glob = c.is_global(i)
        q = self.lin(xq, pre + "wq_du.weight").view(t, hq, d)
        rr = self.lin(xq, pre + "wr_du.weight").view(t, hq, dr)
        gate = self.lin(xq, pre + "gate_attn.weight").view(t, hq, d)
        sink = self.lin(xq, pre + "attn_sink_proj.weight")
        k_raw = self.lin(xk, pre + "wk_dv.weight")
        v_raw = self.lin(xv, pre + "wv_dv.weight")
        if glob:
            yk, lc.conv["k"] = dynamic_conv(k_raw, xk, self.w64(pre + "k_sconv.static_w"),
                                            self.w64(pre + "k_sconv.weight_proj"), hkv, lc.conv["k"], r)
            yv, lc.conv["v"] = dynamic_conv(v_raw, xv, self.w64(pre + "v_sconv.static_w"),
                                            self.w64(pre + "v_sconv.weight_proj"), hkv, lc.conv["v"], r)
        else:
            yk, lc.conv["k"] = static_conv(k_raw, self.w64(pre + "k_sconv.weight")[:, 0, :], lc.conv["k"], r)
            yv, lc.conv["v"] = static_conv(v_raw, self.w64(pre + "v_sconv.weight")[:, 0, :], lc.conv["v"], r)
        k_hat = r.bf16(l2clip(yk.view(t, hkv, d), c.qk_clip))
        yv = yv.view(t, hkv, d)
        v_hat = r.bf16(yv * torch.rsqrt((yv * yv).mean(-1, keepdim=True) + c.eps))
        q_hat = r.bf16(l2clip(q * self.w64(pre + "q_l2_clip.scale"), c.qk_clip))
        r_hat = r.bf16(l2clip(rr * self.w64(pre + "rel_l2_clip.scale"), c.rel_clip))
        if glob:
            tau = log_tau(pos, c.log_scaling_n_floor, c.log_scaling_alpha, r)[:, None, None]
            q_hat = r.bf16(q_hat * tau)
            r_hat = r.bf16(r_hat * tau)
        if glob and self.em.kv8:  # every query reads the int8 store, the writing step's own rows included
            k_hat, v_hat = kv8_roundtrip(k_hat), kv8_roundtrip(v_hat)
        lc.k = torch.cat([lc.k, k_hat])
        lc.v = torch.cat([lc.v, v_hat])
        k_pos = torch.arange(lc.k_start, lc.k_start + lc.k.shape[0])
        rel_proj = self.w64(pre + "rel_logits_proj.proj")
        window = None if glob else c.sliding_window
        outs = []
        for a in range(0, t, self.query_chunk):  # bounds the [Hq, T, S] score memory on long prompts
            b = min(a + self.query_chunk, t)
            bias = relative_bias(r_hat[a:b], rel_proj, pos[a:b], k_pos, r)
            args = (q_hat[a:b], lc.k, lc.v, bias, gate[a:b], sink[a:b], pos[a:b], k_pos, window, r)
            if self.em.attn_bf16_p:
                chunk = self.em.chunk_global if glob else self.em.chunk_local
                outs.append(attention_core_tiled(*args, chunk=chunk, tile=self.em.tile))
            else:
                outs.append(attention_core(*args))
        out = torch.cat(outs).reshape(t, hq * d)
        self._tap(i, "core", out)
        if not glob:  # ring: keep what the next query (position pos[-1] + 1) can see
            keep = c.sliding_window - 1
            drop = max(0, lc.k.shape[0] - keep)
            lc.k, lc.v, lc.k_start = lc.k[drop:], lc.v[drop:], lc.k_start + drop
        wo = self.w64(pre + "wo_ud.weight")
        cols = hq * d // self.tp
        partials = [r.bf16(self.matmul(out[:, j * cols:(j + 1) * cols], wo[:, j * cols:(j + 1) * cols]))
                    for j in range(self.tp)]
        return allreduce(partials, r)

    # MoE ----------------------------------------------------------------------------------------------------------
    def router(self, i: int, x: torch.Tensor):
        """§3.13 of layer i on rows x: (fp32 logits [T, E], idx [T, k], weights [T, k], margin [T]) (see `route`)."""
        c, r = self.cfg, self.r
        pre = f"model.llm.layers.{i}.mlp."
        logits = r.f32(self.matmul(x, self.w64(pre + "gate.weight")))  # [T, E]
        return (logits, *route(logits, self.w.tensor(pre + "gate.bias").to(F64), c.top_k, c.route_scale, r))

    @staticmethod
    def expert_jobs(x: torch.Tensor, idx: torch.Tensor):
        """Per expert picked in idx [T, k] (ascending): the job (expert, its rows of x as numpy) and (tokens, slots)."""
        x_np = x.numpy()
        jobs, rows = [], []
        for e in sorted(set(idx.flatten().tolist())):
            tok, slot = torch.nonzero(idx == e, as_tuple=True)
            jobs.append((e, x_np[tok.numpy()]))
            rows.append((tok, slot))
        return jobs, rows

    def combine(self, jobs, rows, outs, w: torch.Tensor, shape) -> torch.Tensor:
        """The expert outputs times their routing weights, summed per owning rank (E mod tp) in float64, each rank's
        partial rounded to bf16, then the all-reduce (§2, NUMERICS U2)."""
        acc = torch.zeros(self.tp, *shape, dtype=F64)
        for (e, _), (tok, slot), y in zip(jobs, rows, outs):
            acc[e % self.tp].index_add_(0, tok, torch.from_numpy(y) * w[tok, slot][:, None])
        return allreduce([self.r.bf16(acc[j]) for j in range(self.tp)], self.r)

    def moe(self, i: int, x: torch.Tensor, rec: dict | None) -> torch.Tensor:
        c = self.cfg
        logits, idx, w, margin = self.router(i, x)
        self._tap(i, "router_logits", logits)
        if rec is not None:
            rec["idx"].append(idx.to(torch.int32))
            rec["w"].append(w.to(self.out_dtype))
            rec["margin"].append(margin.to(self.out_dtype))
        jobs, rows = self.expert_jobs(x, idx)
        outs = self.experts.ffn_many(i, jobs, c.gate_softcap, c.up_softcap, self.r, fp16_in=self.em.moe_fp16)
        return self.combine(jobs, rows, outs, w, x.shape)

    # forward ------------------------------------------------------------------------------------------------------
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        if (ids == IMAGE_PAD_ID).any():
            raise ValueError("image placeholder tokens need the vision tower, which the oracle does not implement")
        e = self.w.rows("model.llm.embed.weight", ids).to(F64)
        return rmsnorm_w(e, self.w64("model.llm.embed_norm.weight"), self.cfg.eps, self.r)

    def unembed(self, h: torch.Tensor) -> torch.Tensor:
        c, r = self.cfg, self.r
        out = []
        for a in range(0, c.unpadded_vocab_size, self.unembed_chunk):
            b = min(a + self.unembed_chunk, c.unpadded_vocab_size)
            z = r.f32(h @ self.w.row_range("model.llm.unembed.weight", a, b).to(F64).T)
            out.append(r.f32(c.final_softcap * torch.tanh(z / c.final_softcap)))
        return torch.cat(out, -1).to(self.out_dtype)

    def layer(self, i: int, p: torch.Tensor, blocks: list[torch.Tensor], pos: torch.Tensor, lc: LayerCache,
              rec: dict | None = None) -> torch.Tensor:
        """§3.2 schedule of layer i: P entering the layer -> P leaving it. `blocks` (the completed blocks) gains P when
        layer i starts a block; `lc` is the layer's cache (updated in place)."""
        c, r = self.cfg, self.r
        ln = f"model.llm.layers.{i}."
        glob = c.is_global(i)
        norms = ([self.w64(ln + f"{n}_input_norm.weight") for n in "qkv"] if glob
                 else [self.w64(ln + "attn_norm.weight")])
        if i == 0:
            xs = [rmsnorm_w(p, w, c.eps, r) for w in norms]
        else:
            xs = self.mix(blocks, p, self.query(f"model.llm.attn_res.attn_queries.{i - 1}"), norms)
        xq, xk, xv = (xs[0], xs[1], xs[2]) if glob else (xs[0], xs[0], xs[0])
        a_raw = self.attention(i, xq, xk, xv, pos, lc)
        a, lc.conv["attn"] = static_conv(a_raw, self.w64(ln + "attn_sconv.weight")[:, 0, :], lc.conv["attn"], r)
        self._tap(i, "attn", a)
        if i % c.layers_per_block == 0:
            blocks.append(p)
            part = a
        else:
            part = r.f32(p + a)  # fp32, not rounded to bf16 (§3.2)
        mlp_in = self.mix(blocks, part, self.query(f"model.llm.attn_res.mlp_queries.{i}"),
                          [self.w64(ln + "mlp_norm.weight")])[0]
        self._tap(i, "mlp_in", mlp_in)
        m_raw = self.moe(i, mlp_in, rec)
        self._tap(i, "moe", m_raw)
        m, lc.conv["mlp"] = static_conv(m_raw, self.w64(ln + "mlp_sconv.weight")[:, 0, :], lc.conv["mlp"], r)
        return r.bf16(r.f32(part + m))

    def forward(self, ids: Sequence[int] | torch.Tensor, cache: Cache | None = None, record: bool = False) -> ForwardOut:
        """Run T new tokens. With `cache`, they continue the cached sequence (the cache is updated in place);
        without, they are a whole sequence from position 0."""
        c = self.cfg
        ids = torch.as_tensor(ids, dtype=torch.long).reshape(-1)
        cache = cache if cache is not None else self.new_cache()
        t = ids.shape[0]
        pos = torch.arange(cache.pos, cache.pos + t)
        rec = {"idx": [], "w": [], "margin": []} if record else None
        hidden = []
        blocks: list[torch.Tensor] = []
        p = self.embed(ids)
        for i in range(c.num_layers):
            if record:
                hidden.append(p)
            p = self.layer(i, p, blocks, pos, cache.layers[i], rec)
        h = self.mix(blocks, p, self.query("model.llm.attn_res.output_query"), [self.w64("model.llm.norm.weight")])[0]
        cache.pos += t
        out = ForwardOut(logits=self.unembed(h), h_final=h.to(self.store_dtype))
        if record:
            hidden.append(p)
            out.hidden = torch.stack(hidden).to(self.store_dtype)
            out.router_idx = torch.stack(rec["idx"])
            out.router_w = torch.stack(rec["w"])
            out.router_margin = torch.stack(rec["margin"])
        return out

    def drafter_features(self, out: ForwardOut) -> torch.Tensor:
        """§4.1: the 8 target features per token (P entering layers 5, 11, ..., 41, then h) as bf16 [T, 8, H]."""
        if out.hidden is None:
            raise ValueError("forward(record=True) is needed for the drafter features")
        ids = self.cfg.drafter_target_layers
        feats = [out.hidden[j] for j in ids[:-1]] + [out.h_final]
        return torch.stack(feats, 1)


def greedy(logits: torch.Tensor) -> torch.Tensor:
    """First-max argmax (NUMERICS U14): the lowest vocabulary index among equal maxima; NaN counts as -inf."""
    x = torch.nan_to_num(logits.to(F64), nan=float("-inf"))
    return torch.argmax(x, dim=-1)


def generate(model: Inkling2Ref, prompt: Sequence[int], n_new: int, record: bool = True) -> dict:
    """Prefill the prompt, then greedy-decode n_new >= 1 tokens one at a time from the cache. `logits` row t is the
    prediction for token t + 1 (len(tokens) - 1 rows)."""
    if n_new < 1:
        raise ValueError("generate needs n_new >= 1")
    cache = model.new_cache()
    outs = [model.forward(prompt, cache, record=record)]
    tokens = list(prompt)
    for _ in range(n_new):
        nxt = int(greedy(outs[-1].logits[-1:])[0])
        tokens.append(nxt)
        if len(tokens) - len(prompt) == n_new:
            break
        outs.append(model.forward([nxt], cache, record=record))
    cat = lambda f: torch.cat([getattr(o, f) for o in outs], dim=1 if f.startswith(("hidden", "router")) else 0)  # noqa: E731
    res = {"tokens": torch.tensor(tokens, dtype=torch.int64), "prompt_len": len(prompt),
           "logits": cat("logits"), "h_final": cat("h_final")}
    if record:
        for f in ("hidden", "router_idx", "router_w", "router_margin"):
            res[f] = cat(f)
    return res
