"""Inkling 2 DSpark drafter reference ("dspark oracle"): the 5-layer block drafter and the `rnn_hidden` Markov head of
docs/PORT-SPEC.md §4 (`mtp.safetensors`), on the CPU, in float64.

The contract is the drafter reference spec shared with the REXL3 port (rounding points, TP rounding, fixture files); it
transcribes vLLM PR #67325 at `vllm-project/vllm@1448919f` (cited below as V:<path>:<lines>, with
`n/` = `vllm/models/inkling2/nvidia/`). Numerics follow docs/NUMERICS.md: every op between two storage points runs in
float64 from the stored values, and values are rounded to bf16 (via fp32) or fp32 exactly where the spec stores them.
Results therefore do not depend on batch size or chunking: incremental commits equal one commit bit for bit.

Flow (one sequence):
  * `commit(state, F[pos0:pos0+n], pos0)`: the context K/V of committed tokens from their 8 target features
    (V:n/dspark.py:250-266 `store_context_kv`, 513-526 `precompute_and_store_context_kv`). The cache is indexed by
    absolute position; a commit below the current length overwrites (rollback) and truncates to pos0 + n.
  * `propose(state, anchor)`: the anchor (the bonus token at p = state.length) plus 7 mask rows through the 5 draft
    layers (V:n/dspark.py:290-319, 423-439), then the Markov head over hd rows 1..7 (V:n/dspark.py:322-385, 540-570;
    V:vllm/models/inkling2/speculator.py:49-77) with the first-max greedy argmax (NUMERICS U14). 7 draft tokens.

Tensor parallelism (`tp`) only changes where the o_proj and down_proj row-parallel partials are rounded: rank r owns
q heads [r*NH/tp, ...) and MLP columns [r*I/tp, ...); each partial is rounded to bf16, summed in rank order in fp32 and
added to the residual with one bf16 rounding; the following norm reads that bf16 residual (V:n/dspark.py:72-116, the
MNNVL fused all-reduce + residual + RMSNorm path).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from inkling2_ref import F64, Rounding, WeightSource, greedy, rmsnorm_w

# ----------------------------------------------------------------------------------------------------------------------
# config


@dataclass(frozen=True)
class DSparkConfig:
    """Pack `mtp_config` values as defaults (P config.json; PORT-SPEC §4)."""

    hidden_size: int = 4608
    num_layers: int = 5
    num_heads: int = 48
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 18432
    local_layer_ids: tuple[int, ...] = (0, 1, 3, 4)
    sliding_window: int = 1024
    block_size: int = 8
    markov_rank: int = 256
    rope_theta: float = 1e6
    eps: float = 1e-6
    target_layer_ids: tuple[int, ...] = (5, 11, 17, 23, 29, 35, 41, 48)
    vocab_size: int = 201024  # padded rows of embed / unembed / markov_w1
    valid_vocab: int = 200058  # logits and argmax run over [0, valid_vocab)
    softcap: float = 30.0

    @property
    def n_taps(self) -> int:
        return len(self.target_layer_ids)

    @property
    def n_drafts(self) -> int:
        return self.block_size - 1

    def is_local(self, layer: int) -> bool:
        return layer in self.local_layer_ids

    @classmethod
    def from_hf(cls, config: dict) -> "DSparkConfig":
        """From the pack's config.json (top level, with `mtp_config` and `text_config`); the same checks as V's
        `Inkling2DSparkConfig.from_target` (V:vllm/models/inkling2/configs.py:158-192)."""
        m = config.get("mtp_config")
        if not m:
            raise ValueError("config.json has no mtp_config")
        t = config.get("text_config", {})
        if m.get("markov_head_type") != "rnn_hidden" or m.get("q_bias") or m.get("o_bias"):
            raise ValueError("the oracle implements the rnn_hidden Markov head without q/o bias only")
        taps = tuple(m["target_layer_ids"])
        if list(taps) != sorted(set(taps)) or len(taps) < 2 or ("num_hidden_layers" in t
                                                                and taps[-1] != t["num_hidden_layers"]):
            raise ValueError(f"target_layer_ids {taps} must be sorted ids ending with the target's layer count")
        if "hidden_size" in t and t["hidden_size"] != m["hidden_size"]:
            raise ValueError("mtp_config.hidden_size differs from the target's")
        return cls(
            hidden_size=m["hidden_size"], num_layers=m["num_hidden_layers"], num_heads=m["num_attention_heads"],
            num_kv_heads=m["num_key_value_heads"], head_dim=m["head_dim"], intermediate_size=m["intermediate_size"],
            local_layer_ids=tuple(m["local_layer_ids"]), sliding_window=m["sliding_window_size"],
            block_size=m["block_size"], markov_rank=m["markov_rank"], rope_theta=float(m["rope_theta"]),
            eps=m["rms_norm_eps"], target_layer_ids=taps, vocab_size=t.get("vocab_size", cls.vocab_size),
            valid_vocab=t.get("unpadded_vocab_size", cls.valid_vocab),
            softcap=float(t.get("final_logit_softcapping", cls.softcap)),
        )


# ----------------------------------------------------------------------------------------------------------------------
# state


@dataclass
class DSparkState:
    """Per-sequence drafter state: per draft layer the context K (after k_norm_ctx + RoPE) and V of every committed
    position, indexed by absolute position (rows >= `length` are stale and never read), and the feature row of the last
    committed token (the f_rnn source)."""

    k: list[torch.Tensor]  # [capacity, NKV, D] float64 holding bf16 values
    v: list[torch.Tensor]
    length: int = 0
    last_row: torch.Tensor | None = None  # [taps, H] F[length - 1]

    def ctx_kv(self) -> torch.Tensor:
        """[L, length, 2, NKV, D]: ctx k and v per layer and position (the fixtures' `ctxkv`)."""
        n = self.length
        return torch.stack([torch.stack([k[:n], v[:n]], 1) for k, v in zip(self.k, self.v)])


@dataclass
class Fold:
    """`hidden_out` folded into the unembed rows [0, valid_vocab) (V:n/dspark.py:348-360 `fold`)."""

    weight: torch.Tensor  # [valid, R] fold_W = bf16(U · Wout)
    bias: torch.Tensor  # [valid] fold_b = bf16(U · bout)


@dataclass
class Proposal:
    drafts: list[int]
    trace: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------------------------------
# rope


def rope_table(pos: torch.Tensor, head_dim: int, theta: float, r: Rounding) -> tuple[torch.Tensor, torch.Tensor]:
    """V's `get_rope` fp32 cos/sin cache (V:n/dspark.py:408-414, rope_type default, neox): inv_freq[i] =
    f32(1 / f32(theta ** (2i / D))) with the pow in float64; ang = f32(pos * inv_freq); cos/sin in float64 of that fp32
    angle, rounded to fp32. Returns (cos, sin), each [len(pos), D/2]."""
    i = torch.arange(0, head_dim, 2, dtype=F64)
    inv = r.f32(1.0 / r.f32(torch.pow(torch.tensor(theta, dtype=F64), i / head_dim)))
    ang = r.f32(pos.to(F64)[:, None] * inv[None, :])
    return r.f32(torch.cos(ang)), r.f32(torch.sin(ang))


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, r: Rounding) -> torch.Tensor:
    """Neox RoPE of x [T, heads, D] (bf16 values) at the rows' positions (cos/sin [T, D/2]): o1 = bf16(x1 c - x2 s),
    o2 = bf16(x2 c + x1 s), float64 math, one rounding (V:n/ops/dspark.py:70-165 `_qk_norm_rope_kernel`, `rope_`)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c, s = cos[:, None, :], sin[:, None, :]
    return torch.cat([r.bf16(x1 * c - x2 * s), r.bf16(x2 * c + x1 * s)], -1)


# ----------------------------------------------------------------------------------------------------------------------
# model


class DSparkRef:
    """`weights` serves the unprefixed `mtp.safetensors` names plus the target's `model.llm.embed.weight`,
    `model.llm.embed_norm.weight` and `model.llm.unembed.weight` (row access for embed / unembed / markov_w1)."""

    EMBED = "model.llm.embed.weight"
    EMBED_NORM = "model.llm.embed_norm.weight"
    UNEMBED = "model.llm.unembed.weight"

    def __init__(self, cfg: DSparkConfig, weights: WeightSource, tp: int = 4, rounding: Rounding | None = None,
                 unembed_chunk: int = 16384):
        for what, n in (("q heads", cfg.num_heads), ("kv heads", cfg.num_kv_heads),
                        ("MLP columns", cfg.intermediate_size)):
            if n % tp:
                raise ValueError(f"tp={tp} does not divide the {n} {what}")
        self.cfg = cfg
        self.w = weights
        self.tp = tp
        self.r = rounding or Rounding()
        self.unembed_chunk = unembed_chunk
        self.store_dtype = F64 if self.r.is_exact else torch.bfloat16
        self.out_dtype = F64 if self.r.is_exact else torch.float32
        # fuse_alpha = f32(softmax(fuse_weight)) row-wise; row l < L -> draft layer l, row L -> Markov head
        # (V:n/dspark.py:502-504 attach_target)
        self.alpha = self.r.f32(torch.softmax(self.w64("fuse_weight"), dim=-1))
        self._fold: Fold | None = None

    def with_tp(self, tp: int) -> "DSparkRef":
        """The same weights, rounding and (already computed) fold at another tp; the fold does not depend on tp."""
        other = DSparkRef(self.cfg, self.w, tp, self.r, self.unembed_chunk)
        other._fold = self._fold
        return other

    # weights ------------------------------------------------------------------------------------------------------
    def w64(self, name: str) -> torch.Tensor:
        return self.w.tensor(name).to(F64)

    def lin(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """bf16(x · W^T), float64 accumulate (NUMERICS U3)."""
        return self.r.bf16(x @ self.w64(name).T)

    def norm(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return rmsnorm_w(x, self.w64(name), self.cfg.eps, self.r)

    # ops ----------------------------------------------------------------------------------------------------------
    def mix(self, feats: torch.Tensor, row: int) -> torch.Tensor:
        """feats [n, taps, H] -> rmsnorm_w(Σ_i alpha[row, i] F[:, i], dflare_norm) [n, H], Σ in float64
        (V:n/ops/dspark.py:13-67 `mix_rmsnorm`; V:n/dspark.py:419-421)."""
        name = (f"layers.{row}.dflare_norm.weight" if row < self.cfg.num_layers
                else "markov_head.dflare_norm.weight")
        return self.norm(torch.einsum("nth,t->nh", feats, self.alpha[row]), name)

    def rope_at(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return rope_table(pos, self.cfg.head_dim, self.cfg.rope_theta, self.r)

    def add_norm(self, h: torch.Tensor, partials: Sequence[torch.Tensor], name: str):
        """Fused all-reduce + residual add + RMSNorm: acc = rank-order fp32 sum of the bf16 partials;
        h' = bf16(h + acc); x = rmsnorm_w(h', w) (V:n/dspark.py:72-116). Returns (h', x)."""
        r = self.r
        acc = r.f32(partials[0])
        for p in partials[1:]:
            acc = r.f32(acc + p)
        h = r.bf16(h + acc)
        return h, self.norm(h, name)

    def row_parallel(self, x: torch.Tensor, name: str) -> list[torch.Tensor]:
        """Per-rank bf16 partials of a RowParallelLinear: rank j multiplies its input-column slice."""
        w = self.w64(name)
        cols = w.shape[1] // self.tp
        return [self.r.bf16(x[:, j * cols:(j + 1) * cols] @ w[:, j * cols:(j + 1) * cols].T) for j in range(self.tp)]

    # context ------------------------------------------------------------------------------------------------------
    def new_state(self) -> DSparkState:
        c = self.cfg
        z = lambda: torch.zeros(0, c.num_kv_heads, c.head_dim, dtype=F64)  # noqa: E731
        return DSparkState([z() for _ in range(c.num_layers)], [z() for _ in range(c.num_layers)])

    def context_kv(self, layer: int, feats: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Context K/V of committed rows feats [n, taps, H] at positions pos (V:n/dspark.py:250-266):
        ctx = mix_l(F); k = rope(rmsnorm_w(bf16(ctx Wk_ctx^T) per kv head, k_norm_ctx)); v = bf16(ctx Wv_ctx^T)."""
        c = self.cfg
        a = f"layers.{layer}.self_attn."
        n = feats.shape[0]
        ctx = self.mix(feats, layer)
        k = self.lin(ctx, a + "k_proj_ctx.weight").view(n, c.num_kv_heads, c.head_dim)
        v = self.lin(ctx, a + "v_proj_ctx.weight").view(n, c.num_kv_heads, c.head_dim)
        k = rope(self.norm(k, a + "k_norm_ctx.weight"), *self.rope_at(pos), self.r)
        return k, v

    def commit(self, state: DSparkState, feats: torch.Tensor, pos0: int) -> None:
        """Commit the tokens at positions pos0 .. pos0+n-1 from their target features feats [n, taps, H] (bf16 values):
        writes their context K/V, sets length = pos0 + n and the f_rnn source to the last row. pos0 < length rolls back
        (the rows are rewritten, later rows become stale)."""
        c = self.cfg
        feats = torch.as_tensor(feats).to(F64)
        n = feats.shape[0]
        if n < 1 or feats.shape[1:] != (c.n_taps, c.hidden_size):
            raise ValueError(f"feats must be [n >= 1, {c.n_taps}, {c.hidden_size}], got {tuple(feats.shape)}")
        if not 0 <= pos0 <= state.length:
            raise ValueError(f"commit at {pos0} leaves a gap after length {state.length}")
        end = pos0 + n
        pos = torch.arange(pos0, end)
        for layer in range(c.num_layers):
            if state.k[layer].shape[0] < end:
                grow = max(end, 2 * state.k[layer].shape[0]) - state.k[layer].shape[0]
                pad = torch.zeros(grow, c.num_kv_heads, c.head_dim, dtype=F64)
                state.k[layer] = torch.cat([state.k[layer], pad])
                state.v[layer] = torch.cat([state.v[layer], pad])
            k, v = self.context_kv(layer, feats, pos)
            state.k[layer][pos0:end] = k
            state.v[layer][pos0:end] = v
        state.length = end
        state.last_row = feats[-1].clone()

    # block --------------------------------------------------------------------------------------------------------
    def attention(self, layer: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, state: DSparkState,
                  p: int) -> list[torch.Tensor]:
        """Block attention of layer l (V:n/dspark.py:174, 195-248; block_decode_attention
        V:n/ops/attention.py:1153-1188, mask V:n/ops/attention.py:742, 779-786, 913-914): q/k/v = bf16(x W^T);
        q̂, k̂ = rope(rmsnorm_w(·, q/k_norm));
        every block row sees the ctx keys at [max(0, p - W), p) (local) or [0, p) (global) and all 8 block keys
        (non-causal); exact float64 softmax, scale D^-1/2; out = bf16(P V). Returns the per-rank o_proj partials."""
        c, r = self.cfg, self.r
        a = f"layers.{layer}.self_attn."
        b = x.shape[0]
        q = self.lin(x, a + "q_proj.weight").view(b, c.num_heads, c.head_dim)
        k = self.lin(x, a + "k_proj.weight").view(b, c.num_kv_heads, c.head_dim)
        v = self.lin(x, a + "v_proj.weight").view(b, c.num_kv_heads, c.head_dim)
        q = rope(self.norm(q, a + "q_norm.weight"), cos, sin, r)
        k = rope(self.norm(k, a + "k_norm.weight"), cos, sin, r)
        lo = max(0, p - c.sliding_window) if c.is_local(layer) else 0
        keys = torch.cat([state.k[layer][lo:p], k])
        vals = torch.cat([state.v[layer][lo:p], v])
        grp = c.num_heads // c.num_kv_heads
        s = torch.einsum("bhd,shd->hbs", q, keys.repeat_interleave(grp, 1)) * c.head_dim ** -0.5
        o = torch.einsum("hbs,shd->bhd", torch.softmax(s, -1), vals.repeat_interleave(grp, 1))
        return self.row_parallel(r.bf16(o).reshape(b, c.num_heads * c.head_dim), a + "o_proj.weight")

    def mlp(self, layer: int, x: torch.Tensor) -> list[torch.Tensor]:
        """g, u = bf16(x Wg^T), bf16(x Wu^T); a = bf16(g / (1 + exp(-g)) · u) (V:n/ops/dspark.py:168-189);
        per-rank bf16 down partials."""
        m = f"layers.{layer}.mlp."
        g = self.lin(x, m + "gate_proj.weight")
        u = self.lin(x, m + "up_proj.weight")
        act = self.r.bf16(g / (1.0 + torch.exp(-g)) * u)
        return self.row_parallel(act, m + "down_proj.weight")

    def block(self, state: DSparkState, anchor: int) -> torch.Tensor:
        """hd [B, H]: the final-norm output of the block [anchor, mask × B-1] at positions p .. p+B-1
        (V:n/dspark.py:290-319 Inkling2DraftLayer, 423-439 embed_block / forward)."""
        c = self.cfg
        p = state.length
        b = c.block_size
        cos, sin = self.rope_at(torch.arange(p, p + b))
        e = self.w.rows(self.EMBED, torch.tensor([int(anchor)])).to(F64)
        x0 = rmsnorm_w(e, self.w64(self.EMBED_NORM), c.eps, self.r)
        h = torch.cat([x0, self.w64("mask_embed")[None].expand(b - 1, -1)])
        x = self.norm(h, "layers.0.input_layernorm.weight")
        mlp_parts = None
        for layer in range(c.num_layers):
            ln = f"layers.{layer}."
            if layer > 0:
                h, x = self.add_norm(h, mlp_parts, ln + "input_layernorm.weight")
            attn_parts = self.attention(layer, x, cos, sin, state, p)
            h, x = self.add_norm(h, attn_parts, ln + "post_attention_layernorm.weight")
            mlp_parts = self.mlp(layer, x)
        return self.add_norm(h, mlp_parts, "norm.weight")[1]

    # Markov head --------------------------------------------------------------------------------------------------
    def unembed_chunks(self):
        v = self.cfg.valid_vocab
        for a in range(0, v, self.unembed_chunk):
            b = min(a + self.unembed_chunk, v)
            yield a, b, self.w.row_range(self.UNEMBED, a, b).to(F64)

    def fold(self) -> Fold:
        """fold_W[v] = bf16(U[v] · Wout), fold_b[v] = bf16(U[v] · bout) for v < valid_vocab, float64, computed once
        streaming the unembed in row chunks (V:n/dspark.py:348-360)."""
        if self._fold is None:
            wout = self.w64("markov_head.hidden_out.weight")  # [H, R]
            bout = self.w64("markov_head.hidden_out.bias")
            fw, fb = [], []
            for _, _, u in self.unembed_chunks():
                fw.append(self.r.bf16(u @ wout))
                fb.append(self.r.bf16(u @ bout))
            self._fold = Fold(torch.cat(fw), torch.cat(fb))
        return self._fold

    def base_logits(self, hd: torch.Tensor) -> torch.Tensor:
        """base[i, v] = bf16(hd_i · U[v]) for v < valid_vocab (bf16 GEMM, bf16 output; V:n/dspark.py:540-543)."""
        return torch.cat([self.r.bf16(hd @ u.T) for _, _, u in self.unembed_chunks()], -1)

    def markov_bias(self, latent: torch.Tensor) -> torch.Tensor:
        """bias[v] = bf16(fold_b[v] + latent · fold_W[v]) (V:n/dspark.py:375-385 `step`)."""
        f = self.fold()
        return self.r.bf16(f.bias + f.weight @ latent)

    def markov_context(self, hd: torch.Tensor, f_rnn: torch.Tensor) -> torch.Tensor:
        """ctx_i = f32(f32(hd_i · Wrnn[:, 2R:2R+H]^T + f_rnn · Wrnn[:, 2R+H:]^T) + brnn) [n, 3R]
        (cuBLAS addmm with an fp32 output: fp32 product, then the fp32 bias; V:n/dspark.py:362-373 `context`)."""
        c, r = self.cfg, self.r
        w = self.w64("markov_head.rnn_proj.weight")
        rk, h = c.markov_rank, c.hidden_size
        prod = hd @ w[:, 2 * rk:2 * rk + h].T + f_rnn @ w[:, 2 * rk + h:].T
        return r.f32(r.f32(prod) + self.w64("markov_head.rnn_proj.bias"))

    def markov_gate(self, raw: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """V:n/ops/dspark.py:217-257 `_markov_gate_kernel`: g = bf16(σ(bf16 raw_g)), c = bf16(tanh(bf16 raw_c)),
        o = bf16(raw_o); state' = bf16(bf16(g·state) + bf16(bf16(1-g)·c)); latent = bf16(tanh(o))."""
        r, rk = self.r, self.cfg.markov_rank
        g = r.bf16(torch.sigmoid(r.bf16(raw[:rk])))
        cand = r.bf16(torch.tanh(r.bf16(raw[rk:2 * rk])))
        o = r.bf16(raw[2 * rk:])
        new = r.bf16(r.bf16(g * state) + r.bf16(r.bf16(1.0 - g) * cand))
        return new, r.bf16(torch.tanh(o))

    def softcap_logits(self, base: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """bf16(cap · tanh(bf16(base + bias) / cap)) (V:n/ops/dspark.py:191-215 `add_softcap`)."""
        r, cap = self.r, self.cfg.softcap
        return r.bf16(cap * torch.tanh(r.bf16(base + bias) / cap))

    # proposal -----------------------------------------------------------------------------------------------------
    def propose(self, state: DSparkState, anchor: int, force_prev: Sequence[int] | None = None,
                trace: bool = False) -> Proposal:
        """7 greedy drafts for the anchor token at p = state.length (context [0, p) committed, p >= 1).
        f_rnn = mix_L(F[p-1]) (V:vllm/models/inkling2/speculator.py:49-58); the Markov loop starts from state 0 and
        prev = anchor (V:vllm/models/inkling2/speculator.py:60-77). `force_prev` (7 ids) teacher-forces the token fed
        to the next step (the drafts are still the argmaxes). With `trace`: hd [B, H], f_rnn [H], per-step state,
        latent [7, R] and logits [7, valid_vocab] (bf16 values; float64 in exact mode)."""
        c, r = self.cfg, self.r
        if state.length < 1 or state.last_row is None:
            raise ValueError("propose needs at least one committed position (f_rnn reads the last committed row)")
        nd = c.n_drafts
        if force_prev is not None and len(force_prev) != nd:
            raise ValueError(f"force_prev needs {nd} ids")
        hd = self.block(state, anchor)
        f_rnn = self.mix(state.last_row[None], c.num_layers)[0]
        hid = hd[1:]
        ctx = self.markov_context(hid, f_rnn[None])
        base = self.base_logits(hid)
        w1 = "markov_head.markov_w1.weight"
        wsp = self.w64("markov_head.rnn_proj.weight")[:, :2 * c.markov_rank]
        st = torch.zeros(c.markov_rank, dtype=F64)
        prev = int(anchor)
        drafts: list[int] = []
        states, latents, logits = [], [], []
        for i in range(nd):
            inp = torch.cat([st, self.w.rows(w1, torch.tensor([prev]))[0].to(F64)])
            raw = r.f32(r.f32(wsp @ inp) + ctx[i])
            st, latent = self.markov_gate(raw, st)
            lg = self.softcap_logits(base[i], self.markov_bias(latent))
            d = int(greedy(lg[None])[0])
            drafts.append(d)
            prev = d if force_prev is None else int(force_prev[i])
            if trace:
                states.append(st)
                latents.append(latent)
                logits.append(lg)
        tr = {}
        if trace:
            sd, od = self.store_dtype, self.out_dtype
            tr = {"hd": hd.to(sd), "f_rnn": f_rnn.to(sd), "state": torch.stack(states).to(sd),
                  "latent": torch.stack(latents).to(sd), "logits": torch.stack(logits).to(od)}
        return Proposal(drafts, tr)


def bf16_bits(x: torch.Tensor) -> np.ndarray:
    """bf16 values (any float dtype) -> their uint16 bit patterns (the fixtures' storage)."""
    return x.to(torch.float32).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
