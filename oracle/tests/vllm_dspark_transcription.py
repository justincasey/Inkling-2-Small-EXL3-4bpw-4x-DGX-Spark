"""Independent float64 transcription of the vLLM Inkling 2 DSpark drafter module flow (vllm-project/vllm@1448919f,
`vllm/models/inkling2/nvidia/dspark.py`, `ops/dspark.py`, `ops/attention.py` block_decode_attention,
`vllm/models/inkling2/speculator.py`), without rounding, for checking the op structure of `dspark_ref.DSparkRef` in
`Rounding.exact()` mode. It shares no code with dspark_ref and keeps V's structure:

  * stacked checkpoint weights as V's `load_weights` builds them (`qkv_proj`, `kv_proj_ctx`, `gate_up_proj`);
  * a `get_rope`-style cos/sin cache table [max_pos, D] = [cos | sin];
  * one position-indexed K/V cache per draft layer (V's paged cache with slot = position): `store_context_kv` writes
    committed tokens' ctx K/V, the block forward writes its own K/V at p .. p+7 and then runs block_decode_attention
    with cache_seqlens = p + 8: every query at position seqlen - 1, visible keys 0 <= dist < window, window =
    sliding_window + block on local layers, all keys on the global layer;
  * the residual stream of `Inkling2DraftLayer.forward` (residual = None on layer 0, then fused add + norm);
  * `Inkling2MarkovHead.fold/context/step`, `markov_gate`, `add_softcap` and the `_sample_sequential` loop over the
    mask rows 1..7 (the DFlash 1 + N layout, `sample_from_anchor = False`).
"""

from __future__ import annotations

import torch

F64 = torch.float64


class VDSpark:
    def __init__(self, cfg, weights: dict[str, torch.Tensor], max_pos: int):
        self.c = cfg
        w = {k: v.to(F64) for k, v in weights.items()}
        self.w = w
        L = cfg.num_layers
        self.qkv = [torch.cat([w[f"layers.{i}.self_attn.{n}_proj.weight"] for n in "qkv"]) for i in range(L)]
        self.kv_ctx = [torch.cat([w[f"layers.{i}.self_attn.{n}_proj_ctx.weight"] for n in "kv"]) for i in range(L)]
        self.gate_up = [torch.cat([w[f"layers.{i}.mlp.{n}_proj.weight"] for n in ("gate", "up")]) for i in range(L)]
        self.fuse_alpha = torch.softmax(w["fuse_weight"], dim=-1)
        # get_rope(head_dim, max_position, rope_theta) cos_sin_cache, in float64
        d = cfg.head_dim
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=F64) / d))
        freqs = torch.einsum("i,j->ij", torch.arange(max_pos + cfg.block_size, dtype=F64), inv_freq)
        self.cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
        self.window = [cfg.sliding_window + cfg.block_size if i in cfg.local_layer_ids else 0 for i in range(L)]
        slots = max_pos + cfg.block_size
        self.k_cache = [torch.zeros(slots, cfg.num_kv_heads, d, dtype=F64) for _ in range(L)]
        self.v_cache = [torch.zeros(slots, cfg.num_kv_heads, d, dtype=F64) for _ in range(L)]
        # attach_target: fold hidden_out into the LM head, fp32 rnn bias
        lm = w["model.llm.unembed.weight"]
        self.fold_weight = lm @ w["markov_head.hidden_out.weight"]
        self.fold_bias = lm @ w["markov_head.hidden_out.bias"]
        self.rnn_bias = w["markov_head.rnn_proj.bias"]
        self.f_rnn = None

    # ops/dspark.py ------------------------------------------------------------------------------------------------
    def rmsnorm(self, x, weight):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.c.eps) * weight

    def mix_rmsnorm(self, targets, alpha, weight):
        mixed = (targets * alpha[None, :, None]).sum(1)
        return self.rmsnorm(mixed, weight)

    def _rope_rows(self, x, positions):
        """x [T, heads, D] -> neox rotation by the cache rows of `positions`."""
        half = self.c.head_dim // 2
        cs = self.cos_sin_cache[positions]
        cos, sin = cs[:, None, :half], cs[:, None, half:]
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)

    def qk_norm_rope_(self, qkv, positions, q_w, k_w):
        c = self.c
        t = qkv.shape[0]
        nq = c.num_heads * c.head_dim
        nk = c.num_kv_heads * c.head_dim
        q = qkv[:, :nq].reshape(t, c.num_heads, c.head_dim)
        k = qkv[:, nq:nq + nk].reshape(t, c.num_kv_heads, c.head_dim)
        qkv[:, :nq] = self._rope_rows(self.rmsnorm(q, q_w), positions).reshape(t, nq)
        qkv[:, nq:nq + nk] = self._rope_rows(self.rmsnorm(k, k_w), positions).reshape(t, nk)

    @staticmethod
    def silu_and_mul(x):
        n = x.shape[-1] // 2
        gate, up = x[:, :n], x[:, n:]
        return gate / (1.0 + torch.exp(-gate)) * up

    @staticmethod
    def add_softcap(a, b, softcap):
        return softcap * torch.tanh((a + b) / softcap)

    def markov_gate(self, raw, state):
        r = self.c.markov_rank
        gate = 1.0 / (1.0 + torch.exp(-raw[:, :r]))
        cand = torch.tanh(raw[:, r:2 * r])
        out = raw[:, 2 * r:]
        return gate * state + (1.0 - gate) * cand, torch.tanh(out)

    # ops/attention.py block_decode_attention (draft mode) ---------------------------------------------------------
    def block_decode_attention(self, i, q, seqlen_k):
        c = self.c
        pos0 = seqlen_k - 1  # DRAFT: every query sits at the block's last key
        keys = torch.arange(seqlen_k)
        dist = pos0 - keys
        visible = dist >= 0
        if self.window[i] > 0:
            visible &= dist < self.window[i]
        g = c.num_heads // c.num_kv_heads
        k = self.k_cache[i][:seqlen_k].repeat_interleave(g, 1)
        v = self.v_cache[i][:seqlen_k].repeat_interleave(g, 1)
        s = torch.einsum("thd,shd->ths", q, k) * c.head_dim ** -0.5
        s = s.masked_fill(~visible[None, None, :], float("-inf"))
        return torch.einsum("ths,shd->thd", torch.softmax(s, -1), v).reshape(q.shape[0], -1)

    # dspark.py ----------------------------------------------------------------------------------------------------
    def store_context_kv(self, i, context, positions):
        c = self.c
        kv = context @ self.kv_ctx[i].T
        nk = c.num_kv_heads * c.head_dim
        k, v = kv[:, :nk], kv[:, nk:]
        k = self.rmsnorm(k.reshape(-1, c.head_dim), self.w[f"layers.{i}.self_attn.k_norm_ctx.weight"])
        k = self._rope_rows(k.view(-1, c.num_kv_heads, c.head_dim), positions)
        self.k_cache[i][positions] = k
        self.v_cache[i][positions] = v.view(-1, c.num_kv_heads, c.head_dim)

    def precompute_and_store_context_kv(self, context_states, context_positions):
        c = self.c
        targets = context_states.view(context_states.shape[0], -1, c.hidden_size)
        for i in range(c.num_layers):
            context = self.mix_rmsnorm(targets, self.fuse_alpha[i], self.w[f"layers.{i}.dflare_norm.weight"])
            self.store_context_kv(i, context, context_positions)

    def prepare_context_anchor(self, last_committed_row):
        targets = last_committed_row.view(1, -1, self.c.hidden_size)
        self.f_rnn = self.mix_rmsnorm(targets, self.fuse_alpha[-1], self.w["markov_head.dflare_norm.weight"])

    def self_attn(self, i, positions, x):
        c = self.c
        a = f"layers.{i}.self_attn."
        qkv = x @ self.qkv[i].T
        self.qk_norm_rope_(qkv, positions, self.w[a + "q_norm.weight"], self.w[a + "k_norm.weight"])
        nq, nk = c.num_heads * c.head_dim, c.num_kv_heads * c.head_dim
        q, k, v = qkv[:, :nq], qkv[:, nq:nq + nk], qkv[:, nq + nk:]
        self.k_cache[i][positions] = k.view(-1, c.num_kv_heads, c.head_dim)  # _write_kv at the block's slots
        self.v_cache[i][positions] = v.view(-1, c.num_kv_heads, c.head_dim)
        out = self.block_decode_attention(i, q.view(-1, c.num_heads, c.head_dim), int(positions[-1]) + 1)
        return out @ self.w[a + "o_proj.weight"].T

    def forward(self, input_ids, positions):
        c, w = self.c, self.w
        x = w["mask_embed"].expand(c.block_size, -1).clone()
        x[0] = self.rmsnorm(w["model.llm.embed.weight"][input_ids[0]], w["model.llm.embed_norm.weight"])
        residual = None
        for i in range(c.num_layers):
            ln = f"layers.{i}."
            if residual is None:
                residual = x
                x = self.rmsnorm(x, w[ln + "input_layernorm.weight"])
            else:
                residual = residual + x
                x = self.rmsnorm(residual, w[ln + "input_layernorm.weight"])
            x = self.self_attn(i, positions, x)
            residual = residual + x
            x = self.rmsnorm(residual, w[ln + "post_attention_layernorm.weight"])
            x = self.silu_and_mul(x @ self.gate_up[i].T) @ w[ln + "mlp.down_proj.weight"].T
        residual = residual + x
        return self.rmsnorm(residual, w["norm.weight"])

    def context(self, hidden):
        r = self.c.markov_rank
        inputs = torch.cat([hidden, self.f_rnn.expand_as(hidden)], dim=-1)
        return self.rnn_bias + inputs @ self.w["markov_head.rnn_proj.weight"][:, 2 * r:].T

    def step(self, state, prev, context):
        r = self.c.markov_rank
        inputs = torch.cat([state, self.w["markov_head.markov_w1.weight"][prev][None]], dim=-1)
        raw = context[None] + inputs @ self.w["markov_head.rnn_proj.weight"][:, :2 * r].T
        state, latent = self.markov_gate(raw, state)
        return state, self.fold_bias + latent @ self.fold_weight.T

    def draft(self, anchor: int, p: int):
        """One proposal at block start p; returns (drafts, hd [B, H], logits [7, valid])."""
        c = self.c
        ids = torch.tensor([anchor] + [0] * (c.block_size - 1))  # mask_token_id 0; only ids[0] is read
        head_hidden = self.forward(ids, torch.arange(p, p + c.block_size))
        hidden = head_hidden[1:]  # sample_indices: the mask rows
        base_logits = hidden @ self.w["model.llm.unembed.weight"].T
        context = self.context(hidden)
        state = torch.zeros(1, c.markov_rank, dtype=F64)
        prev = anchor
        drafts, logits = [], []
        for i in range(c.block_size - 1):
            state, bias = self.step(state, prev, context[i])
            lg = self.add_softcap(base_logits[i], bias[0], c.softcap)[: c.valid_vocab]
            prev = int(torch.argmax(lg))
            drafts.append(prev)
            logits.append(lg)
        return drafts, head_hidden, torch.stack(logits)
