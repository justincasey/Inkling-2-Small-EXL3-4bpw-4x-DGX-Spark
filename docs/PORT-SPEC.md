# Inkling 2 Small → REXL3: operation-by-operation port spec

This spec defines the forward pass of the target model, the DSpark drafter and the vision tower. It maps each op to the
tensors in pack `0xSero/Inkling-2-Small-EXL3-4bpw@996dc404`, gives TP4 shapes and the rounding points of the reference
implementations, and inventories what REXL3 already has. The byte-exact reference we build will implement this
document. Where the references disagree, the disagreement is listed under UNKNOWN (§10).

## 0. Sources and citation keys

| Key | Source | Status / use |
|---|---|---|
| **V** | `vllm-project/vllm@1448919ffc267a321c65b32bd51b334075d9a8f0`: head of PR #67325 "[Model] Add Inkling 2 Small" (**OPEN**, not merged) | Primary numerics reference. It writes down every rounding point, and its kernels claim to match "the reference implementation". Paths under `vllm/models/inkling2/nvidia/` are shortened to `n/`. |
| **HF** | `huggingface/transformers@fc00b5b3c9f9306c55908bb546fa3b033a4bfb1d` `src/transformers/models/inkling2/modeling_inkling2.py` (on `main`) | Eager reference. Used to confirm semantics, and the second opinion where rounding differs. |
| **SG** | `sgl-project/sglang` PR #50186 @ `ea3742849fa4f32a5a136f0792f667226d345f25` (OPEN). It contains `python/sglang/kernels/ops/inkling2/*` and `jit/csrc/inkling2/{sconv,prologue,gate_sink,route}.cuh` | Exists. **Not read in depth.** Third opinion for the UNKNOWNs. |
| **OX** | `0xSero/inkling2-exl3@d98291637e55c8573744f121245b0d6f0ba586e8` (exllamav3 overlay, `overlay/exllamav3/modules/inkling2*.py`; `FORK_COMMIT` 744aa465) | Produced this pack. Numerics reference only; its code is **not** copyable and was not read in depth. |
| **P** | Pack `config.json`, `intake/census.json`, `mtp.safetensors` header (read by HTTP Range at `996dc404`) | Tensor names, shapes and dtypes. |
| **R** | REXL3 `upstream/main@06181b1` (v0.1.2), the read-only worktree `/Users/justin/projects/rexl3-inkling2` | Reuse inventory (§9). |

Citation form: `V:n/model.py:258-435` means repo V, path `vllm/models/inkling2/nvidia/model.py`, lines 258-435.

## 1. Global facts that change the plan

1. **The target text model has no RoPE.** Position enters only three ways: (a) a learned, input-conditioned relative-distance bias (§3.9), (b) log-scaling of q and r on the 8 global layers (§3.10), and (c) the causal/window masks. Nothing in V:n/attention.py:107-305 or HF:401-629 rotates anything. The **drafter does use neox RoPE** (θ = 1e6) (V:n/dspark.py:195-266, 409-414; P `mtp_config.rope_theta`). The **vision tower uses 2D axial RoPE** (θ = 1e4) (V:n/ops/vision.py:18-22, 240-276).
2. **The residual stream is not additive across layers.** It is a "block attention residual" (AttnRes). Every sublayer input is a softmax-weighted mix of RMS-normalized completed 6-layer blocks plus the running partial block, using a learned fp32 pseudo-query per input (§3.2). Global layers have **three** attention-input queries (q, k, v), each with its own RMSNorm (`q/k/v_input_norm`). So on global layers **K and V are projected from different inputs than Q**.
3. **Each layer has four short convolutions, not two.** (a) `k_sconv` and `v_sconv` act on the K/V projections before clip/norm. (b) `attn_sconv` acts on the all-reduced attention output and (c) `mlp_sconv` on the all-reduced MoE output. Each output conv is applied to the sublayer output **before** it joins the residual. All four carry a 3-token causal state (§3.4, §6).
4. **Relative logits need no per-key cache.** The bias for query t and key s is `(r̂_t · rel_proj[:, t−s])` for 0 ≤ t−s < 1024, else 0 (global) or masked (local). It depends only on the query's `r̂_t` and the distance (HF:166-189; V:n/ops/sconv.py:466-489). **Confirmed: K and V are the only per-token growing state.**
5. **KV per token is confirmed at 32 KiB/token** for the target (8 global layers × (K 1024 + V 1024) × bf16) and **+4 KiB/token** for drafter layer 2. Two kinds of fixed per-sequence state come on top. (a) Sliding rings on the 40 local layers and 4 local drafter layers. These must hold ≥ 1024 + (spec tokens) positions, not exactly 1024. (b) Conv state: 4 streams × 3 tokens × 48 layers, about 2.7 MiB per sequence per rank, more if rollback history is kept (§6).
6. **The MoE softcap is inside the expert activation and is asymmetric:** `act = 4·tanh(g/4) · sigmoid(g) · 25·tanh(u/25)`. The sigmoid sees the **raw** gate (HF:641-652; V:n/ops/activation.py:47-78). V folds the routing weight into the activation before the down projection; HF applies it after down. REXL3's MoE epilogue is hardcoded `silu(g)·u`, so it must change (§9).
7. **The router is sigmoid + selection bias + renormalize × 8, in fp32** (HF:695-715; V:n/ops/moe.py:31-64). REXL3's `moe_router.cu` emits **fp16** logits and fp16 weights. That works for a lossy port but does not match the reference bits.
8. **The attention sink depends on the input** (`attn_sink_proj`: one logit per head per token), and the output gate is per element (`sigmoid(gate_attn·x)`). Both are applied after attention: `o·σ(gate)·σ(lse − sink)` (V:n/ops/attention.py:249-334).
9. **The drafter is not the MiniCPM DSpark.** It mixes 8 target features per layer (softmax over `fuse_weight` rows), keeps a separate ctx-K/V projection per layer with RoPE, uses a learned `mask_embed`, and adds a recurrent GRU-like Markov head (`rnn_proj`). The block is 1 anchor + 7 masks, giving 7 drafted tokens per step.
10. **The unembed is BF16 × BF16 with fp32 output, then softcap 30, then a slice to 200058.** Vocab is padded to 201024. The embedding table and unembed are separate tensors (not tied).

## 2. Notation, dtypes and TP4 layout

- `T` = tokens in the batch, `H` = 4608, `L` = 48, `E` = 384 experts, `k` = 8, `I` = 2304, `Hq` = 48, `Hkv` = 8, `D` = 128, `d_rel` = 16, `X` = rel_extent = window = 1024, `W` = conv kernel = 4. `eps` = 1e-6 everywhere (P `rms_norm_eps`).
- Global layers `G = {5,11,17,23,29,35,41,47}`; local layers are the other 40 (P `local_layer_ids`).
- **TP4 rank r ∈ {0..3}**: 12 q heads `[12r, 12r+12)` and 2 KV heads `[2r, 2r+2)` per rank (GQA ratio 6), experts `[96r, 96r+96)` (EP = TP, V:n/moe.py:84-130), vocab rows `[50256r, 50256r+50256)` for embed/unembed (`VocabParallelEmbedding`/`ParallelLMHead`, V:n/model.py:208-212, 509-516). The hidden stream `[T, 4608]` is **replicated** on all ranks. The conv state of the attn/mlp output streams is replicated too (`replicate_hidden=True`, V:n/sconv.py:3-9, V:vllm/models/inkling/nvidia/sconv_swa_attn.py:148-173).
- `rmsnorm_w(x, w) = bf16( fp32(x) · rsqrt(mean(fp32(x)²) + eps) · fp32(w) )` (V:n/ops/norm.py:27-44; HF:117-122). In V, rsqrt is `rsqrt.approx.f32` with no FTZ (V:n/ops/norm.py:14-24), and mean = `div_rn(sum, N)`.
- `rmsnorm_plain(x) = x · rsqrt(mean(x²) + eps)` (no gamma; HF:128-134).
- `l2clip(x, c) = x · min(c / (‖x‖₂ + 1e-6), 1)` in fp32. V computes the norm as `s·rsqrt(s)` (0 if s = 0) and uses `div_rn` (V:n/ops/norm.py:73-82, V:n/ops/sconv.py:729-735; HF:137-163).
- `softcap(x, c) = c · tanh(x / c)`.
- **Dense linears** (all BF16 non-expert weights): `y = bf16( Σ_k x_k w_k )` with fp32 accumulate. V's "reference `linear`" accumulates in ordered 1024-wide K groups (V:n/ops/linear.py header, 1-40). The small-M path (≤ 32 rows) uses FlashInfer CuTe `mm_bf16`, and large-M uses cuBLAS, so the **summation order differs by batch size in V**. Our reference fixes one order (UNKNOWN U3).
- **TP all-reduce** (attention `wo` partials, MoE partials): each rank's partial is **rounded once to bf16**, the ranks are summed in **rank order in fp32**, and the result is rounded once to bf16, giving identical bits on all ranks (V:n/ops/allreduce.py:1-13; V:n/moe.py:233-241).

## 3. Target text model, op by op

### 3.1 Embedding + embed_norm

| | |
|---|---|
| Formula | `e_t = rmsnorm_w(embed[id_t], embed_norm)`. Image placeholder positions (`id = 200054`) are then **overwritten** by the vision embeddings, with no extra norm (V:n/model.py:243-244, 651-666; HF:1213-1219) |
| Shapes | id → `[4608]` bf16. TP: each rank looks up its 50256-row vocab shard (other rows give 0) and the shards are all-reduced. For bf16 a lookup-and-broadcast is exact (only one rank is nonzero) |
| Dtypes | embed bf16; norm fp32 math; output bf16 (`out_dtype=None` keeps bf16, V:n/model.py:211-213) |
| Tensors | `model.llm.embed.weight` BF16 [201024,4608]; `model.llm.embed_norm.weight` BF16 [4608] |

### 3.2 Block attention residual (AttnRes) and the residual stream

**State.** The `blocks` list B holds completed blocks (bf16 `[T,H]` each) plus the running **partial** P (bf16). Initially
B = [], P = e (V:n/model.py:283-292; HF:1291-1305).

**Mix op** `mix(B, P [+A], q, w)`, one per (token, query) (V:n/ops/attn_res.py:3-10, 56-162; HF:732-750):

```
sources x_n = [B_0 … B_{nb-1}, P (+ A)]    # all fp32. When an addend A is present, P+A is summed in fp32 and NOT rounded
ρ_n = rsqrt(mean(x_n²) + 1e-6)             # per source
ℓ_n = (q' · x_n) · ρ_n                     # q' = q · H^-0.5, pre-scaled once in fp32 at load (V:n/model.py:105-135)
p_n = exp(ℓ_n − max ℓ) / Σ exp(…)          # V: inv_sum = div_rn(1, Σ); HF: SDPA softmax
m = Σ_n p_n · ρ_n · x_n                    # attn_res_normalize_values=true → mix the RMS-normalized sources
out = bf16( m · rsqrt(mean(m²) + 1e-6) · w )  # the following sublayer's RMSNorm, fused
```

Shapes: q `[Q, 4608]` fp32 with Q = 1 or 3. Output `[Q, T, 4608]` bf16, replicated on every rank. At most 9 sources (8 completed blocks + partial at the final mix).

**Layer schedule**, for layer i (V:n/model.py:302-421; HF:753-868):

```
first = (i % 6 == 0)
# attention input(s)
if i == 0: in_attn = [rmsnorm_w(P, attn_norm)]                       # no mix at layer 0
else:      in_attn = mix(B, P, attn_queries[i-1], {attn_norm | q/k/v_input_norm})
a = attn_sconv( allreduce_bf16( Attn_i(in_attn) ) )                  # §3.4 static conv, bf16 output
if first:  B.append(P);  base = None                                  # block boundary: P becomes a completed block
mlp_in = mix(B, (first ? a : P + a), mlp_queries[i], mlp_norm)        # P+a is fp32 and unrounded (addend)
m = mlp_sconv( allreduce_bf16( MoE_i(mlp_in) ) )                      # §3.4, conv output y_m
P = first ? bf16( fp32(a) + y_m )  :  bf16( (fp32(P) + fp32(a)) + y_m )   # V order: (res + addend) + y (V:n/ops/sconv.py:379-384)
```

**Final**: `h = mix(B, P, output_query, model.llm.norm)` gives bf16 `[T,4608]` (V:n/model.py:423-432; HF:1307-1308). At the end B has 8 entries (one per boundary at layers 0, 6, …, 42).

Query indexing: the checkpoint's `attn_queries.N` belongs to **layer N+1**. Global layers 5, 11, …, 47 ↔ `attn_queries.{4,10,16,22,28,34,40,46}` are `[3,4608]`, rows (q,k,v). The rest are `[4608]` (P census; V:n/model.py:109-122). `mlp_queries.N` belongs to layer N.

Tensors: `model.llm.attn_res.attn_queries.{0..46}` F32, `model.llm.attn_res.mlp_queries.{0..47}` F32 [4608], `model.llm.attn_res.output_query` F32 [4608]. Norm weights: `layers.i.attn_norm` (local), `layers.i.{q,k,v}_input_norm` (global), `layers.i.mlp_norm`, `model.llm.norm`, all BF16 [4608].

`pre_norm_pre_affine=false` / `final_norm_pre_affine=false` mean these RMSNorms are ordinary post-normalization-gamma norms. The "pre-affine" variants are rejected outright (V:vllm/models/inkling2/configs.py:57-58). There is no separate pre-norm: the RMSNorm with gamma is fused into the mix.

### 3.3 Q/K/V/R/gate/sink projections

Inputs: local layers use `x_q = x_k = x_v = in_attn[0]`. Global layers use `x_q, x_k, x_v` = the three mix outputs (V:n/attention.py:241-259, 270-288; HF:563-576).

| Output | Formula | Per token | Per rank | Tensor (rank-r row slice) |
|---|---|---|---|---|
| q | `x_q Wq^T` | [48,128] | [12,128] | `attn.wq_du.weight` BF16 [6144,4608] rows `[1536r, +1536)` |
| r | `x_q Wr^T` | [48,16] | [12,16] | `attn.wr_du.weight` BF16 [768,4608] rows `[192r, +192)` |
| gate | `x_q Wg^T` | [6144] | [1536] | `attn.gate_attn.weight` BF16 [6144,4608] rows `[1536r, +1536)` |
| sink | `x_q Ws^T` | [48] | [12] | `attn.attn_sink_proj.weight` BF16 [48,4608] rows `[12r, +12)` |
| k_raw | `x_k Wk^T` | [8,128] | [2,128] | `attn.wk_dv.weight` BF16 [1024,4608] rows `[256r, +256)` |
| v_raw | `x_v Wv^T` | [8,128] | [2,128] | `attn.wv_dv.weight` BF16 [1024,4608] rows `[256r, +256)` |

All outputs are bf16 (fp32 accumulate). V fuses them into one `qkvr` GEMM, in the order of the table rows (V:n/model.py:477-487).

### 3.4 Short convolutions (W = 4) and their state

**Static conv** (local `k_sconv`/`v_sconv`, and `attn_sconv`/`mlp_sconv` on every layer) (V:n/ops/sconv.py:12-14, 96-183, 218-298; HF:248-303):

```
y_t[c] = bf16( ( Σ_{k=0..3} w[c,k] · x_{t-3+k}[c]  +  x_t[c] ) · 5^-0.5 )   # tap 3 = current token; x_{<0} = 0
```

V accumulates oldest tap first with FMA (`acc = fma(x, w, acc)`), adds `cur`, and multiplies by `(W+1)^-0.5`. HF divides by `sqrt(5)` instead. The two differ at the last ulp (U4). Inputs are rounded to bf16 before tapping, weights are bf16, math is fp32.

- `attn_sconv` / `mlp_sconv`: x = the all-reduced bf16 sublayer output `[T,4608]` (replicated). Weights `layers.i.attn_sconv.weight` / `layers.i.mlp_sconv.weight` BF16 [4608,1,4]. The `mlp_sconv` call also does the residual update `P = bf16((P + a) + y)` (§3.2).
- Local `k_sconv`/`v_sconv`: x = k_raw/v_raw. Weight `attn.k_sconv.weight`/`attn.v_sconv.weight` BF16 [1024,1,4], rows `[256r, +256)` per rank.

**Dynamic conv** (global-layer `k_sconv`/`v_sconv`) (V:n/ops/sconv.py:16-19, 164-183; HF:306-352):

```
dyn_t = bf16( x_{k|v,t} · weight_proj^T )         # [8 heads × 4 lags]; row = head*4 + lag
y_t[c] = bf16( ( Σ_{l=0..3} (static_w[l,c] + dyn_t[head(c), l]) · x_{t-l}[c]  +  x_t[c] ) · 9^-0.5 )   # l = backward distance, newest first
```

The dynamic taps are a function of the **current** token's `x_k` (or `x_v`), applied to its own window. Only past raw inputs need state. Tensors: `attn.{k,v}_sconv.static_w` BF16 [4,1024] (columns `[256r,+256)` per rank, V:n/sconv.py:63-77) and `attn.{k,v}_sconv.weight_proj` BF16 [32,4608] (rows `[8r,+8)` per rank). V computes `weight_proj` in fp32 1024-wide K groups and rounds to bf16 (V:n/sconv.py:96-107).

**Decode state** per layer per sequence: the last 3 **raw conv inputs** of each stream. Streams: K (k_raw), V (v_raw), ATTN (all-reduced attn output) and MLP (all-reduced MoE output). V stores every position in a paged sliding conv cache. The current input is written to its slot first, and taps before the current forward come from the cache (V:n/ops/sconv.py:3-10, 218-263).

### 3.5 K post-processing: l2 clip

`k̂ = bf16( l2clip( fp32(y_k), 22.627417 ) )` per KV head. There is **no** learned scale on K (V:n/ops/sconv.py:728-736; HF:432, 466, 596). It is written to the KV cache as bf16.

### 3.6 V post-processing: value norm without gamma

`v̂ = bf16( y_v · rsqrt(mean(y_v²) + 1e-6) )` per KV head (head_dim 128) (`use_value_norm_no_gamma=true`; V:n/ops/sconv.py:728-733; HF:434, 468). It is written to the KV cache as bf16.

### 3.7 Q l2 clip with learned scale

`q̂ = bf16( l2clip( fp32(q) · fp32(q_scale[h]), 22.627417 ) )` per head (V:n/ops/norm.py:73-82; HF:151-163). Tensor: `attn.q_l2_clip.scale` BF16 [48,128], rows `[12r,+12)`. Threshold 22.627 = √512 bounds |q̂·k̂| ≤ 512, so after the 1/√128 scale the logit is ≤ 45.25 before bias.

### 3.8 R l2 clip with learned scale

`r̂ = bf16( l2clip( fp32(r) · fp32(r_scale[h]), 8.0 ) )` per head (d_rel 16). Tensor: `attn.rel_l2_clip.scale` BF16 [48,16], rows `[12r,+12)`.

### 3.9 Relative-position logits

```
bias_t[h, Δ] = bf16( Σ_{j<16} r̂_t[h,j] · rel_proj[j, Δ] )   for Δ = pos_t − pos_s ∈ [0, 1024)      (fp32 accumulate, bf16 store)
             = 0     for Δ ≥ 1024 on global layers
             = −inf  for Δ ≥ 1024 on local layers (redundant with the window mask) and for Δ < 0 (causal)
```

(HF:166-189; V:n/ops/sconv.py:466-489 `shift_rel_proj`; V:n/ops/norm.py:116-185.) It is added to the **scaled** logit: `s = q̂·k̂ · D^-0.5 + bias`. V computes it as `fma(bias, 1/scale, qk)` and then multiplies by the scale inside the exp2 (V:n/ops/attention.py:725-800). HF adds it after scaling (HF:382-384). Tensor: `attn.rel_logits_proj.proj` BF16 [16,1024], replicated. For local layers V sizes it as `rel_extent = sliding_window_size`, which is also 1024 (V:n/attention.py:140).
**Per-token work**: `[12,16]×[16,1024]` per rank per query token. **Cache**: none. The bias needs only the query's r̂ and integer distances.

### 3.10 Log scaling (global layers only)

`τ_t = 1 + 0.1 · ln( max( (pos_t + 1) / 128000, 1 ) )` in fp32 (V:n/ops/attention.py:337-362; HF:588-592). It is applied **after** clipping and rounding: `q̂ ← bf16(fp32(q̂)·τ)`, `r̂ ← bf16(fp32(r̂)·τ)` (V:n/ops/norm.py:79-82; HF:594-601). K is not scaled. Below position 127999, τ = 1 exactly (a no-op). Local layers have no τ (V:n/model.py:192-194).

### 3.11 Attention core: masks, sinks, gate

Per rank: 12 q heads over 2 KV heads (GQA 6), D = 128, scale `128^-0.5` (V:n/attention.py:125-141).

```
s_{t,s} = q̂_t·k̂_s · 128^-0.5 + bias_t[h, pos_t − pos_s]
visible:  0 ≤ pos_t − pos_s           (global)
          0 ≤ pos_t − pos_s < 1024    (local; FA window (1023, 0), i.e. 1024 keys incl. self)
lse_t = logsumexp_s s_{t,s};   o_t = Σ softmax(s)_s · v̂_s     (fp32)
out_t[h,:] = bf16( o_t · σ(gate_t[h,:]) · σ(lse_t[h] − sink_t[h]) )
```

(V:n/ops/attention.py:249-334, 425-493; HF:367-398, 491-496.) `σ(lse − sink)` is the softmax with one extra zero-value logit `sink` (HF:388-391). V uses `ex2.approx` / `rcp.approx` sigmoids and FA4's online softmax with **bf16 P** and fp32 row sums (V:n/ops/attention.py:725-800). HF uses full fp32 softmax and rounds the weights to bf16 before P·V. The two differ (U5).
Output `[T, 12·128]` bf16 per rank.

### 3.12 Output projection wo

`attn_partial = bf16( out · Wo[:, 1536r:1536r+1536]^T )` gives `[T,4608]`, then the TP all-reduce (§2) and `attn_sconv` (§3.4) (V:n/attention.py:172-178, 303). Tensor `attn.wo_ud.weight` BF16 [4608,6144], column slice per rank.

### 3.13 MoE router

```
logit_e = fp32( mlp_in · Wgate[e]^T )                 # fp32; V: 1024-wide K-group partials summed in order (V:n/ops/moe.py:67-79)
sel_e   = σ(logit_e) + bias_e                          # V: σ = div_rn(1, 1 + exp(−x))
top8    = argtop8(sel); ties → lower expert index      (V:n/ops/moe.py:31-41)
w_j     = 8 · σ(logit_j) / Σ_{j∈top8} σ(logit_j)       # V: softmax of log-sigmoids, pairwise-tree Σ (V:n/ops/moe.py:20-64); HF: /(Σ + 1e-20)
```

(HF:695-715; V:n/moe.py:62-82.) Tensors: `layers.i.mlp.gate.weight` BF16 [384,4608], `layers.i.mlp.gate.bias` F32 [384]. Replicated: every rank routes every token and keeps only the slots whose expert falls in its range.

### 3.14 Experts: softcapped SwiGLU

Per selected (token, expert j):

```
g = x · Wg_j^T, u = x · Wu_j^T          # [2304] each; bf16 in V (NVFP4 GEMM), fp32 in HF's .float() view
a = softcap(g, 4) · σ(g) · softcap(u, 25)   # fp32; σ on RAW g
V:  a' = bf16( a · w_j ),  y_j = a' · Wd_j^T,  combine Σ_j y_j in fp32 (slot order), bf16 once   (V:n/ops/activation.py:47-78; V:n/moe.py:214-241)
HF: a' = bf16(a),          y_j = (a' · Wd_j^T) · w_j,  index_add in fp32, bf16 at end          (HF:641-688)
```

V's `softcap` uses `tanh.approx.f32` on `div_rn(x, cap)`, and σ uses `ex2.approx.ftz`/`rcp.approx.ftz` (V:n/ops/activation.py:19-58).
Bounds: |softcap(g,4)·σ(g)| < 4 and |softcap(u,25)| < 25, so |a| < 100, and |a·w| < 800 since w ≤ 8. **fp16 intermediates are safe** (REXL3's MoE uses fp16 between gate/up and down).
Shapes per rank: 96 local experts; per (token, local slot) `[4608]→[2304]→[4608]`. Partial `[T,4608]` → bf16 → all-reduce → `mlp_sconv`.
Tensors (EXL3, mcg codebook, 4.0 bpw, `out_scales: always`): `layers.i.mlp.experts.e.{gate_proj,up_proj}` = {`trellis` I16 [288,144,64], `suh` F16 [4608], `svh` F16 [2304]}; `…down_proj` = {`trellis` I16 [144,288,64], `suh` F16 [2304], `svh` F16 [4608]}; `….mcg` I32 scalar (P census). V's NVFP4/MXFP8 weights do not apply to this pack. **The expert matmuls cannot match V or HF bit-for-bit.** The byte-exact target is our own EXL3 reference.

### 3.15 Final norm, unembed, softcap, vocab

```
h = mix(B, P, output_query, model.llm.norm)          # §3.2, bf16
z = fp32( h · unembed^T )                            # bf16 × bf16 → fp32 output (V: head_dtype float32, V:vllm/models/inkling2/configs.py:123-125)
logits = 30 · tanh(z / 30)                           # fp32 (V:n/model.py:512-516 LogitsProcessor soft_cap; HF:1385-1386)
logits = logits[:200058]                              # unpadded vocab; rows 200058..201023 are padding
```

HF rounds `lm_head` output to bf16 and then upcasts (`self.lm_head(h).float()`), whereas V keeps fp32 output (U6). TP: vocab-sharded rows (50256 per rank), then all-gather or a distributed argmax over valid columns. Tensor `model.llm.unembed.weight` BF16 [201024,4608]. **Not tied** to `embed` (HF:1317-1318).

## 4. DSpark drafter (mtp.safetensors)

Config (P `mtp_config`): 5 layers, hidden 4608, 48 q / 8 kv heads, D 128, MLP 18432 (dense SiLU), rope θ 1e6, local layers {0,1,3,4} with window 1024, **layer 2 global**, block_size 8, markov_rank 256, `markov_head_type=rnn_hidden`, target_layer_ids [5,11,17,23,29,35,41,48].

### 4.1 Target features consumed

For each `id ∈ target_layer_ids[:-1]` the feature is the **input partial block P at the start of target layer id**: the running residual, bf16, before that layer's AttnRes mix. The final id 48 is the **final-norm output h** (§3.15) (V:n/model.py:239-241, 302-312, 433-434; V:vllm/models/inkling2/configs.py:151-154). V stores all 8 as one `[T, 8·4608]` bf16 row per committed token (V:vllm/models/inkling2/speculator.py:22-27).

### 4.2 Feature fusion

`α = softmax_fp32(fuse_weight)` over the 8 features. `fuse_weight` is BF16 [6,8]: rows 0–4 belong to draft layers 0–4 and row 5 to the Markov head (V:n/dspark.py:402-407, 502-504, 519-530).
`ctx_l = bf16( rmsnorm_w_fp32( Σ_i α[l,i] · f_i , layers.l.dflare_norm ) )`, all fp32 math, using `tl.rsqrt` (V:n/ops/dspark.py:13-67).

### 4.3 Context K/V (what is cached)

Per draft layer l, per committed token (V:n/dspark.py:250-266, 513-526):
`k = rope_neox( rmsnorm_w(ctx_l·Wk_ctx^T per head, k_norm_ctx), pos )`, `v = ctx_l · Wv_ctx^T` (**no** v norm). These are written to the draft layer's own paged KV cache at the token's slot.
Tensors: `layers.l.self_attn.{k_proj_ctx,v_proj_ctx}.weight` BF16 [1024,4608] (rows `[256r,+256)`), `k_norm_ctx.weight` BF16 [128], `dflare_norm.weight` BF16 [4608].
Retention: local draft layers (0,1,3,4) keep the last `1024` positions before the block start, plus the block. Layer 2 keeps the full length, at **4 KiB/token** (2 × 8 × 128 × 2 B) (V:n/dspark.py:119-174).

### 4.4 Block drafting (block_size 8 → 7 draft tokens)

The scheduler requires `num_speculative_tokens = block_size − 1 = 7` (V:n/dspark.py:466-471; V:vllm/models/inkling2/configs.py:143-150: anchor + 7 masks, each predicting the token **at its own position**, `sample_from_anchor = False`).

1. Block input `[8,4608]` bf16: row 0 = `rmsnorm_w(embed[anchor], embed_norm)` (the target's normed embedding); rows 1..7 = `mask_embed` BF16 [4608] (V:n/dspark.py:423-430). The anchor is the last accepted/bonus token at position p; the masks sit at p+1..p+7.
2. Per layer (pre-norm transformer, V:n/dspark.py:290-319, 72-116):
   - `x = rmsnorm_w(h, input_layernorm)`. From layer 1 on this is a fused all-reduce + residual add + norm: `h ← h + allreduce(x)`, fp32 sum rounded once to bf16, and the norm is computed from that sum.
   - `qkv = x·[Wq;Wk;Wv]^T`. Per head: `q = rope(rmsnorm_w(q, q_norm))`, `k = rope(rmsnorm_w(k, k_norm))` (norm rounded to bf16, then RoPE in fp32 math, V:n/ops/dspark.py:119-165). The block K/V are written to the cache at slots p..p+7.
   - Attention is **non-causal within the block**. Each block query sees ctx keys (local: positions ≥ p − 1024; global: all) plus all 8 block keys. Scale `128^-0.5`, no bias, no sink, no gate (V:n/dspark.py:119-248; `block_decode_attention` V:n/ops/attention.py:1153-1188).
   - `o_proj`, then `h ← h + allreduce(·)` and `x = rmsnorm_w(h, post_attention_layernorm)`.
   - MLP: `bf16( silu(x Wg^T) · (x Wu^T) )` with precise exp/div in fp32 and one rounding, then `Wd` (V:n/ops/dspark.py:169-189).
   - Final: `hd = rmsnorm_w(h + allreduce(mlp), norm)`.
   - Per rank: q 12 heads, kv 2 heads, MLP 4608 columns (row-parallel down), each all-reduce fp32 then bf16.
   - Tensors: `layers.l.self_attn.{q_proj [6144,4608], k_proj [1024,4608], v_proj [1024,4608], o_proj [4608,6144], q_norm [128], k_norm [128]}`, `layers.l.mlp.{gate_proj,up_proj} [18432,4608]`, `down_proj [4608,18432]`, `layers.l.{input_layernorm, post_attention_layernorm}` [4608], `norm.weight` [4608], all BF16.
3. **Markov head** (`rnn_hidden`, rank 256) runs sequentially over the 7 mask outputs `hd_i`, i = 1..7 (V:n/dspark.py:322-384, 545-570; V:vllm/models/inkling2/speculator.py:49-77):
   ```
   f_rnn   = bf16( rmsnorm_w( Σ α[5,i] f_i(last committed token), markov_head.dflare_norm ) )
   ctx_i   = fp32( rnn_bias + [hd_i, f_rnn] · Wrnn[:, 512:]^T )         # once for all i
   base_i  = bf16( hd_i · unembed^T )                                     # bf16 GEMM, bf16 output (TP-local vocab)
   state_0 = 0 (bf16 [256]);  prev_0 = anchor id
   for i in 1..7:
     raw   = fp32( ctx_i + [state, markov_w1[prev]] · Wrnn[:, :512]^T )   # [768]
     g = bf16σ(bf16 raw[0:256]); c = bf16 tanh(bf16 raw[256:512]); o = bf16 raw[512:768]
     state = bf16( bf16(g·state) + bf16(bf16(1−g)·c) );  latent = bf16 tanh(o)
     bias  = bf16( fold_b + latent · fold_W^T )   # fold_W = bf16(unembed·hidden_out.W in fp64) [V,256]; fold_b = bf16(unembed·hidden_out.b)
     logit = bf16( 30·tanh( bf16(base_i + bias) / 30 ) )
     prev  = draft_i = argmax_{v<200058} logit   (NaN→−inf, first max)   # greedy; sampling path uses Gumbel
   ```
   (V:n/ops/dspark.py:191-257.) Tensors: `markov_head.rnn_proj.weight` BF16 [768,9728] (input columns `[state 256 | markov_w1(prev) 256 | hidden 4608 | f_rnn 4608]`, output `[gate | cand | out]`), `rnn_proj.bias` [768], `markov_head.markov_w1.weight` [201024,256], `hidden_out.weight` [4608,256], `hidden_out.bias` [4608], `markov_head.dflare_norm.weight` [4608], `fuse_weight` [6,8], `mask_embed` [4608], all BF16 (P mtp header).

### 4.5 Verification and acceptance (vLLM)

The target runs one forward over `[anchor, d1..d7]` (8 query tokens; the decode/verify path treats it as one request with q_len 8, V:n/attention.py:68-98). Acceptance uses vLLM's standard rejection sampler: greedy keeps the longest prefix where the target argmax equals the draft, plus a bonus token. The generic DSpark speculator does sequential Markov sampling (V:vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:3-28, 498-519).
After acceptance, the ctx K/V for the newly committed tokens are computed from their stored target features (`precompute_and_store_context_kv`). `f_rnn` is recomputed from the last committed row, accounting for rejected tokens (V:vllm/models/inkling2/speculator.py:49-58).
**Rollback**: KV and conv state in V are indexed by absolute position, so rejected positions are simply overwritten. REXL3 must give the same rollback semantics for the conv state (§6).

## 5. Vision tower

Architecture (P `vision_config`; V:n/vision.py:1-283; HF:871-1154, 1426-1462): PE-Lang-style ViT, 23 layers, width 1024, 16 heads × 64, MLP 4096 (exact-erf GELU), LayerNorm eps 1e-5 with bias, layer-scale (`ls_1`, `ls_2`), a CLS token per image, then a 2×2 merge projector `proj_in` [4096,4096] → GELU → `proj_out` [4608,4096] (no bias). It runs in bf16. V runs it replicated on every rank; FIT plans rank 0 only.

| Step | Formula / rounding | Tensors |
|---|---|---|
| Preprocess | Decode to RGB. Alpha is composited on an 8-px checkerboard (204/153). **No resize.** Pad to a multiple of 40 px with value −1; grid (rows, cols) = ceil(H/40), ceil(W/40). One 40×40 cell = **one LM token** (V:vllm/models/inkling2/processing.py:42-130) | `processor_config.json`: size 40×40, mean/std 0.5 |
| Pixel norm | `2·( bf16((v/255 − clip_mean)/clip_std′)·clip_std + clip_mean ) − 1`, fp32 without FMA. CLIP constants are listed at V:n/vision.py:28-32. Features ordered (py, px, c) per 20×20 native patch (V:n/ops/vision.py:50-87) | — |
| Patch embed | `bf16(patch · Wp^T)` [1200→1024], no bias | `backbone.embeddings.patch_embedding.weight` [1024,1200] |
| Pos embed | Bilinear resize (align_corners=False, clamped) of the 32×32 grid to the image's native-patch grid (2·rows × 2·cols); `hidden = bf16(bf16(patch) + bf16(lerp))`. CLS = class + cls_position (V:n/ops/vision.py:124-190) | `position_embedding` [1024,1024], `class_embedding`, `cls_position_embedding` [1024] |
| ln_pre | LayerNorm, rounded to bf16; this is the residual | `backbone.ln_pre.{weight,bias}` |
| Layer ×23 | `h = bf16(h + bf16(ls_1·Attn(LN1(h))))`; `h = bf16(h + bf16(ls_2·MLP(LN2(h))))` (V:n/ops/vision.py:298-321) | `layers.n.{ln_1,ln_2}.{weight,bias}`, `ls_{1,2}.gamma` |
| Attention | Fused qkv with bias. **2D axial RoPE** interleaved pairs: pairs 0..15 rotate by x+1, pairs 16..31 by y+1, CLS rotates by 0; inv_freq = 10000^(−4i/64) (V:n/ops/vision.py:18-22, 240-276; HF:871-945). Non-causal per image (varlen), scale 64^-0.5 | `self_attn.{q,k,v,out}_proj.{weight [1024,1024], bias}` |
| MLP | `c_proj(gelu_erf(c_fc(x)))` | `mlp.c_fc` [4096,1024]+b, `mlp.c_proj` [1024,4096]+b |
| Merge | Drop CLS. Gather each 2×2 group of native-patch rows → [4096]; `proj_out(gelu(proj_in(·)))` gives [4608] per LM token (V:n/vision.py:258-260) | `model.visual.proj_in` [4096,4096], `proj_out` [4608,4096] |

**Token stream**: the renderer emits `<|content_image|>` (200005). The processor expands it to `200005` followed by `rows·cols × <|unused_200054|>` (200054). The 200054 positions get the projector outputs **in place of** their normed text embeddings (V:vllm/models/inkling2/processing.py:42-46, 235-250; V:n/model.py:651-666). Chat template: `<|content_image|><|unused_200054|><|end_message|>` inside a user/tool message (P chat_template.jinja:83-87).

## 6. KV / state per token and per sequence

| Item | Layers | What | Per token (cluster) | Per token per rank | Growth |
|---|---|---|---|---|---|
| Target K, V | 8 global | k̂, v̂ bf16, 8 heads × 128 each | 2 × 8 × 128 × 2 B × 8 = **32 KiB** | 8 KiB | ∝ context ✔ confirmed |
| Target K, V | 40 local | same, window 1024 | 4 KiB per layer | 1 KiB per layer | fixed ring ≥ 1024 + 7 positions (verify queries at p..p+7 each need 1024 keys back) → ~40 MiB/seq/rank |
| Drafter K, V | layer 2 | ctx k (RoPE'd), v bf16 | **4 KiB** | 1 KiB | ∝ context ✔ confirmed |
| Drafter K, V | 0,1,3,4 | same, window 1024 + block 8 | — | — | fixed ring ≥ 1032 |
| Relative-position state | all | **none** (bias recomputed from the query's r̂) | 0 | 0 | ✔ confirmed |
| Conv state | 48 | last 3 raw inputs: K raw 1024 + V raw 1024 (sharded: 256 + 256 per rank) + attn-out 4608 + mlp-out 4608 (replicated) | — | 3 × (512 + 9216) × 2 B = 57 KiB per layer → **2.7 MiB/seq/rank** | fixed; ×(3 + 8)/3 if a speculative rollback history is kept |
| Target features for drafter | — | 8 × 4608 bf16 per token, only for tokens whose ctx K/V are not yet written | 72 KiB transient | — | per step, not per context |
| AttnRes stack | — | ≤ 9 × [T,4608] bf16 activations | — | — | transient |

V's KV spec: `AttentionSpec(num_kv_heads per rank, head_dim, dtype bf16)`, with SlidingWindowManager for local layers and FullAttention for global (V:n/attention.py:213, 228-239). The conv state is a separate per-layer sliding cache with `block_size = W` and packed heads `[K | V | attn | mlp]`, padded to a power of two (V:vllm/models/inkling/nvidia/sconv_swa_attn.py:153-173).

## 7. Chat template, reasoning, tool calls

Special tokens (P tokenizer_config): `<|endoftext|>` 199999, `<|message_user|>` 200000, `<|message_model|>` 200001, `<|message_system|>` 200002, `<|message_tool|>` 200003, `<|content_text|>` 200004, `<|content_image|>` 200005, `<|content_model_end_sampling|>` 200006 (**= eos_token_id**), `<|content_thinking|>` 200008, `<|end_message|>` 200010, `<|content_tool_error|>` 200022, `<|content_xml|>` 200024, `<|begin_of_text|>` 200028, `<|content_invoke_tool_json|>` 200049, image pad `<|unused_200054|>` 200054, `<|content_invoke_tool_text|>` 200057.

- **Message framing**: `ROLE_TOKEN [author name text] CONTENT_KIND_TOKEN body <|end_message|>`. Assistant turns end with `<|content_model_end_sampling|>` (P chat_template.jinja:50-121; V:vllm/renderers/inkling_encoding.py:287-313).
- **Generation prompt**: Inkling 2 adds **no** generation prefix. The model emits its own `<|message_model|>` (V:vllm/renderers/inkling_encoding.py:436-441, 444-533; the template has no `add_generation_prompt` branch). `continue_final_message` appends `<|message_model|><|content_text|>` + prefix.
- **Reasoning**: `<|message_model|><|content_thinking|>…<|end_message|>`, followed by the text block `<|message_model|><|content_text|>…<|end_message|>`. Blocks may repeat in any order. Generation stops at `<|content_model_end_sampling|>` (V:vllm/parser/inkling.py:1-23).
- **Effort control**: a system text message `Thinking effort level: X` inserted before the first non-system message (or at the end if there is none). Default 0.9, range [0, 0.99]. Template map: none 0, minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh/max 0.99 (P chat_template.jinja:1-21). V quantizes to milli-effort and prints with `%g` (V:vllm/renderers/inkling_encoding.py:673-680). The template prints the raw float, and prints 0 as `0` (U9).
- **Tools**: declared once as `<|message_system|>tool_declare<|content_xml|>` + JSON (sort_keys, compact) + `<|end_message|>`. A call is `<|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":{…}}<|end_message|>`, and a result is `<|message_tool|>NAME<|content_text|>…<|end_message|>` (P chat_template.jinja:23-38, 96-116). V can append a tool-choice instruction to the system/effort text (V:vllm/renderers/inkling_encoding.py:486-496, 616-632).

## 8. TP4 communication per layer (reference semantics)

| Point | Op | Size per token | Precision |
|---|---|---|---|
| embed | vocab-shard lookup → all-reduce (or owner broadcast) | 4608 | exact |
| after `wo` | all-reduce | 4608 | bf16 partials, fp32 rank-order sum, bf16 once |
| after MoE combine | all-reduce | 4608 | same |
| unembed | all-gather logits or distributed argmax | 50256 per rank | fp32 |
| drafter (×2 per layer + final) | all-reduce fused with add + RMSNorm | 4608 | fp32 sum, bf16 once |

Two all-reduces per target layer → 96 per token, plus 11 for the drafter.

## 9. REXL3 reuse inventory (upstream/main@06181b1)

Paths are relative to the worktree root. The survey was read-only (nothing built). "NEW" means nothing found after searching for `inkling`, `softcap`, `alibi`, `layer_scale`, `rnn_proj`, `fuse_weight`, `mask_embed`, sink and window terms.

| Op (§) | Closest REXL3 | Fit / work |
|---|---|---|
| Embedding (3.1) | `gb_embed` `crates/cuda/csrc/glue.cu:744` | bf16 table, out-of-range rows → 0. Reuse; vocab-shard + sum fits the disjoint-support TP design |
| RMSNorm (3.1, 3.2) | `gb_rmsnorm`/`gb_add_rmsnorm` `glue.cu:755,764` | bf16 in/out, fp32 math, but **fp32 gamma** (ours is bf16: convert at load). Need `div_rn` mean + `rsqrt.approx` to match V |
| AttnRes mix + fused norm (3.2) | — | **NEW** kernel (≤ 9 sources, Q = 1/3, fp32). Simple row kernel |
| Dense BF16 projections (3.3, 3.12, router, drafter) | `gb_bf16_linear` `csrc/dsv4_bf16.cu:190`, `crates/cuda/src/bf16_linear.rs` | Weights bf16 and fp32 accumulate fit, but **input is fp16** and output fp16/fp32. Needs a **bf16-input** variant (bf16 activations here exceed fp16 range: the unnormalized P stream, logits). TP slicing: `crates/exl3/src/tp.rs` (`tile_cut`), `nemotron_h_omni/tp_shard.rs` |
| Static short conv W = 4 + state (3.4) | `pre_kernel`/`conv_commit_kernel` `gdn.cu:92-95,437`; `nemotron_ssm.cu:116-121` | Exists, but fused with silu and with channel counts baked in. **NEW parameterised kernel**: 4608-wide (attn/mlp) and 256-wide (K/V) per rank, the `(Σ+x)·5^-½` residual form, fused residual update |
| Dynamic conv (3.4) | `dynconv_kernel` `dflash2.cu:22-84` | 2 taps only, block-local, no cross-step state. **NEW** (4 taps + state) |
| q/k/r l2-clip, v norm-no-gamma (3.5-3.8, 3.10) | (l2norm inside `gdn.cu:92`, `kda.cu:417`) | **NEW** fused "attention prologue" (clip, τ, rel-logits GEMV, KV write), as in V:n/ops/sconv.py:493-747 |
| Relative-logit bias (3.9) | — | **NEW** (no additive bias in any attention kernel) |
| Attention core: GQA 6, hd 128, window/full, bias, sink, gate (3.11) | `attention.cu:52-57` (GRP 6 allowed; Q24Kv4 exists; **D = 256 constexpr**); `nemotron_attn.cu:33-35` (hd 128, GRP 16, fp16 KV); `dsv4_attn.cu:595` (sinks, MQA hd 512); `dsv4_kvq4.cu:105` (ring); `dflash2.cu:139-203` (hd128 window ring, GQA 4) | No kernel combines them. **NEW decode + prefill kernel** (or generalise `attention.cu` to D = 128 with bias/sink/window). Output gate: `gb_gate_mul` `glue.cu:780` is reusable for `σ(gate)`; the `σ(lse − sink)` factor is new |
| Router (3.13) | `router_select_kernel` `moe_router.cu:168`, `gb_moe_router_logits` `:313`, `MoeRouter::select` `crates/cuda/src/moe.rs:501` | Sigmoid + bias + top-k + renorm·scale fits, but **fp16 logits (σ via a 64K table) and fp16 weights**, and the bias is "centred" (`moe.rs:238`). Needs an fp32 variant for exactness |
| EXL3 MCG experts decode (3.14) | `gb_moe_gemv` `crates/cuda/csrc/moe_experts.cu:521-540`, `MoeLayer` `crates/cuda/src/moe.rs:686,1754` | **MCG supported** (codebook runtime 0..2, K2 even; this pack is K2 = 4). bf16 x accepted, fp16 intermediate, fp32 out. Reuse |
| EXL3 MCG experts prefill | `gb_moe_pf_gateup/down/combine` `moe_prefill.cu:745-793` | MCG supported. Reuse |
| FrostMoE fused (`fmoe.cu:1763`) | — | **MUL1 only. Not usable** for this MCG pack |
| Activation epilogue (3.14) | `gateup_epilogue_kernel` `moe_experts.cu:328-376`; `gateup_kernel` `moe_prefill.cu:341-390` | **Hardcoded `silu(g)·u`** (optional clamp), not pluggable. Add a softcapped-SiTU variant `4tanh(g/4)·σ(g)·25tanh(u/25)` in both. Routing weight is applied **at combine** (`down_combine_kernel` `:381-425`), which matches HF, not V (U2) |
| TP collectives (§8) | `NcclComm::allreduce_sum_f32`/`oneshot_sum_f32` `crates/cuda/src/nccl.rs:663,500`; `gb_tp_sum_slots_f32` `tp_oneshot.cu:63`; `TpComm` `nemotron_h_omni/tp_comm.rs:131,220` | **fp32-only**, fixed rank order. This matches the reference "fp32 rank-order sum, bf16 once" if partials are rounded to bf16 first. No bf16/reduce-scatter/all-gather. RoCE is **2-rank only** (`roce.rs`). Only nemotron_h_omni is N-rank ("R > 2 never measured on a fabric"). Use NCCL |
| Final softcap + vocab-parallel argmax (3.15) | `gb_topk_valid` `glue.cu:1088`, `head_row_cut` `tp_shard.rs:144-184` | Valid-vs-stride handling exists. **Softcap NEW** (tanh hook before sampling) |
| Drafter ctx K/V + RoPE (4.3) | `gb_dflash_ctx_kv` `dflash2.cu:522`; `gb_attn_prep` `glue.cu:790-826` (per-head RMSNorm + neox RoPE, hd 128) | Close. Needs the 8-feature softmax fusion (**NEW**) and the per-layer ctx projections |
| Drafter block attention (4.4) | `gb_dflash_attn_vis` `dflash2.cu:573,585` (causal flag) | **GQA fixed at 4** (`dflash2.cu:158`). Inkling needs GQA 6 (12q/2kv per rank). Generalise |
| Markov RNN head (4.4) | `markov_bias_k` `dsv4_mtp.cu:69`; MiniCPM `markov_w1·markov_w2` walk `minicpmv4_7/dspark.rs` | Different math. **NEW** (`rnn_proj` GRU-like gate, folded `unembed·hidden_out`) |
| Accept/verify (4.5) | `crates/spec/src/lib.rs:33` (`accept`), `arbiter.rs`, `draft_pick.rs` | Family-agnostic keyed accept. Reuse |
| Vision (5) | `vision.cu:281,294,308` (LN, 2D RoPE, GELU-tanh, hd 72); `dsv4_vision.cu` (hd 64, gelu_erf `:109`); `mcv_vision.cu:873` (`merge4`) | Pieces exist but head dims are constexpr (need 64). Layer-scale NEW. GELU must be **erf** |
| Family scaffolding | `crates/model/src/families/mod.rs:6-27`, `ModelFamily` `crates/model/src/family.rs:288-356`; example `minicpmv4_7/model.rs:1267-1420` | Add an `inkling2` family |
| KV/state allocation | `KvLayerClass::gqa` `crates/sched/src/memory.rs:59-72`; `PagedKv` `crates/cuda/src/attention.rs:167,214`; SWA ring via a custom plane `mimo/memory.rs:29` | 256-token pages (layout `[blocks][256][kv][256]` assumes hd 256). Needs hd 128 classes, a ring class for 44 layers, and a conv-state plane (4 streams × 3 + rollback) |

## 10. UNKNOWNs (and what resolves them)

| ID | Unknown | Resolve by |
|---|---|---|
| U1 | **Which reference is canonical for bit-exactness?** V (open PR) and HF differ at several rounding points (U2–U6), and both claim to follow "the original". | Decide in DECISIONS.md. Recommendation: pin V@1448919f semantics for non-expert ops, since every rounding there is explicit. Our EXL3 reference defines the experts. Diff SG's `jit/csrc/inkling2/*.cuh` as a tiebreaker |
| U2 | Routing weight **before** down-proj with bf16 rounding of `a·w` (V) vs **after** down in fp32 (HF). REXL3's combine does after. | Pick one; it is a numerics decision. Check SG `route.cuh`/`routed_quant.cuh` and the OX overlay `inkling2_tmoe.py` (what this pack was calibrated with) |
| U3 | Accumulation order of dense bf16 linears. V switches between CuTe low-M, cuBLAS and "ordered 1024-wide K groups" by batch size. | Our reference fixes one order (e.g. 1024-K groups, fp32), applied regardless of batch, to keep concurrency-invariance (DECISIONS) |
| U4 | Conv scale: V multiplies by `(W+1)^-½` (and `(2W+1)^-½` for dynamic); HF divides by `sqrt`. Also the tap summation order. | Adopt V; one-line test against HF on random inputs to measure the ulp drift |
| U5 | Attention softmax numerics: V uses FA4 (bf16 P, ex2 emulation columns, lazy rescale threshold 8); HF uses exact fp32 softmax with bf16 weights. | Our kernel defines its own fixed-split online softmax; document it as the reference |
| U6 | Unembed output: fp32 straight from the GEMM (V) vs bf16-rounded then upcast (HF). This changes the logits' last bits and can flip argmax ties. | Adopt V (fp32 output), consistent with V's comment "checkpoint's unembedding runs in BF16 with an FP32 output" |
| U7 | Vision padding pixels: the processor fills −1 **raw** values, which then go through the CLIP→PE normalization rather than being masked. HF's processor path was not traced. | Read HF `image_processing_inkling2.py` (fetched, not reviewed) and SG `processor` |
| U8 | Drafter local-layer window boundary: is it exactly 1024 positions before the block start (`window = 1024 + 8` keys, V:n/dspark.py:173-174)? And can the anchor's own features ever be in ctx? | Read `block_decode_attention` (V:n/ops/attention.py:1153-1188) and DFlash `prepare_inputs` in V |
| U9 | Effort-text formatting: template `{{ num }}` (e.g. `0.9`, `0`) vs V `%g` of the milli-quantized value (`0.9`, `0`) vs Inkling-1 `0.0`. | Golden-render tests against V's Rust renderer `rust/src/chat/src/renderer/inkling/inkling2.rs` and its tests |
| U10 | The attn-res query pre-scale `q·H^-½` at load (V) vs scaling the dot product (HF SDPA). This differs at the ulp level. | Adopt V |
| U11 | Whether expert activations (input to EXL3 gate/up) should be the bf16 `mlp_in` or something else. V quantizes it to NVFP4/MXFP8 (n/a for EXL3). REXL3 rotates to fp16 internally. | Fix in our EXL3 reference; check that OX does the same |
| U12 | Prefix-cache alignment interaction with conv state (V's `prefix_align` = full-attention block size). | Only relevant if we implement prefix caching; read V:n/ops/conv_mix.py and V:n/ops/seq_parallel.py |
| U13 | Whether the 8 features passed to the drafter at TP > 1 are gathered from a sequence-parallel shard (V all-gathers under SP). | Irrelevant if REXL3 keeps the hidden stream replicated (planned) |
| U14 | MTP greedy argmax across TP vocab shards: tie-breaking is "first max", but across ranks? | V `dspark_argmax` (V:n/ops/dspark.py:259-314): read the rank-merge rule |
