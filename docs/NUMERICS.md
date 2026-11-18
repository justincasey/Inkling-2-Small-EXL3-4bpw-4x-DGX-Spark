# Numerics of the reference forward (oracle)

This file fixes the numerics of `oracle/inkling2_ref.py`, the CPU reference that the REXL3 `inkling2` port is checked
against. It answers PORT-SPEC §10 U1–U14 and lists the oracle's rounding points. Byte-exact serving is a property of the
engine against its own pinned build. The oracle is the tolerance anchor (`oracle/README.md`, "Tolerances") and the
numeric spec of what each GPU kernel approximates.

**Principle.** The oracle evaluates every op of PORT-SPEC §3 in float64, starting from the stored values. It rounds
only at the spec's storage points:

- to **bf16** (through fp32, `bf16(fp32(x))`, the value an fp32-result kernel with a bf16 store produces) wherever the
  spec stores bf16;
- to **fp32** wherever the spec names an fp32 value that crosses an op boundary.

Transcendentals are precise: `exp`, `tanh`, `sigmoid` and `rsqrt` have no approximation and no FTZ. Products of
bf16/fp16 values are exact in float64, so the result does not depend on summation order, batch size or chunking.
Prefill + decode reproduces a full recompute bit for bit (`oracle/tests/test_model.py`).

## UNKNOWN resolutions

- **U1: canonical reference.** The oracle is the reference. It keeps the PORT-SPEC §2–§3 op order and storage points;
  at every module boundary V and HF already agree on bf16 storage. Inside an op it uses HF-style precise
  math in float64, not V's approximate PTX instructions. Where V and HF place a rounding differently, the oracle takes
  HF. The exceptions are where PORT-SPEC gives a reason to take V (U4, U6, U10) and §3.11's single output rounding
  (U5). *Why:* the shared contract defaults to HF-style fp32 semantics. A precise reference does not depend on
  any kernel's summation order. Byte-exactness is engine-vs-pinned-engine, so the oracle only has to be the most
  faithful anchor. The `Rounding.exact()` mode of the oracle equals an independent float64 transcription of the HF
  eager model to 1e-9 (`tests/hf_transcription.py`), which checks the op structure separately from the rounding.
- **U2: where the routing weight goes.** HF: the weight is applied **after** the down projection, in high precision.
  The order is:
  1. `a' = bf16(act)`;
  2. `y_j = a' · W_d,j` in float64;
  3. rank r forms `Σ_{j: E_j mod 4 = r} w_j · y_j` in float64 and rounds it to bf16;
  4. the ranks are summed in rank order in fp32 and rounded to bf16 (§2).

  *Why:* this is the HF default, and it is what REXL3's existing combine does (`down_combine_kernel` applies the
  weight at combine time). The pack's calibration path (OX) was not read, so nothing argues for V's `bf16(a·w)`.
- **U3: dense linear accumulation.** `y = bf16(fp32(Σ_k x_k w_k))`. The sum is in float64 and therefore
  order-independent. *Why:* fixing one fp32 order in the oracle would be arbitrary. The engine pins its own
  batch-invariant fp32 order. Its result is within a few fp32 ulps of the oracle before the bf16 store, which
  occasionally flips a bf16 ulp. The tolerances cover that.
- **U4: conv scale.** Take V. Multiply by the fp32 constant `fp32((W+1)^-½) = fp32(5^-½)` for the static conv and by
  `fp32(9^-½)` for the dynamic conv. The constant applies to `Σ taps + x`, which is computed in float64; the
  result is rounded once to bf16. *Why:* PORT-SPEC recommends V. Because the tap sum is float64, its order no longer
  matters. HF's `/ sqrt(5)` differs by at most one fp32 ulp before the bf16 store.
- **U5: attention softmax.** Exact softmax in float64 over the visible keys:
  - `s = q̂·k̂·128^-½ + bias`, where `bias` is the stored bf16 relative bias;
  - `lse` and `P` stay in float64, with no bf16 P;
  - `o = P·v̂`;
  - `out = bf16(o · σ(gate) · σ(lse − sink))`, rounded once, as in the §3.11 formula.

  *Why:* HF (bf16 P, bf16 `o` before the gate) and V (FA4 with bf16 P) both approximate this limit. The engine's
  fixed-split online softmax is the engine's own pinned reference and is checked against the oracle by tolerance.
- **U6: unembed output.** Take V. `z = fp32(h · Uᵀ)` (float64 accumulate, one fp32 rounding), then
  `logits = fp32(30 · tanh(z / 30))`, then the slice to 200058. *Why:* PORT-SPEC recommends V, and the checkpoint
  comment says "BF16 with an FP32 output". HF's bf16 rounding of the `lm_head` output creates argmax ties.
- **U7: vision padding.** The raw 0–255 image is padded with −1, then rescaled and normalized like real pixels; the
  padding is not masked. HF does exactly this (`image_processing_inkling2.py@fc00b5b3`, lines 88–91:
  `pad(..., fill_value=-1.0)` followed by `rescale_and_normalize`), and so does V (PORT-SPEC §5). The oracle is text-only
  and refuses image placeholder ids (200054). This entry fixes the semantics for the vision tower.
- **U8: drafter local window.** Take V's `Inkling2DraftSelfAttention` (`vllm/models/inkling2/nvidia/dspark.py@1448919f`,
  lines 173–174: `window = sliding_window_size + block`):
  - a local draft layer at block start p sees ctx keys at positions `[p − 1024, p − 1]` plus the 8 block keys
    `p .. p+7`;
  - the block's K/V are written at `p .. p+7`, so the anchor's own ctx features are never visible;
  - the global draft layer (2) sees every ctx key `< p` plus the block.

  The oracle implements the drafter (`oracle/dspark_ref.py`) with exactly this window; its rounding points are listed
  under "Drafter rounding points" below.
- **U9: effort text.** The pack's `chat_template.jinja` is authoritative. It renders
  `Thinking effort level: ` followed by `0` when the effort is 0.0, else Jinja's `{{ num }}` (Python float repr: `0.1`,
  `0.2`, `0.7`, `0.9`, `0.99`). For the template's map values this equals V's `%g`. Fixtures pin token ids.
  `oracle/prompts.jsonl` holds strings already rendered this way, and the pack tokenizer reads the special-token
  strings in them.
- **U10: AttnRes query pre-scale.** Take V. `q' = fp32(q · fp32(4608^-½))` once at load. Then `ℓ_n = (q'·x_n)·ρ_n` in
  float64. *Why:* PORT-SPEC recommends V. It removes one rounding from every mix.
- **U11: expert input.** The bf16 `mlp_in` (AttnRes mix fused with `mlp_norm`) enters each EXL3 layer unchanged.
  The layer computes `((x ⊙ suh) H_K) W_q H_N ⊙ svh` in float64, the form of gb-exl3's `forward`, with `W_q` the exact
  fp16 codebook values. `g` and `u` are rounded to fp32; the activation is rounded to bf16 before the down projection.
  *Why:* the oracle defines the EXL3 math exactly. REXL3's kernels rotate the input and convert it to fp16. For a
  normalized bf16 input that conversion is exact except for values below fp16's normal range. The fp16
  intermediates are within tolerance. OX's choice is unverified (its code was not read).
- **U12: prefix cache and conv state.** The oracle's state at position p consists of:
  - K/V of every position `< p`; local layers need only the last 1023;
  - each layer's 4 conv streams' raw inputs at `p−3 .. p−1`.

  Any p is a valid split point; the reference has no alignment requirement, and the tests split at every position. An
  engine prefix cache must restore exactly this state. If it can only snapshot conv state at aligned positions, it
  must recompute from the last aligned position.
- **U13: drafter features under TP.** The hidden stream is replicated on every rank; there is no sequence parallelism.
  The 8 features are the full rows of that stream: P entering layers 5, 11, …, 41, plus the final-norm output h.
  `Inkling2Ref.drafter_features` exports them; fixtures carry them as `hidden[l]` and `h_final`.
- **U14: greedy argmax across vocab shards.** Take the first maximum by global vocabulary index (the lowest index among
  equal maxima) over `[0, 200058)`, with NaN treated as −inf. A sharded argmax that merges per-rank
  `(max, lowest index)` pairs by value, then by lowest global index, is identical. V's `dspark_argmax` documents
  "First argmax (NaN as -inf) … gathered" (`ops/dspark.py@1448919f`). The same rule applies to target greedy decoding
  and the drafter.

## Other fixed choices

- **Router (§3.13):**
  - logits are `fp32(x · W_gateᵀ)`;
  - selection uses `σ(logit) + bias` in float64, top-8 by a stable sort, so ties go to the lower expert index;
  - weights are `fp32(8 · σ_j / (Σ σ + 1e-20))` (HF's ε).

  Fixtures record the 8th-minus-9th selection margin so that a routing flip at a near-tie can be told apart from a
  bug.
- **TP emulation:** the default `tp = 4` (`gen_fixtures.py --tp`).
  - Rank r holds q heads `[12r, 12r+12)` (the `wo` column slice) and experts with `E mod 4 == r`.
  - Each rank's partial is rounded to bf16, the partials are summed in rank order in fp32, and the sum is rounded to
    bf16 (§2).
  - `tp = 1` gives the single-device model. Without rounding the two are the same function (equal to 1e-10 in the
    tests); with rounding they differ only by where the bf16 stores fall.
- **Log scaling τ** is an fp32 value: `fp32(1 + 0.1 · ln(max((pos+1)/128000, 1)))`. It is applied after the clip and
  round: `bf16(q̂ · τ)`, `bf16(r̂ · τ)`.
- **l2 clip** `x · min(c / (‖x‖₂ + 1e-6), 1)`, and the value norm `x · rsqrt(mean(x²) + 1e-6)`, are both computed in
  float64 and stored as bf16.

## Rounding points (in forward order)

| Value | Rounding |
|---|---|
| `embed_norm(embed[id])` | bf16 |
| AttnRes mix + following RMSNorm (`attn_norm`, `q/k/v_input_norm`, `mlp_norm`, `norm`) | bf16 (mix in float64) |
| q, r, gate, sink, k_raw, v_raw, dynamic-conv `dyn` | bf16 |
| k/v conv output, k̂ (clip), v̂ (norm), q̂, r̂ (scaled clip), τ-scaled q̂/r̂ | bf16 each |
| relative bias table `r̂ · rel_proj` | bf16 |
| attention output `o·σ(gate)·σ(lse−sink)` | bf16 |
| `wo` rank partial / all-reduce | bf16 / fp32 rank-order sum, then bf16 |
| `attn_sconv` output a | bf16 |
| `P + a` (non-boundary layers) | fp32 |
| router logits / weights | fp32 / fp32 |
| expert g, u / activation | fp32 / bf16 |
| MoE rank partial / all-reduce | bf16 / fp32 rank-order sum, then bf16 |
| `mlp_sconv` output | bf16 |
| new P = `(P + a) + m` | fp32, then bf16 |
| unembed z / soft-capped logits | fp32 / fp32 |

## Drafter rounding points

`oracle/dspark_ref.py` (PORT-SPEC §4, vLLM `@1448919f`). Everything else is float64 from the stored values.

- `alpha = fp32(softmax(fuse_weight))`; feature mix + `dflare_norm`: bf16 (Σ in float64).
- Context K/V: `k_raw`, `v` bf16; `k_norm_ctx` output bf16; RoPE output bf16.
- RoPE table: `inv_freq = fp32(1 / fp32(θ^(2i/D)))`, angle `fp32(pos · inv_freq)`, `fp32(cos)`, `fp32(sin)`.
- Block: `embed_norm(embed[anchor])` and `input_layernorm` (layer 0) bf16; q, k, v bf16; q/k norm bf16; RoPE bf16;
  attention output bf16 (exact float64 softmax, non-causal over ctx window + block).
- `o_proj` / `down_proj` rank partials bf16; rank-order fp32 sum; residual `bf16(h + sum)`; the following norm
  (`post_attention_layernorm`, next `input_layernorm`, `norm`) reads that bf16 residual and is stored bf16.
- MLP: g, u bf16; `silu(g) · u` bf16.
- Markov head: `fold_W`, `fold_b` bf16; `ctx_i` fp32 (fp32 product, then + bias in fp32); base logits bf16; per step
  `raw` fp32 (fp32 product, then + `ctx_i`); `bf16(raw)` → σ / tanh each bf16; `g·state`, `1 − g`, `(1 − g)·c` and
  the new state bf16; `latent = bf16(tanh(o))`; bias bf16; `bf16(base + bias)`, then the softcapped logit bf16.
- Draft: first maximum over `[0, 200058)`, NaN as −inf (U14).
