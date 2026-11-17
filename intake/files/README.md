---
license: apache-2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0
base_model: thinkingmachines/Inkling-2-Small
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: exllamav3
tags:
- exl3
- moe
- quantized
- conversational
---

# Inkling-2-Small EXL3 4.0 bpw (experts only, mcg)

An [EXL3](https://github.com/turboderp-org/exllamav3) quantization of [thinkingmachines/Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small) (594B total, 17B active). Only the routed MoE experts are quantized. Every other tensor is copied unchanged from the source checkpoint: attention, shared weights, router, embeddings, vision tower and the MTP head.

| | |
|---|---|
| Routed experts | EXL3 trellis, **mcg codebook, uniform 4.0 bpw** (every layer, every projection) |
| Everything else | Unchanged from source (BF16) |
| MTP / speculative head | Included unchanged (`mtp.safetensors`) |
| Total size | **313 GB**, against 393.5 GB for the source |

This quant uses a uniform bit width and the mcg codebook on purpose. The layout follows the requirements of [Local Inference Lab](https://github.com/local-inference-lab)'s fused Trellis/EXL3 MoE kernels ([b12x](https://github.com/local-inference-lab/b12x), now part of [FlashInfer](https://github.com/flashinfer-ai/flashinfer/tree/main/flashinfer/experimental/b12x)). Those kernels expect the mcg codebook at a single uniform bit width per layer. For a smaller mixed-width file, see [Inkling-2-Small-EXL3-3.48bpw](https://huggingface.co/0xSero/Inkling-2-Small-EXL3-3.48bpw). This one is higher quality, as the table below shows.

> **Runs today:** an inference engine for this quant is at [github.com/0xSero/inkling2-exl3](https://github.com/0xSero/inkling2-exl3). It has tensor/expert parallelism, [Local Inference Lab](https://github.com/local-inference-lab)'s b12x fused MoE kernels, a rotated 4-bit KV cache and always-on DSpark speculative decoding, with a one-command Docker image. Stock exllamav3 and TabbyAPI do not support Inkling-2 yet.

## Quality

The table reports KL divergence and next-token top-1 agreement against the source checkpoint. The source's NVFP4/MXFP8 experts were dequantized to BF16 weights for the comparison. Everything was measured on a 218K-token held-out set that does not overlap the calibration data. Lower KLD is better.

| Domain | Tokens | KLD (nats) | Top-1 agree | PPL source → quant |
|---|---:|---:|---:|---:|
| Agentic (multi-turn tool use) | 30,695 | 0.0507 | 96.84% | 3.042 → 3.041 |
| Tool calling | 18,420 | 0.0282 | 96.99% | 3.920 → 3.927 |
| Code | 15,388 | 0.0067 | 98.13% | 2.692 → 2.698 |
| Thinking / reasoning | 18,238 | 0.0041 | 97.20% | 3.204 → 3.199 |
| Chat | 7,019 | 0.0062 | 97.35% | 4.573 → 4.575 |
| Multilingual | 17,507 | 0.0187 | 96.08% | 5.708 → 5.715 |
| Wiki / general | 12,909 | 0.0172 | 96.35% | 5.578 → 5.594 |
| Long context (32K rows) | 98,301 | 0.0184 | 97.83% | 2.195 → 2.198 |
| **All** | **218,477** | **0.0213** | **97.35%** | **2.951 → 2.954 (+0.1%)** |

On the same eval, the 3.48 bpw mixed quant scores KLD 0.0240 and 96.82% top-1.

Most of the measured KLD is a numerical floor rather than quantization damage:
- Running the source model with the NVFP4/MXFP8 activation quantization it was designed for shifts it by a similar amount relative to the BF16 reference (KLD ≈ 0.037).
- Compare quants on the same eval protocol; numbers from different eval sets are not comparable.

The per-layer reconstruction error is in `quantization_details.json`.

## How it was made

- **Source:** the official checkpoint, where layers 0–39 experts are NVFP4 and layers 40–47 are MXFP8. Both were dequantized exactly, bit-checked against an independent decoder.
- **Calibration:** 4.17M tokens in 916 rows.
  - About 60% is real multi-turn agentic and coding sessions with tool use.
  - The rest is multilingual text, Wikipedia, reasoning, math, code, structured and function-calling data, and chat.
  - 12% of tokens are in 16K–32K long-context rows.
  - All secrets were redacted before use.
- **Procedure:** layer by layer on the quantized model's own activations, so each layer is calibrated on inputs that already include upstream quantization error. Each expert's Hessian comes only from the tokens the router actually sends to it, which averages about 80K tokens per expert. Rarely routed experts are shrunk toward the layer-wide covariance.
- **Model math:** verified against the reference implementation to about 1e-7 relative error in FP32, with 100% router top-8 agreement.

## Running it

On 4× 96 GB SM120 GPUs (RTX PRO 6000):

```bash
docker run --rm --gpus all --ipc=host -p 8000:8000 -v $HOME/models:/models ghcr.io/0xsero/inkling2-exl3:latest serve
```

This downloads the model on first run and serves an OpenAI-compatible API on port 8000. It runs 16 concurrent sessions at a 380K context each. Speculative decoding (the included DSpark/MTP head) is always on.

Measured 2026-11-15 on 4× RTX PRO 6000 at 300 W, served through the image above (speculative decoding on, temperature 0).

**Prefill** ([LIL-bench](https://github.com/local-inference-lab) `--prefill-only`):

| Prompt | 8K | 64K | 128K |
|---|---:|---:|---:|
| tok/s | 8,150 | 7,716 | 7,016 |
| previous image (11-14) | 4,429 | 3,945 | 3,547 |

A ~300K-token prompt reaches its first token in 42 s.

**Decode**, served rate with prefill excluded, on 16 realistic chat/code/agentic prompts:

| Concurrency | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| tok/s | 224 | 315 | 417 | 523 | 676 |

Prefill is token-sharded: each GPU owns a quarter of a chunk's rows, and activations move between GPUs with copy-engine ring collectives (design after NVIDIA Hyperflare's Inkling-2 decoder). The routed experts run on [Local Inference Lab](https://github.com/local-inference-lab)'s b12x fused Trellis MoE with expert parallelism over 2 GPU pairs; the draft head is FP8 with draft length tuned per concurrency. All 4 GPUs sit at their 300 W cap. Optimization is ongoing; details and the full ledger are at [github.com/0xSero/inkling2-exl3](https://github.com/0xSero/inkling2-exl3).

**KV cache:** 6.46M tokens in total, enough for 16 sessions at 380K. Global-layer KV is a Hadamard-rotated 4-bit format. Against 8-bit KV, it moves served-engine KLD from 0.0135 to 0.0170 on short rows and from 0.0195 to 0.0228 on 32K rows. `--bits-g 8` gives higher fidelity at about 60% of the capacity. Every release passes a coherence gate (math, JSON, French, tool round trip, code run against hidden tests, long prose without loops, 16 parallel sessions, needles at 32K / 128K / ~300K).

## Hardware

The weights are sized for 4× 96 GB GPUs. Only the 8 global-attention layers need a full-length KV cache; the 40 sliding-window layers keep a 1024-token ring. That leaves room for long contexts at high concurrency.

## Credits

- [Thinking Machines](https://huggingface.co/thinkingmachines) for Inkling-2-Small (Apache-2.0).
- [turboderp](https://github.com/turboderp-org/exllamav3) for EXL3 / exllamav3.
- [Local Inference Lab](https://github.com/local-inference-lab) for the b12x fused Trellis MoE kernels (now in FlashInfer) and their EXL3 serving recipes, which set this quant's codebook and bit layout.
