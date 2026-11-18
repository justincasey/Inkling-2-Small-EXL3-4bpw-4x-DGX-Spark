# Fit — 0xSero/Inkling-2-Small-EXL3-4bpw on four Sparks (TP4)

Arithmetic from `intake/census.json`, `config/pack-ranks.tsv` and the `mtp.safetensors` header. No engine number is in
this file; the first boot's capacity line and `MemAvailable` replace every estimate here.

## Pack

311.76 GB (290.3 GiB) in 227 files (`intake/manifest.tsv`). Each rank stages 91.02 GB: its 48 expert files plus 35
shared files (`config/pack-ranks.tsv`).

| Group | GiB (whole pack) | Placement |
|---|---:|---|
| routed experts, EXL3 mcg 4.0 bpw (384 × 48 layers) | 274.09 | rank r owns experts E mod 4 == r |
| attention, BF16 (wq/wk/wv/wr/wo/gate_attn/sink per layer) | 8.78 | split by heads: 12 query / 2 KV per rank |
| embed + unembed, BF16 (201,024 × 4,608 each) | 3.45 | embed on every rank; unembed split by vocab rows |
| vision tower | 0.61 | rank 0 |
| routers and the rest | 0.16 | every rank |
| DSpark drafter (`mtp.safetensors`, 5 layers) | 3.19 | split by heads / FFN columns |

## Per-rank weights

| Component | GiB |
|---|---:|
| routed experts (274.09 / 4) | 68.52 |
| attention (8.78 / 4) | 2.20 |
| embed (whole) | 1.73 |
| unembed (1.73 / 4) | 0.43 |
| routers and the rest | 0.16 |
| drafter (3.19 / 4) | 0.80 |
| vision (rank 0 only) | 0.61 |
| **total, rank 0** | **74.45** |

## KV per token

Only the 8 full-attention layers (5, 11, …, 47) and the drafter's full-attention layer 2 grow with context; the 40
sliding-window layers and drafter layers 0, 1, 3, 4 keep a fixed 1,024-token ring. To be confirmed by
`docs/PORT-SPEC.md` (relative-position logits are expected to add no cached state).

| Part | BF16 per token, whole cluster | per rank (2 of 8 KV heads) | 1,048,576-token lane, per rank |
|---|---:|---:|---:|
| 8 target full-attention layers | 32 KiB | 8 KiB | 8.00 GiB |
| drafter layer 2 | 4 KiB | 1 KiB | 1.00 GiB |
| sliding-window rings (40 + 4 layers) | fixed | fixed | 0.04 GiB |
| **total** | **36 KiB** | **9 KiB** | **9.04 GiB** |

## KV budget per rank

| Item | GiB |
|---|---:|
| `MemTotal` (measured, .121) | 121.69 |
| OS, CUDA context, unaccounted (CYBERFROST: 121.7 − 97 allocated − 13.5 free) | ~11 |
| `MemAvailable` floor kept under load | ~10 |
| engine budget | ~100.7 |
| weights, rank 0 | 74.45 |
| activations, graphs, scratch | ~6 |
| NCCL buffers | ~1 |
| **KV pool** | **~19** (≈23 with a ~6 GiB floor) |

| KV format | per 1M lane per rank | full 1M lanes at 19 / 23 GiB |
|---|---:|---|
| BF16 | 9.04 GiB | 2 / 2 |
| **int8 + per (token, KV head) scale (chosen, `docs/DECISIONS.md`)** | **~4.6 GiB** | **4 / 5** |
| 4-bit (dsv41 q4 style) + scales | ~2.4 GiB | 7 / 9 |

## Profiles (all `PROFILE_MEASURED=0`)

| Profile | Window × lanes | KV per rank | Status |
|---|---|---:|---|
| bootstrap | 1,048,576 × 4, int8 | ~18.4 GiB | unmeasured; first boot sets the real pool from the engine's capacity line |
