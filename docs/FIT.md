# Fit — 0xSero/Inkling-2-Small-EXL3-4bpw on four Sparks (TP4)

Arithmetic from `intake/census.json`, `config/pack-ranks.tsv` and the `mtp.safetensors` header, written before the
first boot. The measured section at the end (capacity line and `MemAvailable` of the recorded boots) supersedes it.

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

## Measured (`bench/records/2026-11-18-inkling2-final-base/`, engine `f4e27ca`)

Capacity line of rank 0 (all ranks plan alike):

```
slots=4 max_context=1048576 kv_pages=16388 kv_tokens=4194304 kv_gib=18.15 lanes_at_max_context=4 weights_gib=73.94
free_after_gib=41.00 floor_gib=8.00 reserve_gib=6.00 state_mib_per_slot=67.7 kv_bytes_per_token=4644 kv_dtype=int8
drafter=dspark:plane=Int8 prefix_entries=16 prefix_pages=4781 prefix_gib=6.03 prefix_reach_tokens=1223936
```

| Item | Estimate above | Measured |
|---|---:|---:|
| weights per rank | 74.45 GiB | 73.94 GiB |
| KV per token per rank | 9 KiB bf16 / ~4.6 KiB int8 | 4,644 B (int8 target 4,128 + int8 draft plane 516) |
| 4 x 1M lanes | ~18.4 GiB | 18.15 GiB |
| per-slot state (rings, conv, drafter) | — | 67.7 MiB |
| prefix cache | — | 6.03 GiB, 16 entries |
| MemAvailable minimum under load | ~10 GiB floor | `memwatch-*.txt` in the record (9.9-12.1 GiB) |

## Profiles

| Profile | Window × lanes | KV per rank | Status |
|---|---|---:|---|
| default (`.env.example`) | 1,048,576 × 4, int8, DSpark, prefix cache, vision | 18.15 GiB | measured, record above |
